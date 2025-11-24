from os import path as osp
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from PIL import Image
import pandas as pd
import random

from . import augmentation
from .masking import MaskGenerator
from . import data_utils as utils


def prompt_generator(
    prompts: dict[str, str],
    text_prompt: float,
    visual_prompt: float,
    apprearance: float,
    position: float,
    action: float,
    pose: float,
    box: float,
    point: float,
) -> str:
    """
    Build a mixed prompt string that may contain text, visual cues, or both.
    - Skips empty/None/"none" entries.
    - Randomly removes at least one available item so we never expose all info.
    """

    def _is_valid(value: Optional[str]) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped or stripped.lower() == "none":
                return False
        return True

    def _sample_partial(items: list[str]) -> list[str]:
        if len(items) <= 1:
            return items[:]
        keep = random.randint(1, len(items) - 1)
        return random.sample(items, keep)

    selected_prompts: list[str] = []

    text_candidates: list[str] = []
    if random.random() < apprearance and _is_valid(prompts.get("apprearance")):
        text_candidates.append(f"apprearance: {prompts['apprearance']}")
    if random.random() < position and _is_valid(prompts.get("position")):
        text_candidates.append(f"position: {prompts['position']}")
    if random.random() < action and _is_valid(prompts.get("action")):
        text_candidates.append(f"action: {prompts['action']}")
    if random.random() < pose and _is_valid(prompts.get("pose")):
        text_candidates.append(f"pose: {prompts['pose']}")

    visual_candidates: list[str] = []
    if random.random() < box and _is_valid(prompts.get("box")):
        visual_candidates.append(f"visual box: {prompts['box']}")
    if random.random() < point and _is_valid(prompts.get("point")):
        visual_candidates.append(f"visual point: {prompts['point']}")

    if text_candidates and random.random() < text_prompt:
        selected_prompts.extend(_sample_partial(text_candidates))

    if visual_candidates and random.random() < visual_prompt:
        selected_prompts.extend(_sample_partial(visual_candidates))

    if not selected_prompts:
        fallback_pool = []
        if text_candidates:
            fallback_pool.append(random.choice(text_candidates))
        if visual_candidates:
            fallback_pool.append(random.choice(visual_candidates))
        if fallback_pool:
            selected_prompts.append(random.choice(fallback_pool))

    random.shuffle(selected_prompts)
    return "; ".join(selected_prompts)


def _sample_point_in_box(
    x_min: float, y_min: float, x_max: float, y_max: float
) -> tuple[float, float]:
    """
    Draw a normalized point inside the bbox, biased toward the center.
    """
    width = max(x_max - x_min, 1e-6)
    height = max(y_max - y_min, 1e-6)
    u = random.betavariate(2, 2)  # center-heavy distribution
    v = random.betavariate(2, 2)
    return x_min + u * width, y_min + v * height


class AnyGazeDataset(Dataset):
    def __init__(
        self,
        image_root: str,
        anno_root: str,
        transform: Callable,
        input_size: int,
        output_size: int,
        quant_labelmap: bool = True,
        is_train: bool = True,
        *,
        mask_generator: Optional[MaskGenerator] = None,
        bbox_jitter: float = 0.0,
        rand_crop: float = 0.0,
        rand_flip: float = 0.0,
        color_jitter: float = 0.0,
        rand_rotate: float = 0.0,
        rand_lsj: float = 0.0,
        visual_text_ratio: float = 0.5,
    ):
        if is_train:
            column_names = [
                "path",
                "idx",
                "gaze_x",
                "gaze_y",
                "head_x_min",
                "head_y_min",
                "head_x_max",
                "head_y_max",
                "inout",
                "source",
                "meta0",
                "meta1",
                "apprearance",
                "position",
                "action",
                "pose",
            ]
            df = pd.read_csv(
                anno_root,
                sep=";",
                names=column_names,
                index_col=False,
                encoding="utf-8-sig",
            )
            df = df.sample(frac=1).reset_index(drop=True)  # shuffle the data
            
            # df = df[
            #     df["inout"] != -1
            # ]  # only use "in" or "out "gaze. (-1 is invalid, 0 is out gaze)
            df.reset_index(inplace=True)
            self.y_train = df[
                [
                    "head_x_min",
                    "head_y_min",
                    "head_x_max",
                    "head_y_max",
                    "gaze_x",
                    "gaze_y",
                    "inout",
                    "apprearance",
                    "position",
                    "action",
                    "pose",
                ]
            ]
            self.X_train = df["path"]
            self.length = len(df)
        else:
            column_names = [
                "path",
                "idx",
                "gaze_x",
                "gaze_y",
                "head_x_min",
                "head_y_min",
                "head_x_max",
                "head_y_max",
                "inout",
                "source",
                "meta0",
                "meta1",
                "apprearance",
                "position",
                "action",
                "pose",
            ]
            df = pd.read_csv(
                anno_root,
                sep=";",
                names=column_names,
                index_col=False,
                encoding="utf-8-sig",
            )
            df = df[
                [
                    "path",
                    "gaze_x",
                    "gaze_y",
                    "head_x_min",
                    "head_y_min",
                    "head_x_max",
                    "head_y_max",
                    "inout",
                    "apprearance",
                    "position",
                    "action",
                    "pose",
                ]
            ].groupby(["path", "head_x_min"])
            self.keys = list(df.groups.keys())
            self.X_test = df
            self.length = len(self.keys)

        self.data_dir = image_root
        self.transform = transform
        self.is_train = is_train

        self.input_size = input_size
        self.output_size = output_size
        self.visual_text_ratio = visual_text_ratio
        
        self.draw_labelmap = (
            utils.draw_labelmap if quant_labelmap else utils.draw_labelmap_no_quant
        )

        if self.is_train:
            ## data augmentation
            self.augment = augmentation.AugmentationList(
                [
                    augmentation.ColorJitter(color_jitter),
                    augmentation.BoxJitter(bbox_jitter),
                    augmentation.RandomCrop(rand_crop),
                    augmentation.RandomFlip(rand_flip),
                    augmentation.RandomRotate(rand_rotate),
                    augmentation.RandomLSJ(rand_lsj),
                ]
            )

            self.mask_generator = mask_generator

    def __getitem__(self, index):
        if not self.is_train:
            g = self.X_test.get_group(self.keys[index])
            cont_gaze = []
            for _, row in g.iterrows():
                path = row["path"]
                x_min = row["head_x_min"]
                y_min = row["head_y_min"]
                x_max = row["head_x_max"]
                y_max = row["head_y_max"]
                gaze_x = row["gaze_x"]
                gaze_y = row["gaze_y"]
                inout = row["inout"]
                apprearance = row["apprearance"]
                position = row["position"]
                action = row["action"]
                pose = row["pose"]
                cont_gaze.append(
                    [float(gaze_x), float(gaze_y)]
                )  # all ground truth gaze are stacked up
            for _ in range(len(cont_gaze), 20):
                cont_gaze.append(
                    [-1, -1]
                )  # pad dummy gaze to match size for batch processing
            cont_gaze = torch.FloatTensor(cont_gaze)
            gaze_inside = bool(inout)
            # gaze_inside = True  # always consider test samples as inside
        else:
            path = self.X_train.iloc[index]
            (
                x_min,
                y_min,
                x_max,
                y_max,
                gaze_x,
                gaze_y,
                inout,
                apprearance,
                position,
                action,
                pose
            ) = self.y_train.iloc[index]
            gaze_inside = bool(inout)

        img = Image.open(osp.join(self.data_dir, path))
        img = img.convert("RGB")
        width, height = img.size
        x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])
        if x_max < x_min:
            x_min, x_max = x_max, x_min
        if y_max < y_min:
            y_min, y_max = y_max, y_min
        # expand face bbox a bit
        k = 0.1
        x_min = max(x_min - k * abs(x_max - x_min), 0)
        y_min = max(y_min - k * abs(y_max - y_min), 0)
        x_max = min(x_max + k * abs(x_max - x_min), width - 1)
        y_max = min(y_max + k * abs(y_max - y_min), height - 1)

        if self.is_train:
            img, bbox, gaze, size = self.augment(
                img,
                (x_min, y_min, x_max, y_max),
                (gaze_x, gaze_y),
                (width, height),
            )
            x_min, y_min, x_max, y_max = bbox
            gaze_x, gaze_y = gaze
            width, height = size
            # center_rate = random.choice([2,3,4])
            # x_center_norm = round((x_min + x_max) / (2 * width), center_rate)
            # y_center_norm = round((y_min + y_max) / (2 * height), center_rate)
            x_min_norm = x_min / width
            y_min_norm = y_min / height
            x_max_norm = x_max / width
            y_max_norm = y_max / height
            box_prompt = f"{x_min_norm:.4f},{y_min_norm:.4f},{x_max_norm:.4f},{y_max_norm:.4f}"
            point_x, point_y = _sample_point_in_box(
                x_min_norm, y_min_norm, x_max_norm, y_max_norm
            )
            point_prompt = f"{point_x:.4f},{point_y:.4f}"
            text = prompt_generator(
                {
                    "apprearance": apprearance,
                    "position": position,
                    "action": action,
                    "pose": pose,
                    "box": box_prompt,
                    "point": point_prompt,
                },
                text_prompt=0.8,
                visual_prompt=0.2,
                apprearance=1.0,
                position=0.5,
                action=0.5,
                pose=0.5,
                box=0.5,
                point=0.5,
            )
        else:
            x_min_norm = x_min / width
            y_min_norm = y_min / height
            x_max_norm = x_max / width
            y_max_norm = y_max / height
            box_prompt = f"{x_min_norm:.4f},{y_min_norm:.4f},{x_max_norm:.4f},{y_max_norm:.4f}"
            point_x, point_y = _sample_point_in_box(
                x_min_norm, y_min_norm, x_max_norm, y_max_norm
            )
            point_prompt = f"{point_x:.4f},{point_y:.4f}"
            text = prompt_generator(
                {
                    "apprearance": apprearance,
                    "position": position,
                    "action": action,
                    "pose": pose,
                    "box": box_prompt,
                    "point": point_prompt,
                },
                text_prompt=1.0,
                visual_prompt=0.0,
                apprearance=1.0,
                position=0.0,
                action=0.0,
                pose=0.0,
                box=0.0,
                point=0.0,
            )

        head_channel = utils.get_head_box_channel(
            x_min,
            y_min,
            x_max,
            y_max,
            width,
            height,
            resolution=self.input_size,
            coordconv=False,
        ).unsqueeze(0)

        if self.is_train and self.mask_generator is not None:
            image_mask = self.mask_generator(
                x_min / width,
                y_min / height,
                x_max / width,
                y_max / height,
                head_channel,
            )

        if self.transform is not None:
            img = self.transform(img)

        # generate the heat map used for deconv prediction
        gaze_heatmap = torch.zeros(
            self.output_size, self.output_size
        )  # set the size of the output
        if not self.is_train:  # aggregated heatmap
            num_valid = 0
            for gaze_x, gaze_y in cont_gaze:
                if gaze_x != -1:
                    num_valid += 1
                    gaze_heatmap += self.draw_labelmap(
                        torch.zeros(self.output_size, self.output_size),
                        [gaze_x * self.output_size, gaze_y * self.output_size],
                        3,
                        type="Gaussian",
                    )
            gaze_heatmap /= num_valid
        else:
            if gaze_inside:
                gaze_heatmap = self.draw_labelmap(
                    gaze_heatmap,
                    [gaze_x * self.output_size, gaze_y * self.output_size],
                    3,
                    type="Gaussian",
                )

        imsize = torch.IntTensor([width, height])

        if self.is_train:
            out_dict = {
                "images": img,
                "heatmaps": gaze_heatmap,
                "gazes": torch.FloatTensor([gaze_x, gaze_y]),
                "bbox": torch.FloatTensor([(x_max + x_min) / (2 * width), (y_max + y_min) / (2 * height), (x_max-x_min) / width, (y_max-y_min) / height]),
                "gaze_inouts": torch.FloatTensor([gaze_inside]),
                "imsize": imsize,
                "image_path": path,
                "texts": text,
            }
            if self.mask_generator is not None:
                out_dict["image_masks"] = image_mask
            return out_dict
        else:
            return {
                "images": img,
                "heatmaps": gaze_heatmap,
                "gazes": cont_gaze,
                "bbox": torch.FloatTensor([(x_max + x_min) / (2 * width), (y_max + y_min) / (2 * height), (x_max-x_min) / width, (y_max-y_min) / height]),
                "bbox_raw": torch.FloatTensor([x_min, y_min, x_max, y_max]),
                "gaze_inouts": torch.FloatTensor([gaze_inside]),
                "imsize": imsize,
                "image_path": path,
                "texts": text,
            }

    def __len__(self):
        return self.length
