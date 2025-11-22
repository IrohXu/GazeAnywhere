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


def random_prompt(
    attribute: str,
    position: str,
    action: str,
    pose: str,
    size_weights: dict[int, float] | None = None,
    rng: random.Random | None = None,
) -> str:
    """
    Build a randomized 'key: value' string using a random subset (size 1..4) and random order.
    - Automatically includes 1-item and 2-item cases.
    - Skips empty/None values.
    - `size_weights` lets you bias how often 1/2/3/4-item prompts appear.
      e.g., {1:1, 2:2, 3:3, 4:4} biases toward longer prompts.
    - Pass a custom `rng` for reproducibility (e.g., rng=random.Random(42)).
    """
    if rng is None:
        rng = random

    fields = {
        "attribute": attribute,
        "position": position,
        "action": action,
        "pose": pose,
    }

    # keep only non-empty values
    available_keys = [k for k, v in fields.items() if v not in (None, "", [])]
    if not available_keys:
        return ""

    # choose subset size (1..len(available))
    if size_weights is None:
        # neutral: all sizes equally likely
        size_weights = {1: 1, 2: 1, 3: 1, 4: 1}

    valid_sizes = [s for s in size_weights if 1 <= s <= len(available_keys)]
    weights = [size_weights[s] for s in valid_sizes]
    n = rng.choices(valid_sizes, weights=weights, k=1)[0]

    # pick random subset and shuffle order
    chosen = rng.sample(available_keys, n)
    rng.shuffle(chosen)

    # format
    parts = [f"{k}: {fields[k]}" for k in chosen]
    return "; ".join(parts)


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
                "attribute",
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
                    "attribute",
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
                "attribute",
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
                    "attribute",
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
                attribute = row["attribute"]
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
                attribute,
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
            if self.visual_text_ratio < random.random():
                text = random_prompt(
                    attribute,
                    position,
                    action,
                    pose,
                    size_weights = {1: 0, 2: 1, 3: 1, 4: 1}
                )
            else:
                center_rate = random.choice([2,3,4,5])
                x_center_norm = round((x_min + x_max) / (2 * width), center_rate)
                y_center_norm = round((y_min + y_max) / (2 * height), center_rate)
                text = "visual position: " + str(x_center_norm) + " " + str(y_center_norm)
        else:           
            text = random_prompt(
                attribute,
                position,
                action,
                pose,
                size_weights = {1: 0, 2: 0, 3: 0, 4: 1}
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
                "head_channels": head_channel,
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
                "head_channels": head_channel,
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
