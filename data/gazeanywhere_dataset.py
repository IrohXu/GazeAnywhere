import json
import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from . import data_utils as utils
from .augmentation import (
    AugmentationList,
    ColorJitter,
    RandomFlip,
    BoxJitter,
    RandomCrop,
    RandomRotate,
)


def _default_train_augmentations():
    """
    Geometric augmentations that keep (image, bbox, gaze) aligned.
    All run in PIL space on (image, bbox_pixel, gaze_norm, size); bbox/gaze are updated
    by each augmentation so they stay consistent with the transformed image.
    Applied before resize; heatmaps and head_channel are then built from the updated coords.
    """
    return AugmentationList([
        ColorJitter(p=0.5, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        RandomFlip(p=0.5),
        BoxJitter(p=0.5, expansion=0.2),
        RandomCrop(p=0.5),
        RandomRotate(p=0.3, max_angle=15),
    ])


class GazeAnywhereDataset(Dataset):
    def __init__(
        self, 
        json_file, 
        root_dir, 
        input_size=518, 
        output_size=64, 
        quant_labelmap=True, 
        is_train=True,
        transform=None,
        augmentations=None,
    ):
        self.root_dir = root_dir
        self.input_size = input_size
        self.output_size = output_size
        self.is_train = is_train
        self.transform = transform
        # Augmentations run on PIL (image, bbox_pixel, gaze_norm, size); only when is_train
        self.augmentations = augmentations if is_train else None
        if self.augmentations is None and is_train:
            self.augmentations = _default_train_augmentations()
        
        self.draw_labelmap = (
            utils.draw_labelmap if quant_labelmap else utils.draw_labelmap_no_quant
        )
        
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        print(f"Loading data from {json_file}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def _apply_augmentations(self, image_pil, bbox_pixel, gaze_norm, size):
        """Apply augmentation list in PIL space; returns (image, bbox_pixel, gaze_norm, size) all aligned."""
        if self.augmentations is None:
            return image_pil, bbox_pixel, gaze_norm, size
        return self.augmentations(image_pil, bbox_pixel, gaze_norm, size)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        paths = item['path']
        gazes = item['gaze']
        heads = item['head']
        inouts = item['inout']
        
        num_views = len(paths)
        view_indices = list(range(num_views))
        if self.is_train:
            valid_indices = [i for i in view_indices if len(heads[i]) > 0]
            view_idx = random.choice(valid_indices) if valid_indices else 0
        else:
            view_idx = 0
        
        rel_path = paths[view_idx]
        img_path = os.path.join(self.root_dir, rel_path)
        img_pil = Image.open(img_path).convert('RGB')
        original_w, original_h = img_pil.size
        size = (original_w, original_h)
        
        raw_gaze = gazes[view_idx]
        gaze_inside = bool(inouts[view_idx])
        gx_norm = raw_gaze[0] if gaze_inside else 0.5
        gy_norm = raw_gaze[1] if gaze_inside else 0.5
        gaze_norm = (gx_norm, gy_norm)
        
        raw_head = heads[view_idx]
        is_head_valid = len(raw_head) == 4
        if is_head_valid:
            hx1, hy1, hx2, hy2 = raw_head
            bbox_pixel = (
                hx1 * original_w, hy1 * original_h,
                hx2 * original_w, hy2 * original_h,
            )
        else:
            bbox_pixel = (0.0, 0.0, float(original_w), float(original_h))
        
        if self.is_train and self.augmentations is not None:
            img_pil, bbox_pixel, gaze_norm, size = self._apply_augmentations(
                img_pil, bbox_pixel, gaze_norm, size
            )
            aug_w, aug_h = size
            gx_norm, gy_norm = gaze_norm
            hx1 = bbox_pixel[0] / aug_w
            hy1 = bbox_pixel[1] / aug_h
            hx2 = bbox_pixel[2] / aug_w
            hy2 = bbox_pixel[3] / aug_h
        else:
            aug_w, aug_h = original_w, original_h
            if is_head_valid:
                hx1, hy1, hx2, hy2 = raw_head
            else:
                hx1 = hy1 = hx2 = hy2 = -1.0
        
        img_tensor = self.transform(img_pil)
        
        gaze_heatmap = torch.zeros(self.output_size, self.output_size)
        if gaze_inside:
            gaze_heatmap = self.draw_labelmap(
                gaze_heatmap,
                [gx_norm * self.output_size, gy_norm * self.output_size],
                3,
                type="Gaussian",
            )
            gx, gy = gx_norm, gy_norm
        else:
            gx, gy = -1.0, -1.0
        
        head_heatmap = torch.zeros(self.output_size, self.output_size)
        if is_head_valid and (hx1 >= 0 and hy1 >= 0):
            hcx = (hx1 + hx2) / 2.0
            hcy = (hy1 + hy2) / 2.0
            hw = hx2 - hx1
            hh = hy2 - hy1
            head_heatmap = self.draw_labelmap(
                head_heatmap,
                [hcx * self.output_size, hcy * self.output_size],
                3,
                type="Gaussian",
            )
            bbox_vec = torch.tensor([hcx, hcy, hw, hh], dtype=torch.float32)
            head_channel = utils.get_head_box_channel(
                bbox_pixel[0], bbox_pixel[1], bbox_pixel[2], bbox_pixel[3],
                aug_w, aug_h,
                resolution=self.input_size,
                coordconv=False,
            ).unsqueeze(0)
        else:
            bbox_vec = torch.tensor([-1.0, -1.0, -1.0, -1.0])
            head_channel = torch.zeros(1, self.input_size, self.input_size)
        
        if is_head_valid and hx1 >= 0:
            raw_head_tensor = torch.tensor([hx1, hy1, hx2, hy2], dtype=torch.float32)
        else:
            raw_head_tensor = torch.tensor([-1.0, -1.0, -1.0, -1.0])
        
        text_prompt = item.get('apprearance', "")
        # Use consistent dtypes for collate: idx/imsize as long to avoid "can't be cast to Long" in DataLoader
        out_dict = {
            "images": img_tensor,
            "head_channels": head_channel,
            "heatmaps": gaze_heatmap,
            "headmaps": head_heatmap,
            "gazes": torch.tensor([gx, gy], dtype=torch.float32),
            "bbox": bbox_vec,
            "bbox_raw": raw_head_tensor,
            "gaze_inouts": torch.tensor([gaze_inside], dtype=torch.float32),
            "idx": torch.tensor(int(item["idx"]), dtype=torch.long),
            "imsize": torch.tensor([int(aug_w), int(aug_h)], dtype=torch.long),
            "texts": text_prompt,
        }
        return out_dict
