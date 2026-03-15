from os import path as osp
from typing import Literal

from omegaconf import OmegaConf
from detectron2.config import LazyCall as L
from detectron2.config import instantiate
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data import *


data_info = OmegaConf.create()

data_info.gazeanywhere_dataset = OmegaConf.create()

data_info.input_size = 512
data_info.output_size = 64
data_info.quant_labelmap = True
data_info.mean = (0.485, 0.456, 0.406)
data_info.std = (0.229, 0.224, 0.225)
data_info.bbox_jitter = 0.5
data_info.rand_crop = 0.5
data_info.rand_flip = 0.5
data_info.color_jitter = 0.5
data_info.rand_rotate = 0.0
data_info.rand_lsj = 0.0

data_info.mask_size = 24
data_info.mask_scene = False
data_info.mask_head = False
data_info.max_scene_patches_ratio = 0.5
data_info.max_head_patches_ratio = 0.3
data_info.mask_prob = 0.2

# Dataloader(gazeanywhere_dataset, train/val)
def __build_dataloader(
    name: Literal[
        "gazeanywhere_dataset"
    ],
    is_train: bool,
    batch_size: int = 64,
    num_workers: int = 14,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    drop_last: bool = True,
    distributed: bool = False,
    **kwargs,
):
    assert name in [
        "gazeanywhere_dataset"
    ], f'{name} not in ("gazeanywhere_dataset")'

    for k, v in kwargs.items():
        if k in ["train_root", "train_anno", "val_root", "val_anno", "head_root"]:
            data_info[name][k] = v
        else:
            data_info[k] = v

    datasets = {
        "gazeanywhere_dataset": GazeAnywhereDataset,
    }
    dataset = L(datasets[name])(
        json_file=data_info[name]["train_anno" if is_train else "val_anno"],
        root_dir=data_info[name]["train_root" if is_train else "val_root"], 
        transform=get_transform(
            input_resolution=data_info.input_size,
            mean=data_info.mean,
            std=data_info.std,
        ),
        input_size=data_info.input_size,
        output_size=data_info.output_size,
        quant_labelmap=data_info.quant_labelmap,
        is_train=is_train,
    )
    dataset = instantiate(dataset)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        sampler=DistributedSampler(dataset, shuffle=is_train) if distributed else None,
        drop_last=drop_last,
    )


dataloader = OmegaConf.create()
dataloader.gazeanywhere_dataset = OmegaConf.create()
dataloader.gazeanywhere_dataset.train = L(__build_dataloader)(
    name="gazeanywhere_dataset",
    is_train=True,
)
dataloader.gazeanywhere_dataset.val = L(__build_dataloader)(
    name="gazeanywhere_dataset",
    is_train=False,
)
