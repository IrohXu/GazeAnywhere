import torch
from torch import nn
from detectron2.utils.registry import Registry
from typing import Literal, List, Dict, Optional, OrderedDict
from timm.models.resnetv2 import Bottleneck

FUSION_REGISTRY = Registry("FUSION_REGISTRY")
FUSION_REGISTRY.__doc__ = "Registry for fusion module"

from .cross_attention import CrossAttentionBlock

@FUSION_REGISTRY.register()
class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        gesture_dim: int = 256,
        gaze_dim: int = 256,
        dim: int = 256,
        image_size: int = 512,
        patch_size: int = 16,
        num_layers: int = 3
    ) -> None:
        super().__init__()
        self.gesture_dim = gesture_dim
        self.gaze_dim = gaze_dim
        self.dim = dim
        self.mask_size = image_size // patch_size
        
        self.cross_fusion = nn.Sequential(*[CrossAttentionBlock(
            dim,
            num_heads=8,
            mlp_ratio=4,
            drop_path=0.1
        ) for i in range(num_layers-1)])
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.Conv2d(dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x_gesture, x_gaze):
        x_gesture, _ = self.cross_fusion((x_gesture, x_gaze))
        feats_gesture = x_gesture.reshape(x_gesture.shape[0], self.mask_size, self.mask_size, x_gesture.shape[2]).permute(0, 3, 1, 2) # b (h w) c -> b c h w  
        feats = self.decoder(feats_gesture)
        return feats


def build_fusion_module(name, *args, **kwargs):
    return FUSION_REGISTRY.get(name)(*args, **kwargs)


