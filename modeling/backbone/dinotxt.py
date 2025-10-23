# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Union

import torch
import torch.nn.functional as F
from torch import nn

from .src.dinov3.eval.text.text_tower import build_text_model
from .src.dinov3.eval.text.vision_tower import build_vision_model

from .src.dinov3.eval.text.dinotxt_model import DINOTxt, DINOTxtConfig
from .src.dinov3.eval.text.text_transformer import TextTransformer
from .src.dinov3.eval.text.tokenizer import get_tokenizer

from .backbones import dinov3_vitl16, Weights as BackboneWeights, convert_path_or_url_to_url

# from dinov3 import DinoVisionTransformer

bpe_path_or_url = "https://dl.fbaipublicfiles.com/dinov3/thirdparty/bpe_simple_vocab_16e6.txt.gz"

@dataclass
class DINOTxtConfig:
    embed_dim: int
    vision_backbone_config: str | None = None
    text_backbone_config: str | None = None
    vision_backbone_pretrained_weights: str | None = None
    text_backbone_pretrained_weights: str | None = None
    vision_model_freeze_backbone: bool = True
    vision_model_train_img_size: int = 224
    vision_model_use_class_token: bool = True
    vision_model_use_patch_tokens: bool = False
    vision_model_num_head_blocks: int = 0
    vision_model_head_blocks_drop_path: float = 0.3
    vision_model_use_linear_projection: bool = False
    vision_model_patch_tokens_pooler_type: str = "mean"
    vision_model_patch_token_layer: int = 1  # which layer to take patch tokens from
    # 1 - last layer, 2 - second last layer, etc.
    text_model_freeze_backbone: bool = False
    text_model_num_head_blocks: int = 0
    text_model_head_blocks_is_causal: bool = False
    text_model_head_blocks_drop_prob: float = 0.0
    text_model_tokens_pooler_type: str = "first"
    text_model_use_linear_projection: bool = False
    text_vocab_path_or_url: Optional[str] = None
    init_logit_scale: float = math.log(1 / 0.07)
    init_logit_bias: Optional[float] = None
    freeze_logit_scale: bool = False


class DINOTxt(nn.Module):
    def __init__(
        self,
        model_config: DINOTxtConfig,
        vision_backbone: Optional[nn.Module] = None,
        text_backbone: Optional[nn.Module] = None,
        device=None,
    ):
        super().__init__()
        self.model_config = model_config
        self.visual_model = build_vision_model(
            model_config.embed_dim,
            model_config.vision_backbone_config,
            model_config.vision_model_freeze_backbone,
            model_config.vision_model_num_head_blocks,
            model_config.vision_model_head_blocks_drop_path,
            model_config.vision_model_use_class_token,
            model_config.vision_model_use_patch_tokens,
            model_config.vision_model_patch_token_layer,
            model_config.vision_model_patch_tokens_pooler_type,
            model_config.vision_model_use_linear_projection,
            backbone=vision_backbone,
        )
        self.text_model = build_text_model(
            model_config.embed_dim,
            model_config.text_backbone_config,
            model_config.text_model_freeze_backbone,
            model_config.text_model_num_head_blocks,
            model_config.text_model_head_blocks_is_causal,
            model_config.text_model_head_blocks_drop_prob,
            model_config.text_model_tokens_pooler_type,
            model_config.text_model_use_linear_projection,
            backbone=text_backbone,
        )
        self.logit_scale = nn.Parameter(torch.empty(1, device=device))
        if model_config.freeze_logit_scale:
            self.logit_scale.requires_grad = False

    def init_weights(self):
        torch.nn.init.constant(self.logit_scale, self.model_config.init_logit_scale)
        self.visual_model.init_weights()
        self.text_model.init_weights()

    def encode_image_with_patch_tokens(
        self,
        image: torch.Tensor,
        normalize: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, patch_tokens, backbone_patch_tokens = self.visual_model(image)
        return (
            F.normalize(features, dim=-1) if normalize else features,
            patch_tokens,
            backbone_patch_tokens,
        )

    def encode_image(
        self,
        image: torch.Tensor,
        normalize: bool = False,
    ) -> torch.Tensor:
        features, _, _ = self.visual_model(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text: torch.Tensor, normalize: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        features, text_patch_tokens = self.text_model(text)
        return (F.normalize(features, dim=-1), text_patch_tokens) if normalize else (features, text_patch_tokens)

    def get_logits(
        self, image: torch.Tensor, text: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_features, text_patch_tokens = self.encode_text(text, normalize=True)
        image_features = self.encode_image(image, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        text_logits = image_logits.T
        return image_logits, text_logits

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        text_features, text_patch_tokens = self.encode_text(text, normalize=True)
        image_features, patch_tokens, backbone_patch_tokens = (
            self.encode_image_with_patch_tokens(image, normalize=True)
        )
        return (
            image_features,
            text_features,
            self.logit_scale.exp(),
            patch_tokens,
            backbone_patch_tokens,
            text_patch_tokens
        )


def vit_large(patch_size=16, **kwargs):
    dinotxt_config = DINOTxtConfig(
        embed_dim=2048,
        vision_model_freeze_backbone=True,
        vision_model_train_img_size=224,
        vision_model_use_class_token=True,
        vision_model_use_patch_tokens=True,
        vision_model_num_head_blocks=2,
        vision_model_head_blocks_drop_path=0.3,
        vision_model_use_linear_projection=False,
        vision_model_patch_tokens_pooler_type="mean",
        vision_model_patch_token_layer=1,  # which layer to take patch tokens from
        # 1 - last layer, 2 - second last layer, etc.
        text_model_freeze_backbone=False,
        text_model_num_head_blocks=0,
        text_model_head_blocks_is_causal=False,
        text_model_head_blocks_drop_prob=0.0,
        text_model_tokens_pooler_type="argmax",
        text_model_use_linear_projection=True,
        init_logit_scale=math.log(1 / 0.07),
        init_logit_bias=None,
        freeze_logit_scale=False,
    )
    
    vision_backbone = dinov3_vitl16(pretrained=False)
    # vision_backbone = DinoVisionTransformer(
    #     patch_size=patch_size,
    #     embed_dim=1024,
    #     depth=24,
    #     num_heads=16,
    #     ffn_ratio=4,
    #     **kwargs,
    # )
    
    text_backbone = TextTransformer(
        context_length=77,
        vocab_size=49408,
        dim=1280,
        num_heads=20,
        num_layers=24,
        ffn_ratio=4,
        is_causal=True,
        ls_init_value=None,
        dropout_prob=0.0,
    )
    model = DINOTxt(model_config=dinotxt_config, vision_backbone=vision_backbone, text_backbone=text_backbone)
    model.visual_model.backbone = vision_backbone
    
    return model


def build_backbone_dinov3txt(name: Literal["dinov3_large"], **kwargs):
    vit_dict = {
        "dinov3_large": vit_large,
    }
    return vit_dict[name](**kwargs)

def build_tokenizer_dinov3txt(bpe_path: Optional[str] = None):
    if bpe_path is None:
        bpe_path = convert_path_or_url_to_url(bpe_path_or_url)
    tokenizer = get_tokenizer(bpe_path)
    return tokenizer


# if __name__ == "__main__":
#     model, tokenizer = vit_large()
#     weight = torch.load("/projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/pretrained/dinov3_vitl16_dinotxt-a442d8f5.pth", map_location="cpu")
#     model.load_state_dict(weight, strict=False)
#     model = model.cuda()
#     # image = torch.randn(1, 3, 224, 224).cuda()
#     import urllib
#     from PIL import Image

#     def load_image_from_url(url: str) -> Image:
#         with urllib.request.urlopen(url) as f:
#             return Image.open(f).convert("RGB")

#     EXAMPLE_IMAGE_URL = "https://dl.fbaipublicfiles.com/dinov2/images/example.jpg"
#     img_pil = load_image_from_url(EXAMPLE_IMAGE_URL)

#     import torch
#     from src.dinov3.data.transforms import make_classification_eval_transform
    
#     image_preprocess = make_classification_eval_transform()
#     image_tensor = torch.stack([image_preprocess(img_pil)], dim=0).cuda()
    
#     texts = ["photo of dogs", "photo of a chair", "photo of a bowl", "photo of a tupperware"]
#     text =  tokenizer.tokenize(texts).to("cuda")
#     out = model(image_tensor, text)
    
#     image_features, text_features, _, patch_tokens, backbone_patch_tokens, text_patch_tokens = out
#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     text_features /= text_features.norm(dim=-1, keepdim=True)
#     similarity = (
#         text_features.cpu().float().detach().numpy() @ image_features.cpu().float().detach().numpy().T
#     )
#     print(similarity) 
    
