import torch
from torch import nn
from typing import Dict, Union
from ..backbone.vit import Block
import torchvision

import math

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

def repeat_tensors(tensor, repeat_counts):
    repeated_tensors = [tensor[i:i+1].repeat(repeat, *[1] * (tensor.ndim - 1)) for i, repeat in enumerate(repeat_counts)]
    return torch.cat(repeated_tensors, dim=0)


class GazeModelMapper(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        criterion: nn.Module,
        device: Union[torch.device, str] = "cuda",
        freeze_backbone: bool = True,
        dim: int = 256,
        inout: bool = True,
        num_layers: int = 3,
        image_size: int = 518,
        patch_size: int = 14,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.criterion = criterion
        self.device = torch.device(device)
        self.dim = dim
        self.linear = nn.Conv2d(self.backbone.embed_dim, self.dim, 1)
        self.head_token = nn.Embedding(1, self.dim)
        self.mask_size = image_size // patch_size
        self.register_buffer("pos_embed", positionalencoding2d(self.dim, self.mask_size, self.mask_size).squeeze(dim=0).squeeze(dim=0))
        self.inout = inout
        self.out_size = (64, 64)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.transformer = nn.Sequential(*[
            Block(
                dim=self.dim, 
                num_heads=8, 
                mlp_ratio=4, 
                drop_path=0.1) for i in range(num_layers)]
        )
        
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.Conv2d(dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        if self.inout:
            self.inout_head = nn.Sequential(
                nn.Linear(self.dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
                # nn.Sigmoid()
            )
            self.inout_token = nn.Embedding(1, self.dim)

    def forward(self, x):
        (
            scenes,
            heads,
            gt_heatmaps,
            gt_inouts,
            head_masks,
            image_masks,
        ) = self.preprocess_inputs(x)
        # Calculate patch weights based on head position
        
        embedded_heads = torchvision.transforms.functional.resize(heads, (self.mask_size, self.mask_size))
        embedded_heads = torch.where(embedded_heads > 0, 1., 0.)
                
        # Get out-dict
        x = self.backbone(
            scenes,
            image_masks,
            None,
        )
        
        # Apply patch weights to get the final feats and attention maps
        feats = x.get("last_feat", None)
        feats = self.linear(feats)
        feats = feats + self.pos_embed

        # head_map_embeddings = embedded_heads.unsqueeze(dim=1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
        head_map_embeddings = embedded_heads * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
        feats = feats + head_map_embeddings
        feats = feats.flatten(start_dim=2).permute(0, 2, 1) # "b c h w -> b (h w) c"
        
        if self.inout:
            feats = torch.cat([self.inout_token.weight.unsqueeze(dim=0).repeat(feats.shape[0], 1, 1), feats], dim=1)

        feats = self.transformer(feats)
        
        if self.inout:
            inout_tokens = feats[:, 0, :] 
            inout_preds = self.inout_head(inout_tokens).squeeze(dim=-1)
            # inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
            feats = feats[:, 1:, :] # slice off inout tokens from scene tokens
        
        feats = feats.reshape(feats.shape[0], self.mask_size, self.mask_size, feats.shape[2]).permute(0, 3, 1, 2) # b (h w) c -> b c h w        
        feats = self.heatmap_head(feats)
        
        # feats = self.heatmap_head(feats).squeeze(dim=1)
        feats = torchvision.transforms.functional.resize(feats, self.out_size)
        heatmap_preds = feats
            
        if self.training:
            return self.criterion(
                heatmap_preds,
                inout_preds,
                gt_heatmaps,
                gt_inouts
            )
        # Inference
                
        return heatmap_preds, inout_preds.sigmoid()
        # return heatmap_preds, inout_preds.sigmoid()

    def preprocess_inputs(self, batched_inputs: Dict[str, torch.Tensor]):
        return (
            batched_inputs["images"].to(self.device),
            batched_inputs["head_channels"].to(self.device),
            batched_inputs["heatmaps"].to(self.device)
            if "heatmaps" in batched_inputs.keys()
            else None,
            batched_inputs["gaze_inouts"].to(self.device)
            if "gaze_inouts" in batched_inputs.keys()
            else None,
            batched_inputs["head_masks"].to(self.device)
            if "head_masks" in batched_inputs.keys()
            else None,
            batched_inputs["image_masks"].to(self.device)
            if "image_masks" in batched_inputs.keys()
            else None,
        )