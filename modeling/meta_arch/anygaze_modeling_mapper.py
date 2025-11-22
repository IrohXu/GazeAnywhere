import torch
from torch import nn
from typing import Dict, Union
from ..backbone.vit import Block
from ..fusion.cross_attention import CrossAttentionBlock
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


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of the positions
    :return: d_model*length position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, length)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos = torch.arange(0., length).unsqueeze(1)
    pe[0::2, :] = torch.sin(pos * div_term).transpose(0, 1)
    pe[1::2, :] = torch.cos(pos * div_term).transpose(0, 1)
    return pe


def repeat_tensors(tensor, repeat_counts):
    repeated_tensors = [tensor[i:i+1].repeat(repeat, *[1] * (tensor.ndim - 1)) for i, repeat in enumerate(repeat_counts)]
    return torch.cat(repeated_tensors, dim=0)


class AnyGazeModelMapper(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        tokenizer: nn.Module,
        criterion: nn.Module,
        device: Union[torch.device, str] = "cuda",
        freeze_backbone: bool = True,
        freeze_visual_encoder: bool = True,
        freeze_text_encoder: bool = True,
        dim: int = 256,
        inout: bool = True,
        fusion_layers: int = 3,
        num_layers: int = 3,
        linear_dim: int = 1024,
        linear_txt_dim: int = 2048,
        image_size: int = 512,
        patch_size: int = 16,
        max_text_seq: int = 77, # --- ADDED --- Max text tokens (e.g., 77 for CLIP)
        # --- New DETR parameters ---
        num_head_queries: int = 1, # Max number of heads to detect
        num_detr_layers: int = 6, # Number of decoder layers
        detr_nhead: int = 8,  # Num heads in decoder MHA
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.criterion = criterion
        self.device = torch.device(device)
        self.dim = dim
        self.linear = nn.Linear(linear_dim, self.dim)
        self.linear_txt = nn.Linear(linear_txt_dim, self.dim)
        # self.head_token = nn.Embedding(1, self.dim)
        self.mask_size = image_size // patch_size
        self.register_buffer("pos_embed", positionalencoding2d(self.dim, self.mask_size, self.mask_size).squeeze(dim=0).squeeze(dim=0))
        self.register_buffer("text_pos_embed", positionalencoding1d(self.dim, max_text_seq))
        
        self.max_text_seq = max_text_seq
        
        self.inout = inout
        self.out_size = (64, 64)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            if freeze_visual_encoder == False:
                for param in self.backbone.visual_model.parameters():
                    param.requires_grad = True
            if freeze_text_encoder == False:
                for param in self.backbone.text_model.parameters():
                    param.requires_grad = True
                
        
        self.num_head_queries = num_head_queries
        
        self.head_queries = nn.Embedding(self.num_head_queries, self.dim)

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
        
        self.head_bbox_head = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, 4),
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
            texts,
            gt_heads,
            head_channels,
            gt_heatmaps,
            gt_inouts,
            image_masks,
        ) = self.preprocess_inputs(x)
        
        B = scenes.size(0)
        
        # Get out-dict
        x = self.backbone(
            scenes,
            texts,
        )
        
        out = self.backbone(scenes, texts)
        img_feats = out["img_feat"]          # (B, N_img, C_in)
        txt_feats = out["text_feat"]         # (B, N_txt, C_in)
        txt_emb_feats = out["text_emb"]      # (B, C_txt_in)
        img_emb_feats= out["img_emb"]      # (B, C_img_in)
        
        # project
        img_feats = self.linear(img_feats)        # (B, N_img, dim)
        img_emb_feats = self.linear(img_emb_feats)  # (B, dim)

        txt_feats = self.linear_txt(txt_feats)  # (B, N_txt, dim)
        txt_emb_feats = self.linear_txt(txt_emb_feats)  # (B, dim)
        
        # image pos (assumes N_img == mask_size*mask_size)
        img_pos = self.pos_embed.flatten(1).permute(1, 0).unsqueeze(0)  # (1, N_img, dim)
        img_feats = img_feats + img_pos
        
        # text pos
        txt_pos = self.text_pos_embed.permute(1, 0).unsqueeze(0)  # (1, N_txt, dim)
        txt_feats = txt_feats + txt_pos
        
        base_head = self.head_queries.weight.unsqueeze(0).repeat(B, 1, 1)   # (B, 1, dim)
        head_token = base_head + txt_emb_feats.unsqueeze(1)   

        base_inout = self.inout_token.weight.unsqueeze(dim=0).repeat(B, 1, 1)
        inout_token = base_inout + img_emb_feats.unsqueeze(1)                 # (B, 1, dim)

        if self.inout:
            feats = torch.cat([head_token, txt_feats, img_feats, inout_token], dim=1)
        else:
            feats = torch.cat([head_token, txt_feats, img_feats], dim=1)

        feats = self.transformer(feats)
        
        bbox_token = feats[:, 0, :]           # (B, dim)
        pred_boxes = self.head_bbox_head(bbox_token)  # (B, 4)
        
        if self.inout:
            inout_tokens = feats[:, -1, :] 
            inout_preds = self.inout_head(inout_tokens).squeeze(dim=-1)
            # inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
            feats = feats[:, 1+txt_feats.shape[1]:-1, :] # slice off inout tokens from scene tokens
        
        feats = feats.reshape(feats.shape[0], self.mask_size, self.mask_size, feats.shape[2]).permute(0, 3, 1, 2) # b (h w) c -> b c h w        
        feats = self.heatmap_head(feats)
        
        # feats = self.heatmap_head(feats).squeeze(dim=1)
        feats = torchvision.transforms.functional.resize(feats, self.out_size)
        heatmap_preds = feats
            
        if self.training:
            return self.criterion(
                heatmap_preds,
                inout_preds,
                pred_boxes, 
                gt_heatmaps,
                gt_inouts,
                gt_heads
            )
        # Inference
                
        return heatmap_preds, inout_preds.sigmoid(), pred_boxes
    
    
    def inference(self, scenes, texts):
        
        texts = self.tokenizer.tokenize(texts)
                
        # --- MODIFIED --- Ensure text tokens are padded/truncated to max_text_seq
        B, N_txt = texts.shape
        if N_txt > self.max_text_seq:
            texts = texts[:, :self.max_text_seq]
        elif N_txt < self.max_text_seq:
            # Assuming 0 is the padding token index
            padding = torch.zeros((B, self.max_text_seq - N_txt), dtype=torch.long, device=texts.device)
            texts = torch.cat([texts, padding], dim=1)
            
        scenes = scenes.to(self.device)
        texts = texts.to(self.device)
        
        B = scenes.size(0)
        
        # Get out-dict
        x = self.backbone(
            scenes,
            texts,
        )
        
        out = self.backbone(scenes, texts)
        img_feats = out["img_feat"]          # (B, N_img, C_in)
        txt_feats = out["text_feat"]         # (B, N_txt, C_in)
        txt_emb_feats = out["text_emb"]      # (B, C_txt_in)
        img_emb_feats= out["img_emb"]      # (B, C_img_in)
        
        # project
        img_feats = self.linear(img_feats)        # (B, N_img, dim)
        img_emb_feats = self.linear(img_emb_feats)  # (B, dim)

        txt_feats = self.linear_txt(txt_feats)  # (B, N_txt, dim)
        txt_emb_feats = self.linear_txt(txt_emb_feats)  # (B, dim)
        
        # image pos (assumes N_img == mask_size*mask_size)
        img_pos = self.pos_embed.flatten(1).permute(1, 0).unsqueeze(0)  # (1, N_img, dim)
        img_feats = img_feats + img_pos
        
        # text pos
        txt_pos = self.text_pos_embed.permute(1, 0).unsqueeze(0)  # (1, N_txt, dim)
        txt_feats = txt_feats + txt_pos
        
        base_head = self.head_queries.weight.unsqueeze(0).repeat(B, 1, 1)   # (B, 1, dim)
        head_token = base_head + txt_emb_feats.unsqueeze(1)   

        base_inout = self.inout_token.weight.unsqueeze(dim=0).repeat(B, 1, 1)
        inout_token = base_inout + img_emb_feats.unsqueeze(1)                 # (B, 1, dim)

        if self.inout:
            feats = torch.cat([head_token, txt_feats, img_feats, inout_token], dim=1)
        else:
            feats = torch.cat([head_token, txt_feats, img_feats], dim=1)

        feats = self.transformer(feats)
        
        bbox_token = feats[:, 0, :]           # (B, dim)
        pred_boxes = self.head_bbox_head(bbox_token)  # (B, 4)
        
        if self.inout:
            inout_tokens = feats[:, -1, :] 
            inout_preds = self.inout_head(inout_tokens).squeeze(dim=-1)
            # inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
            feats = feats[:, 1+txt_feats.shape[1]:-1, :] # slice off inout tokens from scene tokens
        
        feats = feats.reshape(feats.shape[0], self.mask_size, self.mask_size, feats.shape[2]).permute(0, 3, 1, 2) # b (h w) c -> b c h w        
        feats = self.heatmap_head(feats)
        
        # feats = self.heatmap_head(feats).squeeze(dim=1)
        feats = torchvision.transforms.functional.resize(feats, self.out_size)
        heatmap_preds = feats
                            
        return heatmap_preds, inout_preds.sigmoid(), pred_boxes


    def preprocess_inputs(self, batched_inputs: Dict[str, torch.Tensor]):
        texts = batched_inputs["texts"]
        texts = self.tokenizer.tokenize(texts)
                
        # --- MODIFIED --- Ensure text tokens are padded/truncated to max_text_seq
        B, N_txt = texts.shape
        if N_txt > self.max_text_seq:
            texts = texts[:, :self.max_text_seq]
        elif N_txt < self.max_text_seq:
            # Assuming 0 is the padding token index
            padding = torch.zeros((B, self.max_text_seq - N_txt), dtype=torch.long, device=texts.device)
            texts = torch.cat([texts, padding], dim=1)
        
        return (
            batched_inputs["images"].to(self.device),
            texts.to(self.device),
            batched_inputs["bbox"].to(self.device),
            batched_inputs["head_channels"].to(self.device),
            batched_inputs["heatmaps"].to(self.device)
            if "heatmaps" in batched_inputs.keys()
            else None,
            batched_inputs["gaze_inouts"].to(self.device)
            if "gaze_inouts" in batched_inputs.keys()
            else None,
            batched_inputs["image_masks"].to(self.device)
            if "image_masks" in batched_inputs.keys()
            else None,
        )