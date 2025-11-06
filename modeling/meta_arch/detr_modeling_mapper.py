import torch
from torch import nn
from typing import Dict, Union
from ..backbone.vit import Block
from ..fusion.cross_attention import CrossAttentionBlock
import torchvision
import math

# --- (Helper functions positionalencoding2d and repeat_tensors remain unchanged) ---

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


class DETRModelMapper(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        tokenizer: nn.Module,
        criterion: nn.Module,
        device: Union[torch.device, str] = "cuda",
        freeze_backbone: bool = True,
        dim: int = 256,
        inout: bool = True,
        fusion_layers: int = 3,
        num_layers: int = 3,
        linear_dim: int = 1024,
        linear_txt_dim: int = 2048,
        image_size: int = 512,
        patch_size: int = 16,
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

        self.mask_size = image_size // patch_size
        self.register_buffer("pos_embed", positionalencoding2d(self.dim, self.mask_size, self.mask_size))
        self.inout = inout
        self.out_size = (64, 64)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.fusion = nn.Sequential(*[
            CrossAttentionBlock(
                dim=self.dim,
                num_heads=8,
                mlp_ratio=4,
                drop_path=0.1) for i in range(fusion_layers)]
        )

        # ========== New DETR Decoder Components ==========
        self.num_head_queries = num_head_queries
        
        # 1. Object queries and their positional embeddings
        self.head_queries = nn.Embedding(self.num_head_queries, self.dim)
        self.query_pos_embed = nn.Embedding(self.num_head_queries, self.dim)
        
        # 2. DETR Transformer Decoder Layers
        # We use a ModuleList to easily access intermediate layer outputs
        self.detr_decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=self.dim,
                nhead=detr_nhead,
                dim_feedforward=self.dim * 4, # 4x expansion
                dropout=0.1,
                activation='relu',
                batch_first=True # Ensures (B, N, C) format
            ) for _ in range(num_detr_layers)
        ])
        self.detr_norm = nn.LayerNorm(self.dim)

        # 3. Prediction Heads (MLPs)
        # Class head: predicts 2 classes (head vs. no-object)
        # self.head_class_head = nn.Linear(self.dim, 2) 
        
        # BBox head: predicts 4 values (cx, cy, w, h)
        # We use a standard 3-layer MLP with ReLU
        self.head_bbox_head = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, 4),
            nn.Sigmoid() # Normalizes box coords to [0, 1]
        )
        # =================================================

    def forward(self, x):
        (
            scenes,
            texts,
            gt_heads,      # Ground truth head channels (for loss)
            head_channels,
            gt_heatmaps,
            gt_inouts,
            image_masks,
        ) = self.preprocess_inputs(x)
        
        B = scenes.shape[0] # Get batch size
        
        x = self.backbone(
            scenes,
            texts,
        )
        
        img_feats = x.get("img_feat", None)
        txt_feats = x.get("text_feat", None)
        img_feats = self.linear(img_feats)
        txt_feats = self.linear_txt(txt_feats)

        feats, _ = self.fusion((img_feats, txt_feats))

        # Pass all tokens (inout + image) through the transformer
        # feats = self.transformer(feats)        
        # --- (DETR Decoder) Path ---
        
        # Prepare inputs for the decoder
        # Memory is the output from the transformer (gaze_feats)
        decoder_memory = feats # Shape: (B, N, C)
        
        # Positional embedding for memory (from self.pos_embed)
        # (C, H, W) -> (C, N) -> (N, C) -> (B, N, C)
        memory_pos = self.pos_embed.flatten(1).permute(1, 0).unsqueeze(0).repeat(B, 1, 1)
        
        # Object queries (tgt)
        # (num_queries, C) -> (B, num_queries, C)
        head_queries = self.head_queries.weight.unsqueeze(0).repeat(B, 1, 1)
        
        # Positional embedding for queries
        # (num_queries, C) -> (B, num_queries, C)
        query_pos = self.query_pos_embed.weight.unsqueeze(0).repeat(B, 1, 1)

        # Initialize the decoder's target (queries) as zeros
        detr_output = head_queries

        intermediate_outputs = [] # To store outputs from all layers
        
        
        for layer in self.detr_decoder_layers:
            # --- START OF MODIFICATION ---
            # Add positional embeddings to query and memory *before* passing them
            tgt_with_pos = detr_output + query_pos
            memory_with_pos = decoder_memory + memory_pos
            
            detr_output = layer(
                tgt=tgt_with_pos,
                memory=memory_with_pos,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None
            )
            # --- END OF MODIFICATION ---
            
            intermediate_outputs.append(self.detr_norm(detr_output))

        # Stack outputs: (L, B, num_queries, C)
        detr_output_stack = torch.stack(intermediate_outputs)
        
        # Apply prediction heads to all layer outputs
        pred_boxes = self.head_bbox_head(detr_output_stack)   # (L, B, Q, 4)

        # Collate head predictions into a dict
        # The criterion will use this for loss calculation
        head_preds = pred_boxes
        
        if self.training:
            return self.criterion(
                head_preds,     # Pass new head predictions
                gt_heads,
            )
        
        # Inference: return final layer's predictions
        final_head_preds = {
            'pred_boxes': head_preds[-1]  # (B, Q, 4)
        }
            
        return final_head_preds
    

    def preprocess_inputs(self, batched_inputs: Dict[str, torch.Tensor]):
        texts = batched_inputs["texts"]
        texts = self.tokenizer.tokenize(texts)
        return (
            batched_inputs["images"].to(self.device),
            texts.to(self.device),
            batched_inputs["bbox"].to(self.device),
            batched_inputs["head_channels"].to(self.device), # This is the GT for the decoder
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