from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from fvcore.nn import sigmoid_focal_loss_jit
from torchvision.ops import generalized_box_iou_loss, box_convert


class GazeAnywhereMapperCriterion(nn.Module):
    def __init__(
        self,
        heatmap_weight: float = 1000,
        inout_weight: float = 100,
        weight_bbox=50.0, 
        weight_giou=20.0,
        use_focal_loss: bool = False,
        alpha: float = -1,
        gamma: float = 2,
    ):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.inout_weight = inout_weight
        self.bbox_weight = weight_bbox
        self.giou_weight = weight_giou
        
        self.bbox_loss = nn.L1Loss()
        self.heatmap_loss = nn.BCELoss() # nn.MSELoss(reduce=False)

        if use_focal_loss:
            self.inout_loss = partial(
                sigmoid_focal_loss_jit, alpha=alpha, gamma=gamma, reduction="mean"
            )
        else:
            self.inout_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        pred_heatmap,
        pred_inout,
        pred_boxes,
        gt_heatmap,
        gt_inout,
        gt_boxes
    ):
        loss_dict = {}
                
        pred_heatmap = F.interpolate(
            pred_heatmap,
            size=tuple(gt_heatmap.shape[-2:]),
            mode="bilinear",
            align_corners=True,
        )

        bbox_loss = (self.bbox_loss(pred_boxes, gt_boxes) * self.bbox_weight)
        
        pred_xyxy = box_convert(pred_boxes, "cxcywh", "xyxy")
        gt_xyxy   = box_convert(gt_boxes,   "cxcywh", "xyxy")

        # returns (B,) if reduction="none"
        giou_per_box = generalized_box_iou_loss(
            pred_xyxy, gt_xyxy, reduction="none"
        )  # shape (B,)

        giou_loss = (giou_per_box.mean() * self.giou_weight)

        heatmap_loss = (
            self.heatmap_loss(pred_heatmap.squeeze(1), gt_heatmap) * self.heatmap_weight
        )
        
        inout_loss = (
            self.inout_loss(pred_inout.reshape(-1), gt_inout.reshape(-1))
            * self.inout_weight
        )
        
        loss_dict["regression loss"] = heatmap_loss
        loss_dict["classification loss"] = inout_loss
        loss_dict["bbox loss"] = bbox_loss
        loss_dict["giou loss"] = giou_loss

        return loss_dict
