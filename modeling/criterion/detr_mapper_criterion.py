import torch
from torch import nn
from torchvision.ops import generalized_box_iou_loss, box_convert

class DETRMapperCriterion(nn.Module):
    def __init__(self, weight_bbox=5.0, weight_giou=2.0):
        super().__init__()
        self.bbox_loss = nn.L1Loss()
        self.weight_dict = {
            "loss_bbox": weight_bbox,
            "loss_giou": weight_giou,
        }

    def forward(self, pred_boxes, gt_boxes):
        """
        pred_boxes: (B, 4) in cxcywh, normalized
        gt_boxes:   (B, 4) in cxcywh, normalized
        """
        loss_dict = {}

        # 1) L1 on cxcywh
        loss_bbox = self.bbox_loss(pred_boxes, gt_boxes)
        loss_dict["loss_bbox"] = loss_bbox * self.weight_dict["loss_bbox"]

        # 2) GIoU on xyxy
        pred_xyxy = box_convert(pred_boxes, "cxcywh", "xyxy")
        gt_xyxy   = box_convert(gt_boxes,   "cxcywh", "xyxy")

        # returns (B,) if reduction="none"
        giou_per_box = generalized_box_iou_loss(
            pred_xyxy, gt_xyxy, reduction="none"
        )  # shape (B,)

        loss_giou = giou_per_box.mean()
        loss_dict["loss_giou"] = loss_giou * self.weight_dict["loss_giou"]

        return loss_dict