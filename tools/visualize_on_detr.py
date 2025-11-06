import sys
import os
from os import path as osp
import argparse
import warnings
import torch
import numpy as np
from PIL import Image
from detectron2.config import instantiate, LazyConfig

from typing import List, Tuple, Optional, Union

sys.path.append(osp.dirname(osp.dirname(__file__)))
from utils import *


warnings.simplefilter(action="ignore", category=FutureWarning)
Color = Tuple[int, int, int]  # BGR (OpenCV)


def overlay_heatmap_on_image(image_bgr: np.ndarray,
                             boxes: Tuple[int, int, int, int],
                             boxes_ref: Tuple[int, int, int, int],
                             color: Tuple[int, int, int] = (0, 255, 0),  # BGR
                             color_ref: Tuple[int, int, int] = (255, 0, 0),  # BGR
                             thickness: int = 4,               # ignore tiny heat
                             ) -> np.ndarray:
    """
    Draws:
      1) Heatmap overlay (masked by mask_thresh)
      2) Head bounding box
      3) A green line from head-box center to gaze point
      4) A thick green dot at the gaze point

    Returns:
      overlay_bgr: image with overlays
      colored_heat_bgr: colorized heatmap (BGR)
    """
    H, W = image_bgr.shape[:2]

    bbox_center_x = boxes[0] * W
    bbox_center_y = boxes[1] * H
    
    bbox_w = boxes[2] * W
    bbox_h = boxes[3] * H

    boxes_pred = [bbox_center_x - bbox_w / 2, bbox_center_y - bbox_h / 2, bbox_center_x + bbox_w / 2, bbox_center_y + bbox_h / 2]

    # --- heatmap -> resized -> [0,1] -> uint8 for colormap ---

    # --- inputs to float32 for blending ---
    img_u8 = image_bgr if image_bgr.dtype == np.uint8 else np.clip(image_bgr, 0, 255).astype(np.uint8)

    overlay_bgr = img_u8

    # --- robust single-box handling ---
    box = np.array(boxes_pred, dtype=np.float32).reshape(-1)
    if box.size != 4:
        raise ValueError(f"'boxes' must be 4 numbers, got shape {np.array(boxes_pred).shape} / size {box.size}")
    x1, y1, x2, y2 = map(int, box)
    x1 = int(max(0, min(W - 1, x1)));  y1 = int(max(0, min(H - 1, y1)))
    x2 = int(max(0, min(W - 1, x2)));  y2 = int(max(0, min(H - 1, y2)))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1

    # Draw head bounding box
    cv2.rectangle(overlay_bgr, (x1, y1), (x2, y2), color, thickness)
    

    # boxes_ref = [boxes_ref[0] * W, boxes_ref[1] * H, boxes_ref[2] * W, boxes_ref[3] * H]
    # --- robust single-box handling ---
    box = np.array(boxes_ref, dtype=np.float32).reshape(-1)
    if box.size != 4:
        raise ValueError(f"'boxes' must be 4 numbers, got shape {np.array(boxes_ref).shape} / size {box.size}")
    x1, y1, x2, y2 = map(int, box)
    x1 = int(max(0, min(W - 1, x1)));  y1 = int(max(0, min(H - 1, y1)))
    x2 = int(max(0, min(W - 1, x2)));  y2 = int(max(0, min(H - 1, y2)))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1

    # Draw head bounding box
    cv2.rectangle(overlay_bgr, (x1, y1), (x2, y2), color_ref, thickness)

    return overlay_bgr


def print_model_size(model: torch.nn.Module, verbose: bool = False) -> None:
    """
    Prints the total and trainable parameter count of a PyTorch model.

    Args:
        model (torch.nn.Module): your model
        verbose (bool): if True, also prints per‐module parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params:     {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable params: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    if verbose:
        print("\nPer‐module parameter breakdown:")
        for name, module in model.named_modules():
            pm = sum(p.numel() for p in module.parameters(recurse=False))


def do_test(cfg, model, visualization_dir, use_dark_inference=False):
    val_loader = instantiate(cfg.dataloader.val)
    
    gaze_threshold = 0.3

    model.train(False)
    AUC = []
    min_dist = []
    avg_dist = []
    with torch.no_grad():
        for data in val_loader:
            bbox_pred = model(data)['pred_boxes']
            image_path = data["image_path"]  
            head_bbox_gt = data["bbox_raw"]                               

            head_bbox_pred = bbox_pred.squeeze(1).cpu().detach().numpy()

            input_images = (
                data["images"].squeeze(1).cpu().detach().numpy()
            )
            
            # go through each data point and record AUC, min dist, avg dist
            for b_i in range(len(head_bbox_pred)):
                # remove padding and recover valid ground truth points
                visualization_save_path = osp.join(
                    visualization_dir,
                    image_path[b_i]
                )
                
                head_bbox_ref = head_bbox_gt[b_i].cpu().numpy()

                os.makedirs(osp.dirname(visualization_save_path), exist_ok=True)
                
                image = cv2.imread(osp.join(cfg.dataloader.val.val_root, image_path[b_i]))
                overlay_bgr = overlay_heatmap_on_image(image, head_bbox_pred[b_i], head_bbox_ref)
                # overlay_bgr, colored_heat_bgr = overlay_heatmap_on_image(image, scaled_heatmap, head_bbox, alpha=0.1)
                visualization_pred = overlay_bgr
                cv2.imwrite(visualization_save_path, visualization_pred)

def main(args):
    cfg = LazyConfig.load(args.config_file)
    visualization_dir = args.output_path # osp.join(args.output_path, "visualization")
    if not osp.exists(visualization_dir):
        os.makedirs(visualization_dir)
    model: torch.Module = instantiate(cfg.model)
    model.load_state_dict(torch.load(args.model_weights, weights_only=False)["model"])
    model.to(cfg.train.device)
    do_test(cfg, model, visualization_dir, use_dark_inference=args.use_dark_inference)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="config file")
    parser.add_argument("--output_path", type=str, help="output path")
    parser.add_argument(
        "--model_weights",
        type=str,
        help="model weights",
    )
    parser.add_argument("--use_dark_inference", action="store_true")
    args = parser.parse_args()
    main(args)

