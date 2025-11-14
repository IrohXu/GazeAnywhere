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
import time


warnings.simplefilter(action="ignore", category=FutureWarning)
Color = Tuple[int, int, int]  # BGR (OpenCV)


def overlay_heatmap_on_image(image_bgr: np.ndarray,
                             heatmap_01: np.ndarray,
                             boxes: Tuple[int, int, int, int],
                             gaze_xy: Tuple[float, float],            # NEW: (x, y)
                             inpout: bool = True,                     # NEW: whether in‐out prediction
                             color: Tuple[int, int, int] = (0, 255, 0),  # BGR
                             color_inout: Tuple[int, int, int] = (255, 0, 0),  # BGR for in-out
                             thickness: int = 4,
                             alpha: float = 0.3,                       # small blend
                             colormap: int = cv2.COLORMAP_JET,
                             mask_thresh: float = 0.10                  # ignore tiny heat
                             ) -> Tuple[np.ndarray, np.ndarray]:
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

    # # --- heatmap -> resized -> [0,1] -> uint8 for colormap ---
    # heat_resized = cv2.resize(heatmap_01.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    # heat_resized = np.clip(heat_resized, 0.0, 1.0)
    # heat_uint8 = (heat_resized * 255).round().astype(np.uint8)
    # colored_heat_bgr = cv2.applyColorMap(heat_uint8, colormap)

    # --- inputs to float32 for blending ---
    img_u8 = image_bgr if image_bgr.dtype == np.uint8 else np.clip(image_bgr, 0, 255).astype(np.uint8)
    # img_f  = img_u8.astype(np.float32)
    # heat_f = colored_heat_bgr.astype(np.float32)

    # # --- standard blended image ---
    # blended = cv2.addWeighted(heat_f, alpha, img_f, 1 - alpha, 0.0)

    # # --- only apply where heat is meaningful ---
    # mask = (heat_resized >= mask_thresh).astype(np.float32)[..., None]  # (H,W,1)
    # overlay_f = np.where(mask > 0, blended, img_f)
    # overlay_bgr = np.clip(overlay_f, 0, 255).astype(np.uint8)
    
    overlay_bgr = img_u8

    # --- robust single-box handling ---
    b = np.array(boxes, dtype=np.float32).reshape(-1)
    if b.size != 4:
        raise ValueError(f"'boxes' must be 4 numbers, got shape {np.array(boxes).shape} / size {b.size}")
    x1, y1, x2, y2 = map(int, b)
    x1 = int(max(0, min(W - 1, x1)));  y1 = int(max(0, min(H - 1, y1)))
    x2 = int(max(0, min(W - 1, x2)));  y2 = int(max(0, min(H - 1, y2)))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    
    if inpout:
        # Draw head bounding box
        cv2.rectangle(overlay_bgr, (x1, y1), (x2, y2), color, thickness)

        # --- compute head center ---
        cx = int(round((x1 + x2) * 0.5))
        cy = int(round((y1 + y2) * 0.5))

        # --- handle gaze point (clip to image and draw) ---
        gx, gy = gaze_xy
        gx_i = int(round(max(0, min(W - 1, gx))))
        gy_i = int(round(max(0, min(H - 1, gy))))

        # Draw line from head center to gaze point
        line_thickness = max(2, thickness)  # slightly robust
        cv2.line(overlay_bgr, (cx, cy), (gx_i, gy_i), color, line_thickness, lineType=cv2.LINE_AA)

        # Draw a thicker filled dot at the gaze point
        gaze_radius = max(12, thickness * 2)
        cv2.circle(overlay_bgr, (gx_i, gy_i), gaze_radius, color, thickness=-1, lineType=cv2.LINE_AA)
    else:
        # Draw head bounding box in different color for out-of-frame
        cv2.rectangle(overlay_bgr, (x1, y1), (x2, y2), color_inout, thickness)

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
            start_time = time.time()
            val_gaze_heatmap_pred, val_inout_pred, val_bbox_pred = model(data)
            end_time = time.time()
            inference_time = end_time - start_time
            print(f"Inference time for batch: {inference_time:.4f} seconds")
            
            image_path = data["image_path"]  
            # head_bboxes = data["bbox_raw"]                               
            val_gaze_heatmap_pred = (
                val_gaze_heatmap_pred.squeeze(1).cpu().detach().numpy()
            )
            
            val_inout_pred = (
                val_inout_pred.cpu().detach().numpy()
            )
            
            head_bboxes = (
                val_bbox_pred.cpu().detach().numpy()
            )
            
            input_images = (
                data["images"].squeeze(1).cpu().detach().numpy()
            )
            
            val_subject_images = (
                data["head_channels"].squeeze(1).cpu().detach().numpy()
            )
            
            # go through each data point and record AUC, min dist, avg dist
            for b_i in range(len(val_gaze_heatmap_pred)):
                # remove padding and recover valid ground truth points
                visualization_save_path = osp.join(
                    visualization_dir,
                    image_path[b_i]
                )
                
                inpout = (val_inout_pred[b_i] > 0.5)
                
                head_bbox = [0.0, 0.0, data["imsize"][b_i][0].cpu().numpy(), data["imsize"][b_i][1].cpu().numpy()]
                
                head_bbox[0] = (head_bboxes[b_i][0] - head_bboxes[b_i][2] / 2) * data["imsize"][b_i][0].cpu().numpy()  # x1
                head_bbox[1] = (head_bboxes[b_i][1] - head_bboxes[b_i][3] / 2) * data["imsize"][b_i][1].cpu().numpy()  # y1
                head_bbox[2] = (head_bboxes[b_i][0] + head_bboxes[b_i][2] / 2) * data["imsize"][b_i][0].cpu().numpy()  # x2
                head_bbox[3] = (head_bboxes[b_i][1] + head_bboxes[b_i][3] / 2) * data["imsize"][b_i][1].cpu().numpy()  # y2   
                
                os.makedirs(osp.dirname(visualization_save_path), exist_ok=True)
                
                valid_gaze = data["gazes"][b_i]
                valid_gaze = valid_gaze[valid_gaze != -1].view(-1, 2)
                # AUC: area under curve of ROC
                multi_hot = multi_hot_targets(data["gazes"][b_i], data["imsize"][b_i])
                if use_dark_inference:
                    pred_x, pred_y = dark_inference(val_gaze_heatmap_pred[b_i])
                else:
                    pred_x, pred_y = argmax_pts(val_gaze_heatmap_pred[b_i])
                norm_p = [
                    pred_x / val_gaze_heatmap_pred[b_i].shape[-2],
                    pred_y / val_gaze_heatmap_pred[b_i].shape[-1],
                ]
                
                scaled_heatmap = np.array(
                    Image.fromarray(val_gaze_heatmap_pred[b_i]).resize(
                        tuple(data["imsize"][b_i].cpu().detach().numpy()),
                        resample=Image.BILINEAR,
                    )
                )
                
                scaled_head = np.array(
                    Image.fromarray(val_subject_images[b_i]).resize(
                        tuple(data["imsize"][b_i].cpu().detach().numpy()),
                        resample=Image.BILINEAR,
                    )
                )
                
                flat_index = np.argmax(scaled_heatmap)
                gaze_x, gaze_y = np.unravel_index(flat_index, scaled_head.shape)
                
                image = cv2.imread(osp.join(cfg.dataloader.val.val_root, image_path[b_i]))
                overlay_bgr = overlay_heatmap_on_image(image, scaled_heatmap, head_bbox, gaze_xy=(gaze_y, gaze_x), inpout=inpout)
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

