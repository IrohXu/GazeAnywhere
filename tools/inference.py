import sys
from os import path as osp
import argparse
import warnings
import torch
import numpy as np
from PIL import Image
from detectron2.config import instantiate, LazyConfig
from torchvision import transforms

sys.path.append(osp.dirname(osp.dirname(__file__)))
from utils import *

warnings.simplefilter(action="ignore", category=FutureWarning)


def overlay_heatmap_on_image(image_bgr: np.ndarray,
                             heatmap_01: np.ndarray,
                             boxes: Tuple[int, int, int, int],
                             gaze_xy: Tuple[float, float],            # NEW: (x, y)
                             inout: bool = True,                     # NEW: whether in‐out prediction
                             color: Tuple[int, int, int] = (0, 255, 0),  # BGR
                             color_inout: Tuple[int, int, int] = (255, 0, 0),  # BGR for in-out
                             thickness: int = 4,
                             alpha: float = 0.2,                       # small blend
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
    heat_resized = cv2.resize(heatmap_01.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    heat_resized = np.clip(heat_resized, 0.0, 1.0)
    heat_uint8 = (heat_resized * 255).round().astype(np.uint8)
    colored_heat_bgr = cv2.applyColorMap(heat_uint8, colormap)

    # --- inputs to float32 for blending ---
    img_u8 = image_bgr if image_bgr.dtype == np.uint8 else np.clip(image_bgr, 0, 255).astype(np.uint8)
    img_f  = img_u8.astype(np.float32)
    heat_f = colored_heat_bgr.astype(np.float32)

    # # --- standard blended image ---
    blended = cv2.addWeighted(heat_f, alpha, img_f, 1 - alpha, 0.0)

    # # --- only apply where heat is meaningful ---
    mask = (heat_resized >= mask_thresh).astype(np.float32)[..., None]  # (H,W,1)
    overlay_f = np.where(mask > 0, blended, img_f)
    overlay_bgr = np.clip(overlay_f, 0, 255).astype(np.uint8)
    
    # overlay_bgr = img_u8

    # --- robust single-box handling ---
    b = np.array(boxes, dtype=np.float32).reshape(-1)
    if b.size != 4:
        raise ValueError(f"'boxes' must be 4 numbers, got shape {np.array(boxes).shape} / size {b.size}")
    x1, y1, x2, y2 = map(int, b)
    x1 = int(max(0, min(W - 1, x1)));  y1 = int(max(0, min(H - 1, y1)))
    x2 = int(max(0, min(W - 1, x2)));  y2 = int(max(0, min(H - 1, y2)))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    
    if inout:
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


def main(args):
    cfg = LazyConfig.load(args.config_file)
    model: torch.Module = instantiate(cfg.model)
    model.load_state_dict(torch.load(args.model_weights, weights_only=False)["model"])
    model = model.cuda()
    model.train(False)
    
    use_dark_inference = args.use_dark_inference
    
    image_path = args.image_path
    save_path = args.save_path
    img = Image.open(image_path).convert("RGB")
    img_size = img.size
    text = args.text
            
    image_transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    img = image_transform(img).unsqueeze(0).cuda()
    
    
    with torch.no_grad():
        gaze_heatmap_pred, inout_pred, bbox_pred = model.inference(img, [text])
                
        gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1).cpu().detach().numpy()[0]
        inout_pred = inout_pred.cpu().detach().numpy()[0]
        bbox_pred = bbox_pred.cpu().detach().numpy()[0]
        
        inout = (inout_pred > 0.5)
        
        head_bbox = [0.0, 0.0, img_size[0], img_size[1]]
        
        head_bbox[0] = (bbox_pred[0] - bbox_pred[2] / 2) * img_size[0]  # x1
        head_bbox[1] = (bbox_pred[1] - bbox_pred[3] / 2) * img_size[1]  # y1
        head_bbox[2] = (bbox_pred[0] + bbox_pred[2] / 2) * img_size[0]  # x2
        head_bbox[3] = (bbox_pred[1] + bbox_pred[3] / 2) * img_size[1]  # y2   

        if use_dark_inference:
            pred_x, pred_y = dark_inference(gaze_heatmap_pred)
        else:
            pred_x, pred_y = argmax_pts(gaze_heatmap_pred)
        
        scaled_heatmap = np.array(Image.fromarray(gaze_heatmap_pred).resize(
                img_size,
                resample=Image.BILINEAR,
            )
        )
        image = cv2.imread(image_path)
                
        gaze_x = pred_x / model.out_size[0] * img_size[0]
        gaze_y = pred_y / model.out_size[1] * img_size[1]
        
        visualization_pred = overlay_heatmap_on_image(image, scaled_heatmap, head_bbox, gaze_xy=(gaze_x, gaze_y), inout=inout)
        # overlay_bgr, colored_heat_bgr = overlay_heatmap_on_image(image, scaled_heatmap, head_bbox, alpha=0.1)
        cv2.imwrite(save_path, visualization_pred)
        
        output = {
            "gaze": (gaze_x, gaze_y),
            "inout": inout
        }
        
        return output
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="config file")
    parser.add_argument(
        "--model_weights",
        type=str,
        help="model weights",
    )
    parser.add_argument("--image_path", type=str, help="image path")
    parser.add_argument("--text", type=str, help="prompt")
    parser.add_argument("--save_path", type=str, help="save path")
    parser.add_argument("--use_dark_inference", action="store_true")
    args = parser.parse_args()
    out = main(args)
    print(out)
    