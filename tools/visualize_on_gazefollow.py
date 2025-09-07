import sys
import os
from os import path as osp
import argparse
import warnings
import torch
import numpy as np
from PIL import Image
from detectron2.config import instantiate, LazyConfig

sys.path.append(osp.dirname(osp.dirname(__file__)))
from utils import *


warnings.simplefilter(action="ignore", category=FutureWarning)


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
            val_gaze_heatmap_pred, _ = model(data)
            image_path = data["image_path"]                                 
            val_gaze_heatmap_pred = (
                val_gaze_heatmap_pred.squeeze(1).cpu().detach().numpy()
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
                
                scaled_heatmap[scaled_heatmap > gaze_threshold] = 1
                scaled_heatmap[scaled_heatmap <= gaze_threshold] = 0
                
                image = input_images[b_i]
                image = image.transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = image * std + mean
                image = np.clip(image, 0, 1)
                image = image * 255
                image = image.astype("uint8")
                image = cv2.resize(image, tuple(data["imsize"][b_i].cpu().detach().numpy()))
                
                visualization_pred = image.copy()
                
                # cv2.rectangle(visualization_pred, (subject_bbox[0], subject_bbox[1]), (subject_bbox[2], subject_bbox[3]), (0, 255, 0), 2)
                
                overlay = image.copy()
                overlay[scaled_heatmap >= 0.5] = [0, 0, 255]
                alpha = 0.3  # Transparency of red mask
                if np.sum(scaled_heatmap >= 0.5) > 0:
                    visualization_pred[scaled_heatmap > 0.5] = cv2.addWeighted(image[scaled_heatmap > 0.5], 1 - alpha, overlay[scaled_heatmap > 0.5], alpha, 0)
                
                overlay = image.copy()
                overlay[scaled_head >= 0.5] = [0, 255, 0]
                visualization_pred[scaled_head > 0.5] = cv2.addWeighted(image[scaled_head > 0.5], 1 - alpha, overlay[scaled_head > 0.5], alpha, 0)
                cv2.imwrite(visualization_save_path, visualization_pred)

                
                # auc_score = auc(scaled_heatmap, multi_hot)
                # AUC.append(auc_score)
                # # min distance: minimum among all possible pairs of <ground truth point, predicted point>
                # all_distances = []
                # for gt_gaze in valid_gaze:
                #     all_distances.append(L2_dist(gt_gaze, norm_p))
                # min_dist.append(min(all_distances))
                # # average distance: distance between the predicted point and human average point
                # mean_gt_gaze = torch.mean(valid_gaze, 0)
                # avg_distance = L2_dist(mean_gt_gaze, norm_p)
                # avg_dist.append(avg_distance)
    
    # print_model_size(model)
    # print("|AUC   |min dist|avg dist|")
    # print(
    #     "|{:.4f}|{:.4f}  |{:.4f}  |".format(
    #         torch.mean(torch.tensor(AUC)),
    #         torch.mean(torch.tensor(min_dist)),
    #         torch.mean(torch.tensor(avg_dist)),
    #     )
    # )


def main(args):
    cfg = LazyConfig.load(args.config_file)
    visualization_dir = osp.join(args.output_path, "visualization")
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

