import sys
from os import path as osp
import argparse
import warnings
from typing import Tuple
import torch
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
from detectron2.config import instantiate, LazyConfig
import os

from transformers import Owlv2Processor, Owlv2ForObjectDetection

sys.path.append(osp.dirname(osp.dirname(__file__)))
from utils import *

import cv2

warnings.simplefilter(action="ignore", category=FutureWarning)

def to_numpy(tensor: torch.Tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().detach().numpy()
    elif type(tensor).__module__ != "numpy":
        raise ValueError("Cannot convert {} to numpy array".format(type(tensor)))
    return tensor


def to_torch(ndarray: np.ndarray):
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray

def get_bbox_channel(
    x_min, y_min, x_max, y_max, width, height, resolution, coordconv=False
):
    head_box = (
        np.array([x_min / width, y_min / height, x_max / width, y_max / height])
        * resolution
    )
    int_head_box = head_box.astype(int)
    int_head_box = np.clip(int_head_box, 0, resolution - 1)
    if int_head_box[0] == int_head_box[2]:
        if int_head_box[0] == 0:
            int_head_box[2] = 1
        elif int_head_box[2] == resolution - 1:
            int_head_box[0] = resolution - 2
        elif abs(head_box[2] - int_head_box[2]) > abs(head_box[0] - int_head_box[0]):
            int_head_box[2] += 1
        else:
            int_head_box[0] -= 1
    if int_head_box[1] == int_head_box[3]:
        if int_head_box[1] == 0:
            int_head_box[3] = 1
        elif int_head_box[3] == resolution - 1:
            int_head_box[1] = resolution - 2
        elif abs(head_box[3] - int_head_box[3]) > abs(head_box[1] - int_head_box[1]):
            int_head_box[3] += 1
        else:
            int_head_box[1] -= 1
    head_box = int_head_box
    if coordconv:
        unit = np.array(range(0, resolution), dtype=np.float32)
        head_channel = []
        for i in unit:
            head_channel.append([unit + i])
        head_channel = np.squeeze(np.array(head_channel)) / float(np.max(head_channel))
        head_channel[head_box[1] : head_box[3], head_box[0] : head_box[2]] = 0
    else:
        head_channel = np.zeros((resolution, resolution), dtype=np.float32)
        head_channel[head_box[1] : head_box[3], head_box[0] : head_box[2]] = 1
    head_channel = torch.from_numpy(head_channel)
    return head_channel


def ovod_head_detection(image_path, processor, model):
    image = Image.open(image_path)
    text_labels = [["head of a child"]]
    inputs = processor(text=text_labels, images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.tensor([(image.height, image.width)])
    # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=0.1, text_labels=text_labels
    )
    # Retrieve predictions for the first image for the corresponding text queries
    result = results[0]
    boxes, scores, text_labels = result["boxes"], result["scores"], result["text_labels"]
    output_boxes = []
    for box, score, text_label in zip(boxes, scores, text_labels):
        box = [round(i, 2) for i in box.tolist()]
        output_boxes.append((round(score.item(), 3), box))

    output_boxes.sort(key=lambda x: x[0], reverse=True)
    if output_boxes != []:
        return output_boxes[0][1]
    else:
        return None


def inference_gaze(image_path, subject_bbox, model):
    img = Image.open(image_path)
    img = img.convert("RGB")
    
    model.train(False)
    
    width, height = img.size
    
    x_min, y_min, x_max, y_max = subject_bbox
    
    subject_channel = get_bbox_channel(
        x_min,
        y_min,
        x_max,
        y_max,
        width,
        height,
        resolution=518,
        coordconv=False,
    ).unsqueeze(0).cuda()
    
    # head_masks[head_region > 0] = 1.0
    # head_masks = head_masks.unsqueeze(0).cuda()
    
    image_transform = transforms.Compose(
        [
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    
    img = image_transform(img).cuda()
    img = img.unsqueeze(0)
    
    with torch.no_grad():
        gaze_heatmap_pred, in_out_pred = model.inference(img, subject_channel)
        
        gaze_heatmap_pred = (
            gaze_heatmap_pred.squeeze(1).cpu().detach().numpy()
        )
    
    output = gaze_heatmap_pred[0]
    output = np.clip(output, 0, 1)
    output = output * 255
    output = output.astype("uint8")
    
    output = cv2.resize(output, (width, height))
    
    # cv2.imwrite("output_gaze.jpg", output)
    return output
    
                
def main(args):
    cfg = LazyConfig.load(args.config_file)
    model: torch.Module = instantiate(cfg.model)
    model.load_state_dict(torch.load(args.model_weights, weights_only=False)["model"])
    model.to(cfg.train.device)
    inference_gaze(image_path, subject_bbox, model)


if __name__ == "__main__":
    config_file = "./configs/gazefollow_gaze_vit_large.py"
    model_weights = "/projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/checkpoints/dinov2_gaze_vit_large.pth"
    visualization_save_path = "output_gaze.jpg"
    processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14-ensemble")
    head_detection_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-large-patch14-ensemble")
    cfg = LazyConfig.load(config_file)
    model = instantiate(cfg.model)
    model.load_state_dict(torch.load(model_weights, weights_only=False)["model"])
    model.to(cfg.train.device)
    
    image_path = "/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/WeillCornell_PedabyteProject_processed/for_annotations/all/PWC007_2019_02_23_T1_panasonic_ESCS_merged_12129_15508/clipped_frames/0720.jpg"
    subject_bbox = ovod_head_detection(image_path, processor, head_detection_model)
    # subject_bbox = [422.6, 249.82, 714.94, 671.29]
    scaled_heatmap = inference_gaze(image_path, subject_bbox, model)
    
    image = Image.open(image_path)
    image = image.convert("RGB")
    image = np.array(image)
    # cv2.imwrite("image.jpg", image)
    
    visualization_pred = image.copy()    
    overlay = image.copy()

    scaled_heatmap = scaled_heatmap / 255
    overlay[scaled_heatmap >= 0.5] = [0, 0, 255]
    alpha = 0.3  # Transparency of red mask
    if np.sum(scaled_heatmap >= 0.5) > 0:
        visualization_pred[scaled_heatmap > 0.5] = cv2.addWeighted(image[scaled_heatmap > 0.5], 1 - alpha, overlay[scaled_heatmap > 0.5], alpha, 0)

    cv2.imwrite(visualization_save_path, visualization_pred)
    