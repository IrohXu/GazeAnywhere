import json
import glob
import re
import threading
import random
import string
import traceback
import logging

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from tqdm import tqdm
import cv2
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

# Project-specific imports
from tools.utils import *
from torchvision import transforms
from detectron2.config import instantiate, LazyConfig

app = FastAPI()
model = None
processor = None
model_lock = threading.Lock()

MODEL_OPT = "configs/gazefollow_gaze_vit_large.py"
MODEL_ID = "/projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/checkpoints/dinov2_gaze_vit_large.pth"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)

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

def inference_gaze(image_path, subject_bbox, model):
    img = Image.open(image_path)
    img = img.convert("RGB")
    
    model.train(False)
    
    width, height = img.size
    
    x_min, y_min, x_max, y_max = subject_bbox
    x_min = float(x_min)
    y_min = float(y_min)
    x_max = float(x_max)
    y_max = float(y_max)
    
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
        
        in_out_pred = in_out_pred.cpu().detach().numpy()
    
    output = gaze_heatmap_pred[0]
    output = np.clip(output, 0, 1)
    if in_out_pred[0] >= 0.5:
        output = output * 255
    else:
        output = output * 0
    output = output.astype("uint8")
    
    output = cv2.resize(output, (width, height))
    return output

@app.on_event("startup")
def load_model():
    global model
    print("Loading Gaze model...")
    # Defaults

    cfg = LazyConfig.load(MODEL_OPT)
    model = instantiate(cfg.model)
    model.load_state_dict(torch.load(MODEL_ID, weights_only=False)["model"])
    model.to(cfg.train.device)
    print("Gaze model is ready for image analysis.")

@app.post("/gaze/")
async def detect(
        image_path: str = Form(...),
        subject_bbox: list = Form(...),
        save_path: str = Form(...)
    ):
    try:
        global model
        if not image_path:
            print(image_path)
            return "Detection failed: No image provided."
        
        if subject_bbox is None:
            print(f"Skipping this frame due to no detected head.")
            return "Detection failed: No subject bounding box provided."
        
        scaled_heatmap = inference_gaze(image_path, subject_bbox, model)
        
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = np.array(image)
        
        image_visualized = image.copy()    
        overlay = image.copy()

        scaled_heatmap = scaled_heatmap / np.max(scaled_heatmap)
        scaled_heatmap[scaled_heatmap < 0.6] = 0
        gray = (255 * scaled_heatmap).astype(np.uint8)
        heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
        # visualization_pred = cv2.hconcat([image, heatmap])
        # image_visualized = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
        
        overlay[scaled_heatmap >= 0.6] = [0, 0, 255]
        alpha = 0.3  # Transparency of red mask
        if np.sum(scaled_heatmap >= 0.6) > 0:
            image_visualized[scaled_heatmap >= 0.6] = cv2.addWeighted(image[scaled_heatmap >= 0.6], 1 - alpha, overlay[scaled_heatmap >= 0.6], alpha, 0)
        
        visualization_pred = cv2.hconcat([image_visualized, heatmap])
        cv2.imwrite(save_path, visualization_pred)
        
        return JSONResponse(content=save_path)
        

    except Exception as e:
        tb = traceback.format_exc()
        logging.error(f"Error during /gaze/: {e}\n{tb}")
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "trace": tb
            }
        )    