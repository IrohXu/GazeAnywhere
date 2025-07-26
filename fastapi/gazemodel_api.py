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
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

# Project-specific imports
from detectron2.config import instantiate, LazyConfig

app = FastAPI()
model = None
processor = None
model_lock = threading.Lock()

MODEL_OPT = "/projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/configs/gazefollow_gaze_vit_large.py"
MODEL_ID = "/projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/checkpoints/dinov2_gaze_vit_large.pth"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)

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