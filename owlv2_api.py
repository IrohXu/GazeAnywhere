# Standard library imports
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
from transformers import Owlv2Processor, Owlv2ForObjectDetection

app = FastAPI()
model = None
processor = None
model_lock = threading.Lock()

MODEL_ID = "google/owlv2-large-patch14-ensemble"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)

@app.on_event("startup")
def load_model():
    global model
    global processor
    print("Loading Google OWLv2 model...")
    # Defaults

    processor = Owlv2Processor.from_pretrained(MODEL_ID)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID)
    print("Google OWLv2 is ready for image analysis.")

@app.post("/detection/")
async def detect(
        image_path: str = Form(...)
    ):
    try:
        global model
        global processor
        if not image_path:
            print(image_path)
            return "Detection failed: No image provided."
        
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
            return JSONResponse(content=output_boxes[0][1])
        else:
            return JSONResponse(content=None)
    
    except Exception as e:
        tb = traceback.format_exc()
        logging.error(f"Error during /detection/: {e}\n{tb}")
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "trace": tb
            }
        )    