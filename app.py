import gradio as gr
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from detectron2.config import instantiate, LazyConfig
import cv2
import sys
import os

# Add the tools directory to the python path to import from it
sys.path.append(os.path.join(os.path.dirname(__file__), "tools"))
try:
    from utils import *
    from inference import overlay_heatmap_on_image
except ImportError:
    print("Warning: Could not import tools module. Make sure you run this script from the project root.")

# Global model variable
model = None

def load_model(config_file, model_weights):
    global model
    if model is None:
        try:
            cfg = LazyConfig.load(config_file)
            model = instantiate(cfg.model)
            model.load_state_dict(torch.load(model_weights, weights_only=False, map_location="cpu")["model"])
            if torch.cuda.is_available():
                model = model.cuda()
            model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise gr.Error(f"Error loading model: {e}")

def infer(image, text, config_file, model_weights, use_dark_inference=False):
    global model
    if image is None:
        raise gr.Error("Please upload an image.")
    if not text:
        raise gr.Error("Please enter a text prompt.")
        
    try:
        load_model(config_file, model_weights)
    except Exception as e:
        raise gr.Error(f"Failed to load model: {e}")

    img_size = image.size
    
    image_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    img_tensor = image_transform(image).unsqueeze(0)
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
        
    with torch.no_grad():
        gaze_heatmap_pred, inout_pred, bbox_pred = model.inference(img_tensor, [text])
        
        gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1).cpu().detach().numpy()[0]
        inout_pred = inout_pred.cpu().detach().numpy()[0]
        bbox_pred = bbox_pred.cpu().detach().numpy()[0]
        
        inout = bool(inout_pred > 0.5)
        
        head_bbox = [0.0, 0.0, img_size[0], img_size[1]]
        head_bbox[0] = (bbox_pred[0] - bbox_pred[2] / 2) * img_size[0]
        head_bbox[1] = (bbox_pred[1] - bbox_pred[3] / 2) * img_size[1]
        head_bbox[2] = (bbox_pred[0] + bbox_pred[2] / 2) * img_size[0]
        head_bbox[3] = (bbox_pred[1] + bbox_pred[3] / 2) * img_size[1]
        
        if use_dark_inference:
            pred_x, pred_y = dark_inference(gaze_heatmap_pred)
        else:
            pred_x, pred_y = argmax_pts(gaze_heatmap_pred)
            
        scaled_heatmap = np.array(Image.fromarray(gaze_heatmap_pred).resize(img_size, resample=Image.BILINEAR))
        
        # Convert PIL image to BGR for cv2
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        gaze_x = pred_x / model.out_size[0] * img_size[0]
        gaze_y = pred_y / model.out_size[1] * img_size[1]
        
        visualization_pred = overlay_heatmap_on_image(
            image_bgr, 
            scaled_heatmap, 
            head_bbox, 
            gaze_xy=(gaze_x, gaze_y), 
            inout=inout
        )
        
        # Convert back to RGB for Gradio
        visualization_pred_rgb = cv2.cvtColor(visualization_pred, cv2.COLOR_BGR2RGB)
        
        status_text = f"Gaze Point: ({gaze_x:.2f}, {gaze_y:.2f}) | In-frame: {inout}"
        
        return visualization_pred_rgb, status_text

# Create Gradio interface
with gr.Blocks(title="GazeAnywhere Web UI") as demo:
    gr.Markdown("# GazeAnywhere Web UI")
    gr.Markdown("Interactive Gradio interface for local inference.")
    
    with gr.Row():
        with gr.Column():
            config_input = gr.Textbox(label="Config File Path", value="configs/base.py")
            weights_input = gr.Textbox(label="Model Weights Path", value="checkpoints/model.pth")
            image_input = gr.Image(type="pil", label="Input Image")
            text_input = gr.Textbox(label="Text Prompt", placeholder="Describe the person whose gaze you want to predict")
            dark_infer_checkbox = gr.Checkbox(label="Use Dark Inference", value=False)
            submit_btn = gr.Button("Predict Gaze", variant="primary")
            
        with gr.Column():
            output_image = gr.Image(label="Prediction Visualization")
            output_status = gr.Textbox(label="Status")
            
    submit_btn.click(
        fn=infer,
        inputs=[image_input, text_input, config_input, weights_input, dark_infer_checkbox],
        outputs=[output_image, output_status]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
