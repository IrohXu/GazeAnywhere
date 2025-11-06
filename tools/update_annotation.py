import os
import cv2
import json
import pandas as pd
from PIL import Image

def update_annotations(image_dir, annotation_file, output_file):
    # Load existing annotations
    annotations = pd.read_csv(annotation_file)
    
    updated_annotations = []
    

    for _, annotation in annotations.iterrows():
        print(f"Processing image: {annotation['idx']}")
        image_path = os.path.join(image_dir, annotation['path'])
        if not os.path.exists(image_path):
            print(f"Image {annotation['path']} not found, skipping.")
            continue
        x_min = annotation['head_x_min']
        y_min = annotation['head_y_min']
        x_max = annotation['head_x_max']
        y_max = annotation['head_y_max']
        # Load image to get its dimensions
        img = Image.open(image_path)
        img = img.convert("RGB")
        width, height = img.size
        x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])
        if x_max < x_min:
            x_min, x_max = x_max, x_min
        if y_max < y_min:
            y_min, y_max = y_max, y_min

        # Update annotation with image dimensions
        # annotation['width'] = width
        # annotation['height'] = height

        head_x_center = round((x_min + x_max) / (2 * width), 3)
        head_y_center = round((y_min + y_max) / (2 * height), 3)

        annotation["text5"] = f"head position: {head_x_center} {head_y_center}"
        annotation["text10"] = f"head position: {head_x_center} {head_y_center}"

        updated_annotations.append(annotation)

    pd.DataFrame(updated_annotations).to_csv(output_file, index=False)
    print(f"Updated annotations saved to {output_file}")
    
if __name__ == "__main__":
    image_directory = "/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/gaze_datasets"
    annotation_json = "/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/gaze_datasets/gazefollow_test_annotations.txt"
    output_json = "/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/gaze_datasets/gazefollow_test_pseudo_annotations.txt"

    update_annotations(image_directory, annotation_json, output_json)