import json
import os
import random
import pandas as pd

data_dir = "/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/Gaze-Co/data"

annotation_paths = [
    "/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/Gaze-Co/anygaze_train_annotations.txt",
    "/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/Gaze-Co/benchmark/anygaze_gazefollow_test_annotations.txt",
    "/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/Gaze-Co/benchmark/anygaze_childplay_test_annotations.txt",
    "/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/Gaze-Co/benchmark/anygaze_videoattentiontarget_test_annotations.txt"
]

all_data = []
for annotation_path in annotation_paths:
    with open(annotation_path, 'r') as f:
        all_data.extend([line.strip().split(";")[0] for line in f])

all_image_paths = set()
for item in all_data:
    all_image_paths.add(os.path.join(data_dir, item))

for image_path in os.walk(data_dir):
    for filename in image_path[2]:
        full_path = os.path.join(image_path[0], filename)
        if full_path not in all_image_paths:
            os.remove(full_path)
            print(f"Removed: {full_path}")