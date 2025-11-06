import os
import glob
import pandas as pd
import numpy as np
import torch
from PIL import Image

datasets_root = "/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/gaze_datasets"
videoattentiontarget_training_dir = "/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/gaze_datasets/videoattentiontarget/annotations/test"

output_path = "/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/gaze_datasets/videoattentiontarget_test_annotations.txt"

output_dict = {
    "path": [],
    "idx": [],
    "gaze_x": [],
    "gaze_y": [],
    "head_x_min": [],
    "head_y_min": [],
    "head_x_max": [],
    "head_y_max": [],
    "inout": [],
    "source": [],
    "meta0": [],
    "meta1": [],
}

        
def process_videoattentiontarget(anno_root):
    idx = 0
    for show_dir in glob.glob(os.path.join(anno_root, "*")):
        for sequence_path in glob.glob(os.path.join(show_dir, "*", "*.txt")):
            df = pd.read_csv(
                sequence_path,
                header=None,
                index_col=False,
                names=[
                    "path",
                    "x_min",
                    "y_min",
                    "x_max",
                    "y_max",
                    "gaze_x",
                    "gaze_y",
                ],
            )
            
            df = df.sample(frac=0.25, random_state=42)
            
            # coords = torch.tensor(
            #     np.array(
            #         (
            #             df["x_min"].values,
            #             df["y_min"].values,
            #             df["x_max"].values,
            #             df["y_max"].values,
            #         )
            #     ).transpose(1, 0)
            # )
            # valid_bboxes = (coords[:, 2:] >= coords[:, :2]).all(dim=1)
            # df = df.loc[valid_bboxes.tolist(), :]
            # df.reset_index(inplace=True)

            show_name = sequence_path.split("/")[-3]
            clip = sequence_path.split("/")[-2]
            df["path"] = df["path"].apply(
                lambda path: os.path.join("videoattentiontarget", "images", show_name, clip, path)
            )
            
            for idx, row in df.iterrows():
                path = row["path"]
                img = Image.open(os.path.join(datasets_root, path))
                img = img.convert("RGB")
                width, height = img.size
                gaze_x = row["gaze_x"]
                gaze_y = row["gaze_y"]

                inout = 0 if gaze_x < 0 or gaze_y < 0 else 1

                head_x_min = row["x_min"]
                head_y_min = row["y_min"]
                head_x_max = row["x_max"]
                head_y_max = row["y_max"]
                if head_x_max < head_x_min:
                    head_x_min, head_x_max = head_x_max, head_x_min
                if head_y_max < head_y_min:
                    head_y_min, head_y_max = head_y_max, head_y_min
                gaze_x, gaze_y = gaze_x / width, gaze_y / height
                if gaze_x < 0 or gaze_y < 0:
                    gaze_x, gaze_y = -1, -1
                
                source = "videoattentiontarget"
                meta0 = f"{show_name}"
                meta1 = f"{clip}"

                output_dict["path"].append(path)
                output_dict["idx"].append(idx)
                output_dict["gaze_x"].append(gaze_x)
                output_dict["gaze_y"].append(gaze_y)
                output_dict["head_x_min"].append(head_x_min)
                output_dict["head_y_min"].append(head_y_min)
                output_dict["head_x_max"].append(head_x_max)
                output_dict["head_y_max"].append(head_y_max)
                output_dict["inout"].append(inout)
                output_dict["source"].append(source)
                output_dict["meta0"].append(meta0)
                output_dict["meta1"].append(meta1)
                idx += 1

process_videoattentiontarget(videoattentiontarget_training_dir)

idx = 1
for i in range(len(output_dict["idx"])):
    output_dict["idx"][i] = idx
    idx += 1
    
df = pd.DataFrame(output_dict)
df.to_csv(output_path, index=False)

