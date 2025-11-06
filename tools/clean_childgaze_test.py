import os
import glob
import pandas as pd
import numpy as np
import torch
from PIL import Image

datasets_root = "/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/gaze_datasets"
childgaze_training_path = "/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/gaze_datasets/childgaze/test_annotations_release.txt"

output_path = "/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/gaze_datasets/test_annotations_childgaze.txt"

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
    "text5" : [],
    "text10" : [],
}


def process_childgaze(anno_root):
    column_names = [
        "path",
        "idx",
        "body_bbox_x",
        "body_bbox_y",
        "body_bbox_w",
        "body_bbox_h",
        "eye_x",
        "eye_y",
        "gaze_x",
        "gaze_y",
        "head_x_min",
        "head_y_min",
        "head_x_max",
        "head_y_max",
        # "inout",
        "meta0",
        "meta1",
        "text5",
        "text10"
    ]
    df = pd.read_csv(
        anno_root,
        sep=",",
        names=column_names,
        index_col=False,
        encoding="utf-8-sig",
    )
    # df = df[
    #     df["inout"] != -1
    # ]  # only use "in" or "out "gaze. (-1 is invalid, 0 is out gaze)

    for idx, row in df.iterrows():
        path = "childgaze/" + row["path"]
        idx = row["idx"]
        gaze_x = row["gaze_x"]
        gaze_y = row["gaze_y"]
        head_x_min = row["head_x_min"]
        head_y_min = row["head_y_min"]
        head_x_max = row["head_x_max"]
        head_y_max = row["head_y_max"]
        inout = 1
        source = "childgaze"
        meta0 = row["meta0"]
        meta1 = row["meta1"]
        text5 = "a child"
        text10 = "a child"

        output_dict["path"].append(path)
        output_dict["idx"].append(idx)
        output_dict["gaze_x"].append(gaze_x)
        output_dict["gaze_y"].append(gaze_y)
        output_dict["head_x_min"].append(head_x_min)
        output_dict["head_y_min"].append(head_y_min)
        output_dict["head_x_max"].append(head_x_max)
        output_dict["head_y_max"].append(head_y_max)
        output_dict["inout"].append(1)
        output_dict["source"].append(source)
        output_dict["meta0"].append(meta0)
        output_dict["meta1"].append(meta1)
        output_dict["text5"].append(text5)
        output_dict["text10"].append(text10)
    

process_childgaze(childgaze_training_path)

idx = 1
for i in range(len(output_dict["idx"])):
    output_dict["idx"][i] = idx
    idx += 1
    
df = pd.DataFrame(output_dict)
df.to_csv(output_path, index=False)

