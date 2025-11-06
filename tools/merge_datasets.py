import os
import glob
import pandas as pd
import numpy as np
import torch
from PIL import Image

datasets_root = "/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/gaze_datasets"
gazefollow_training_path = "/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/gaze_datasets/gazefollow/train_annotations_release.txt"
videoattentiontarget_training_dir = "/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/gaze_datasets/videoattentiontarget/annotations/train"
childplay_training_path = "/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/gaze_datasets/child_play_train.csv"
childplay_val_path = "/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/gaze_datasets/child_play_val.csv"
childplay_test_path = "/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/gaze_datasets/child_play_test.csv"

output_path = "/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/gaze_datasets/test_annotations_childplay.txt"

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


def process_gazefollow(anno_root):
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
        "inout",
        "meta0",
        "meta1",
    ]
    df = pd.read_csv(
        anno_root,
        sep=",",
        names=column_names,
        index_col=False,
        encoding="utf-8-sig",
    )
    df = df[
        df["inout"] != -1
    ]  # only use "in" or "out "gaze. (-1 is invalid, 0 is out gaze)

    for idx, row in df.iterrows():
        path = "gazefollow/" + row["path"]
        idx = row["idx"]
        gaze_x = row["gaze_x"]
        gaze_y = row["gaze_y"]
        head_x_min = row["head_x_min"]
        head_y_min = row["head_y_min"]
        head_x_max = row["head_x_max"]
        head_y_max = row["head_y_max"]
        inout = row["inout"]
        source = "gazefollow"
        meta0 = row["meta0"]
        meta1 = row["meta1"]

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
        
        
def process_childplay(anno_root):
    column_names = [
        "path",
        "idx",
        "gaze_x",
        "gaze_y",
        "head_x_min",
        "head_y_min",
        "head_x_max",
        "head_y_max",
        "inout",
        "source",
        "meta0",
        "meta1",
    ]
    df = pd.read_csv(
        anno_root,
        sep=",",
        names=column_names,
        index_col=False,
        encoding="utf-8-sig",
    )

    for idx, row in df.iterrows():
        path = row["path"]
        idx = row["idx"]
        gaze_x = row["gaze_x"]
        gaze_y = row["gaze_y"]
        head_x_min = row["head_x_min"]
        head_y_min = row["head_y_min"]
        head_x_max = row["head_x_max"]
        head_y_max = row["head_y_max"]
        inout = row["inout"]
        source = "childplay"
        meta0 = row["meta0"]
        meta1 = row["meta1"]

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

# process_gazefollow(gazefollow_training_path)
# process_videoattentiontarget(videoattentiontarget_training_dir)
# process_childplay(childplay_training_path)
# process_childplay(childplay_val_path)

process_childplay(childplay_test_path)

idx = 1
for i in range(len(output_dict["idx"])):
    output_dict["idx"][i] = idx
    idx += 1
    
df = pd.DataFrame(output_dict)
df.to_csv(output_path, index=False)

