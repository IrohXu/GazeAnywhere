import json
import os

import pandas as pd

raw_annotations_path = '/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/gaze_datasets/gazefollow_test_annotations.txt'
text_annotations_path = '/projects/illinois/eng/cs/jrehg/users/houzey2/CVPR2026/Gemini_API/gazefollow_test_word.jsonl'

output_path = '/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/gaze_datasets/gazefollow_example_annotations.txt'

with open(text_annotations_path, 'r') as f:
    text_data = [json.loads(line) for line in f]

text_dict = {item["idx"]: item for item in text_data}

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
    raw_annotations_path,
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
    source = row["source"]
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
    output_dict["text5"].append(text_dict.get(idx, {}).get("phrase5", ""))
    output_dict["text10"].append(text_dict.get(idx, {}).get("desc10", ""))


df = pd.DataFrame(output_dict)
df.to_csv(output_path, index=False)
