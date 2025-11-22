import json
import os
import random
import pandas as pd

raw_annotations_path = '/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/gaze_datasets/train_annotations.txt'
text_annotations_paths = [
    "/projects/illinois/eng/cs/jrehg/users/houzey2/CVPR2026/Gemini_API/New_Annotation/train_gf_0-5000.jsonl",
    "/projects/illinois/eng/cs/jrehg/users/houzey2/CVPR2026/Gemini_API/New_Annotation/train_gf_5000-10000.jsonl",
    "/projects/illinois/eng/cs/jrehg/users/houzey2/CVPR2026/Gemini_API/New_Annotation/train_gf_10000-15000.jsonl",
    "/projects/illinois/eng/cs/jrehg/users/houzey2/CVPR2026/Gemini_API/New_Annotation/train_gf_15000-20000.jsonl",
    "/projects/illinois/eng/cs/jrehg/users/houzey2/CVPR2026/Gemini_API/New_Annotation/train_gf_20000-25000.jsonl",
    "/projects/illinois/eng/cs/jrehg/users/houzey2/CVPR2026/Gemini_API/New_Annotation/train_gf_25000-30000.jsonl",
    "/projects/illinois/eng/cs/jrehg/users/houzey2/CVPR2026/Gemini_API/New_Annotation/train_gf_30000-35000.jsonl",
    "/projects/illinois/eng/cs/jrehg/users/houzey2/CVPR2026/Gemini_API/New_Annotation/train_gf_35000-40000.jsonl",
    "/projects/illinois/eng/cs/jrehg/users/houzey2/CVPR2026/Gemini_API/New_Annotation/train_gf_40000-45000.jsonl",
    "/projects/illinois/eng/cs/jrehg/users/houzey2/CVPR2026/Gemini_API/New_Annotation/train_gf_45000-50000.jsonl",
    "/projects/illinois/eng/cs/jrehg/users/houzey2/CVPR2026/Gemini_API/New_Annotation/train_gf_50000-55000.jsonl",
    "/projects/illinois/eng/cs/jrehg/users/houzey2/CVPR2026/Gemini_API/New_Annotation/train_gf_55000-60000.jsonl",
    "/projects/illinois/eng/cs/jrehg/users/houzey2/CVPR2026/Gemini_API/New_Annotation/train_gf_60000-65000.jsonl",
    "/projects/illinois/eng/cs/jrehg/users/houzey2/CVPR2026/Gemini_API/New_Annotation/train_gf_65000-70000.jsonl",
    "/projects/illinois/eng/cs/jrehg/users/houzey2/CVPR2026/Gemini_API/New_Annotation/train_gf_70000-75000.jsonl",
    "/projects/illinois/eng/cs/jrehg/users/houzey2/CVPR2026/Gemini_API/New_Annotation/train_gf_75000-80000.jsonl",
    "/projects/illinois/eng/cs/jrehg/users/houzey2/CVPR2026/Gemini_API/New_Annotation/train_gf_80000-83494.jsonl",
    "/projects/illinois/eng/cs/jrehg/users/houzey2/CVPR2026/Gemini_API/New_Annotation/train_vat_83494-88493.jsonl",
    "/projects/illinois/eng/cs/jrehg/users/houzey2/CVPR2026/Gemini_API/New_Annotation/train_vat_88493-93494.jsonl",
    "/projects/illinois/eng/cs/jrehg/users/houzey2/CVPR2026/Gemini_API/New_Annotation/train_vat_93494-98494.jsonl",
    "/projects/illinois/eng/cs/jrehg/users/houzey2/CVPR2026/Gemini_API/New_Annotation/train_vat_98494-103494.jsonl",
    "/projects/illinois/eng/cs/jrehg/users/houzey2/CVPR2026/Gemini_API/New_Annotation/train_vat_103494-108494.jsonl",
    "/projects/illinois/eng/cs/jrehg/users/houzey2/CVPR2026/Gemini_API/New_Annotation/train_cp_107044-112044.jsonl",
    "/projects/illinois/eng/cs/jrehg/users/houzey2/CVPR2026/Gemini_API/New_Annotation/train_cp_112044-117044.jsonl",
    "/projects/illinois/eng/cs/jrehg/users/houzey2/CVPR2026/Gemini_API/New_Annotation/train_cp_117044-end.jsonl",
]

output_path = '/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/gaze_datasets/anygaze_train_annotations_new.txt'

# raw_annotations_path = '/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/gaze_datasets/videoattentiontarget_test_annotations.txt'
# raw_annotations_path = '/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/gaze_datasets/test_annotations_childplay.txt'
# text_annotations_paths = [
#     # "/projects/illinois/eng/cs/jrehg/users/houzey2/CVPR2026/Gemini_API/New_Annotation/test_gf_all.jsonl",
#     # "/projects/illinois/eng/cs/jrehg/users/houzey2/CVPR2026/Gemini_API/New_Annotation/test_cp_all.jsonl",
#     # "/projects/illinois/eng/cs/jrehg/users/houzey2/CVPR2026/Gemini_API/New_Annotation/test_vat_all.jsonl"
# ]

# output_path = '/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/gaze_datasets/anygaze_videoattentiontarget_test_annotations.txt'
# output_path = '/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/gaze_datasets/anygaze_childplay_test_annotations.txt'

text_data = []
for text_annotations_path in text_annotations_paths:
    with open(text_annotations_path, 'r') as f:
        text_data.extend([json.loads(line) for line in f])

random.shuffle(text_data)

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
    "attribute" : [],
    "position": [],
    "action": [],
    "pose": []
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

exist_set = set()

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
    
    if idx in exist_set:
        continue
    
    if idx not in text_dict:
        continue

    if text_dict[idx]["attribute"] == "" or text_dict[idx]["position"] == "" or text_dict[idx]["action"] == "" or text_dict[idx]["pose"] == "":
        continue
     
    if text_dict[idx]["action"] == "none" and text_dict[idx]["pose"] == "none":
        continue
    
    if text_dict[idx]["attribute"] == "none" and text_dict[idx]["position"] == "none":
        continue
    
    # if '[PARSE_ERROR]' in text_dict[idx]["phrase5"] or '[PARSE_ERROR]' in text_dict[idx]["desc10"]:
    #     continue
    
    # if '[NO_CANDIDATES]' in text_dict[idx]["phrase5"] or '[NO_CANDIDATES]' in text_dict[idx]["desc10"]:
    #     continue

    # if '[API_ERROR]' in text_dict[idx]["phrase5"] or '[API_ERROR]' in text_dict[idx]["desc10"]:
    #     continue

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
    
    output_dict["attribute"].append(text_dict.get(idx, {}).get("attribute", ""))
    output_dict["position"].append(text_dict.get(idx, {}).get("position", ""))
    output_dict["action"].append(text_dict.get(idx, {}).get("action", ""))
    output_dict["pose"].append(text_dict.get(idx, {}).get("pose", ""))
    
    exist_set.add(idx)


df = pd.DataFrame(output_dict)
df.to_csv(output_path, index=False, sep=";")
