import requests

head_detection_result = requests.post(
    "http://172.29.130.184:8001/detection/",
    data={
        "image_path": "/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/WeillCornell_PedabyteProject_processed/for_annotations/all/PWC007_2019_02_23_T1_panasonic_ESCS_merged_12129_15508/clipped_frames/2072.jpg",
    },
    verify=False
)

print(head_detection_result.json())

gaze_result = requests.post(
    "http://172.29.130.184:8002/gaze/",
    data={
        "image_path": "/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/WeillCornell_PedabyteProject_processed/for_annotations/all/PWC007_2019_02_23_T1_panasonic_ESCS_merged_12129_15508/clipped_frames/2072.jpg",
        "subject_bbox": head_detection_result.json(),
        "save_path": "test.jpg"
    },
    verify=False
)

print(gaze_result.json())
