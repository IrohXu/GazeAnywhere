# GazeAnywhere: Gaze Target Estimation Anywhere with Concepts

### UIUC Rehg Lab | Google XR

## Installation
* Create a conda virtual env and activate it.

  ```
  conda env create -f environment.yml
  conda activate anygaze
  ```

* Install [detectron2](https://github.com/facebookresearch/detectron2) , follow its [documentation](https://detectron2.readthedocs.io/en/latest/), or

  ```
  pip install "git+https://github.com/facebookresearch/detectron2.git@017abbfa5f2c2a2afa045200c2af9ccf2fc6227f#egg=detectron2"
  ```

## Train/Eval
### Pre-training/Fine-tuning/Testing Dataset Preprocessing

Use our internal dataset.    

### Pretrained Model

DINOv3-txt:

```
/projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/pretrained/dinov3_vitl16_dinotxt.pth
```

### Training    

```
python -u tools/train.py --config-file ./configs/anygaze_dinov3txt_large_text_concept_64.py --num-gpu 2
```

```
python -u tools/train.py --config-file ./configs/anygaze_dinov3txt_large_text_concept_128.py --num-gpu 2
```

```
python -u tools/train.py --config-file ./configs/anygaze_dinov3txt_large_text_concept_256.py --num-gpu 2
```

```
python -u tools/train.py --config-file ./configs/anygaze_dinov3txt_large_text_concept_512.py --num-gpu 2
```

```
python -u tools/train.py --config-file ./configs/anygaze_dinov3txt_large_text_concept_1024.py --num-gpu 2
```

```
python -u tools/train.py --config-file ./configs/anygaze_siglip2_large_text_concept.py --num-gpu 1
```



## Evaluation


```
python tools/eval_on_gazefollow2.py \
  --config_file ./configs/anygaze_dinov3txt_large_text_concept.py \
  --model_weights ./output/anygaze_dinov3txt_large_text_concept/model_final.pth \
  --use_dark_inference
```

```
python tools/eval_on_videoattentiontarget2.py \
  --config_file ./configs/anygaze_dinov3txt_large_text_concept_vat.py \
  --model_weights ./output/anygaze_dinov3txt_large_text_concept/model_final.pth \
  --use_dark_inference
```

```
python tools/eval_on_videoattentiontarget2.py \
  --config_file ./configs/anygaze_dinov3txt_large_text_concept_childplay.py \
  --model_weights ./output/anygaze_dinov3txt_large_text_concept/model_final.pth \
  --use_dark_inference
```

```
python tools/visualize_on_gazefollow5.py \
  --config_file /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/configs/anygaze_dinov3txt_large_text_concept_deployment.py \
  --output_path /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/visualization/anygaze_dinov3txt_large_text_concept_deployment \
  --model_weights /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/output/anygaze_dinov3txt_large_text_concept_256_new/model_final.pth
```

python tools/latency_calculate.py \
  --config_file /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/configs/anygaze_dinov3txt_large_text_concept_vat_latency.py \
  --output_path /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/visualization/anygaze_dinov3txt_large_text_concept_latency \
  --model_weights /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/output/anygaze_dinov3txt_large_text_concept_256_new/model_final.pth


python tools/latency_calculate.py \
  --config_file /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/configs/anygaze_clip_large_text_concept_vat_latency.py \
  --output_path /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/visualization/anygaze_clip_large_text_concept_latency \
  --model_weights /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/output/anygaze_clip_large_text_concept_256/model_final.pth



```
python tools/visualize_on_gazefollow5_new.py \
  --config_file /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/configs/anygaze_dinov3txt_large_text_concept_256.py \
  --input_path /projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/WeillCornell_PedabyteProject_processed/for_annotations/PWC/PWC054_2019_11_13_T1_panasonic_ESCS_merged_9715_14975/clipped_frames \
  --text "attribute: a child with gold brown hair; position: center left of the image" \
  --output_path /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/visualization/PWC054_2019_11_13_T1_panasonic_ESCS_merged_9715_14975 \
  --model_weights /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/output/anygaze_dinov3txt_large_text_concept_256_new/model_final.pth
```


```
python tools/visualize_on_gazefollow5_new.py \
  --config_file /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/configs/anygaze_dinov3txt_large_text_concept_256.py \
  --input_path /projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/WeillCornell_PedabyteProject_processed/for_annotations/PWC/PWC041_2019_09_06_T1_panasonic_ESCS_merged_66052_72116/clipped_frames \
  --text "attribute: child with black hair; position: center of the image" \
  --output_path /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/visualization/PWC041_2019_09_06_T1_panasonic_ESCS_merged_66052_72116 \
  --model_weights /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/output/anygaze_dinov3txt_large_text_concept_256_new/model_final.pth
```

```
python tools/convert_visualization_to_video.py \
  --input /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/visualization/PWC022_2019_06_13_T1_panasonic_ESCS_merged_35448_40627 \
  --output /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/visualization/PWC022_2019_06_13_T1_panasonic_ESCS_merged_35448_40627.mp4 \
  --fps 30
```

```
python tools/cut_video.py /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/visualization/PWC022_2019_06_13_T1_panasonic_ESCS_merged_35448_40627.mp4 -s 45 -e 55 -o sample_1.mp4
```

```
python tools/convert_visualization_to_video.py \
  --input /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/visualization/PWC007_2019_02_23_T1_panasonic_ESCS_merged_43518_49870 \
  --output /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/visualization/PWC007_2019_02_23_T1_panasonic_ESCS_merged_43518_49870.mp4 \
  --fps 30
```

```
python tools/cut_video.py /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/visualization/PWC007_2019_02_23_T1_panasonic_ESCS_merged_43518_49870.mp4 -s 74 -e 83 -o sample_3.mp4
```


```
python tools/convert_visualization_to_video.py \
  --input /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/visualization/PWC041_2019_09_06_T1_panasonic_ESCS_merged_66052_72116 \
  --output /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/visualization/PWC041_2019_09_06_T1_panasonic_ESCS_merged_66052_72116.mp4 \
  --fps 30
```


```
python tools/cut_video.py /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/visualization/PWC041_2019_09_06_T1_panasonic_ESCS_merged_66052_72116.mp4 -s 142 -e 148 -o sample_4.mp4
```


```
python tools/merge_videos.py -o merged.mp4 sample_1.mp4 sample_2.mp4 sample_3.mp4 sample_4.mp4
```