# AnyGaze: Gaze Target Estimation with Concepts

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


