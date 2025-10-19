#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1 
#SBATCH --job-name="asd_gaze"
#SBATCH --partition=HCESC-H100-normal
#SBATCH --gres=gpu:1
#SBATCH --mem=128g
#SBATCH --output=./evaluation_dinov3_base_all.log

source /projects/illinois/eng/cs/jrehg/users/xucao2/miniconda3/etc/profile.d/conda.sh
export PATH=/projects/illinois/eng/cs/jrehg/users/xucao2/miniconda3/bin:$PATH
export HF_HOME=/projects/illinois/eng/cs/jrehg/users/xucao2/checkpoints/huggingface

conda activate smolagent
export PATH=/projects/illinois/eng/cs/jrehg/users/xucao2/miniconda3/envs/smolagent/bin:$PATH
python=/projects/illinois/eng/cs/jrehg/users/xucao2/miniconda3/envs/smolagent/bin/python

nvidia-smi

python tools/eval_on_gazefollow.py \
  --config_file /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/configs/gazefollow_gaze_dinov3_base.py \
  --model_weights /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/output/all_gaze_dinov3_base/model_final.pth \
  --use_dark_inference

python tools/eval_on_video_attention_target.py \
  --config_file /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/configs/videoattentiontarget_gaze_dinov3_base.py \
  --model_weights /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/output/all_gaze_dinov3_base/model_final.pth \
  --use_dark_inference
