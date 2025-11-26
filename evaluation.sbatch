#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1 
#SBATCH --job-name="asd_gaze"
#SBATCH --partition=HCESC-H100-normal
#SBATCH --gres=gpu:1
#SBATCH --mem=128g
#SBATCH --output=./evaluation_anygaze_dinov3.log

source /projects/illinois/eng/cs/jrehg/users/xucao2/miniconda3/etc/profile.d/conda.sh
export PATH=/projects/illinois/eng/cs/jrehg/users/xucao2/miniconda3/bin:$PATH
export HF_HOME=/projects/illinois/eng/cs/jrehg/users/xucao2/checkpoints/huggingface

conda activate smolagent
export PATH=/projects/illinois/eng/cs/jrehg/users/xucao2/miniconda3/envs/smolagent/bin:$PATH
python=/projects/illinois/eng/cs/jrehg/users/xucao2/miniconda3/envs/smolagent/bin/python

nvidia-smi

python tools/eval_on_gazefollow.py \
  --config_file /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/configs/gazeanywhere_dinov3txt_large_text_concept.py \
  --model_weights /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/output/gazeanywhere_dinov3txt_large_text_concept_next/model_final.pth \
  --use_dark_inference
