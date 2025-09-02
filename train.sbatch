#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1 
#SBATCH --job-name="asd_gaze"
#SBATCH --partition=HCESC-H200-normal
#SBATCH --gres=gpu:1
#SBATCH --mem=128g
#SBATCH --output=./train_dinov2_large.log

source /projects/illinois/eng/cs/jrehg/users/xucao2/miniconda3/etc/profile.d/conda.sh
export PATH=/projects/illinois/eng/cs/jrehg/users/xucao2/miniconda3/bin:$PATH
export HF_HOME=/projects/illinois/eng/cs/jrehg/users/xucao2/checkpoints/huggingface

conda activate smolagent
export PATH=/projects/illinois/eng/cs/jrehg/users/xucao2/miniconda3/envs/smolagent/bin:$PATH
python=/projects/illinois/eng/cs/jrehg/users/xucao2/miniconda3/envs/smolagent/bin/python

nvidia-smi

python -u tools/train.py \
  --config-file ./configs/childgaze_gaze_vit_large.py \
  --num-gpu 1
