#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1 
#SBATCH --job-name="asd_gaze"
#SBATCH --partition=HCESC-L40S-normal
#SBATCH --gres=gpu:1
#SBATCH --mem=128g
#SBATCH --output=./visualization.log

source /projects/illinois/eng/cs/jrehg/users/xucao2/miniconda3/etc/profile.d/conda.sh
export PATH=/projects/illinois/eng/cs/jrehg/users/xucao2/miniconda3/bin:$PATH
export HF_HOME=/projects/illinois/eng/cs/jrehg/users/xucao2/checkpoints/huggingface

conda activate smolagent
export PATH=/projects/illinois/eng/cs/jrehg/users/xucao2/miniconda3/envs/smolagent/bin:$PATH
python=/projects/illinois/eng/cs/jrehg/users/xucao2/miniconda3/envs/smolagent/bin/python

nvidia-smi

python tools/visualize_on_gazefollow2.py \
  --config_file /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/configs/childgaze_gaze_dinov3_large_boscc.py \
  --output_path /projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/ChildGaze/visualization_child_large_boscc \
  --model_weights /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/output/all_gaze_dinov3_large/model_final.pth \
  --use_dark_inference

python tools/visualize_on_gazefollow2.py \
  --config_file /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/configs/childgaze_gaze_dinov3_large.py \
  --output_path /projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/ChildGaze/visualization_child_large \
  --model_weights /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/output/childgaze_gaze_dinov3_large/model_final.pth \
  --use_dark_inference