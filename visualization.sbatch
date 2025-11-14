#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1 
#SBATCH --job-name="asd_gaze"
#SBATCH --partition=HCESC-H200-normal
#SBATCH --gres=gpu:1
#SBATCH --mem=128g
#SBATCH --output=./visualization_dinov3.log

source /projects/illinois/eng/cs/jrehg/users/xucao2/miniconda3/etc/profile.d/conda.sh
export PATH=/projects/illinois/eng/cs/jrehg/users/xucao2/miniconda3/bin:$PATH
export HF_HOME=/projects/illinois/eng/cs/jrehg/users/xucao2/checkpoints/huggingface

conda activate smolagent
export PATH=/projects/illinois/eng/cs/jrehg/users/xucao2/miniconda3/envs/smolagent/bin:$PATH
python=/projects/illinois/eng/cs/jrehg/users/xucao2/miniconda3/envs/smolagent/bin/python

nvidia-smi

python tools/visualize_on_gazefollow5.py \
  --config_file /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/configs/anygaze_dinov3txt_large_text_concept_deployment.py \
  --output_path /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/visualization/anygaze_dinov3txt_large_text_concept_deployment \
  --model_weights /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/output/anygaze_dinov3txt_large_text_concept_256_new/model_final.pth

# python tools/visualize_on_gazefollow2.py \
#   --config_file /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/configs/childgaze_gaze_dinov3_large_boscc.py \
#   --output_path /projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/ChildGaze/visualization_child_large_boscc \
#   --model_weights /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/output/all_gaze_dinov3_large/model_final.pth \
#   --use_dark_inference

# python tools/visualize_on_gazefollow2.py \
  # --config_file /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/configs/childgaze_gaze_dinov3_large.py \
  # --output_path /projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/ChildGaze/visualization_child_large \
  # --model_weights /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/output/childgaze_gaze_dinov3_large/model_final.pth \
  # --use_dark_inference

# python tools/visualize_on_gazefollow3.py \
#   --config_file /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/configs/anygaze_siglip2_large.py \
#   --output_path /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/visualization/siglip2 \
#   --model_weights /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/output/anygaze_siglip2_large/model_final.pth \
#   --use_dark_inference

# python tools/visualize_on_gazefollow3.py \
#   --config_file /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/configs/childgaze_dinov3txt_large.py \
#   --output_path /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/visualization/256 \
#   --model_weights /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/output/anygaze_dinov3txt_large_256/model_final.pth \
#   --use_dark_inference

# python tools/visualize_on_detr.py \
#   --config_file /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/configs/owl_siglip2_large.py \
#   --output_path /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/visualization/siglip2_owl \
#   --model_weights /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/output/owl_siglip2_large/model_0002144.pth

# python tools/visualize_on_gazefollow3.py \
#   --config_file /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/configs/anygaze_dinov3txt_large_text_concept.py \
#   --output_path /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/visualization/anygaze_dinov3txt_large_text_concept \
#   --model_weights /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/output/anygaze_dinov3txt_large_text_concept/model_final.pth


# python tools/visualize_on_gazefollow3.py \
#   --config_file /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/configs/anygaze_dinov3txt_large_text_concept_512.py \
#   --output_path /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/visualization/anygaze_dinov3txt_large_text_concept_512 \
#   --model_weights /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/output/anygaze_dinov3txt_large_text_concept_512/model_final.pth

# python tools/visualize_on_gazefollow4.py \
#   --config_file /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/configs/anygaze_dinov3txt_large_text_concept_childgaze.py \
#   --output_path /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/visualization/anygaze_dinov3txt_large_text_concept_childgaze \
#   --model_weights /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/output/anygaze_dinov3txt_large_text_concept_256_new/model_final.pth

# python tools/visualize_on_gazefollow4.py \
#   --config_file /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/configs/anygaze_dinov3txt_large_text_concept_gazefollow.py \
#   --output_path /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/visualization/anygaze_dinov3txt_large_text_concept_gazefollow \
#   --model_weights /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/output/anygaze_dinov3txt_large_text_concept_256_new/model_final.pth

# python tools/visualize_on_gazefollow4.py \
#   --config_file /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/configs/anygaze_dinov3txt_large_text_concept_vat.py \
#   --output_path /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/visualization/anygaze_dinov3txt_large_text_concept_vat \
#   --model_weights /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/output/anygaze_dinov3txt_large_text_concept_256_new/model_final.pth

# python tools/visualize_on_gazefollow4.py \
#   --config_file /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/configs/anygaze_dinov3txt_large_text_concept_childplay.py \
#   --output_path /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/visualization/anygaze_dinov3txt_large_text_concept_childplay \
#   --model_weights /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/output/anygaze_dinov3txt_large_text_concept_256_new/model_final.pth