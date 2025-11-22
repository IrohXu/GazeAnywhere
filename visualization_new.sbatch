#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1 
#SBATCH --job-name="asd_gaze"
#SBATCH --partition=HCESC-L40S-normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64g
#SBATCH --output=./visualization_dinov3.log

source /projects/illinois/eng/cs/jrehg/users/xucao2/miniconda3/etc/profile.d/conda.sh
export PATH=/projects/illinois/eng/cs/jrehg/users/xucao2/miniconda3/bin:$PATH
export HF_HOME=/projects/illinois/eng/cs/jrehg/users/xucao2/checkpoints/huggingface

conda activate smolagent
export PATH=/projects/illinois/eng/cs/jrehg/users/xucao2/miniconda3/envs/smolagent/bin:$PATH
python=/projects/illinois/eng/cs/jrehg/users/xucao2/miniconda3/envs/smolagent/bin/python

nvidia-smi

# python tools/visualize_on_gazefollow5_new.py \
#   --config_file /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/configs/anygaze_dinov3txt_large_text_concept_256.py \
#   --input_path /projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/WeillCornell_PedabyteProject_processed/for_annotations/PWC/PWC022_2019_06_13_T1_panasonic_ESCS_merged_35448_40627/clipped_frames \
#   --text "attribute: child with gold brown hair, wearing a dark red and gray top; position: center left of the image" \
#   --output_path /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/visualization/PWC022_2019_06_13_T1_panasonic_ESCS_merged_35448_40627 \
#   --model_weights /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/output/anygaze_dinov3txt_large_text_concept_256_new/model_final.pth

# python tools/visualize_on_gazefollow5_new.py \
#   --config_file /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/configs/anygaze_dinov3txt_large_text_concept_256.py \
#   --input_path /projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/WeillCornell_PedabyteProject_processed/for_annotations/PWC/PWC007_2019_02_23_T1_panasonic_ESCS_merged_43518_49870/clipped_frames \
#   --text "attribute: child with black hair, wearing a white and purple top; position: center left of the image" \
#   --output_path /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/visualization/PWC007_2019_02_23_T1_panasonic_ESCS_merged_43518_49870 \
#   --model_weights /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/output/anygaze_dinov3txt_large_text_concept_256_new/model_final.pth

python tools/visualize_on_gazefollow5_new.py \
  --config_file /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/configs/anygaze_dinov3txt_large_text_concept_256.py \
  --input_path /projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/WeillCornell_PedabyteProject_processed/for_annotations/PWC/PWC041_2019_09_06_T1_panasonic_ESCS_merged_66052_72116/clipped_frames \
  --text "attribute: child with black hair; position: center of the image" \
  --output_path /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/visualization/PWC041_2019_09_06_T1_panasonic_ESCS_merged_66052_72116 \
  --model_weights /projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/output/anygaze_dinov3txt_large_text_concept_256_new/model_final.pth