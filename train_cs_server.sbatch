#!/bin/bash
#SBATCH -A jrehg-cs-eng
#SBATCH --time=64:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1 
#SBATCH --job-name="asd_gaze"
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --gres=gpu:4
#SBATCH --mem=256g
#SBATCH --output=./0028-train_anygaze-textconcept-dinov3.log

source /projects/illinois/eng/cs/jrehg/users/xucao2/miniconda3/etc/profile.d/conda.sh
export PATH=/projects/illinois/eng/cs/jrehg/users/xucao2/miniconda3/bin:$PATH
export HF_HOME=/projects/illinois/eng/cs/jrehg/users/xucao2/checkpoints/huggingface

conda activate smolagent
export PATH=/projects/illinois/eng/cs/jrehg/users/xucao2/miniconda3/envs/smolagent/bin:$PATH
python=/projects/illinois/eng/cs/jrehg/users/xucao2/miniconda3/envs/smolagent/bin/python

nvidia-smi

# python -u tools/train.py \
#   --config-file ./configs/all_gaze_dinov3_huge.py \
#   --num-gpu 2

# pick a free ephemeral port
export MASTER_PORT=$(python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)

# python -u tools/train.py --config-file ./configs/anygaze_dinov3txt_large_768.py --num-gpu 4 --dist-url tcp://127.0.0.1:${MASTER_PORT}
# python -u tools/train.py --config-file ./configs/anygaze_siglip2_large.py --num-gpu 4 --dist-url tcp://127.0.0.1:${MASTER_PORT}
# python -u tools/train.py --config-file ./configs/anygaze_dinov3txt_large_text_concept_256.py --num-gpu 4 --dist-url tcp://127.0.0.1:${MASTER_PORT}
# python -u tools/train.py --config-file ./configs/anygaze_dinov3txt_large_text_concept_512.py --num-gpu 2 --dist-url tcp://127.0.0.1:${MASTER_PORT}
python -u tools/train.py --config-file ./configs/anygaze_dinov3txt_large_text_concept_256_no_inout.py --num-gpu 4 --dist-url tcp://127.0.0.1:${MASTER_PORT}
# python -u tools/train.py --config-file ./configs/anygaze_clip_large_text_concept_256.py --num-gpu 2 --dist-url tcp://127.0.0.1:${MASTER_PORT}
