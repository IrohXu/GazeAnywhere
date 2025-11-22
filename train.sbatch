#!/bin/bash
#SBATCH --time=128:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1 
#SBATCH --job-name="asd_gaze"
#SBATCH --partition=HCESC-H100-normal
#SBATCH --gres=gpu:4
#SBATCH --mem=256g
#SBATCH --output=./0050-train_anygaze-textconcept-dinov3-all.log

source /projects/illinois/eng/cs/jrehg/users/xucao2/miniconda3/etc/profile.d/conda.sh
export PATH=/projects/illinois/eng/cs/jrehg/users/xucao2/miniconda3/bin:$PATH
export HF_HOME=/projects/illinois/eng/cs/jrehg/users/xucao2/checkpoints/huggingface

conda activate smolagent
export PATH=/projects/illinois/eng/cs/jrehg/users/xucao2/miniconda3/envs/smolagent/bin:$PATH
python=/projects/illinois/eng/cs/jrehg/users/xucao2/miniconda3/envs/smolagent/bin/python

nvidia-smi

# pick a free ephemeral port
export MASTER_PORT=$(python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)

python -u tools/train.py --config-file ./configs/anygaze_dinov3txt_large_text_concept.py --num-gpu 4 --dist-url tcp://127.0.0.1:${MASTER_PORT}

