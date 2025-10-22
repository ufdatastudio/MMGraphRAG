#!/bin/bash

#SBATCH --job-name=mmgraph-daniyal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=hpg-default
#SBATCH --account=ufdatastudios
#SBATCH --gpus=0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abbasidaniyal@ufl.edu
#SBATCH --time=1:00:00

# pip install -U "magic-pdf[full]" -q
pip install -U "magic-pdf[full]" "mineru[core]" -q

ml conda

# cache/requirements.sh
uv sync

uv pip install modelscope -q
# uv run modelscope download --model sentence-transformers/all-MiniLM-L6-v2 --cache_dir ./cache/all-MiniLM-L6-v2
uv run modelscope download --model sentence-transformers/all-MiniLM-L6-v2

echo "DONE"

uv run mmgraphrag/mmgraphrag_test.py

echo "JOB COMPLETED"
