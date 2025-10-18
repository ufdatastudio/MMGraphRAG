#!/bin/bash

#SBATCH --job-name=mmgraph-daniyal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=hpg-default
#SBATCH --gpus=0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abbasidaniyal@ufl.edu
#SBATCH --time=1:00:00


pip install -U "magic-pdf[full]"
pip install -U "mineru[core]"

ml conda

# cache/requirements.sh
uv sync

uv pip install modelscope
uv run modelscope download --model sentence-transformers/all-MiniLM-L6-v2



uv run mmgraphrag/mmgraphrag_test.py

