#!/bin/bash

#SBATCH --job-name=mmgraph-daniyal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=hpg-default
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abbasidaniyal@ufl.edu
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --time=4:00:00


pip install -U "magic-pdf[full]" -q
# pip install -U "magic-pdf[full]" "mineru[core]" -q

ml conda

# cache/requirements.sh
uv sync

uv pip install modelscope -q
uv run modelscope download --model sentence-transformers/all-MiniLM-L6-v2 --local_dir ./cache/all-MiniLM-L6-v2



uv run mmgraphrag/mmgraphrag_test.py

