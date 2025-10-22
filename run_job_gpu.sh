#!/bin/bash

#SBATCH --job-name=mmgraph-daniyal-gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --account=ufdatastudios
#SBATCH --qos=ufdatastudios
#SBATCH --partition=hpg-b200
#SBATCH --gpus=2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abbasidaniyal@ufl.edu
#SBATCH --time=1:00:00
#SBATCH --output=logs/%j_mmgraph_rag_test.log   # Standard output and error log

module purge

ml conda

uv sync

# pip install -U "magic-pdf[full]" -q
pip install -U "magic-pdf[full]" "mineru[core]" -q



# uv pip install modelscope -q
modelscope download --model sentence-transformers/all-MiniLM-L6-v2 --cache_dir ./cache/all-MiniLM-L6-v2
# uv run modelscope download --model sentence-transformers/all-MiniLM-L6-v2

echo "DONE"

LOG_LEVEL=DEBUG uv run mmgraphrag/mmgraphrag_test.py

echo "JOB COMPLETED"
