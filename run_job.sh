#!/bin/bash

#SBATCH --job-name=mmgraph-daniyal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=hpg-default
#SBATCH --account=ufdatastudios
#SBATCH --qos=ufdatastudios
#SBATCH --gpus=0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abbasidaniyal@ufl.edu
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j_mmgraph_rag_test.log   # Standard output and error log

set -e

cache/requirements.sh

echo "Requirements installed. Starting the job..."

which python

LOG_LEVEL=DEBUG python mmgraphrag/mmgraphrag_test.py

echo "JOB COMPLETED"
