#!/bin/bash
#SBATCH --job-name=test
#SBATCH --cpus-per-task=128
#SBATCH --nodes=1
#SBATCH --partition=all

module load cuda/cuda-11.0

source ~/venv/bin/activate

python3 -u ray_train.py --config=$1 --name=$2 $3 $4
