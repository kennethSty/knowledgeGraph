#!/bin/bash
#SBATCH --output=slurm_embedding_output_%j.txt
#SBATCH --time=10:00
#SBATCH --nodes=1
#SBATCH --mem=1000
#SBATCH -p dev_gpu_4
#SBATCH --gres=gpu:1
srun -u python -m data_preprocessing.text2vec
