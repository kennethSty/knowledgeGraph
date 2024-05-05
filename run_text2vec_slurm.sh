#!/bin/bash
#SBATCH --output=slurm_embedding_output_%j.txt
#SBATCH --time=14:00:00
#SBATCH --nodes=1
#SBATCH --mem=20000
#SBATCH -p gpu_4
#SBATCH --gres=gpu:1
srun -u python -m data_preprocessing.text2vec
