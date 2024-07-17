#!/bin/bash
#SBATCH --output=slurm_graphtransf_output_%j.txt
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --mem=90000
#SBATCH -p dev_gpu_4
#SBATCH --gres=gpu:1
srun -u python manual_construction.py
