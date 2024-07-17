#!/bin/bash
#SBATCH --output=slurm_chunking_output_%j.txt
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --mem=1500
#SBATCH -p single
srun -u python -m data_preprocessing.page2paragraphs
