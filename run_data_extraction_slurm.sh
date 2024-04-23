#!/bin/bash
#SBATCH --output=data_extraction/slurm_extraction_output_%j.txt
#SBATCH --time=10:00
#SBATCH --nodes=1
#SBATCH --mem=100
#SBATCH -p dev_single
srun -u python -m data_extraction.extract_wikipedia
