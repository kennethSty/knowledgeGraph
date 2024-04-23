#!/bin/bash
#SBATCH --output=data_extraction/slurm_extraction_output_%j.txt
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --mem=1000
#SBATCH -p fat
srun -u python -m data_extraction.extract_wikipedia
