#!/bin/bash
#SBATCH --output=data_extraction/slurm_extraction_output_%j.txt
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=1000
#SBATCH -p single
srun -u python -m data_extraction.extract_wikipedia
