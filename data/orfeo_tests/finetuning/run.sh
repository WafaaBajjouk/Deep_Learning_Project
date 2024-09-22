#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --exclude=gpu[003]
#SBATCH --time=1:00:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=48
#SBATCH --gpus=2
#SBATCH --output=slurm.out

#exit on error
set -eo pipefail

#print header
date; echo $SLURM_JOB_ID; srun hostname; echo -e "-----------------------------\n"

#activate Python environment
source ~/.bashrc
conda activate deep-gpu

python finetune.py

#print footer
echo -e "\n-----------------------------\nDone"; date
