#!/bin/bash
#SBATCH --job-name=fine_tune_gpu
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --partition=GPU

pip install torch transformers datasets evaluate accelerate peft scikit-learn
python -m torch.distributed.run --nproc_per_node=1 fine_tune.py
