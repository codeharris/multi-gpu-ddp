#!/bin/bash
#SBATCH --job-name=single_gpu_baseline
#SBATCH --output=logs/single_gpu_%j.out
#SBATCH --error=logs/single_gpu_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Load modules / env as required on your cluster
# module load cuda/11.8
# module load anaconda
# source activate YOUR_ENV_NAME

echo "Running train.py on a single GPU (or CPU if no GPU)..."

python src/train.py --config configs/baseline_single_gpu.yaml
