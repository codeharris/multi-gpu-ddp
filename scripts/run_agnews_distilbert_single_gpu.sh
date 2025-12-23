#!/bin/bash
#SBATCH --job-name=agnews_distilbert_1gpu
#SBATCH --output=logs/agnews_distilbert_1gpu_%j.out
#SBATCH --error=logs/agnews_distilbert_1gpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=gpu

# Load modules
module purge
module load PyTorch/2.3.0

# Activate virtual environment
source .venv_cluster/bin/activate

# Create logs directory
mkdir -p logs

# Print environment info
echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Config: configs/ag_news_distilbert_single_gpu.yaml"
echo ""

# Run single-GPU training with DistilBERT
python src/train.py \
    --config configs/ag_news_distilbert_single_gpu.yaml

echo ""
echo "=== Training Complete ==="
echo "Results saved to: experiments/exp045_agnews_distilbert_1gpu/"
