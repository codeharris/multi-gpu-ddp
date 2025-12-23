#!/bin/bash
#SBATCH --job-name=agnews_distilbert_ddp_4gpu
#SBATCH --output=logs/agnews_distilbert_ddp_4gpu_%j.out
#SBATCH --error=logs/agnews_distilbert_ddp_4gpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
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
echo "Config: configs/ag_news_distilbert_ddp_4gpu.yaml"
echo ""

# Run DDP training with DistilBERT (4 GPUs)
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    src/train.py \
    --config configs/ag_news_distilbert_ddp_4gpu.yaml

echo ""
echo "=== Training Complete ==="
echo "Results saved to: experiments/exp046_agnews_distilbert_ddp_4gpu/"
