#!/bin/bash
#SBATCH --job-name=ddp_single_node
#SBATCH --output=logs/ddp_single_node_%j.out
#SBATCH --error=logs/ddp_single_node_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4            # adjust to the number of GPUs per node
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

module load cuda/11.8   # or whatever your cluster uses
# activate your env here, e.g.:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate hpc-ml

GPUS_PER_NODE=4  # <- adjust to match --gres=gpu:X

echo "Running DDP on a single node with $GPUS_PER_NODE GPUs"

torchrun --nproc_per_node=$GPUS_PER_NODE \
         src/train.py \
         --config configs/ddp_single_node.yaml
