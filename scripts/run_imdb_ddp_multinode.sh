#!/bin/bash
#SBATCH -J imdb_ddp_2node
#SBATCH -N 2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --time=0-01:30:00
#SBATCH -p gpu
#SBATCH --output=ddp_2node_%j.out
#SBATCH --error=ddp_2node_%j.err

set -euo pipefail

echo "=========================================="
echo "Multi-node DDP job started at: $(date)"
echo "Job ID:          $SLURM_JOB_ID"
echo "Node list:       $SLURM_JOB_NODELIST"
echo "Num nodes:       $SLURM_JOB_NUM_NODES"
echo "GPUs per node:   4"
echo "Total GPUs:      $((SLURM_JOB_NUM_NODES * 4))"
echo "=========================================="
echo ""

cd "$SLURM_SUBMIT_DIR" || exit 1

# Activate venv (your working environment)
source .venv_cluster/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo ""

echo "Python path (head node): $(which python)"
python - <<'EOF'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
EOF

echo ""
echo "=========================================="
echo "Node sanity check (each node should print once)"
echo "=========================================="
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -lc '
  echo "NODE=$(hostname) PWD=$(pwd)"
  source .venv_cluster/bin/activate
  python - <<EOF
import torch
print("NODE_OK", "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
EOF
'

echo ""
echo "=========================================="
echo "Torchrun smoke test (all ranks should print HELLO)"
echo "=========================================="

# This launches 1 torchrun agent per node; each agent spawns 4 ranks on its node.
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -lc "
  source .venv_cluster/bin/activate
  export PYTHONUNBUFFERED=1
  torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    - << 'EOF'
import os, socket
print('HELLO', 'host=', socket.gethostname(),
      'RANK=', os.environ.get('RANK'),
      'LOCAL_RANK=', os.environ.get('LOCAL_RANK'),
      'WORLD_SIZE=', os.environ.get('WORLD_SIZE'),
      flush=True)
EOF
"

echo ""
echo "=========================================="
echo "Starting IMDB multi-node training"
echo "=========================================="

START_TIME=$(date +%s)

# Make sure output dir differs from exp011
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -lc "
  source .venv_cluster/bin/activate
  export PYTHONUNBUFFERED=1
  torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    src/train.py \
    --config configs/imdb_ddp_multinode.yaml
"

END_TIME=$(date +%s)

echo ""
echo "=========================================="
echo "Multi-node DDP training completed!"
echo "Job ended at: $(date)"
echo "Total elapsed time: $((END_TIME - START_TIME)) seconds"
echo "=========================================="

echo ""
echo "Output dirs under experiments/:"
ls -lh experiments || echo "No experiments directory?"
