#!/bin/sh -l
#SBATCH -J transformer_ddp_1node      # Job name
#SBATCH -N 1                          # Number of nodes = 1
#SBATCH --ntasks-per-node=4           # One task per GPU (4 GPUs)
#SBATCH --gres=gpu:4                  # GPUs per node
#SBATCH -c 4                          # CPU cores per task
#SBATCH --time=0-00:30:00             # Max time
#SBATCH -p gpu                        # GPU partition on Iris
#SBATCH --output=ddp_1node_%j.out
#SBATCH --error=ddp_1node_%j.err

echo "=========================================="
echo "Single-node DDP job started at: $(date)"
echo "Job ID:          $SLURM_JOB_ID"
echo "Node list:       $SLURM_JOB_NODELIST"
echo "Num nodes:       $SLURM_JOB_NUM_NODES"
echo "Tasks per node:  $SLURM_NTASKS_PER_NODE"
echo "GPUs per node:   (from --gres=gpu:X) 4"
echo "=========================================="
echo ""

# ---------- Load modules / environment ----------
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0

cd "$SLURM_SUBMIT_DIR"

echo "Python path: $(which python)"
python - << 'EOF'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
EOF
echo ""
# ------------------------------------------------

echo "=========================================="
echo "Setting up single-node DDP run"
echo "=========================================="
echo ""

# Number of nodes and GPUs from Slurm
NNODES=$SLURM_JOB_NUM_NODES        # should be 1
GPUS_PER_NODE=$SLURM_NTASKS_PER_NODE   # should be 4

# MASTER_ADDR = first node in the allocation
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

echo "MASTER_ADDR:     $MASTER_ADDR"
echo "MASTER_PORT:     $MASTER_PORT"
echo "NNODES:          $NNODES"
echo "GPUS_PER_NODE:   $GPUS_PER_NODE"
echo ""

# ---------- Optional: verify CUDA on the node ----------
echo "Verifying CUDA on the node..."
srun bash -c 'echo "--- Node: $(hostname) ---"; \
              python -c "import torch; print(\"  cuda.is_available:\", torch.cuda.is_available()); \
                                  print(\"  device count:\", torch.cuda.device_count())"'
echo ""
# -------------------------------------------------------

echo "Starting DDP training..."
START_TIME=$(date +%s)

srun torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    src/train.py \
    --config configs/ddp_single_node.yaml

END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "DDP training completed!"
echo "Job ended at: $(date)"
echo "Total elapsed time: ${ELAPSED_TIME} seconds ($((${ELAPSED_TIME} / 60)) minutes)"
echo "=========================================="

echo ""
echo "Output dirs under experiments/:"
ls -lh experiments || echo "No experiments directory?"

