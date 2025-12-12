#!/bin/sh -l
#SBATCH -J transformer_single_gpu      # Job name
#SBATCH -N 1                           # One node
#SBATCH --ntasks-per-node=1           # One task = one process
#SBATCH --gres=gpu:1                  # One GPU
#SBATCH -c 4                          # CPU cores for this task
#SBATCH --time=0-00:20:00             # Max time
#SBATCH -p gpu                        # GPU partition (adapt to your cluster)
#SBATCH --output=single_gpu_%j.out
#SBATCH --error=single_gpu_%j.err

echo "=========================================="
echo "Single-GPU job started at: $(date)"
echo "Job ID:          $SLURM_JOB_ID"
echo "Node list:       $SLURM_JOB_NODELIST"
echo "Num nodes:       $SLURM_JOB_NUM_NODES"
echo "Tasks per node:  $SLURM_NTASKS_PER_NODE"
echo "GPUs per node:   (from --gres=gpu:X) 1"
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
echo "Running single-GPU baseline (no DDP)"
echo "=========================================="
echo ""

START_TIME=$(date +%s)

python src/train.py --config configs/baseline_single_gpu.yaml

END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "Training completed!"
echo "Job ended at: $(date)"
echo "Total elapsed time: ${ELAPSED_TIME} seconds ($((${ELAPSED_TIME} / 60)) minutes)"
echo "=========================================="

echo ""
echo "Output dirs under experiments/:"
ls -lh experiments || echo "No experiments directory?"

