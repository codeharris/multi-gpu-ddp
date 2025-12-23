#!/bin/bash
#SBATCH -J amazon_hf_single
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --time=0-02:00:00
#SBATCH -p gpu
#SBATCH --output=%j.out
#SBATCH --error=%j.err

echo "=========================================="
echo "Single-node DDP job started at: $(date)"
echo "Job ID:          $SLURM_JOB_ID"
echo "Node list:       $SLURM_JOB_NODELIST"
echo "Num nodes:       $SLURM_JOB_NUM_NODES"
echo "GPUs requested:  4"
echo "=========================================="
echo ""

# Always run from submission directory
cd "$SLURM_SUBMIT_DIR" || exit 1

# Load Python (adapt if needed)
module load lang/Python/3.11.5-GCCcore-13.2.0
module load data/scikit-learn
module load vis/matplotlib
module load bio/Seaborn/0.13.2-gfbf-2023b
module load lib/mpi4py/3.1.5-gompi-2023b
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0

# Activate venv
source .venv_cluster/bin/activate

# Make Python output unbuffered
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print environment sanity check
echo "Python path: $(which python)"
python - <<EOF
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
EOF

echo ""
echo "=========================================="
echo "Starting torchrun (single-node DDP)"
echo "=========================================="

START_TIME=$(date +%s)

torchrun \
  --nproc_per_node=1 \
  src/train.py \
  --config configs/amazon_hf_single_node_20ep.yaml

END_TIME=$(date +%s)

echo ""
echo "=========================================="
echo "DDP training completed!"
echo "Job ended at: $(date)"
echo "Total elapsed time: $((END_TIME - START_TIME)) seconds"
echo "=========================================="

echo ""
echo "Output dirs under experiments/:"
ls -lh experiments || echo "No experiments directory?"
