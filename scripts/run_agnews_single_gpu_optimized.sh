#!/bin/bash
#SBATCH -J agnews_1gpu_opt
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --time=0-02:00:00
#SBATCH -p gpu
#SBATCH --output=agnews_1gpu_opt_%j.out
#SBATCH --error=agnews_1gpu_opt_%j.err

echo "=========================================="
echo "AG News single-GPU OPTIMIZED job started at: $(date)"
echo "Job ID:          $SLURM_JOB_ID"
echo "Node list:       $SLURM_JOB_NODELIST"
echo "Num nodes:       $SLURM_JOB_NUM_NODES"
echo "GPUs requested:  1"
echo "Config:          ag_news_single_gpu_optimized.yaml"
echo "Model:           d_model=256, layers=4, FFN=1024"
echo "Batch size:      128"
echo "=========================================="
echo ""

# Always run from submission directory
cd "$SLURM_SUBMIT_DIR" || exit 1

# Load Python (same as your working script)
module load lang/Python/3.11.5-GCCcore-13.2.0

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
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("GPU memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
EOF

echo ""
echo "=========================================="
echo "Starting training (single GPU - OPTIMIZED)"
echo "=========================================="

START_TIME=$(date +%s)

python src/train.py --config configs/ag_news_single_gpu_optimized.yaml

END_TIME=$(date +%s)

echo ""
echo "=========================================="
echo "Training completed!"
echo "Job ended at: $(date)"
echo "Total elapsed time: $((END_TIME - START_TIME)) seconds"
echo "=========================================="

echo ""
echo "Output dirs under experiments/:"
ls -lh experiments || echo "No experiments directory?"
