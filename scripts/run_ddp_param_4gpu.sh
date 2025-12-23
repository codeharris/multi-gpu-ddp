#!/bin/bash
# Parameterized single-node DDP runner (config + GPUs)
# Usage:
#   ./scripts/run_imdb_ddp_single_node.sh <config_path> <gpus>
# Examples:
#   ./scripts/run_imdb_ddp_single_node.sh configs/imdb_hf_ddp_single_node_20ep.yaml 4
#   sbatch -p gpu --gres=gpu:4 -c 4 scripts/run_imdb_ddp_single_node.sh configs/amazon_polarity_hf_ddp_single_node.yaml 4

#SBATCH -J ddp_1node_param
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH -c 7
#SBATCH --time=0-02:00:00
#SBATCH -p gpu
#SBATCH --output=ddp_1node_4gpu_%j.out
#SBATCH --error=ddp_1node_4gpu_%j.err

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config_path>" >&2
  exit 1
fi

CONFIG="$1"

if [[ ! -f "$CONFIG" ]]; then
  echo "Error: config file not found: $CONFIG" >&2
  exit 1
fi

echo "=========================================="
echo "Single-node DDP job started at: $(date)"
echo "Job ID:          ${SLURM_JOB_ID:-N/A}"
echo "Node list:       ${SLURM_JOB_NODELIST:-N/A}"
echo "Num nodes:       ${SLURM_JOB_NUM_NODES:-1}"
echo "GPUs requested (torchrun):  ${GPUS}"
echo "Config:          ${CONFIG}"
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
if [[ -f .venv_cluster/bin/activate ]]; then
  source .venv_cluster/bin/activate
elif [[ -f .venv_cluster/Scripts/activate ]]; then
  source .venv_cluster/Scripts/activate
else
  echo "Warning: .venv_cluster not found; continuing without activation" >&2
fi

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
--nproc_per_node=4 \
src/train.py \
--config "$CONFIG"


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