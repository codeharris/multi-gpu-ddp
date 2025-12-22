# AG News Optimized Experiments - HPC Execution Guide

## Overview
This guide provides instructions for running optimized AG News experiments to address the negative scaling observed in the original small model configuration.

## Problem Summary
**Original AG News Results (Small Model):**
- 1 GPU: 19.53s/epoch
- 4 GPUs: 42.33s/epoch
- Speedup: 0.46× (NEGATIVE - slower with more GPUs!)
- Efficiency: 11.5%

**Root Cause:**
- Model too small (d_model=128, 2 layers)
- Batch size too small (32 total → 8 per GPU)
- Communication overhead > compute time

## Optimization Strategy

### Model Changes (2x scale-up)
```yaml
Original → Optimized
- d_model: 128 → 256          (2x wider)
- num_layers: 2 → 4           (2x deeper)
- n_heads: 4 → 8              (2x more attention)
- dim_feedforward: 256 → 1024 (4x larger FFN)
- max_seq_len: 128 → 256      (2x longer context)
```

### Training Changes (4x batch size)
```yaml
- batch_size: 32 → 128
- DDP: 128 total = 32 per GPU (good GPU utilization)
- learning_rate: scaled from 0.0005 → 0.001
```

### Expected Improvements
- **Compute/Communication ratio:** 0.5 → 4.0+ (8x improvement)
- **Target speedup:** 2.5-3.0× on 4 GPUs
- **Target efficiency:** 60-75%
- **Accuracy:** Should maintain ~88% on AG News

## Files Created

### Configuration Files
1. `configs/ag_news_single_gpu_optimized.yaml` (exp043)
2. `configs/ag_news_ddp_4gpu_optimized.yaml` (exp044)

### SLURM Scripts
1. `scripts/run_agnews_single_gpu_optimized.sh`
2. `scripts/run_agnews_ddp_optimized.sh`

## Execution Instructions

### Step 1: Transfer to HPC
```bash
# From your local machine, sync the project to HPC
rsync -avz --exclude='.venv' --exclude='__pycache__' \
  /path/to/project-ml/ username@hpc.cluster:/path/to/project-ml/
```

### Step 2: Submit Baseline Job (1 GPU)
```bash
# SSH to HPC cluster
ssh username@hpc.cluster
cd /path/to/project-ml

# Submit single GPU job
sbatch scripts/run_agnews_single_gpu_optimized.sh

# Check job status
squeue -u $USER

# Monitor output (replace JOBID with actual job ID)
tail -f agnews_1gpu_opt_JOBID.out
```

### Step 3: Submit DDP Job (4 GPUs)
```bash
# After single GPU job completes, submit DDP job
sbatch scripts/run_agnews_ddp_optimized.sh

# Monitor
squeue -u $USER
tail -f agnews_ddp_opt_JOBID.out
```

### Step 4: Check Results
```bash
# After both jobs complete
ls -lh experiments/exp043_agnews_1gpu_optimized/
ls -lh experiments/exp044_agnews_ddp_4gpu_optimized/

# Check metrics
cat experiments/exp043_agnews_1gpu_optimized/metrics.csv
cat experiments/exp044_agnews_ddp_4gpu_optimized/metrics.csv
```

### Step 5: Download Results
```bash
# From your local machine
rsync -avz username@hpc.cluster:/path/to/project-ml/experiments/ \
  /local/path/project-ml/experiments/
```

### Step 6: Generate Updated Plots
```bash
# On your local machine with results downloaded
source .venv/bin/activate

python scripts/plot_comprehensive_analysis.py \
  --root experiments \
  --out experiments/plots \
  --warmup 1 \
  --datasets IMDB AGNews
```

This will regenerate all plots including the new optimized AG News results.

## Expected Timeline
- Single GPU job: ~45-60 minutes (larger model + larger batch)
- 4 GPU DDP job: ~20-30 minutes (target: 2.5× speedup)
- Total HPC time: ~1.5 hours

## Troubleshooting

### Out of Memory (OOM) Errors
If you get CUDA OOM with batch_size=128:
1. Reduce to batch_size=96 or 64
2. Update learning_rate proportionally: lr = 0.001 * (new_batch / 128)

### NCCL Timeout Errors
If DDP hangs or times out:
1. Check GPU visibility: `nvidia-smi`
2. Verify NCCL works: `python -c "import torch; torch.distributed.init_process_group('nccl')"`
3. Check SLURM logs: `sacct -j JOBID --format=JobID,State,ExitCode`

### Slow Training Despite Optimizations
If speedup is still < 2.0×:
1. Increase model size further (d_model=512, layers=6)
2. Increase batch size to 256 (64 per GPU)
3. Enable gradient accumulation in config

## Success Criteria
✅ Single GPU: epoch time 40-50s (slower than original due to larger model)
✅ 4 GPU DDP: epoch time 15-20s (faster than single GPU)
✅ Speedup: > 2.5×
✅ Efficiency: > 60%
✅ Accuracy: ~88% (maintained from original)

## Next Steps After Successful Runs
1. Compare original vs optimized results
2. Update LaTeX report with new findings
3. Create before/after comparison plots
4. Document lessons learned about model size and scaling
