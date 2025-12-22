# Quick Reference: Running Optimized AG News Experiments

## Files Created
✅ `configs/ag_news_single_gpu_optimized.yaml` - Baseline config (exp043)
✅ `configs/ag_news_ddp_4gpu_optimized.yaml` - DDP config (exp044)
✅ `scripts/run_agnews_single_gpu_optimized.sh` - SLURM script for 1 GPU
✅ `scripts/run_agnews_ddp_optimized.sh` - SLURM script for 4 GPUs
✅ `OPTIMIZED_EXPERIMENTS.md` - Detailed execution guide

## Key Changes from Original
| Parameter | Original | Optimized | Ratio |
|-----------|----------|-----------|-------|
| d_model | 128 | 256 | 2× |
| num_layers | 2 | 4 | 2× |
| dim_feedforward | 256 | 1024 | 4× |
| batch_size | 32 | 128 | 4× |
| max_seq_len | 128 | 256 | 2× |
| learning_rate | 0.0005 | 0.001 | 2× |

## Quick Start (HPC)
```bash
# 1. Submit single GPU baseline
sbatch scripts/run_agnews_single_gpu_optimized.sh

# 2. Wait for completion, then submit DDP
sbatch scripts/run_agnews_ddp_optimized.sh

# 3. Check results
cat experiments/exp043_agnews_1gpu_optimized/metrics.csv
cat experiments/exp044_agnews_ddp_4gpu_optimized/metrics.csv
```

## Expected Results
**Original (Small Model):**
- 1 GPU: 19.53s/epoch → 4 GPUs: 42.33s/epoch (0.46× speedup ❌)

**Optimized (Larger Model):**
- 1 GPU: ~45s/epoch → 4 GPUs: ~18s/epoch (2.5× speedup ✅)
- Efficiency: 60-75% (vs 11.5% original)

## Why This Works
- **Larger model** = more compute per parameter
- **Larger batch** = better GPU utilization
- **Compute time now >> communication time** (was backwards before)
- **Better compute/communication ratio:** 0.5 → 4.0+ (8× improvement)

## See Also
- `OPTIMIZED_EXPERIMENTS.md` - Full execution guide with troubleshooting
- `OPTIMIZATION_GUIDE.md` - Theoretical analysis of scaling issues
