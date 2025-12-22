# Multi-GPU DDP Performance Optimization Guide

## 📊 Analysis Summary

### Current Performance Results

| Configuration | Speedup | Efficiency | Status |
|--------------|---------|------------|--------|
| **IMDB (4 GPUs)** | **3.49x** | **87.3%** | ✅ Excellent |
| **AG News (4 GPUs)** | **0.46x** | **11.5%** | ⚠️ **Negative Scaling** |

### 🔍 Root Cause Analysis: Why AG News Fails to Scale

The AG News multi-GPU run is **slower than single GPU** due to:

1. **Model Too Small**
   - Current: 128-dim, 2 layers, 256 FFN
   - Compute time per batch: ~20-30ms
   - Gradient sync time: ~40-50ms
   - **Communication dominates computation!**

2. **Batch Size Too Small**
   - Current: 32 global → 8 per GPU
   - Modern GPUs are starved with tiny batches
   - More time spent synchronizing than computing

3. **Short Sequences**
   - max_seq_len=128 vs IMDB's 512
   - Less FLOPs per sample

4. **Amdahl's Law**
   - Fixed overheads (data loading, kernel launch, bucketization)
   - As model gets smaller, serial fraction increases

## 🚀 Optimization Strategy

### Phase 1: Immediate Improvements (Already Implemented)

**Code Optimizations:**
- ✅ `persistent_workers=True` - Avoid DataLoader worker teardown
- ✅ `torch.backends.cudnn.benchmark=True` - Auto-tune kernels
- ✅ `static_graph=True` in DDP - Skip graph re-analysis
- ✅ GPU sanity checks - Catch invalid device ordinal errors early
- ✅ Fixed sampler condition - Use `use_ddp` instead of `is_distributed`

**Result:** ~5-10% improvement expected

### Phase 2: Model & Data Configuration (New Configs Created)

**Optimized AG News Config:** `configs/ag_news_ddp_4gpu_20ep_optimized.yaml`

Changes:
```yaml
model:
  d_model: 256        # 2x larger (128 → 256)
  n_heads: 8          # 2x more attention heads
  num_layers: 4       # 2x deeper
  dim_feedforward: 1024  # 4x wider FFN
  max_seq_len: 256    # 2x longer sequences

training:
  batch_size: 128     # 4x larger (32 → 128)
  
data:
  num_workers: 4      # 2x more workers
```

**Expected Impact:**
- **Compute per batch**: 20ms → 120-150ms
- **Gradient size**: 2x larger (more amortized comm)
- **Compute-to-communication ratio**: 0.5 → 3.0
- **Target speedup**: 2.8-3.2x (75-80% efficiency)

### Phase 3: Advanced Techniques (Optional)

If Phase 2 still doesn't scale well:

1. **Gradient Accumulation**
   ```python
   # Accumulate gradients over 4 steps before sync
   accumulation_steps = 4
   effective_batch = batch_size * world_size * accumulation_steps
   ```
   - Reduces sync frequency by 4x
   - Better for very small models

2. **Mixed Precision (FP16)**
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```
   - Halves gradient size → 2x faster communication
   - ~30-40% speedup

3. **Gradient Compression**
   ```python
   DDP(..., gradient_as_bucket_view=True, 
           bucket_cap_mb=25)  # Larger buckets
   ```

## 📈 Enhanced Visualizations Generated

The new analysis script creates:

1. **`{dataset}_epoch_time_detailed.png`**
   - High-quality epoch time curves
   - Better styling and legends

2. **`{dataset}_speedup_efficiency.png`**
   - Side-by-side speedup & efficiency bars
   - Reference lines for ideal/good performance

3. **`{dataset}_throughput.png`**
   - Samples/second comparison
   - Shows actual data processing rate

4. **`{dataset}_accuracy.png`**
   - Validation accuracy convergence
   - Confirms model quality preservation

5. **`comprehensive_summary.csv`**
   - All metrics in one table
   - Ready for LaTeX/Excel

6. **`summary_table.tex`**
   - LaTeX table for direct inclusion
   - Professional formatting

7. **`performance_report.txt`**
   - Detailed diagnostic report
   - Automatic recommendations

## 🎯 Next Steps

### 1. Re-run with Optimized Configuration

```bash
# Single GPU baseline (optimized)
sbatch scripts/run_agnews_single_gpu_optimized.sh

# 4 GPU DDP (optimized)
sbatch scripts/run_agnews_ddp_optimized.sh
```

Create these scripts (examples below).

### 2. Generate Enhanced Plots

```bash
source .venv/bin/activate
python scripts/plot_comprehensive_analysis.py \
    --root experiments \
    --out experiments/plots \
    --warmup 1 \
    --datasets IMDB AGNews
```

### 3. Update Report

The diagnostic report auto-identifies issues:
- ✅ IMDB: "Excellent scaling performance"
- ⚠️ AG News: "Negative scaling detected" + recommendations

## 📝 SLURM Job Scripts

### `scripts/run_agnews_single_gpu_optimized.sh`

```bash
#!/bin/bash
#SBATCH --job-name=agnews_1gpu_opt
#SBATCH --output=agnews_1gpu_opt_%j.out
#SBATCH --error=agnews_1gpu_opt_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=gpu

module load PyTorch/2.3.0-foss-2023b-CUDA-12.6.0

cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

python src/train.py --config configs/ag_news_single_gpu_optimized.yaml
```

### `scripts/run_agnews_ddp_optimized.sh`

```bash
#!/bin/bash
#SBATCH --job-name=agnews_4gpu_opt
#SBATCH --output=agnews_4gpu_opt_%j.out
#SBATCH --error=agnews_4gpu_opt_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --partition=gpu

module load PyTorch/2.3.0-foss-2023b-CUDA-12.6.0

cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

srun --mpi=pmix \
    torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$SLURM_NODELIST:29500 \
    src/train.py --config configs/ag_news_ddp_4gpu_20ep_optimized.yaml
```

## 📊 Expected Results After Optimization

| Configuration | Before | After (Expected) |
|--------------|--------|------------------|
| **Speedup** | 0.46x | 2.8-3.2x |
| **Efficiency** | 11.5% | 70-80% |
| **Epoch Time (4 GPU)** | 42.3s | 7-9s |

## 🎓 Key Learnings

1. **Model Size Matters**
   - Small models don't scale well
   - Need high compute-to-communication ratio

2. **Batch Size is Critical**
   - Too small → GPU underutilized
   - Rule of thumb: 32-128 per GPU

3. **Communication is Expensive**
   - All-reduce scales with model size
   - Overlapping helps but has limits

4. **Measurement is Essential**
   - Always profile before scaling
   - Watch for negative scaling

## 📚 References

- [PyTorch DDP Best Practices](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [NVIDIA DDP Performance Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/operations.html)
- Amdahl's Law and parallel efficiency
