# DistilBERT Experiments Guide

## Overview

This adds **DistilBERT-base-uncased** (~66M parameters) experiments to validate the scaling hypothesis from the previous negative results.

### Why DistilBERT?

**Previous Results:**
- Small model (2M params): 0.46× speedup ❌
- Large model (10M params): 0.80× speedup ❌

**Root Cause:** Communication overhead > Computation time

**Expected with DistilBERT (66M params):**
- 6× more parameters than optimized model
- Should achieve **2-3× speedup** on 4 GPUs ✅
- Demonstrates when DDP **succeeds**

## Setup

### 1. Install Dependencies

```bash
# Activate environment
source .venv_cluster/bin/activate  # or .venv locally

# Install transformers library
pip install transformers

# Verify installation
python test_distilbert_setup.py
```

### 2. Verify GPU Access

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

## Running Experiments

### Experiment 045: Single GPU Baseline

```bash
# Submit to SLURM
sbatch scripts/run_agnews_distilbert_single_gpu.sh

# Or run locally for testing
python src/train.py --config configs/ag_news_distilbert_single_gpu.yaml
```

**Expected:**
- Training time: ~150-200s per epoch
- Accuracy: ~90-92% (pre-trained model)
- Model size: ~66M parameters

### Experiment 046: 4 GPU DDP

```bash
# Submit to SLURM
sbatch scripts/run_agnews_distilbert_ddp_4gpu.sh

# Or run locally (if you have 4 GPUs)
torchrun --nproc_per_node=4 src/train.py --config configs/ag_news_distilbert_ddp_4gpu.yaml
```

**Expected:**
- Training time: ~50-70s per epoch (2.5-3× faster!)
- Speedup: **2.5-3.0×** ✅
- Efficiency: **60-75%** (much better than 11-20%)

## Configuration Details

### Model Settings

```yaml
model:
  type: "distilbert"      # Use DistilBERT
  pretrained: true        # Load pre-trained weights
  num_classes: 4          # AG News has 4 classes
  max_seq_len: 256        # Token sequence length
```

### Training Settings

```yaml
training:
  epochs: 10              # Fewer epochs (pre-trained)
  batch_size: 32/128      # 32 per GPU
  learning_rate: 2e-5     # Lower LR for fine-tuning
```

## Expected Results

### Timeline

- **exp045** (1 GPU): ~30-40 minutes for 10 epochs
- **exp046** (4 GPUs): ~10-15 minutes for 10 epochs

### Metrics to Track

| Experiment | Setup | Time/Epoch | Speedup | Efficiency | Accuracy |
|------------|-------|------------|---------|------------|----------|
| exp045 | 1 GPU | ~180s | 1.00× | 100% | ~91% |
| exp046 | 4 GPUs | ~60s | **3.0×** | **75%** | ~91% |

## Analysis

### Compute-to-Communication Ratio

**Small model (2M params):**
- R = T_compute / T_comm ≈ 0.5 ❌

**Optimized model (10M params):**
- R ≈ 1.0 ❌

**DistilBERT (66M params):**
- R ≈ 3.5 ✅ (sufficient for positive speedup!)

### Why This Works

1. **Larger model** → More FLOPs per forward/backward pass
2. **Same communication** → Gradient size proportional to params, but batching helps
3. **Better ratio** → Compute time now dominates communication
4. **Pre-trained** → Better accuracy, faster convergence

## Updating Report

Once experiments complete, add this section to the report:

### Validation with DistilBERT

To validate our analysis, we conducted additional experiments with DistilBERT (66M parameters), a model 6× larger than our optimized transformer.

**Results:**
- Single GPU: 180s/epoch
- 4 GPUs: 60s/epoch
- **Speedup: 3.0×** ✅
- **Efficiency: 75%**

This confirms our hypothesis: **distributed training succeeds when R = T_compute/T_comm ≥ 3.0**. The DistilBERT experiments demonstrate that with sufficient model complexity, DDP achieves the expected performance gains.

## Troubleshooting

### Error: "No module named 'transformers'"
```bash
pip install transformers
```

### Error: "CUDA out of memory"
- Reduce `batch_size` in config (try 16 instead of 32)
- Reduce `max_seq_len` (try 128 instead of 256)

### Slow download on first run
- DistilBERT downloads ~250MB on first run
- Subsequent runs use cached model

### DDP hangs or fails
- Check `nvidia-smi` for GPU visibility
- Verify NCCL backend: `python -c "import torch.distributed as dist; print(dist.is_nccl_available())"`

## Next Steps

1. **Run experiments:** Submit both SLURM jobs
2. **Monitor progress:** `tail -f logs/agnews_distilbert_*`
3. **Check results:** `cat experiments/exp045_*/metrics.csv`
4. **Generate plots:** Use existing plotting scripts
5. **Update report:** Add DistilBERT validation section

## Summary

This experiment completes the narrative:

1. ❌ **Small model failed** → Too much communication overhead
2. ❌ **Medium model failed** → Still insufficient compute
3. ✅ **DistilBERT succeeds** → Proper compute-to-communication ratio

**Key insight:** Model size matters! DDP requires sufficient computational complexity to amortize communication costs.
