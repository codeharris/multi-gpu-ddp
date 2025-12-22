# Multi-GPU Distributed Training for Transformers

This project demonstrates Distributed Data Parallel (DDP) training of transformer models using PyTorch on HPC systems with multiple GPUs.

## 🎯 Project Overview

An investigation into distributed deep learning with PyTorch DDP, focusing on understanding when multi-GPU training succeeds and when it fails. The project includes:

- Implementation of transformer-based text classification with DDP
- Systematic comparison of model sizes and their scaling behavior
- Analysis of communication overhead vs computation time
- Comprehensive theoretical and empirical analysis

**Key Finding**: Both small (2M params) and medium (10M params) models exhibited negative scaling (0.46× and 0.80× speedup respectively) on 4 GPUs, demonstrating that model size and compute-to-communication ratio are critical for effective distributed training.

## 📊 Results Summary

| Configuration | 1 GPU | 4 GPUs | Speedup | Efficiency |
|---------------|-------|---------|---------|------------|
| Small Model (128-dim, 2 layers) | 19.53s | 42.34s | 0.46× | 11.5% |
| Large Model (256-dim, 4 layers) | 99.94s | 124.95s | 0.80× | 20.0% |

**Dataset**: AG News (120k samples, 4-class topic classification)

## 🏗️ Project Structure

```
.
├── configs/                          # YAML configuration files
│   ├── ag_news_single_gpu_optimized.yaml
│   ├── ag_news_ddp_4gpu_optimized.yaml
│   └── ...
├── src/
│   ├── train.py                     # Main training script
│   ├── distributed.py               # DDP setup and utilities
│   ├── config.py                    # Configuration parser
│   ├── data/
│   │   └── dataset.py               # Data loading with DistributedSampler
│   ├── models/
│   │   └── transformer_model.py     # Transformer implementation
│   └── utils/
│       ├── logging.py
│       └── metrics.py
├── scripts/
│   ├── run_agnews_single_gpu_optimized.sh
│   ├── run_agnews_ddp_optimized.sh
│   ├── generate_final_plots.py      # Generate report figures
│   └── plot_comprehensive_analysis.py
├── experiments/                      # Experiment results
│   ├── exp043_agnews_1gpu_optimized/
│   ├── exp044_agnews_ddp_4gpu_optimized/
│   └── plots/
│       └── final_report/            # Final report figures
├── docs/
│   ├── project_overview.md
│   ├── system_design.md
│   └── hpc_setup.md
├── HPC_Project_Final_Report.tex     # Complete LaTeX report
├── OVERLEAF_INSTRUCTIONS.md         # How to compile report
├── FINAL_SUMMARY.md                 # Project summary
└── requirements.txt
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- PyTorch 2.3+
- CUDA-capable GPUs
- SLURM (for HPC clusters)

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/project-ml.git
cd project-ml

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # or `.venv_cluster` on HPC

# Install dependencies
pip install -r requirements.txt
```

### Running Experiments

**Single GPU:**
```bash
python src/train.py --config configs/ag_news_single_gpu_optimized.yaml
```

**4 GPUs with DDP:**
```bash
torchrun --nproc_per_node=4 src/train.py --config configs/ag_news_ddp_4gpu_optimized.yaml
```

**On HPC with SLURM:**
```bash
sbatch scripts/run_agnews_single_gpu_optimized.sh
sbatch scripts/run_agnews_ddp_optimized.sh
```

## 📈 Generating Plots

```bash
# Generate comprehensive analysis plots
python scripts/plot_comprehensive_analysis.py \
  --root experiments \
  --out experiments/plots \
  --warmup 1 \
  --datasets AGNews

# Generate final report figures
python scripts/generate_final_plots.py
```

Plots will be saved to:
- `experiments/plots/` - Comprehensive analysis
- `experiments/plots/final_report/` - Report-ready figures

## 📄 Report

The complete project report is available as a LaTeX document:
- **Main document**: `HPC_Project_Final_Report.tex`
- **Compilation instructions**: `OVERLEAF_INSTRUCTIONS.md`
- **Quick summary**: `FINAL_SUMMARY.md`

### Report Highlights

- Theoretical analysis of D-SGD and Ring All-Reduce
- Communication complexity analysis
- Amdahl's Law application to explain efficiency losses
- Root cause analysis: compute-to-communication ratio
- Practical guidelines for when DDP succeeds/fails

## 🔬 Key Findings

### 1. Negative Scaling Observed
Both model configurations showed slower training on 4 GPUs vs 1 GPU due to communication overhead dominating computation time.

### 2. Compute-to-Communication Ratio Critical
For positive speedup: R = T_compute / T_comm ≥ 3.0

Our results:
- Small model: R ≈ 0.5 (bad)
- Large model: R ≈ 1.0 (still insufficient)

### 3. When DDP Succeeds
Distributed training achieves positive speedup when:
- Model size: 100M+ parameters (vs our 2-10M)
- Batch size: 64+ per GPU (vs our 8-32)
- Sequence length: 512+ tokens (vs our 128-256)
- Interconnect: NVLink or InfiniBand (vs PCIe)

### 4. Lessons Learned
- Small models (< 50M params) often better on single GPU
- Always profile T_compute and T_comm before deploying DDP
- Consider gradient accumulation as alternative
- Mixed precision (FP16) reduces communication by 2×

## 📚 Documentation

- **[Project Overview](docs/project_overview.md)**: High-level goals and design
- **[System Design](docs/system_design.md)**: Architecture and implementation
- **[HPC Setup](docs/hpc_setup.md)**: Cluster configuration
- **[Optimization Guide](OPTIMIZATION_GUIDE.md)**: Detailed scaling analysis
- **[Overleaf Instructions](OVERLEAF_INSTRUCTIONS.md)**: Compile the report

## 🛠️ Configuration

Example configuration (see `configs/` for complete files):

```yaml
experiment_name: "ag_news_ddp_4gpu_optimized"
model:
  d_model: 256
  n_heads: 8
  num_layers: 4
  dim_feedforward: 1024

data:
  dataset: "ag_news"
  num_workers: 4

training:
  epochs: 20
  batch_size: 128  # 32 per GPU in DDP
  learning_rate: 0.001

distributed:
  use_ddp: true
  backend: "nccl"
```

## 🧪 Experiments

| Experiment | Description | Speedup |
|------------|-------------|---------|
| exp041 | Small model, 1 GPU | 1.00× |
| exp042 | Small model, 4 GPUs | 0.46× ❌ |
| exp043 | Large model, 1 GPU | 1.00× |
| exp044 | Large model, 4 GPUs | 0.80× ❌ |

## 👥 Authors

- Amine, Franco, Heriel, Ludovic

**Course**: High-Performance Computing  
**Date**: December 2024

---

**⚠️ Important**: This project demonstrates that distributed training is not always beneficial. The negative scaling results provide valuable insights into when DDP should (and should not) be used.
