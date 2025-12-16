# Multi-GPU Transformer Training with PyTorch DDP

This project demonstrates single-GPU and multi-GPU training of a Transformer-based model on an HPC system using PyTorch Distributed Data Parallel (DDP).

---

## Repository Structure

project-ml/
├── configs/
├── src/
├── scripts/
├── experiments/
│   ├── exp020_imdb_single_gpu_20ep/
│   ├── exp021_imdb_ddp_single_node_20ep/
│   └── plots/
├── REPORT.md
├── README.md

---

## Environment Setup

### Load PyTorch Module (HPC)
module load PyTorch/2.3.0

---

### Create and Activate Virtual Environment
python -m venv .venv_cluster  
source .venv_cluster/bin/activate

---

### Install Dependencies
pip install torch datasets transformers pandas matplotlib

---

## Dataset
The IMDB dataset is downloaded automatically using the Hugging Face datasets library when training starts. No manual download is required.

---

## Running Experiments

### Single-GPU Training
python src/train.py --config configs/imdb_single_gpu.yaml

---

### Multi-GPU Training (Single Node, 4 GPUs)
torchrun --nproc_per_node=4 src/train.py --config configs/imdb_ddp_single_node.yaml

---

## Outputs and Metrics
Each experiment creates a directory under experiments/ containing:
- metrics.csv with per-epoch statistics

Recorded metrics include:
- Epoch
- Training loss
- Training time per epoch
- Validation loss
- Validation accuracy

---

## Plotting Results
To generate performance plots:
python scripts/plot_results.py

Plots are saved under:
experiments/plots/

---

## Reproducibility
- All experiments run for 20 epochs
- Same model, dataset, and hyperparameters
- Only the number of GPUs changes between runs

---

## Summary
This project shows how PyTorch Distributed Data Parallel can significantly reduce training time on HPC systems while maintaining model accuracy, demonstrating the practical benefits of parallelism for deep learning workloads.
