# Multi-GPU Transformer Training with PyTorch DDP

This project demonstrates single-GPU and multi-GPU training on HPC with PyTorch Distributed Data Parallel (DDP). It supports two model families:

- A lightweight custom Transformer encoder classifier
- Hugging Face pretrained sequence classifiers (fine-tuning), e.g., DistilBERT

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
- Minimal runtime (training only):
```bash
pip install -r requirements.txt
```

- With plotting utilities:
```bash
pip install pandas matplotlib
```

---

## Datasets
- IMDB: Balanced movie reviews, ~25k train/25k test. Two variants:
	- `imdb_hash`: whitespace + hash trick tokenizer (fast, vocabulary-free)
	- `imdb_hf_tok`: Hugging Face tokenizer (pairs with HF models)
- Amazon Reviews Polarity: Large-scale sentiment dataset (~3.6M train/~400k test):
	- `amazon_polarity_hash`: hash tokenizer
	- `amazon_polarity_hf_tok`: HF tokenizer

Datasets are fetched automatically via the Hugging Face `datasets` library; no manual download is needed.

---

## Running Experiments

### Custom Transformer (synthetic IMDB-like)
- Single GPU:
```bash
python src/train.py --config configs/baseline_single_gpu.yaml
```

- Single node DDP (4 GPUs example):
```bash
torchrun --nproc_per_node=4 src/train.py --config configs/ddp_single_node.yaml
```

### Hugging Face Fine-Tuning (IMDB)
- Single GPU (20 epochs):
```bash
python src/train.py --config configs/imdb_hf_single_gpu_20ep.yaml
```

- Single node DDP (4 GPUs example):
```bash
torchrun --nproc_per_node=4 src/train.py --config configs/imdb_hf_ddp_single_node_20ep.yaml
```

### Hugging Face Fine-Tuning (Amazon Polarity – large)
- Single node DDP (4 GPUs example):
```bash
torchrun --nproc_per_node=4 src/train.py --config configs/amazon_polarity_hf_ddp_single_node.yaml
```

Tip: Enable mixed precision (AMP) for GPU runs by adding `training.amp: true` to your config. Reduce per-GPU batch size if you hit OOM with HF models.

---

## Outputs and Metrics
Each experiment writes under `experiments/<exp_name>/`:
- `metrics.csv`: per-epoch statistics
- `metrics.json`: rich summary for plots/tables (config, dataset sizes, DDP info, per-epoch series, aggregates)

Recorded metrics include:
- Epoch
- Training loss
- Training time per epoch
- Validation loss
- Validation accuracy

---

## Plotting Results
To generate performance plots (IMDB 1 GPU vs 4 GPUs example):
```bash
python scripts/plot_results.py
```

Plots are saved under:
experiments/plots/

---

## Reproducibility
- Config-driven: hyperparameters, dataset, model, and DDP settings are stored in YAML.
- Distributed sharding: `DistributedSampler` ensures non-overlapping shards per rank.
- For strict reproducibility, set seeds across Python/NumPy/PyTorch and add `training.amp: false` (AMP can introduce small numeric differences). Note: deterministic cuDNN may reduce throughput.

---

## HPC Notes
### Single Node
- Use `torchrun` with `--nproc_per_node=<NUM_GPUS>`.
- Ensure the CUDA module/wheel matches your node’s driver/CUDA version.

### Multi-Node (general guidance)
- Set rendezvous environment variables on all ranks:
	- `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`, `LOCAL_RANK` (or use a launcher that exports them).
- Backend: this repo defaults to `nccl` for GPU runs.
- Example `torchrun` (rank/env typically handled by scheduler/launcher):
```bash
torchrun \
	--nnodes=$WORLD_SIZE \
	--node_rank=$RANK \
	--nproc_per_node=<NUM_GPUS> \
	--master_addr=$MASTER_ADDR \
	--master_port=$MASTER_PORT \
	src/train.py --config configs/imdb_hf_ddp_single_node_20ep.yaml
```

### Recommended Settings
- Prefer HF configs for accuracy and strong baselines; use Amazon Polarity for long HPC jobs.
- Turn on AMP (`training.amp: true`) for speed/throughput improvements on NVIDIA GPUs.
- Start with per-GPU `batch_size: 16` for DistilBERT-like models; tune to your hardware.

## Summary
This project illustrates single- and multi-GPU training with DDP, supporting both a custom Transformer and Hugging Face fine-tuning. It includes large-scale datasets (Amazon Polarity), AMP support, config-based experiments, and CSV/JSON metrics to streamline analysis and scaling studies.
