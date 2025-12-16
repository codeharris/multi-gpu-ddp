# Multi-GPU Accelerated Transformer Training with PyTorch DDP

## 1. Project Overview

### Problem Statement and Goals
Transformer-based models are computationally expensive to train due to their large parameter count and iterative optimization process. The goal of this project is to demonstrate how high-performance computing (HPC) resources can be leveraged to accelerate training using PyTorch Distributed Data Parallel (DDP).

The project compares:
- Single-GPU training
- Multi-GPU training on a single node using 4 GPUs

The main objectives are to measure training time, compute speedup and parallel efficiency, and analyze scalability while preserving model accuracy.

---

### Dataset Description
The IMDB movie review dataset is used for binary sentiment classification (positive vs. negative). The dataset is loaded using the Hugging Face `datasets` library and represents a realistic NLP workload suitable for performance evaluation.

---

### Key Design Ideas
- Use a Transformer-based neural network to ensure a realistic compute-intensive workload
- Apply data parallelism via PyTorch Distributed Data Parallel
- Keep model architecture, dataset, and hyperparameters constant across experiments
- Log per-epoch metrics for detailed performance analysis

---

### Why HPC and Parallelism Matter
Single-GPU training underutilizes modern HPC systems. By distributing training across multiple GPUs, training time can be significantly reduced, enabling faster experimentation and more scalable deep learning workflows.

---

## 2. System Design

### Architecture Diagram

Single-node multi-GPU execution model:

+--------------------------------------------------+
|                  Compute Node                   |
|                                                  |
|  +---------+  +---------+  +---------+  +-----+ |
|  | GPU 0   |  | GPU 1   |  | GPU 2   |  | GPU3| |
|  | Rank 0  |  | Rank 1  |  | Rank 2  |  |Rank3| |
|  +----+----+  +----+----+  +----+----+  +--+--+ |
|       |             |             |         |   |
|       +-------------+-------------+---------+   |
|                     NCCL All-Reduce            |
|                                                  |
|           PyTorch Distributed Data Parallel     |
+--------------------------------------------------+

---

### Execution Model
- One process per GPU
- Each process holds a full replica of the model
- Input batches are partitioned across GPUs
- Gradients are synchronized using NCCL all-reduce
- Parameters remain consistent across processes

This follows a Single Program Multiple Data (SPMD) execution model.

---

## 3. Implementation Details

### Training Workflow
1. Load IMDB dataset
2. Tokenize text inputs
3. Initialize Transformer model
4. Wrap model with DistributedDataParallel for multi-GPU runs
5. Train for 20 epochs
6. Log metrics to CSV files

---

### Parallel Strategy
- Data parallelism
- DistributedSampler used for dataset partitioning
- Each GPU processes a different mini-batch
- Gradient synchronization occurs after each backward pass

---

### Libraries and Tools
- PyTorch
- torch.distributed (DDP)
- NCCL backend
- Hugging Face datasets
- Pandas and Matplotlib for analysis

---

## 4. Performance Evaluation

### Experimental Setup
- HPC GPU cluster
- Single node with 4 GPUs
- Experiments conducted:
  - 1 GPU, 20 epochs
  - 4 GPUs (DDP), 20 epochs
- Identical hyperparameters for all runs

---

### Metrics
- Training time per epoch
- Average training time
- Validation accuracy
- Speedup
- Parallel efficiency

---

### Scaling Metrics
Speedup is computed as:

Speedup = T_1GPU / T_4GPU

Parallel efficiency is computed as:

Efficiency = Speedup / 4

Results show substantial reduction in training time with multi-GPU execution and high parallel efficiency, with minor losses due to communication overhead.

---

### Generated Plots
- Training time per epoch (1 GPU vs 4 GPUs)
- Average epoch training time
- Speedup bar chart
- Parallel efficiency bar chart

---

## 5. Discussion

### Bottlenecks
- Gradient synchronization overhead
- Communication latency during all-reduce
- Minor I/O overhead during dataset loading

---

### What Worked Well
- PyTorch DDP provided near-linear scaling
- Minimal changes required to scale from single GPU to multi-GPU
- Model accuracy remained stable across configurations

---

### Trade-offs
- Increased setup and debugging complexity
- Communication overhead limits ideal linear scaling
- Efficiency decreases as GPU count increases

---

### Possible Extensions
- Multi-node scaling
- Larger Transformer models
- Mixed precision training
- Gradient accumulation

---

### Lessons Learned
This project demonstrates that Distributed Data Parallel training is an effective and practical approach for accelerating deep learning workloads on HPC systems while maintaining accuracy and scalability.
