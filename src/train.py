import argparse
import time
import os
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from config import load_config
from distributed import setup_distributed, cleanup_distributed
from data.dataset import build_dataloaders
from models.transformer_model import build_model
from utils.metrics import AverageMeter
from utils.logging import print_rank0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transformer training with optional PyTorch DDP."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_single_gpu.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()

    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), n=inputs.size(0))

    epoch_time = time.time() - start_time
    return loss_meter.avg, epoch_time


def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_meter.update(loss.item(), n=inputs.size(0))

            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    avg_loss = loss_meter.avg
    accuracy = correct / max(total, 1)

    return avg_loss, accuracy


def main():
    # 1. Parse args and load config
    args = parse_args()
    cfg = load_config(args.config)

    # 2. Setup device / DDP state
    dist_state = setup_distributed(cfg["distributed"])
    device = dist_state.device

    # 3. Prepare output directory (rank 0 only)
    output_dir = Path(cfg["output_dir"])
    if dist_state.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.csv"
        # Write header fresh each run
        with metrics_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_time", "val_loss", "val_acc"])
    else:
        metrics_path = None  # not used on non-main ranks

    # 4. Build dataloaders
    train_loader, val_loader = build_dataloaders(cfg, dist_state)

    # 5. Build model, loss, optimizer
    model = build_model(cfg["model"])
    model = model.to(device)

    if dist_state.use_ddp:
        # Wrap with DistributedDataParallel
        model = DDP(
            model,
            device_ids=[dist_state.local_rank] if device.type == "cuda" else None,
            output_device=dist_state.local_rank if device.type == "cuda" else None,
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=3e-4,
    )

    num_epochs = cfg["training"]["epochs"]

    # 6. Training loop
    for epoch in range(num_epochs):
        # If using DistributedSampler, ensure different shuffling each epoch
        if dist_state.use_ddp and hasattr(train_loader, "sampler") and train_loader.sampler is not None:
            try:
                train_loader.sampler.set_epoch(epoch)
            except AttributeError:
                pass

        train_loss, train_time = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Print to console (main process only)
        print_rank0(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train_loss={train_loss:.4f} (time {train_time:.2f}s) | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}",
            is_main_process=dist_state.is_main_process,
        )

        # Append to metrics.csv (main process only)
        if dist_state.is_main_process and metrics_path is not None:
            with metrics_path.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        epoch + 1,
                        f"{train_loss:.6f}",
                        f"{train_time:.6f}",
                        f"{val_loss:.6f}",
                        f"{val_acc:.6f}",
                    ]
                )

    # 7. Cleanup (for DDP)
    cleanup_distributed(dist_state)


if __name__ == "__main__":
    main()
