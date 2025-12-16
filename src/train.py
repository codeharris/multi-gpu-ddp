import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from config import load_config
from distributed import setup_distributed, cleanup_distributed
from data.dataset import build_dataloaders
from models.transformer_model import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, v, n=1):
        self.sum += float(v) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    meter = AverageMeter()
    start = time.time()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        meter.update(loss.item(), x.size(0))

    return meter.avg, time.time() - start


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    meter = AverageMeter()
    correct = total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model(x)
        loss = criterion(out, y)
        meter.update(loss.item(), x.size(0))

        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return meter.avg, correct / max(total, 1)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    dist_state = setup_distributed(cfg["distributed"])
    device = dist_state.device

    train_loader, val_loader = build_dataloaders(cfg, dist_state)

    model = build_model(cfg["model"]).to(device)
    if dist_state.use_ddp:
        model = DDP(model, device_ids=[dist_state.local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    out_dir = Path(cfg["output_dir"])
    metrics_path = out_dir / "metrics.csv"

    if dist_state.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "train_time", "val_loss", "val_acc"]
            )
        print("🛠 Main process active, starting training", flush=True)

    epochs = cfg["training"]["epochs"]

    for epoch in range(epochs):
        if dist_state.use_ddp:
            train_loader.sampler.set_epoch(epoch)

        tr_loss, tr_time = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if dist_state.is_main_process:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"train_loss={tr_loss:.4f} (time {tr_time:.2f}s) | "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}",
                flush=True,
            )
            with metrics_path.open("a", newline="") as f:
                csv.writer(f).writerow(
                    [epoch + 1, tr_loss, tr_time, val_loss, val_acc]
                )

    cleanup_distributed(dist_state)


if __name__ == "__main__":
    main()
