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
from models.distilbert_model import DistilBertClassifier


# ---------------------------
# Argument parsing
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


# ---------------------------
# Utility: Average meter
# ---------------------------
class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


# ---------------------------
# Training loop (1 epoch)
# ---------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, use_distilbert=False):
    model.train()
    meter = AverageMeter()
    start_time = time.time()

    for batch in loader:
        if use_distilbert:
            # DistilBERT returns (input_ids, attention_mask, label)
            x, mask, y = batch
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(x, attention_mask=mask)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        else:
            # Transformer returns (input_ids, label)
            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        meter.update(loss.item(), y.size(0))

    elapsed = time.time() - start_time
    return meter.avg, elapsed


# ---------------------------
# Validation
# ---------------------------
@torch.no_grad()
def validate(model, loader, criterion, device, use_distilbert=False):
    model.eval()
    meter = AverageMeter()
    correct = 0
    total = 0

    for batch in loader:
        if use_distilbert:
            # DistilBERT returns (input_ids, attention_mask, label)
            x, mask, y = batch
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            outputs = model(x, attention_mask=mask)
        else:
            # Transformer returns (input_ids, label)
            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            outputs = model(x)

        loss = criterion(outputs, y)
        meter.update(loss.item(), y.size(0))

        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    accuracy = correct / max(total, 1)
    return meter.avg, accuracy


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Distributed setup
    dist_state = setup_distributed(cfg.get("distributed", {}))
    device = dist_state.device

    # Enable cuDNN auto-tuner for fixed input sizes
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Check if using DistilBERT
    model_type = cfg["model"].get("type", "transformer")
    use_distilbert = (model_type == "distilbert")
    
    # Data
    train_loader, val_loader = build_dataloaders(cfg, dist_state)

    # Model
    if use_distilbert:
        num_classes = int(cfg["model"]["num_classes"])
        pretrained = cfg["model"].get("pretrained", True)
        model = DistilBertClassifier(num_classes=num_classes, pretrained=pretrained).to(device)
        
        if dist_state.is_main_process:
            total_params, trainable_params = model.count_parameters()
            print(f"📊 DistilBERT - Total params: {total_params:,} | Trainable: {trainable_params:,}")
    else:
        model = build_model(cfg["model"]).to(device)
    
    if dist_state.use_ddp:
        model = DDP(
            model, 
            device_ids=[dist_state.local_rank],
            static_graph=True  # Model graph doesn't change during training
        )

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    lr = float(cfg["training"].get("learning_rate", 3e-4))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Output directory
    out_dir = Path(cfg["output_dir"])
    metrics_path = out_dir / "metrics.csv"

    if dist_state.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["epoch", "train_loss", "train_time", "val_loss", "val_acc"]
            )
        print("🛠 Main process active, starting training", flush=True)

    epochs = int(cfg["training"]["epochs"])

    # ---------------------------
    # Training loop
    # ---------------------------
    for epoch in range(epochs):

        # ✅ SAFE sampler handling (fixes your crash)
        sampler = getattr(train_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        train_loss, train_time = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_distilbert
        )

        val_loss, val_acc = validate(
            model, val_loader, criterion, device, use_distilbert
        )

        if dist_state.is_main_process:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train_loss={train_loss:.4f} (time {train_time:.2f}s) | "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}",
                flush=True,
            )

            with metrics_path.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [epoch + 1, train_loss, train_time, val_loss, val_acc]
                )

    # Cleanup
    cleanup_distributed(dist_state)


# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    main()
