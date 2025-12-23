import argparse
import csv
import time
import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

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


def _move_to_device(batch, device):
    if isinstance(batch, dict):
        return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    return batch.to(device, non_blocking=True)


def _forward_logits(model, x):
    out = model(**x) if isinstance(x, dict) else model(x)
    # Hugging Face models return an object with .logits
    if hasattr(out, "logits"):
        return out.logits
    return out


def _batch_num_bytes(batch):
    def tensor_bytes(t):
        try:
            return t.numel() * t.element_size()
        except Exception:
            return 0

    if isinstance(batch, dict):
        total = 0
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                total += tensor_bytes(v)
        return total
    if isinstance(batch, torch.Tensor):
        return tensor_bytes(batch)
    return 0


def train_one_epoch(model, loader, criterion, optimizer, device, amp_enabled: bool, scaler: GradScaler):
    model.train()
    meter = AverageMeter()
    start = time.time()
    transfer_time = 0.0
    transfer_bytes = 0

    for x, y in loader:
        t0 = time.time()
        x = _move_to_device(x, device)
        y = y.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        transfer_time += (time.time() - t0)
        transfer_bytes += int(_batch_num_bytes(x)) + int(y.numel() * y.element_size())

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp_enabled):
            logits = _forward_logits(model, x)
            loss = criterion(logits, y)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bsz = y.size(0)
        meter.update(loss.item(), bsz)

    return meter.avg, time.time() - start, transfer_time, transfer_bytes


@torch.no_grad()
def validate(model, loader, criterion, device, amp_enabled: bool):
    model.eval()
    meter = AverageMeter()
    correct = total = 0
    transfer_time = 0.0
    transfer_bytes = 0

    for x, y in loader:
        t0 = time.time()
        x = _move_to_device(x, device)
        y = y.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        transfer_time += (time.time() - t0)
        transfer_bytes += int(_batch_num_bytes(x)) + int(y.numel() * y.element_size())

        with autocast(enabled=amp_enabled):
            logits = _forward_logits(model, x)
            loss = criterion(logits, y)
        meter.update(loss.item(), y.size(0))

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    # Aggregate across ranks if using DDP
    if dist.is_initialized():
        # Reduce sums and counts for loss
        sum_loss = torch.tensor(meter.sum, dtype=torch.double, device=device)
        count_loss = torch.tensor(meter.count, dtype=torch.double, device=device)
        correct_t = torch.tensor(correct, dtype=torch.double, device=device)
        total_t = torch.tensor(total, dtype=torch.double, device=device)
        trans_time_t = torch.tensor(transfer_time, dtype=torch.double, device=device)
        trans_bytes_t = torch.tensor(transfer_bytes, dtype=torch.double, device=device)

        dist.all_reduce(sum_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(trans_time_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(trans_bytes_t, op=dist.ReduceOp.SUM)

        global_val_loss = (sum_loss / torch.clamp(count_loss, min=1.0)).item()
        global_val_acc = (correct_t / torch.clamp(total_t, min=1.0)).item()
        global_transfer_time = trans_time_t.item()
        global_transfer_bytes = int(trans_bytes_t.item())
        return global_val_loss, global_val_acc, global_transfer_time, global_transfer_bytes

    return meter.avg, correct / max(total, 1), transfer_time, transfer_bytes


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
    # Use AdamW for HF models by default
    if cfg["model"].get("type") == "hf_sequence_classifier":
        optimizer = optim.AdamW(model.parameters(), lr=3e-5)
    else:
        optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # AMP configuration
    amp_cfg = cfg.get("training", {}).get("amp", False)
    amp_enabled = bool(amp_cfg) and (device.type == "cuda")
    scaler = GradScaler(enabled=amp_enabled)

    out_dir = Path(cfg["output_dir"])
    metrics_path = out_dir / "metrics.csv"
    json_path = out_dir / "metrics.json"

    if dist_state.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "train_time", "val_loss", "val_acc"]
            )
        print("🛠 Main process active, starting training", flush=True)

    epochs = cfg["training"]["epochs"]

    # Accumulators for JSON summary
    per_epoch = []
    total_train_time = 0.0
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_val_acc_epoch = 0

    run_start_time = time.time()

    for epoch in range(epochs):
        if dist_state.use_ddp:
            train_loader.sampler.set_epoch(epoch)

        tr_loss, tr_time, tr_transfer_time, tr_transfer_bytes = train_one_epoch(
            model, train_loader, criterion, optimizer, device, amp_enabled, scaler
        )
        # Aggregate training transfer metrics across ranks for global totals
        if dist.is_initialized():
            tr_time_t = torch.tensor(tr_transfer_time, dtype=torch.double, device=device)
            tr_bytes_t = torch.tensor(tr_transfer_bytes, dtype=torch.double, device=device)
            dist.all_reduce(tr_time_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(tr_bytes_t, op=dist.ReduceOp.SUM)
            tr_transfer_time = tr_time_t.item()
            tr_transfer_bytes = int(tr_bytes_t.item())
        val_loss, val_acc, val_transfer_time, val_transfer_bytes = validate(
            model, val_loader, criterion, device, amp_enabled
        )

        total_train_time += float(tr_time)
        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            best_val_acc_epoch = epoch + 1
        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)

        # Track per-epoch metrics for JSON
        per_epoch.append({
            "epoch": int(epoch + 1),
            "train_loss": float(tr_loss),
            "train_time": float(tr_time),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "train_data_transfer_time_sec": float(tr_transfer_time),
            "train_data_transfer_bytes": int(tr_transfer_bytes),
            "val_data_transfer_time_sec": float(val_transfer_time),
            "val_data_transfer_bytes": int(val_transfer_bytes),
        })

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

    # Write JSON summary (rank 0 only)
    if dist_state.is_main_process:
        try:
            avg_train_time = total_train_time / max(len(per_epoch), 1)
            # Aggregate data transfer across epochs
            agg_train_transfer_time = float(sum(e.get("train_data_transfer_time_sec", 0.0) for e in per_epoch))
            agg_val_transfer_time = float(sum(e.get("val_data_transfer_time_sec", 0.0) for e in per_epoch))
            agg_train_transfer_bytes = int(sum(e.get("train_data_transfer_bytes", 0) for e in per_epoch))
            agg_val_transfer_bytes = int(sum(e.get("val_data_transfer_bytes", 0) for e in per_epoch))

            summary = {
                "experiment_name": cfg.get("experiment_name", None),
                "output_dir": str(out_dir),
                "seed": cfg.get("seed", None),
                "timestamp": int(time.time()),
                "distributed": {
                    "use_ddp": bool(dist_state.use_ddp),
                    "world_size": int(dist_state.world_size),
                    "backend": cfg.get("distributed", {}).get("backend", None),
                },
                "data": {
                    "dataset": cfg.get("data", {}).get("dataset", None),
                    "num_train_samples": int(len(train_loader.dataset)),
                    "num_val_samples": int(len(val_loader.dataset)),
                },
                "model": cfg.get("model", {}),
                "training": cfg.get("training", {}),
                "per_epoch": per_epoch,
                "aggregate": {
                    "total_epochs": int(epochs),
                    "total_train_time": float(total_train_time),
                    "avg_train_time": float(avg_train_time),
                    "best_val_acc": float(best_val_acc),
                    "best_val_acc_epoch": int(best_val_acc_epoch),
                    "best_val_loss": float(best_val_loss),
                    "final_val_acc": float(per_epoch[-1]["val_acc"]) if per_epoch else None,
                    "final_val_loss": float(per_epoch[-1]["val_loss"]) if per_epoch else None,
                    "wall_time_sec": float(time.time() - run_start_time),
                    "data_transfer": {
                        "train_total_time_sec": float(agg_train_transfer_time),
                        "val_total_time_sec": float(agg_val_transfer_time),
                        "train_total_bytes": int(agg_train_transfer_bytes),
                        "val_total_bytes": int(agg_val_transfer_bytes),
                    },
                },
            }
            with json_path.open("w", encoding="utf-8") as jf:
                json.dump(summary, jf, indent=2)
        except Exception as e:
            # Do not crash training on metrics write
            print(f"Warning: failed to write JSON metrics: {e}", flush=True)

    cleanup_distributed(dist_state)


if __name__ == "__main__":
    main()
