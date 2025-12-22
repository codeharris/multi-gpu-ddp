#!/usr/bin/env python3
"""
scripts/plot_results.py

Generates clean, report-ready plots from experiments/*/metrics.csv
for 1-GPU vs 4-GPU DDP runs on IMDB and AG News.

What it produces (saved under experiments/plots/):
- imdb_epoch_time_curve.png
- imdb_avg_epoch_time_bar.png
- imdb_speedup_efficiency_bar.png
- imdb_throughput_bar.png
- imdb_valacc_curve.png
- agnews_epoch_time_curve.png
- agnews_avg_epoch_time_bar.png
- agnews_speedup_efficiency_bar.png
- agnews_throughput_bar.png
- agnews_valacc_curve.png
- summary_table.csv   (nice for LaTeX)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Helpers
# ----------------------------
def read_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize types
    for col in ["epoch", "train_loss", "train_time", "val_loss", "val_acc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["epoch", "train_time"])
    df = df.sort_values("epoch").reset_index(drop=True)
    return df


def mean_std(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(x)), float(np.std(x, ddof=1)) if x.size > 1 else 0.0


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_fig(out_path: Path) -> None:
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def compute_throughput(dataset_train_size: int, epoch_times: np.ndarray) -> np.ndarray:
    # samples/sec per epoch (global, i.e., dataset processed per epoch)
    return dataset_train_size / np.asarray(epoch_times, dtype=float)


def bar_with_labels(ax, xlabels, values, ylabel, title, ylim=None):
    bars = ax.bar(xlabels, values, edgecolor="black", linewidth=1.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    # Annotate bars
    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height(),
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)


# ----------------------------
# Plotting per dataset
# ----------------------------
def plot_dataset(
    name: str,
    df_1: pd.DataFrame,
    df_4: pd.DataFrame,
    out_dir: Path,
    train_size: int,
    warmup_epochs: int,
    gpus_1: int = 1,
    gpus_4: int = 4,
) -> Dict[str, float]:
    """
    Returns summary stats dict.
    """
    ensure_dir(out_dir)

    # Exclude warmup epochs from summary metrics (but keep full curves)
    def slice_after_warmup(df: pd.DataFrame) -> pd.DataFrame:
        if warmup_epochs <= 0:
            return df.copy()
        return df[df["epoch"] > warmup_epochs].copy()

    df_1_s = slice_after_warmup(df_1)
    df_4_s = slice_after_warmup(df_4)

    t1 = df_1_s["train_time"].to_numpy()
    t4 = df_4_s["train_time"].to_numpy()

    mean_t1, std_t1 = mean_std(t1)
    mean_t4, std_t4 = mean_std(t4)

    speedup = mean_t1 / mean_t4
    efficiency = speedup / float(gpus_4)

    thr_1 = compute_throughput(train_size, t1)
    thr_4 = compute_throughput(train_size, t4)
    mean_thr1, std_thr1 = mean_std(thr_1)
    mean_thr4, std_thr4 = mean_std(thr_4)

    # ----------------------------
    # 1) Epoch time curves
    # ----------------------------
    plt.figure(figsize=(9, 5.5))
    plt.plot(df_1["epoch"], df_1["train_time"], marker="o", linewidth=2, label=f"{gpus_1} GPU")
    plt.plot(df_4["epoch"], df_4["train_time"], marker="s", linewidth=2, label=f"{gpus_4} GPUs (DDP)")
    plt.xlabel("Epoch")
    plt.ylabel("Training time per epoch [s]")
    plt.title(f"{name}: Training Time per Epoch")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    save_fig(out_dir / f"{name.lower()}_epoch_time_curve.png")

    # ----------------------------
    # 2) Avg epoch time bar (with error bars)
    # ----------------------------
    plt.figure(figsize=(7.5, 5.5))
    ax = plt.gca()
    x = [f"{gpus_1} GPU", f"{gpus_4} GPUs (DDP)"]
    means = [mean_t1, mean_t4]
    errs = [std_t1, std_t4]
    bars = ax.bar(x, means, yerr=errs, capsize=8, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Avg epoch time [s]")
    ax.set_title(f"{name}: Avg Epoch Time (epochs > {warmup_epochs})")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2.0, b.get_height(), f"{v:.2f}s", ha="center", va="bottom", fontsize=11)
    save_fig(out_dir / f"{name.lower()}_avg_epoch_time_bar.png")

    # ----------------------------
    # 3) Speedup + Efficiency bars (clearly shows baseline 1 GPU)
    # ----------------------------
    plt.figure(figsize=(8.0, 5.5))
    ax = plt.gca()
    labels = [f"{gpus_1} GPU (baseline)", f"{gpus_4} GPUs (DDP)"]
    speedups = [1.0, speedup]
    effs = [1.0, efficiency]  # baseline defined as 1.0 for visual reference

    width = 0.38
    x_pos = np.arange(len(labels))

    b1 = ax.bar(x_pos - width / 2, speedups, width=width, edgecolor="black", linewidth=1.5, label="Speedup")
    b2 = ax.bar(x_pos + width / 2, effs, width=width, edgecolor="black", linewidth=1.5, label="Parallel efficiency")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Value")
    ax.set_title(f"{name}: Speedup & Efficiency (vs 1 GPU)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    for b, v in zip(b1, speedups):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{v:.2f}", ha="center", va="bottom", fontsize=11)
    for b, v in zip(b2, effs):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{v:.2f}", ha="center", va="bottom", fontsize=11)

    save_fig(out_dir / f"{name.lower()}_speedup_efficiency_bar.png")

    # ----------------------------
    # 4) Throughput bar (samples/sec)
    # ----------------------------
    plt.figure(figsize=(7.5, 5.5))
    ax = plt.gca()
    x = [f"{gpus_1} GPU", f"{gpus_4} GPUs (DDP)"]
    means = [mean_thr1, mean_thr4]
    errs = [std_thr1, std_thr4]
    bars = ax.bar(x, means, yerr=errs, capsize=8, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Throughput [samples/sec]")
    ax.set_title(f"{name}: Throughput (epochs > {warmup_epochs})")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{v:.0f}", ha="center", va="bottom", fontsize=11)
    save_fig(out_dir / f"{name.lower()}_throughput_bar.png")

    # ----------------------------
    # 5) Validation accuracy curve (quality check)
    # ----------------------------
    if "val_acc" in df_1.columns and "val_acc" in df_4.columns:
        plt.figure(figsize=(9, 5.5))
        plt.plot(df_1["epoch"], df_1["val_acc"], marker="o", linewidth=2, label=f"{gpus_1} GPU")
        plt.plot(df_4["epoch"], df_4["val_acc"], marker="s", linewidth=2, label=f"{gpus_4} GPUs (DDP)")
        plt.xlabel("Epoch")
        plt.ylabel("Validation accuracy")
        plt.title(f"{name}: Validation Accuracy")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        save_fig(out_dir / f"{name.lower()}_valacc_curve.png")

    # Summary dictionary (for table)
    return {
        "dataset": name,
        "train_size": int(train_size),
        "warmup_excluded_epochs": int(warmup_epochs),
        "gpus_1": int(gpus_1),
        "gpus_4": int(gpus_4),
        "mean_epoch_time_1gpu": mean_t1,
        "std_epoch_time_1gpu": std_t1,
        "mean_epoch_time_4gpu": mean_t4,
        "std_epoch_time_4gpu": std_t4,
        "speedup_4gpu_vs_1gpu": speedup,
        "efficiency_4gpu": efficiency,
        "mean_throughput_1gpu_sps": mean_thr1,
        "std_throughput_1gpu_sps": std_thr1,
        "mean_throughput_4gpu_sps": mean_thr4,
        "std_throughput_4gpu_sps": std_thr4,
        "final_val_acc_1gpu": float(df_1["val_acc"].dropna().iloc[-1]) if "val_acc" in df_1.columns and len(df_1["val_acc"].dropna()) else float("nan"),
        "final_val_acc_4gpu": float(df_4["val_acc"].dropna().iloc[-1]) if "val_acc" in df_4.columns and len(df_4["val_acc"].dropna()) else float("nan"),
    }


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="experiments", help="experiments root directory")
    parser.add_argument("--out", type=str, default="experiments/plots", help="output plots directory")
    parser.add_argument("--warmup", type=int, default=1, help="exclude first N epochs from averages (warmup)")
    parser.add_argument("--imdb1", type=str, default="exp020_imdb_single_gpu_20ep", help="IMDB 1-GPU exp dir")
    parser.add_argument("--imdb4", type=str, default="exp021_imdb_ddp_single_node_20ep", help="IMDB 4-GPU exp dir")
    parser.add_argument("--ag1", type=str, default="exp041_agnews_1gpu_20ep", help="AG News 1-GPU exp dir")
    parser.add_argument("--ag4", type=str, default="exp042_agnews_ddp_single_node_20ep", help="AG News 4-GPU exp dir")

    # If you used train_limit/val_limit, set these accordingly.
    # Defaults are full dataset sizes.
    parser.add_argument("--imdb_train_size", type=int, default=25000)
    parser.add_argument("--agnews_train_size", type=int, default=120000)

    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    # ---- Load metrics ----
    imdb1 = read_metrics(root / args.imdb1 / "metrics.csv")
    imdb4 = read_metrics(root / args.imdb4 / "metrics.csv")
    ag1 = read_metrics(root / args.ag1 / "metrics.csv")
    ag4 = read_metrics(root / args.ag4 / "metrics.csv")

    summaries = []

    summaries.append(
        plot_dataset(
            name="IMDB",
            df_1=imdb1,
            df_4=imdb4,
            out_dir=out_dir,
            train_size=args.imdb_train_size,
            warmup_epochs=args.warmup,
        )
    )

    summaries.append(
        plot_dataset(
            name="AG News",
            df_1=ag1,
            df_4=ag4,
            out_dir=out_dir,
            train_size=args.agnews_train_size,
            warmup_epochs=args.warmup,
        )
    )

    # ---- Write summary table ----
    summary_df = pd.DataFrame(summaries)
    summary_path = out_dir / "summary_table.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[OK] Wrote plots to: {out_dir}")
    print(f"[OK] Wrote summary table: {summary_path}")

    # Print a compact console summary too
    for s in summaries:
        print("\n==============================")
        print(f"Dataset: {s['dataset']}")
        print(f"Train size: {s['train_size']}")
        print(f"Mean epoch time 1GPU: {s['mean_epoch_time_1gpu']:.3f} ± {s['std_epoch_time_1gpu']:.3f} s")
        print(f"Mean epoch time 4GPU: {s['mean_epoch_time_4gpu']:.3f} ± {s['std_epoch_time_4gpu']:.3f} s")
        print(f"Speedup (4 vs 1): {s['speedup_4gpu_vs_1gpu']:.2f}x")
        print(f"Efficiency (4 GPUs): {s['efficiency_4gpu']:.2f}")
        print(f"Throughput 1GPU: {s['mean_throughput_1gpu_sps']:.0f} samples/s")
        print(f"Throughput 4GPU: {s['mean_throughput_4gpu_sps']:.0f} samples/s")
        if not np.isnan(s["final_val_acc_1gpu"]) and not np.isnan(s["final_val_acc_4gpu"]):
            print(f"Final val acc 1GPU: {s['final_val_acc_1gpu']:.3f}")
            print(f"Final val acc 4GPU: {s['final_val_acc_4gpu']:.3f}")


if __name__ == "__main__":
    main()
