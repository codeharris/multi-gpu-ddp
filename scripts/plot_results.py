import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Paths to experiment metrics
# -----------------------------
SINGLE_GPU_CSV = Path("experiments/exp020_imdb_single_gpu_20ep/metrics.csv")
DDP_4GPU_CSV   = Path("experiments/exp021_imdb_ddp_single_node_20ep/metrics.csv")

OUT_DIR = Path("experiments/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")


def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["epoch", "train_time", "train_loss", "val_loss", "val_acc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def main():
    # -----------------------------
    # Load data
    # -----------------------------
    single = load_metrics(SINGLE_GPU_CSV)
    ddp = load_metrics(DDP_4GPU_CSV)

    avg_time_single = single["train_time"].mean()
    avg_time_ddp = ddp["train_time"].mean()

    speedup_4 = avg_time_single / avg_time_ddp
    efficiency_4 = speedup_4 / 4.0  # 4 GPUs

    print("=== Summary (IMDB, 20 epochs) ===")
    print(f"Avg epoch time 1 GPU : {avg_time_single:.3f} s")
    print(f"Avg epoch time 4 GPUs: {avg_time_ddp:.3f} s")
    print(f"Speedup (4 vs 1 GPU) : {speedup_4:.3f}x")
    print(f"Efficiency (4 GPUs)  : {efficiency_4:.3f}")

    # -------------------------------------------------
    # 1) Training time per epoch (line plot)
    # -------------------------------------------------
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(
        single["epoch"], single["train_time"],
        marker="o", linewidth=2, label="1 GPU"
    )
    plt.plot(
        ddp["epoch"], ddp["train_time"],
        marker="s", linewidth=2, label="4 GPUs (DDP)"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Training time per epoch [s]")
    plt.title("Training Time per Epoch on IMDB")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "training_time_per_epoch.png", dpi=200)
    plt.close()

    # -------------------------------------------------
    # 2) Average epoch time (1 GPU vs 4 GPUs) – bars
    # -------------------------------------------------
    plt.figure(figsize=(6.5, 4.5))
    labels = ["1 GPU", "4 GPUs"]
    times = [avg_time_single, avg_time_ddp]
    bars = plt.bar(labels, times, edgecolor="black", linewidth=1.4)

    for bar, value in zip(bars, times):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(times) * 0.03,
            f"{value:.2f}s",
            ha="center", va="bottom", fontsize=11, weight="bold"
        )

    plt.ylabel("Average epoch time [s]")
    plt.title("Average Training Time (IMDB, 20 epochs)")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "avg_epoch_time_1gpu_vs_4gpu.png", dpi=200)
    plt.close()

    # -------------------------------------------------
    # 3) Speedup – bars for 1 GPU baseline & 4 GPUs
    # -------------------------------------------------
    plt.figure(figsize=(6.5, 4.5))
    speedup_values = [1.0, speedup_4]
    labels = ["1 GPU (baseline)", "4 GPUs"]
    bars = plt.bar(labels, speedup_values, edgecolor="black", linewidth=1.4)

    for bar, value in zip(bars, speedup_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.05,
            f"{value:.2f}×",
            ha="center", va="bottom", fontsize=11, weight="bold"
        )

    plt.ylabel("Speedup  T₁ / Tₙ")
    plt.title("Speedup vs Number of GPUs (IMDB)")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.ylim(0, max(speedup_values) * 1.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "speedup_bars.png", dpi=200)
    plt.close()

    # -------------------------------------------------
    # 4) Parallel efficiency – ideal vs actual 4 GPUs
    # -------------------------------------------------
    plt.figure(figsize=(6.5, 4.5))
    eff_labels = ["Ideal (100%)", "4 GPUs"]
    eff_values = [1.0, efficiency_4]
    bars = plt.bar(eff_labels, eff_values, edgecolor="black", linewidth=1.4)

    for bar, value in zip(bars, eff_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.03,
            f"{value:.2f}",
            ha="center", va="bottom", fontsize=11, weight="bold"
        )

    plt.ylabel("Parallel efficiency (Speedup / N)")
    plt.title("Parallel Efficiency (IMDB, 4 GPUs)")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.ylim(0, 1.2)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "efficiency_bars.png", dpi=200)
    plt.close()

    print("Plots saved in:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
