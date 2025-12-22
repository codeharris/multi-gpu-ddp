import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 11

# Output directory
output_dir = Path("experiments/plots/final_report")
output_dir.mkdir(parents=True, exist_ok=True)

# Read data
exp041 = pd.read_csv("experiments/exp041_agnews_1gpu_20ep/metrics.csv")
exp042 = pd.read_csv("experiments/exp042_agnews_ddp_4gpu_20ep/metrics.csv")
exp043 = pd.read_csv("experiments/exp043_agnews_1gpu_optimized/metrics.csv")
exp044 = pd.read_csv("experiments/exp044_agnews_ddp_4gpu_optimized/metrics.csv")

# Calculate metrics (excluding warmup epoch 1)
configs = {
    'Small 1 GPU': {'data': exp041, 'color': '#1f77b4', 'marker': 'o'},
    'Small 4 GPUs': {'data': exp042, 'color': '#ff7f0e', 'marker': 's'},
    'Large 1 GPU': {'data': exp043, 'color': '#2ca02c', 'marker': '^'},
    'Large 4 GPUs': {'data': exp044, 'color': '#d62728', 'marker': 'D'}
}

# ========== Plot 1: Epoch Time Comparison ==========
fig, ax = plt.subplots(figsize=(12, 6))

for name, cfg in configs.items():
    data = cfg['data']
    epochs = data['epoch'].values[1:]  # Skip warmup
    times = data['train_time'].values[1:]
    ax.plot(epochs, times, label=name, color=cfg['color'], 
            marker=cfg['marker'], markersize=6, linewidth=2, alpha=0.8)

ax.set_xlabel('Epoch', fontweight='bold')
ax.set_ylabel('Training Time (seconds)', fontweight='bold')
ax.set_title('Training Time per Epoch: Model Size Impact on Multi-GPU Scaling', 
             fontweight='bold', pad=20)
ax.legend(loc='best', framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.set_xlim(1.5, 20.5)

plt.tight_layout()
plt.savefig(output_dir / 'epoch_time_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'epoch_time_comparison.png'}")
plt.close()

# ========== Plot 2: Speedup Comparison ==========
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate speedups
t1_small = exp041['train_time'].iloc[1:].mean()
t4_small = exp042['train_time'].iloc[1:].mean()
t1_large = exp043['train_time'].iloc[1:].mean()
t4_large = exp044['train_time'].iloc[1:].mean()

speedup_small = t1_small / t4_small
speedup_large = t1_large / t4_large

x = np.array([1, 4])
ideal = x

# Plot ideal scaling
ax.plot(x, ideal, 'k--', linewidth=2, label='Ideal Linear Scaling', alpha=0.7)

# Plot actual speedups
ax.plot([1, 4], [1, speedup_small], 'o-', color='#ff7f0e', 
        linewidth=3, markersize=12, label=f'Small Model ({speedup_small:.2f}×)')
ax.plot([1, 4], [1, speedup_large], 's-', color='#d62728', 
        linewidth=3, markersize=12, label=f'Large Model ({speedup_large:.2f}×)')

# Annotations
ax.annotate(f'{speedup_small:.2f}×\n(11.5% eff)', xy=(4, speedup_small), 
            xytext=(3.5, 1.2), fontsize=10, ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='orange', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))

ax.annotate(f'{speedup_large:.2f}×\n(20% eff)', xy=(4, speedup_large), 
            xytext=(3.5, 1.8), fontsize=10, ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))

ax.set_xlabel('Number of GPUs', fontweight='bold')
ax.set_ylabel('Speedup', fontweight='bold')
ax.set_title('Speedup vs Number of GPUs: Both Models Exhibit Negative Scaling', 
             fontweight='bold', pad=20)
ax.set_xticks([1, 2, 3, 4])
ax.set_ylim(0, 4.5)
ax.legend(loc='upper left', framealpha=0.95)
ax.grid(True, alpha=0.3)

# Add red zone for negative scaling
ax.axhspan(0, 1, alpha=0.1, color='red', label='Negative Scaling Zone')
ax.text(2.5, 0.5, 'NEGATIVE SCALING', fontsize=14, color='red', 
        ha='center', fontweight='bold', alpha=0.5)

plt.tight_layout()
plt.savefig(output_dir / 'speedup_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'speedup_comparison.png'}")
plt.close()

# ========== Plot 3: Efficiency Comparison ==========
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Small model
eff_small = (speedup_small / 4) * 100
ax1.bar(['1 GPU', '4 GPUs'], [100, eff_small], 
        color=['#1f77b4', '#ff7f0e'], alpha=0.7, edgecolor='black', linewidth=2)
ax1.axhline(100, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Ideal (100%)')
ax1.set_ylabel('Parallel Efficiency (%)', fontweight='bold')
ax1.set_title('Small Model (128-dim, 2 layers)', fontweight='bold')
ax1.set_ylim(0, 120)
ax1.text(1, eff_small + 5, f'{eff_small:.1f}%', ha='center', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Large model
eff_large = (speedup_large / 4) * 100
ax2.bar(['1 GPU', '4 GPUs'], [100, eff_large], 
        color=['#2ca02c', '#d62728'], alpha=0.7, edgecolor='black', linewidth=2)
ax2.axhline(100, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Ideal (100%)')
ax2.set_ylabel('Parallel Efficiency (%)', fontweight='bold')
ax2.set_title('Large Model (256-dim, 4 layers)', fontweight='bold')
ax2.set_ylim(0, 120)
ax2.text(1, eff_large + 5, f'{eff_large:.1f}%', ha='center', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Parallel Efficiency Comparison', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'efficiency_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'efficiency_comparison.png'}")
plt.close()

# ========== Plot 4: Accuracy Comparison ==========
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Small model accuracy
for name, cfg in [('Small 1 GPU', configs['Small 1 GPU']), 
                   ('Small 4 GPUs', configs['Small 4 GPUs'])]:
    data = cfg['data']
    ax1.plot(data['epoch'], data['val_acc'] * 100, label=name, 
            color=cfg['color'], linewidth=2, marker=cfg['marker'], 
            markersize=5, alpha=0.8)

ax1.set_xlabel('Epoch', fontweight='bold')
ax1.set_ylabel('Validation Accuracy (%)', fontweight='bold')
ax1.set_title('Small Model: Accuracy Preserved', fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(82, 92)

# Large model accuracy
for name, cfg in [('Large 1 GPU', configs['Large 1 GPU']), 
                   ('Large 4 GPUs', configs['Large 4 GPUs'])]:
    data = cfg['data']
    ax2.plot(data['epoch'], data['val_acc'] * 100, label=name, 
            color=cfg['color'], linewidth=2, marker=cfg['marker'], 
            markersize=5, alpha=0.8)

ax2.set_xlabel('Epoch', fontweight='bold')
ax2.set_ylabel('Validation Accuracy (%)', fontweight='bold')
ax2.set_title('Large Model: Accuracy Preserved', fontweight='bold')
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(82, 92)

plt.suptitle('Model Accuracy: DDP Maintains Correctness Despite Poor Scaling', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'accuracy_comparison.png'}")
plt.close()

# ========== Summary Statistics ==========
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(f"\nSmall Model (d_model=128, 2 layers, batch=32):")
print(f"  1 GPU:  {t1_small:.2f}s/epoch")
print(f"  4 GPUs: {t4_small:.2f}s/epoch")
print(f"  Speedup: {speedup_small:.2f}×")
print(f"  Efficiency: {eff_small:.1f}%")

print(f"\nLarge Model (d_model=256, 4 layers, batch=128):")
print(f"  1 GPU:  {t1_large:.2f}s/epoch")
print(f"  4 GPUs: {t4_large:.2f}s/epoch")
print(f"  Speedup: {speedup_large:.2f}×")
print(f"  Efficiency: {eff_large:.1f}%")

print(f"\n✓ All plots saved to: {output_dir}")
print("="*70)
