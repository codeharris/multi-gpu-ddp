#!/usr/bin/env python3
"""
Enhanced plotting and analysis for multi-GPU DDP experiments.
Generates comprehensive visualizations and diagnostic metrics.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Tuple

plt.style.use('seaborn-v0_8-darkgrid')
COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

def read_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["epoch", "train_loss", "train_time", "val_loss", "val_acc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["epoch", "train_time"])
    df = df.sort_values("epoch").reset_index(drop=True)
    return df

def compute_stats(times: np.ndarray, warmup: int = 1) -> Dict:
    """Compute statistics excluding warmup epochs"""
    times_no_warmup = times[warmup:] if warmup < len(times) else times
    return {
        'mean': float(np.mean(times_no_warmup)),
        'std': float(np.std(times_no_warmup)),
        'min': float(np.min(times_no_warmup)),
        'max': float(np.max(times_no_warmup)),
        'total': float(np.sum(times))
    }

def plot_epoch_time_comparison(df_dict: Dict[str, pd.DataFrame], out_dir: Path, dataset_name: str):
    """Plot epoch time curves for all configurations"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (label, df) in enumerate(df_dict.items()):
        ax.plot(df['epoch'], df['train_time'], 
                marker='o', linewidth=2.5, markersize=6,
                label=label, color=COLORS[i % len(COLORS)])
    
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Training Time (seconds)', fontsize=13, fontweight='bold')
    ax.set_title(f'{dataset_name}: Training Time per Epoch', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(out_dir / f'{dataset_name.lower()}_epoch_time_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_speedup_analysis(stats_dict: Dict[str, Dict], out_dir: Path, dataset_name: str, baseline_key: str):
    """Create detailed speedup and efficiency visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    labels = [k.replace(baseline_key, '1 GPU').replace('4 GPUs', '4 GPUs (DDP)') for k in stats_dict.keys()]
    mean_times = [stats_dict[k]['mean'] for k in stats_dict.keys()]
    std_times = [stats_dict[k]['std'] for k in stats_dict.keys()]
    
    baseline_time = stats_dict[baseline_key]['mean']
    speedups = [baseline_time / t for t in mean_times]
    efficiencies = [s / (i + 1) for i, s in enumerate(speedups)]
    
    # Speedup bars
    bars1 = ax1.bar(labels, speedups, color=COLORS[:len(labels)], 
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline')
    ax1.axhline(y=4.0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Ideal (4x)')
    ax1.set_ylabel('Speedup', fontsize=13, fontweight='bold')
    ax1.set_title(f'{dataset_name}: Speedup Analysis', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, axis='y', alpha=0.3)
    
    for bar, val in zip(bars1, speedups):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}x', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Efficiency bars
    bars2 = ax2.bar(labels, efficiencies, color=COLORS[:len(labels)], 
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Ideal (100%)')
    ax2.axhline(y=0.85, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='Good (85%)')
    ax2.set_ylabel('Parallel Efficiency', fontsize=13, fontweight='bold')
    ax2.set_title(f'{dataset_name}: Parallel Efficiency', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1.2)
    ax2.legend(fontsize=10)
    ax2.grid(True, axis='y', alpha=0.3)
    
    for bar, val in zip(bars2, efficiencies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val*100:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(out_dir / f'{dataset_name.lower()}_speedup_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_throughput_comparison(stats_dict: Dict[str, Dict], train_size: int, out_dir: Path, dataset_name: str):
    """Plot throughput (samples/second) comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = list(stats_dict.keys())
    throughputs = [train_size / stats_dict[k]['mean'] for k in labels]
    
    bars = ax.bar(labels, throughputs, color=COLORS[:len(labels)], 
                  edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.set_ylabel('Throughput (samples/sec)', fontsize=13, fontweight='bold')
    ax.set_title(f'{dataset_name}: Training Throughput', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    for bar, val in zip(bars, throughputs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(out_dir / f'{dataset_name.lower()}_throughput.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_accuracy_comparison(df_dict: Dict[str, pd.DataFrame], out_dir: Path, dataset_name: str):
    """Plot validation accuracy curves"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (label, df) in enumerate(df_dict.items()):
        if 'val_acc' in df.columns:
            ax.plot(df['epoch'], df['val_acc'], 
                    marker='s', linewidth=2.5, markersize=6,
                    label=label, color=COLORS[i % len(COLORS)])
    
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Validation Accuracy', fontsize=13, fontweight='bold')
    ax.set_title(f'{dataset_name}: Model Accuracy Convergence', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(out_dir / f'{dataset_name.lower()}_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table(experiments: Dict[str, Dict], out_dir: Path):
    """Generate comprehensive summary table"""
    rows = []
    for exp_name, data in experiments.items():
        row = {
            'Configuration': exp_name,
            'Dataset': data['dataset'],
            'Train Size': data['train_size'],
            'Mean Epoch Time (s)': f"{data['stats']['mean']:.2f} ± {data['stats']['std']:.2f}",
            'Total Time (s)': f"{data['stats']['total']:.1f}",
            'Speedup': f"{data.get('speedup', 1.0):.2f}x",
            'Efficiency': f"{data.get('efficiency', 1.0)*100:.1f}%",
            'Throughput (sps)': f"{data['throughput']:.0f}",
            'Final Val Acc': f"{data.get('final_acc', 0.0):.3f}"
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'comprehensive_summary.csv', index=False)
    
    # Also create LaTeX table
    latex_table = df.to_latex(index=False, float_format="%.2f", escape=False)
    with open(out_dir / 'summary_table.tex', 'w') as f:
        f.write("% Auto-generated summary table\n")
        f.write("\\begin{table}[H]\n")
        f.write("    \\centering\n")
        f.write("    \\caption{Comprehensive Performance Summary}\n")
        f.write("    \\label{tab:comprehensive}\n")
        f.write("    \\small\n")
        f.write(latex_table)
        f.write("\\end{table}\n")
    
    return df

def generate_diagnostic_report(experiments: Dict[str, Dict], out_dir: Path):
    """Generate text report with diagnostics and recommendations"""
    report = []
    report.append("="*80)
    report.append("MULTI-GPU DDP PERFORMANCE ANALYSIS REPORT")
    report.append("="*80)
    report.append("")
    
    for exp_name, data in experiments.items():
        report.append(f"\n{'='*60}")
        report.append(f"Configuration: {exp_name}")
        report.append(f"{'='*60}")
        report.append(f"Dataset: {data['dataset']} (train size: {data['train_size']})")
        report.append(f"Mean epoch time: {data['stats']['mean']:.2f} ± {data['stats']['std']:.2f} s")
        report.append(f"Total training time: {data['stats']['total']:.1f} s ({data['stats']['total']/60:.1f} min)")
        report.append(f"Throughput: {data['throughput']:.0f} samples/sec")
        
        if 'speedup' in data:
            report.append(f"Speedup: {data['speedup']:.2f}x")
            report.append(f"Parallel efficiency: {data['efficiency']*100:.1f}%")
            
            # Diagnostics
            if data['speedup'] < 0.8:
                report.append("\n⚠️  WARNING: Negative scaling detected!")
                report.append("   Recommendations:")
                report.append("   - Increase model size (more layers, wider dimensions)")
                report.append("   - Increase batch size to improve compute-to-communication ratio")
                report.append("   - Consider using gradient accumulation instead of DDP")
            elif data['efficiency'] < 0.7:
                report.append("\n⚠️  Low parallel efficiency detected")
                report.append("   Possible causes:")
                report.append("   - Communication overhead dominates")
                report.append("   - Load imbalance or stragglers")
                report.append("   - Small batch size per GPU")
            elif data['efficiency'] >= 0.85:
                report.append("\n✓ Excellent scaling performance!")
        
        if 'final_acc' in data:
            report.append(f"Final validation accuracy: {data['final_acc']:.3f}")
        
        report.append("")
    
    report_path = out_dir / 'performance_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print('\n'.join(report))
    return report

def main():
    parser = argparse.ArgumentParser(description='Comprehensive DDP performance analysis')
    parser.add_argument('--root', type=str, default='experiments', help='experiments root')
    parser.add_argument('--out', type=str, default='experiments/plots', help='output directory')
    parser.add_argument('--warmup', type=int, default=1, help='warmup epochs to exclude')
    parser.add_argument('--datasets', nargs='+', default=['IMDB', 'AGNews'], help='dataset names')
    
    args = parser.parse_args()
    
    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Define experiment configurations
    experiment_configs = {
        'IMDB': {
            '1 GPU': ('exp020_imdb_single_gpu_20ep', 25000),
            '4 GPUs': ('exp021_imdb_ddp_single_node_20ep', 25000)
        },
        'AGNews': {
            '1 GPU': ('exp041_agnews_1gpu_20ep', 120000),
            '4 GPUs': ('exp042_agnews_ddp_4gpu_20ep', 120000)
        }
    }
    
    all_experiments = {}
    
    for dataset_name in args.datasets:
        if dataset_name not in experiment_configs:
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name} experiments...")
        print(f"{'='*60}")
        
        configs = experiment_configs[dataset_name]
        df_dict = {}
        stats_dict = {}
        baseline_key = None
        
        for config_name, (exp_dir, train_size) in configs.items():
            metrics_path = root / exp_dir / 'metrics.csv'
            if not metrics_path.exists():
                print(f"⚠️  Warning: {metrics_path} not found, skipping...")
                continue
            
            df = read_metrics(metrics_path)
            df_dict[config_name] = df
            
            times = df['train_time'].to_numpy()
            stats = compute_stats(times, args.warmup)
            stats_dict[config_name] = stats
            
            if '1 GPU' in config_name:
                baseline_key = config_name
            
            # Store for summary
            exp_full_name = f"{dataset_name}_{config_name}"
            all_experiments[exp_full_name] = {
                'dataset': dataset_name,
                'train_size': train_size,
                'stats': stats,
                'throughput': train_size / stats['mean'],
                'final_acc': float(df['val_acc'].iloc[-1]) if 'val_acc' in df.columns else 0.0
            }
        
        if not df_dict:
            print(f"No valid experiments found for {dataset_name}")
            continue
        
        # Compute speedup and efficiency
        if baseline_key and len(stats_dict) > 1:
            baseline_time = stats_dict[baseline_key]['mean']
            for config_name in stats_dict.keys():
                if config_name == baseline_key:
                    all_experiments[f"{dataset_name}_{config_name}"]["speedup"] = 1.0
                    all_experiments[f"{dataset_name}_{config_name}"]["efficiency"] = 1.0
                else:
                    speedup = baseline_time / stats_dict[config_name]['mean']
                    n_gpus = 4 if '4' in config_name else 1
                    efficiency = speedup / n_gpus
                    all_experiments[f"{dataset_name}_{config_name}"]["speedup"] = speedup
                    all_experiments[f"{dataset_name}_{config_name}"]["efficiency"] = efficiency
        
        # Generate plots
        plot_epoch_time_comparison(df_dict, out_dir, dataset_name)
        plot_accuracy_comparison(df_dict, out_dir, dataset_name)
        
        if baseline_key:
            plot_speedup_analysis(stats_dict, out_dir, dataset_name, baseline_key)
            plot_throughput_comparison(stats_dict, train_size, out_dir, dataset_name)
    
    # Generate summary outputs
    if all_experiments:
        print("\n" + "="*60)
        print("Generating comprehensive summary...")
        print("="*60)
        summary_df = create_summary_table(all_experiments, out_dir)
        print("\nSummary Table:")
        print(summary_df.to_string(index=False))
        
        generate_diagnostic_report(all_experiments, out_dir)
        
        print(f"\n✓ All outputs saved to: {out_dir}")
        print(f"  - Plots: *_detailed.png, *_speedup_efficiency.png, *_throughput.png, *_accuracy.png")
        print(f"  - Summary: comprehensive_summary.csv, summary_table.tex")
        print(f"  - Report: performance_report.txt")

if __name__ == "__main__":
    main()
