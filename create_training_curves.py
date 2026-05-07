#!/usr/bin/env python3
"""
Generate training curve figures for representative DML experiments.

Produces 2×2 panels showing loss vs epoch for:
  - poly_trig d=5 (DML success case)
  - poly_trig d=50 (high-d DML success)
  - trig d=5 (DML marginal case)
  - bachelier d=10 (finance success case)

Each panel overlays vanilla, dml_fixed, dml_gradnorm, dml_relobralo.

Usage:
    python create_training_curves.py
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Style
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

METHOD_LABELS = {
    'vanilla': 'Vanilla',
    'dml_fixed': r'DML ($\lambda=1$)',
    'dml_gradnorm': 'DML+GradNorm',
    'dml_relobralo': 'DML+ReLoBRaLo',
}

METHOD_COLORS = {
    'vanilla': '#1f77b4',
    'dml_fixed': '#d62728',
    'dml_gradnorm': '#2ca02c',
    'dml_relobralo': '#ff7f0e',
}

METHOD_STYLES = {
    'vanilla': '-',
    'dml_fixed': '-',
    'dml_gradnorm': '--',
    'dml_relobralo': ':',
}


def load_results_with_logs(results_dir, func_type, dim, seed=2000, n_samples=1024):
    """Load results that have training logs for a specific config."""
    results = {}
    for method in ['vanilla', 'dml_fixed', 'dml_gradnorm', 'dml_relobralo']:
        pattern = f"{func_type}_d{dim}_n{n_samples}_noise0.0_s{seed}_{method}.json"
        path = Path(results_dir) / pattern
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                if data.get('training_logs') and len(data['training_logs']) > 0:
                    results[method] = data
    return results


def find_best_seed_with_logs(results_dirs, func_type, dim, n_samples=1024):
    """Find a seed that has training logs for all methods."""
    # tier4 (seeds 2000-6000) was run with --save-logs
    seeds_to_try = [2000, 3000, 4000, 5000, 6000, 42, 123, 456, 789, 1000]
    for results_dir in results_dirs:
        for seed in seeds_to_try:
            results = load_results_with_logs(results_dir, func_type, dim, seed, n_samples)
            if len(results) >= 3:  # At least vanilla + dml_fixed + one adaptive
                return results, seed, results_dir
    return {}, None, None


def smooth(y, window=5):
    """Simple moving average smoothing."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='valid')


def plot_training_curves(results, ax, title, metric='val_loss', smooth_window=5):
    """Plot training curves for all methods on a single axis."""
    for method in ['vanilla', 'dml_fixed', 'dml_gradnorm', 'dml_relobralo']:
        if method not in results:
            continue
        logs = results[method]['training_logs']
        epochs = [l['epoch'] for l in logs]
        values = [l[metric] for l in logs]
        
        # Smooth
        if smooth_window > 1 and len(values) > smooth_window:
            smoothed = smooth(np.array(values), smooth_window)
            ep_smoothed = epochs[smooth_window-1:]
        else:
            smoothed = values
            ep_smoothed = epochs
        
        ax.semilogy(ep_smoothed, smoothed,
                     color=METHOD_COLORS[method],
                     linestyle=METHOD_STYLES[method],
                     linewidth=1.5,
                     label=METHOD_LABELS[method],
                     alpha=0.85)
    
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Epoch')
    ax.grid(True, alpha=0.3)


def main():
    results_dirs = [
        'results/tier4_benchmark',
        'results/tier1_benchmark',
        'results/tier2_benchmark',
        'results/tier3_benchmark',
    ]
    
    # Configs to plot
    configs = [
        ('poly_trig', 5, 'poly_trig d=5\n(DML success)'),
        ('poly_trig', 50, 'poly_trig d=50\n(high-d success)'),
        ('trig', 5, 'trig d=5\n(DML marginal)'),
        ('bachelier', 10, 'bachelier d=10\n(finance)'),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    
    any_plotted = False
    for idx, (func, dim, title) in enumerate(configs):
        results, seed, rdir = find_best_seed_with_logs(results_dirs, func, dim)
        
        if not results:
            # Try n_samples=4096
            results, seed, rdir = find_best_seed_with_logs(results_dirs, func, dim, n_samples=4096)
        
        if not results:
            axes[idx].text(0.5, 0.5, f'No training logs found\nfor {func} d={dim}',
                          ha='center', va='center', transform=axes[idx].transAxes,
                          fontsize=10, color='gray')
            axes[idx].set_title(title)
            continue
        
        any_plotted = True
        plot_training_curves(results, axes[idx], 
                            f"{title}\n(seed={seed}, n=1024)",
                            metric='val_loss', smooth_window=10)
        
        if idx == 0:
            axes[idx].legend(loc='best', framealpha=0.9)
        axes[idx].set_ylabel('Validation Loss (log)')
    
    fig.suptitle('Training Curves: DML vs Vanilla', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/training_curves.pdf')
    fig.savefig('figures/training_curves.png')
    print("Saved figures/training_curves.pdf and .png")
    
    # === Figure 2: Value vs Derivative Loss decomposition ===
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 9))
    axes2 = axes2.flatten()
    
    for idx, (func, dim, title) in enumerate(configs):
        results, seed, rdir = find_best_seed_with_logs(results_dirs, func, dim)
        if not results:
            results, seed, rdir = find_best_seed_with_logs(results_dirs, func, dim, n_samples=4096)
        if not results:
            axes2[idx].set_title(title)
            continue
        
        ax = axes2[idx]
        # Plot value + derivative loss for dml_fixed
        if 'dml_fixed' in results:
            logs = results['dml_fixed']['training_logs']
            epochs = [l['epoch'] for l in logs]
            val_loss = smooth(np.array([l.get('val_value_loss', l['val_loss']) for l in logs]), 10)
            deriv_loss = smooth(np.array([l.get('val_deriv_loss', 0) for l in logs]), 10)
            ep = epochs[9:]  # After smoothing
            
            ax.semilogy(ep, val_loss, '-', color='#d62728', linewidth=1.5, label='Value loss', alpha=0.85)
            if any(d > 0 for d in deriv_loss):
                ax.semilogy(ep, deriv_loss, '--', color='#9467bd', linewidth=1.5, label='Deriv loss', alpha=0.85)
        
        # Overlay vanilla value loss for comparison
        if 'vanilla' in results:
            logs = results['vanilla']['training_logs']
            epochs = [l['epoch'] for l in logs]
            van_val = smooth(np.array([l['val_loss'] for l in logs]), 10)
            ep = epochs[9:]
            ax.semilogy(ep, van_val, '-', color='#1f77b4', linewidth=1.5, label='Vanilla loss', alpha=0.6)
        
        ax.set_title(f"{title}\n(seed={seed})", fontsize=11)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (log)')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc='best', framealpha=0.9)
    
    fig2.suptitle('Loss Decomposition: Value vs Derivative Components', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig2.savefig('figures/training_curves_decomposition.pdf')
    fig2.savefig('figures/training_curves_decomposition.png')
    print("Saved figures/training_curves_decomposition.pdf and .png")
    
    plt.close('all')


if __name__ == '__main__':
    main()
