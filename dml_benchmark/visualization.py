"""
Visualization utilities for DML Benchmark.

Provides Nature-quality figures for publication and Plotly for interactive analysis.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None
    warnings.warn("Plotly not available for interactive plots")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# ============================================================================
# NATURE-STYLE MATPLOTLIB CONFIGURATION
# ============================================================================

def setup_nature_style():
    """Configure matplotlib for Nature-quality figures."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    # Nature figure sizes (in inches)
    # Single column: 89mm = 3.5in
    # Double column: 183mm = 7.2in
    
    plt.rcParams.update({
        # Font
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        
        # Lines
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        
        # Axes
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        
        # Grid
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        
        # Figure
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.format': 'pdf',
        
        # Colors - use a professional palette
        'axes.prop_cycle': plt.cycler(color=[
            '#0072B2',  # Blue
            '#D55E00',  # Orange
            '#009E73',  # Green
            '#CC79A7',  # Pink
            '#F0E442',  # Yellow
            '#56B4E9',  # Light blue
            '#E69F00',  # Gold
        ])
    })
    
    if SEABORN_AVAILABLE:
        sns.set_style("whitegrid")


def nature_figure(
    n_rows: int = 1,
    n_cols: int = 1,
    width: str = "single"
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create Nature-style figure.
    
    Args:
        n_rows: Number of subplot rows
        n_cols: Number of subplot columns
        width: "single" (3.5in) or "double" (7.2in)
        
    Returns:
        (fig, axes)
    """
    setup_nature_style()
    
    if width == "single":
        fig_width = 3.5
    elif width == "double":
        fig_width = 7.2
    else:
        fig_width = float(width)
    
    # Golden ratio for height
    fig_height = fig_width / 1.618 * n_rows / n_cols
    fig_height = min(fig_height, 10)  # Cap at 10 inches
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    plt.tight_layout()
    
    return fig, axes


# ============================================================================
# TRAINING CURVE PLOTS
# ============================================================================

def plot_training_curves(
    training_logs: List[Dict[str, float]],
    title: str = "Training Progress",
    save_path: Optional[Path] = None,
    show_components: bool = True
) -> plt.Figure:
    """
    Plot training curves with loss decomposition.
    
    Args:
        training_logs: List of log dicts from training
        title: Plot title
        save_path: Optional path to save figure
        show_components: Show value/deriv loss components
        
    Returns:
        matplotlib Figure
    """
    setup_nature_style()
    
    epochs = [log['epoch'] for log in training_logs]
    train_loss = [log['train_loss'] for log in training_logs]
    val_loss = [log['val_loss'] for log in training_logs]
    
    if show_components and 'train_value_loss' in training_logs[0]:
        n_plots = 2
        fig, axes = nature_figure(1, 2, "double")
        
        # Total loss
        axes[0].plot(epochs, train_loss, label='Train', linewidth=1.5)
        axes[0].plot(epochs, val_loss, label='Val', linewidth=1.5)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_yscale('log')
        axes[0].legend()
        axes[0].set_title('Total Loss')
        
        # Components
        train_value = [log['train_value_loss'] for log in training_logs]
        train_deriv = [log['train_deriv_loss'] for log in training_logs]
        
        axes[1].plot(epochs, train_value, label='Value Loss', linewidth=1.5)
        axes[1].plot(epochs, train_deriv, label='Deriv Loss', linewidth=1.5)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_yscale('log')
        axes[1].legend()
        axes[1].set_title('Loss Components')
    else:
        fig, ax = nature_figure(1, 1, "single")
        ax.plot(epochs, train_loss, label='Train', linewidth=1.5)
        ax.plot(epochs, val_loss, label='Val', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.legend()
        ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


# ============================================================================
# DML ADVANTAGE HEATMAP
# ============================================================================

def plot_dml_advantage_heatmap(
    results: Dict[Tuple[int, int], Dict[str, float]],
    metric: str = "value_mse",
    title: str = "DML Advantage (% improvement)",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot heatmap of DML advantage across dimensions and sample sizes.
    
    Args:
        results: Dict mapping (dim, n_samples) to metrics dict
        metric: Which metric to use
        title: Plot title
        save_path: Optional save path
        
    Returns:
        matplotlib Figure
    """
    setup_nature_style()
    
    # Extract dimensions and sample sizes
    dims = sorted(set(k[0] for k in results.keys()))
    samples = sorted(set(k[1] for k in results.keys()))
    
    # Build advantage matrix
    advantage = np.zeros((len(dims), len(samples)))
    
    for i, d in enumerate(dims):
        for j, n in enumerate(samples):
            key = (d, n)
            if key in results:
                vanilla = results[key].get(f'vanilla_{metric}', 1.0)
                dml = results[key].get(f'dml_{metric}', 1.0)
                # Percentage improvement
                advantage[i, j] = 100 * (vanilla - dml) / vanilla
    
    fig, ax = nature_figure(1, 1, "single")
    
    # Use diverging colormap centered at 0
    vmax = max(abs(advantage.min()), abs(advantage.max()))
    
    im = ax.imshow(
        advantage. T,
        cmap='RdYlGn',
        aspect='auto',
        vmin=-vmax,
        vmax=vmax,
        origin='lower'
    )
    
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(dims)
    ax.set_yticks(range(len(samples)))
    ax.set_yticklabels(samples)
    ax.set_xlabel('Input Dimension')
    ax.set_ylabel('Sample Size')
    ax.set_title(title)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('% Improvement')
    
    # Annotate cells
    for i in range(len(dims)):
        for j in range(len(samples)):
            text = f'{advantage[i, j]:.1f}'
            color = 'black' if abs(advantage[i, j]) < vmax * 0.5 else 'white'
            ax.text(i, j, text, ha='center', va='center', color=color, fontsize=7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


# ============================================================================
# NOISE ROBUSTNESS CURVES
# ============================================================================

def plot_noise_robustness(
    noise_levels: List[float],
    vanilla_mse: List[float],
    dml_mse: List[float],
    title: str = "Derivative Noise Robustness",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot how performance degrades with derivative noise (Gap C).
    
    Args:
        noise_levels: List of noise levels tested
        vanilla_mse: Vanilla MSE at each noise level
        dml_mse: DML MSE at each noise level
        title: Plot title
        save_path: Optional save path
        
    Returns:
        matplotlib Figure
    """
    setup_nature_style()
    
    fig, ax = nature_figure(1, 1, "single")
    
    ax.plot(noise_levels, vanilla_mse, 'o-', label='Vanilla', linewidth=1.5, markersize=5)
    ax.plot(noise_levels, dml_mse, 's-', label='DML', linewidth=1.5, markersize=5)
    
    ax.set_xlabel('Derivative Noise Level')
    ax.set_ylabel('Test MSE')
    ax.set_title(title)
    ax.legend()
    
    # Mark crossover point if exists
    for i in range(len(noise_levels) - 1):
        if dml_mse[i] < vanilla_mse[i] and dml_mse[i+1] > vanilla_mse[i+1]:
            ax.axvline(x=(noise_levels[i] + noise_levels[i+1])/2, 
                      color='red', linestyle='--', alpha=0.5, label='Crossover')
            break
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


# ============================================================================
# PLOTLY INTERACTIVE VISUALIZATIONS
# ============================================================================

def interactive_training_curves(
    training_logs: List[Dict[str, float]],
    title: str = "Training Progress"
) -> go.Figure:
    """
    Interactive training curve plot with Plotly.
    """
    if not PLOTLY_AVAILABLE:
        raise RuntimeError("Plotly not available")
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Total Loss', 'Loss Components'))
    
    epochs = [log['epoch'] for log in training_logs]
    
    # Total loss
    fig.add_trace(
        go.Scatter(x=epochs, y=[log['train_loss'] for log in training_logs],
                   name='Train Loss', mode='lines'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=[log['val_loss'] for log in training_logs],
                   name='Val Loss', mode='lines'),
        row=1, col=1
    )
    
    # Components
    if 'train_value_loss' in training_logs[0]:
        fig.add_trace(
            go.Scatter(x=epochs, y=[log['train_value_loss'] for log in training_logs],
                       name='Value Loss', mode='lines'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=[log['train_deriv_loss'] for log in training_logs],
                       name='Deriv Loss', mode='lines'),
            row=1, col=2
        )
    
    fig.update_yaxes(type='log')
    fig.update_layout(title=title, height=400)
    
    return fig


def interactive_benchmark_explorer(
    results_df,  # pandas DataFrame with results
    x_col: str = "dim",
    y_col: str = "n_samples",
    z_col: str = "dml_advantage"
) -> go.Figure:
    """
    Interactive 3D surface plot for exploring benchmark results.
    """
    if not PLOTLY_AVAILABLE:
        raise RuntimeError("Plotly not available")
    
    fig = go.Figure(data=[
        go.Surface(
            x=results_df[x_col].unique(),
            y=results_df[y_col].unique(),
            z=results_df.pivot(index=y_col, columns=x_col, values=z_col).values,
            colorscale='RdYlGn',
            colorbar=dict(title='DML Advantage %')
        )
    ])
    
    fig.update_layout(
        title='DML Advantage Across Dimensions and Sample Sizes',
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col
        )
    )
    
    return fig
