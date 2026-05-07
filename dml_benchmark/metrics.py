"""
Metrics and evaluation utilities for DML Benchmark.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class BenchmarkMetrics:
    """Container for benchmark evaluation metrics."""
    value_mse: float
    grad_mse: float
    value_rmse: float
    grad_rmse: float
    value_relative_error: float
    grad_relative_error: float
    training_time_s: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'value_mse': self.value_mse,
            'grad_mse': self.grad_mse,
            'value_rmse': self.value_rmse,
            'grad_rmse': self.grad_rmse,
            'value_relative_error': self.value_relative_error,
            'grad_relative_error': self.grad_relative_error,
            'training_time_s': self.training_time_s
        }


def compute_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    dydx_pred: np.ndarray,
    dydx_true: np.ndarray,
    training_time_s: float = 0.0
) -> BenchmarkMetrics:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_pred: Predicted values
        y_true: True values
        dydx_pred: Predicted gradients
        dydx_true: True gradients
        training_time_s: Training time in seconds
        
    Returns:
        BenchmarkMetrics object
    """
    # Value metrics
    value_mse = np.mean((y_pred - y_true) ** 2)
    value_rmse = np.sqrt(value_mse)
    value_relative_error = np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-8))
    
    # Gradient metrics
    grad_mse = np.mean((dydx_pred - dydx_true) ** 2)
    grad_rmse = np.sqrt(grad_mse)
    grad_relative_error = np.mean(np.abs(dydx_pred - dydx_true) / (np.abs(dydx_true) + 1e-8))
    
    return BenchmarkMetrics(
        value_mse=float(value_mse),
        grad_mse=float(grad_mse),
        value_rmse=float(value_rmse),
        grad_rmse=float(grad_rmse),
        value_relative_error=float(value_relative_error),
        grad_relative_error=float(grad_relative_error),
        training_time_s=float(training_time_s)
    )


def compute_dml_advantage(
    vanilla_metrics: BenchmarkMetrics,
    dml_metrics: BenchmarkMetrics
) -> Dict[str, float]:
    """
    Compute relative improvement of DML over vanilla.
    
    Positive values = DML is better.
    
    Args:
        vanilla_metrics: Vanilla NN metrics
        dml_metrics: DML metrics
        
    Returns:
        Dict with percentage improvements
    """
    def pct_improvement(vanilla, dml):
        if vanilla < 1e-8:
            return 0.0
        return 100.0 * (vanilla - dml) / vanilla
    
    return {
        'value_mse_improvement': pct_improvement(vanilla_metrics.value_mse, dml_metrics.value_mse),
        'grad_mse_improvement': pct_improvement(vanilla_metrics.grad_mse, dml_metrics.grad_mse),
        'value_rmse_improvement': pct_improvement(vanilla_metrics.value_rmse, dml_metrics.value_rmse),
        'grad_rmse_improvement': pct_improvement(vanilla_metrics.grad_rmse, dml_metrics.grad_rmse),
        'time_overhead_pct': 100.0 * (dml_metrics.training_time_s - vanilla_metrics.training_time_s) / (vanilla_metrics.training_time_s + 1e-8)
    }


def aggregate_seed_results(
    results: List[BenchmarkMetrics]
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across multiple seeds.
    
    Args:
        results: List of metrics from different seeds
        
    Returns:
        Dict with mean and std for each metric
    """
    if not results:
        return {}
    
    aggregated = {}
    
    for field in ['value_mse', 'grad_mse', 'value_rmse', 'grad_rmse', 
                  'value_relative_error', 'grad_relative_error', 'training_time_s']:
        values = [getattr(r, field) for r in results]
        aggregated[field] = {
            'mean': float(np.mean(values)),
            # J2 (2026-04-16): sample std (ddof=1) per F-L1 — prior ddof=0
            # underestimated at small n (12% at n=5, 5% at n=10).
            'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
    
    return aggregated


# ============================================================================
# RESULT MANAGEMENT
# ============================================================================

class ResultsManager:
    """Manage benchmark results with persistence."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, Any] = {}
    
    def add_result(
        self,
        key: str,
        config: Dict[str, Any],
        metrics: BenchmarkMetrics
    ):
        """Add a result to the collection."""
        self.results[key] = {
            'config': config,
            'metrics': metrics.to_dict()
        }
    
    def save(self, filename: str = "benchmark_results.json"):
        """Save all results to JSON."""
        path = self.results_dir / filename
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def load(self, filename: str = "benchmark_results.json"):
        """Load results from JSON."""
        path = self.results_dir / filename
        if path.exists():
            with open(path, 'r') as f:
                self.results = json.load(f)
    
    def get_completed_keys(self) -> List[str]:
        """Get list of completed experiment keys."""
        return list(self.results.keys())
    
    def is_completed(self, key: str) -> bool:
        """Check if an experiment is already completed."""
        return key in self.results


def make_experiment_key(
    function_type: str,
    dim: int,
    n_samples: int,
    lambda_: float,
    noise_level: float,
    seed: int,
    method: str = "dml_fixed"
) -> str:
    """Create unique key for an experiment."""
    return f"{function_type}_d{dim}_n{n_samples}_l{lambda_}_noise{noise_level}_s{seed}_{method}"
