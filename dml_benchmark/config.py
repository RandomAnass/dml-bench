"""
Configuration module for DML Benchmark Suite.

Contains all experimental settings in one place for consistency.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import json
from pathlib import Path


@dataclass
class QuickExplorationConfig:
    """Fast iteration config - use for debugging and initial exploration."""
    dimensions: List[int] = field(default_factory=lambda: [2, 10, 50])
    sample_sizes: List[int] = field(default_factory=lambda: [512, 2048])
    lambda_values: List[float] = field(default_factory=lambda: [0, 0.1, 1])
    derivative_noise: List[float] = field(default_factory=lambda: [0.0, 0.1])
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    n_epochs: int = 200
    batch_size: int = 256
    
    # Model architecture
    n_layers: int = 4
    hidden_size: int = 256
    activation: str = "softplus"
    lr: float = 0.005
    
    # Function types to test
    function_types: List[str] = field(default_factory=lambda: ["trig", "poly_trig"])


@dataclass
class FullBenchmarkConfig:
    """Complete benchmark config - use for final publishable results."""
    dimensions: List[int] = field(default_factory=lambda: [1, 2, 5, 10, 20, 50, 100])
    sample_sizes: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048, 4096, 8192])
    lambda_values: List[float] = field(default_factory=lambda: [0, 0.001, 0.01, 0.1, 1, 10])
    derivative_noise: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.10, 0.20, 0.50])
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1000])
    n_epochs: int = 1000
    batch_size: int = 1024
    
    # Model architecture
    n_layers: int = 4
    hidden_size: int = 256
    activation: str = "softplus"
    lr: float = 0.005
    
    # All function types including finance
    function_types: List[str] = field(default_factory=lambda: [
        "trig", "poly_trig", "step", "bachelier", "black_scholes", "heston"
    ])


@dataclass  
class ModelConfig:
    """Fixed model architecture settings."""
    n_layers: int = 4
    hidden_size: int = 256
    activation: str = "softplus"
    optimizer: str = "Adam"
    lr: float = 0.005
    regularization_scale: float = 0.0


@dataclass
class SmokeTestConfig:
    """Minimal config for pipeline validation. Should finish in <2 minutes."""
    dimensions: List[int] = field(default_factory=lambda: [2])
    sample_sizes: List[int] = field(default_factory=lambda: [256])
    lambda_values: List[float] = field(default_factory=lambda: [0, 1])
    derivative_noise: List[float] = field(default_factory=lambda: [0.0])
    seeds: List[int] = field(default_factory=lambda: [42])
    n_epochs: int = 30
    batch_size: int = 64
    n_layers: int = 2
    hidden_size: int = 64
    activation: str = "softplus"
    lr: float = 0.005
    function_types: List[str] = field(default_factory=lambda: ["trig"])
    methods: List[str] = field(default_factory=lambda: [
        "vanilla", "dml_fixed", "dml_gradnorm", "dml_relobralo"
    ])


def get_config(mode: str = "quick") -> Dict[str, Any]:
    """Get configuration dictionary.
    
    Args:
        mode: "quick" for exploration, "full" for final benchmark
        
    Returns:
        Configuration dictionary
    """
    if mode == "smoke":
        cfg = SmokeTestConfig()
    elif mode == "quick":
        cfg = QuickExplorationConfig()
    elif mode == "full":
        cfg = FullBenchmarkConfig()
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'smoke', 'quick', or 'full'.")
    
    result = {
        "dimensions": cfg.dimensions,
        "sample_sizes": cfg.sample_sizes,
        "lambda_values": cfg.lambda_values,
        "derivative_noise": cfg.derivative_noise,
        "seeds": cfg.seeds,
        "n_epochs": cfg.n_epochs,
        "batch_size": cfg.batch_size,
        "n_layers": cfg.n_layers,
        "hidden_size": cfg.hidden_size,
        "activation": cfg.activation,
        "lr": cfg.lr,
        "function_types": cfg.function_types,
    }
    
    # Add methods if config defines them
    if hasattr(cfg, 'methods'):
        result["methods"] = cfg.methods
    
    return result


def save_config(config: Dict[str, Any], path: Path):
    """Save configuration to JSON file."""
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


def load_config(path: Path) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)
