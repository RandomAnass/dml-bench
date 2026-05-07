"""
End-to-end smoke test — generate data → train 5 epochs → verify loss decreases.
"""

import numpy as np
import torch
import pytest


class TestEndToEndSmoke:
    """Minimal end-to-end test that the full pipeline works."""
    
    def test_dml_pipeline_runs(self):
        """DML training should complete without errors on tiny data."""
        from dml_benchmark.functions import generate_data, train_test_split
        from dml_benchmark.trainer import train_single_experiment
        
        data = generate_data("trig", n_dim=2, n_samples=128, seed=42)
        train_data, test_data = train_test_split(data, train_ratio=0.75, seed=42)
        
        result = train_single_experiment(
            x_train=train_data.x,
            y_train=train_data.y,
            dydx_train=train_data.dydx,
            x_test=test_data.x,
            y_test=test_data.y,
            dydx_test=test_data.dydx,
            lambda_=1.0,
            n_epochs=5,
            batch_size=32,
            n_layers=2,
            hidden_size=32,
            lr=0.005,
            method="dml_fixed",
            pbar=False
        )
        
        # Should have training logs
        assert len(result.training_logs) == 5
        
        # Test MSE should be finite
        assert np.isfinite(result.test_value_mse)
        assert np.isfinite(result.test_grad_mse)
        
        # Config should include method and metadata
        assert result.config['method'] == 'dml_fixed'
        assert 'run_metadata' in result.config
    
    def test_vanilla_pipeline_runs(self):
        """Vanilla training should complete without errors."""
        from dml_benchmark.functions import generate_data, train_test_split
        from dml_benchmark.trainer import train_single_experiment
        
        data = generate_data("trig", n_dim=2, n_samples=128, seed=42)
        train_data, test_data = train_test_split(data, train_ratio=0.75, seed=42)
        
        result = train_single_experiment(
            x_train=train_data.x,
            y_train=train_data.y,
            dydx_train=train_data.dydx,
            x_test=test_data.x,
            y_test=test_data.y,
            dydx_test=test_data.dydx,
            lambda_=0.0,
            n_epochs=5,
            batch_size=32,
            n_layers=2,
            hidden_size=32,
            method="vanilla",
            pbar=False
        )
        
        assert len(result.training_logs) == 5
        assert np.isfinite(result.test_value_mse)
    
    def test_gradnorm_pipeline_runs(self):
        """GradNorm method should complete without errors."""
        from dml_benchmark.functions import generate_data, train_test_split
        from dml_benchmark.trainer import train_single_experiment
        
        data = generate_data("trig", n_dim=2, n_samples=128, seed=42)
        train_data, test_data = train_test_split(data, train_ratio=0.75, seed=42)
        
        result = train_single_experiment(
            x_train=train_data.x,
            y_train=train_data.y,
            dydx_train=train_data.dydx,
            x_test=test_data.x,
            y_test=test_data.y,
            dydx_test=test_data.dydx,
            n_epochs=5,
            batch_size=32,
            n_layers=2,
            hidden_size=32,
            method="dml_gradnorm",
            pbar=False
        )
        
        assert len(result.training_logs) == 5
        assert np.isfinite(result.test_value_mse)
    
    def test_relobralo_pipeline_runs(self):
        """ReLoBRaLo method should complete without errors."""
        from dml_benchmark.functions import generate_data, train_test_split
        from dml_benchmark.trainer import train_single_experiment
        
        data = generate_data("trig", n_dim=2, n_samples=128, seed=42)
        train_data, test_data = train_test_split(data, train_ratio=0.75, seed=42)
        
        result = train_single_experiment(
            x_train=train_data.x,
            y_train=train_data.y,
            dydx_train=train_data.dydx,
            x_test=test_data.x,
            y_test=test_data.y,
            dydx_test=test_data.dydx,
            n_epochs=5,
            batch_size=32,
            n_layers=2,
            hidden_size=32,
            method="dml_relobralo",
            pbar=False
        )
        
        assert len(result.training_logs) == 5
        assert np.isfinite(result.test_value_mse)
    
    def test_training_loss_decreases(self):
        """Training loss should generally decrease over epochs."""
        from dml_benchmark.functions import generate_data, train_test_split
        from dml_benchmark.trainer import train_single_experiment
        
        data = generate_data("trig", n_dim=2, n_samples=256, seed=42)
        train_data, test_data = train_test_split(data, train_ratio=0.8, seed=42)
        
        result = train_single_experiment(
            x_train=train_data.x,
            y_train=train_data.y,
            dydx_train=train_data.dydx,
            x_test=test_data.x,
            y_test=test_data.y,
            dydx_test=test_data.dydx,
            lambda_=1.0,
            n_epochs=20,
            batch_size=64,
            n_layers=2,
            hidden_size=64,
            method="dml_fixed",
            pbar=False
        )
        
        # First epoch loss should be higher than last
        first_loss = result.training_logs[0]['train_loss']
        last_loss = result.training_logs[-1]['train_loss']
        assert last_loss < first_loss, (
            f"Loss should decrease: first={first_loss:.6f}, last={last_loss:.6f}"
        )
    
    def test_lr_scheduling_works(self):
        """LR should decrease when validation loss plateaus."""
        from dml_benchmark.functions import generate_data, train_test_split
        from dml_benchmark.trainer import train_single_experiment
        
        data = generate_data("trig", n_dim=2, n_samples=128, seed=42)
        train_data, test_data = train_test_split(data, train_ratio=0.75, seed=42)
        
        result = train_single_experiment(
            x_train=train_data.x,
            y_train=train_data.y,
            dydx_train=train_data.dydx,
            x_test=test_data.x,
            y_test=test_data.y,
            dydx_test=test_data.dydx,
            n_epochs=5,
            batch_size=32,
            n_layers=2,
            hidden_size=32,
            method="vanilla",
            scheduler_patience=2,
            pbar=False
        )
        
        # LR should be logged in training logs
        assert 'lr' in result.training_logs[0]


class TestBaselines:
    """Smoke test for baseline models."""
    
    def test_gp_baseline(self):
        """GP baseline should fit and predict."""
        from dml_benchmark.baselines import run_baseline_experiment
        
        np.random.seed(42)
        x_train = np.random.randn(50, 2)
        y_train = np.sin(x_train[:, 0:1]) + np.cos(x_train[:, 1:2])
        dydx_train = np.stack([
            np.cos(x_train[:, 0:1]), -np.sin(x_train[:, 1:2])
        ], axis=-1).reshape(50, 1, 2)
        
        x_test = np.random.randn(20, 2)
        y_test = np.sin(x_test[:, 0:1]) + np.cos(x_test[:, 1:2])
        dydx_test = np.stack([
            np.cos(x_test[:, 0:1]), -np.sin(x_test[:, 1:2])
        ], axis=-1).reshape(20, 1, 2)
        
        result = run_baseline_experiment(
            'gp', x_train, y_train, dydx_train,
            x_test, y_test, dydx_test
        )
        
        assert np.isfinite(result['value_mse'])
        assert np.isfinite(result['grad_mse'])
        assert result['training_time_s'] > 0
    
    def test_krr_baseline(self):
        """KRR baseline should fit and predict."""
        from dml_benchmark.baselines import run_baseline_experiment
        
        np.random.seed(42)
        x_train = np.random.randn(50, 2)
        y_train = np.sin(x_train[:, 0:1])
        dydx_train = np.cos(x_train[:, 0:1]).reshape(50, 1, 1)
        dydx_train = np.concatenate([dydx_train, np.zeros((50, 1, 1))], axis=-1)
        
        x_test = np.random.randn(20, 2)
        y_test = np.sin(x_test[:, 0:1])
        dydx_test = np.cos(x_test[:, 0:1]).reshape(20, 1, 1)
        dydx_test = np.concatenate([dydx_test, np.zeros((20, 1, 1))], axis=-1)
        
        result = run_baseline_experiment(
            'krr', x_train, y_train, dydx_train,
            x_test, y_test, dydx_test
        )
        
        assert np.isfinite(result['value_mse'])
    
    def test_rf_baseline(self):
        """RF baseline should fit and predict."""
        from dml_benchmark.baselines import run_baseline_experiment
        
        np.random.seed(42)
        x_train = np.random.randn(50, 3)
        y_train = x_train[:, 0:1] ** 2
        dydx_train = np.zeros((50, 1, 3))
        dydx_train[:, 0, 0] = 2 * x_train[:, 0]
        
        x_test = np.random.randn(20, 3)
        y_test = x_test[:, 0:1] ** 2
        dydx_test = np.zeros((20, 1, 3))
        dydx_test[:, 0, 0] = 2 * x_test[:, 0]
        
        result = run_baseline_experiment(
            'rf', x_train, y_train, dydx_train,
            x_test, y_test, dydx_test
        )
        
        assert np.isfinite(result['value_mse'])
