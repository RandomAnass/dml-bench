"""
Tests for DML model — forward/backward pass shapes, gradient computation.
"""

import torch
import numpy as np
import pytest


class TestDmlFeedForward:
    """Test the DML neural network."""
    
    def test_forward_shape(self):
        """Forward pass should produce correct output shape."""
        from dml_benchmark.model import DmlFeedForward
        
        model = DmlFeedForward(input_dim=5, output_dim=1, n_layers=3, hidden_size=64)
        x = torch.randn(32, 5)
        y = model(x)
        assert y.shape == (32, 1)
    
    def test_forward_with_greek_shapes(self):
        """forward_with_greek should return (y, dydx) with correct shapes."""
        from dml_benchmark.model import DmlFeedForward
        
        model = DmlFeedForward(input_dim=5, output_dim=1, n_layers=3, hidden_size=64)
        x = torch.randn(32, 5)
        y, dydx = model.forward_with_greek(x)
        
        assert y.shape == (32, 1)
        assert dydx.shape == (32, 1, 5)
    
    def test_gradient_requires_grad(self):
        """Predicted gradients should have grad_fn (needed for backward)."""
        from dml_benchmark.model import DmlFeedForward
        
        model = DmlFeedForward(input_dim=3, output_dim=1, n_layers=2, hidden_size=32)
        x = torch.randn(16, 3)
        y, dydx = model.forward_with_greek(x)
        
        assert y.requires_grad
        assert dydx.requires_grad
    
    def test_activations(self):
        """All supported activations should work."""
        from dml_benchmark.model import DmlFeedForward
        
        for act in ["softplus", "relu", "sigmoid", "tanh"]:
            model = DmlFeedForward(
                input_dim=3, output_dim=1, n_layers=2, 
                hidden_size=32, activation=act
            )
            x = torch.randn(8, 3)
            y = model(x)
            assert y.shape == (8, 1), f"Failed for activation: {act}"
    
    def test_invalid_activation_raises(self):
        """Unknown activation should raise ValueError."""
        from dml_benchmark.model import DmlFeedForward
        
        with pytest.raises(ValueError, match="Unknown activation"):
            DmlFeedForward(input_dim=3, output_dim=1, n_layers=2, 
                          hidden_size=32, activation="leaky_gelu")
    
    def test_weight_init_kaiming_for_relu(self):
        """ReLU/softplus should use kaiming init (fan_in variance ~ 2/fan_in)."""
        from dml_benchmark.model import DmlFeedForward
        
        model = DmlFeedForward(
            input_dim=100, output_dim=1, n_layers=3, 
            hidden_size=256, activation="relu"
        )
        
        # Check first hidden layer weight variance
        w = model.layers[0].weight.data
        fan_in = w.shape[1]
        expected_var = 2.0 / fan_in  # Kaiming
        actual_var = w.var().item()
        
        # Should be roughly in the right ballpark (within 3x)
        assert 0.3 * expected_var < actual_var < 3.0 * expected_var
    
    def test_weight_init_xavier_for_tanh(self):
        """Tanh should use xavier init."""
        from dml_benchmark.model import DmlFeedForward
        
        model = DmlFeedForward(
            input_dim=100, output_dim=1, n_layers=3, 
            hidden_size=256, activation="tanh"
        )
        
        w = model.layers[0].weight.data
        fan_in = w.shape[1]
        fan_out = w.shape[0]
        expected_var = 2.0 / (fan_in + fan_out)  # Xavier
        actual_var = w.var().item()
        
        assert 0.3 * expected_var < actual_var < 3.0 * expected_var


class TestDmlDataset:
    """Test the dataset class."""
    
    def test_dataset_length(self):
        from dml_benchmark.model import DmlDataset
        
        x = np.random.randn(100, 5)
        y = np.random.randn(100, 1)
        dydx = np.random.randn(100, 5)  # 2D input
        
        ds = DmlDataset(x, y, dydx)
        assert len(ds) == 100
    
    def test_dataset_shapes(self):
        from dml_benchmark.model import DmlDataset
        
        x = np.random.randn(50, 3)
        y = np.random.randn(50, 1)
        dydx = np.random.randn(50, 3)  # 2D → should be reshaped to (50, 1, 3)
        
        ds = DmlDataset(x, y, dydx)
        item = ds[0]
        
        assert item['x'].shape == (3,)
        assert item['y'].shape == (1,)
        assert item['dydx'].shape == (1, 3)
    
    def test_dataset_dtype(self):
        from dml_benchmark.model import DmlDataset
        
        x = np.random.randn(10, 2).astype(np.float64)
        y = np.random.randn(10, 1).astype(np.float64)
        dydx = np.random.randn(10, 2).astype(np.float64)
        
        ds = DmlDataset(x, y, dydx)
        assert ds.x.dtype == torch.float32
