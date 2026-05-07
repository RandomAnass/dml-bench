"""
Tests for loss functions — DmlLoss, VanillaLoss, GradNorm, ReLoBRaLo.
"""

import torch
import numpy as np
import pytest


class TestDmlLoss:
    """Test DML loss function."""
    
    def test_loss_components_exist(self):
        """DmlLoss should return all four components."""
        from dml_benchmark.model import DmlLoss
        
        loss_fn = DmlLoss(lambda_=1.0, input_dim=3)
        
        y_pred = torch.randn(16, 1)
        y_true = torch.randn(16, 1)
        dydx_pred = torch.randn(16, 1, 3)
        dydx_true = torch.randn(16, 1, 3)
        
        result = loss_fn(y_pred, y_true, dydx_pred, dydx_true)
        
        assert hasattr(result, 'total')
        assert hasattr(result, 'value_loss')
        assert hasattr(result, 'deriv_loss')
        assert hasattr(result, 'reg_loss')
    
    def test_vanilla_lambda_zero_weights(self):
        """With lambda=0, derivative loss weight should be 0."""
        from dml_benchmark.model import DmlLoss
        
        loss_fn = DmlLoss(lambda_=0.0, input_dim=3)
        
        assert loss_fn.ml_loss_scale == 1.0
        assert loss_fn.dml_loss_scale == 0.0
    
    def test_large_lambda_weights(self):
        """With large lambda, derivative loss should dominate."""
        from dml_benchmark.model import DmlLoss
        
        loss_fn = DmlLoss(lambda_=100.0, input_dim=3)
        
        # dml_loss_scale should be close to 1
        assert loss_fn.dml_loss_scale > 0.99
    
    def test_loss_is_differentiable(self):
        """Total loss should be differentiable (needed for backward)."""
        from dml_benchmark.model import DmlLoss, DmlFeedForward
        
        model = DmlFeedForward(input_dim=3, output_dim=1, n_layers=2, hidden_size=32)
        loss_fn = DmlLoss(lambda_=1.0, input_dim=3)
        
        x = torch.randn(8, 3)
        y_true = torch.randn(8, 1)
        dydx_true = torch.randn(8, 1, 3)
        
        y_pred, dydx_pred = model.forward_with_greek(x)
        result = loss_fn(y_pred, y_true, dydx_pred, dydx_true, model)
        
        # Should be able to call backward without error
        result.total.backward()
        
        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None


class TestVanillaLoss:
    """Test vanilla (MSE-only) loss."""
    
    def test_vanilla_ignores_derivatives(self):
        """VanillaLoss should have zero derivative loss."""
        from dml_benchmark.model import VanillaLoss
        
        loss_fn = VanillaLoss()
        y_pred = torch.randn(16, 1)
        y_true = torch.randn(16, 1)
        
        result = loss_fn(y_pred, y_true)
        
        assert result.deriv_loss.item() == 0.0
        assert result.total.item() == result.value_loss.item()
    
    def test_vanilla_mse_correct(self):
        """VanillaLoss should compute correct MSE."""
        from dml_benchmark.model import VanillaLoss
        
        loss_fn = VanillaLoss()
        y_pred = torch.tensor([[1.0], [2.0], [3.0]])
        y_true = torch.tensor([[1.5], [2.5], [3.5]])
        
        result = loss_fn(y_pred, y_true)
        expected = torch.mean((y_pred - y_true) ** 2)
        
        torch.testing.assert_close(result.total, expected)


class TestGradNormLoss:
    """Test GradNorm adaptive loss."""
    
    def test_gradnorm_forward(self):
        """GradNorm loss should produce valid output."""
        from dml_benchmark.loss_balancing import GradNormDmlLoss
        
        loss_fn = GradNormDmlLoss(input_dim=3)
        
        y_pred = torch.randn(16, 1)
        y_true = torch.randn(16, 1)
        dydx_pred = torch.randn(16, 1, 3)
        dydx_true = torch.randn(16, 1, 3)
        
        result = loss_fn(y_pred, y_true, dydx_pred, dydx_true)
        
        assert not torch.isnan(result.total)
        assert result.total.item() > 0
    
    def test_gradnorm_weights_accessible(self):
        """Should be able to get current task weights."""
        from dml_benchmark.loss_balancing import GradNormDmlLoss
        
        loss_fn = GradNormDmlLoss(input_dim=3)
        weights = loss_fn.get_weights()
        
        assert "value_weight" in weights
        assert "deriv_weight" in weights


class TestReLoBRaLoLoss:
    """Test ReLoBRaLo adaptive loss."""
    
    def test_relobralo_forward(self):
        """ReLoBRaLo loss should produce valid output."""
        from dml_benchmark.loss_balancing import ReLoBRaLoDmlLoss
        
        loss_fn = ReLoBRaLoDmlLoss(input_dim=3)
        loss_fn.train()
        
        y_pred = torch.randn(16, 1)
        y_true = torch.randn(16, 1)
        dydx_pred = torch.randn(16, 1, 3)
        dydx_true = torch.randn(16, 1, 3)
        
        # First call — no history yet
        result1 = loss_fn(y_pred, y_true, dydx_pred, dydx_true)
        assert not torch.isnan(result1.total)
        
        # Second call — now has history for lookback
        result2 = loss_fn(y_pred, y_true, dydx_pred, dydx_true)
        assert not torch.isnan(result2.total)
    
    def test_relobralo_history_grows(self):
        """Loss history should grow during training."""
        from dml_benchmark.loss_balancing import ReLoBRaLoDmlLoss
        
        loss_fn = ReLoBRaLoDmlLoss(input_dim=3)
        loss_fn.train()
        
        for _ in range(5):
            loss_fn(torch.randn(8, 1), torch.randn(8, 1),
                    torch.randn(8, 1, 3), torch.randn(8, 1, 3))
        
        assert len(loss_fn.loss_history) == 5
    
    def test_relobralo_reset(self):
        """Reset should clear history and weights."""
        from dml_benchmark.loss_balancing import ReLoBRaLoDmlLoss
        
        loss_fn = ReLoBRaLoDmlLoss(input_dim=3)
        loss_fn.train()
        
        loss_fn(torch.randn(8, 1), torch.randn(8, 1),
                torch.randn(8, 1, 3), torch.randn(8, 1, 3))
        
        assert len(loss_fn.loss_history) > 0
        
        loss_fn.reset()
        assert len(loss_fn.loss_history) == 0
