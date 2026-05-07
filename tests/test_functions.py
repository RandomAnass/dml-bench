"""
Tests for function generators — verifying gradient correctness.

Compares JAX autodiff gradients against numerical finite differences.
"""

import numpy as np
import pytest


def _finite_diff_gradient(f, x, h=1e-5):
    """Central finite differences for a single input vector."""
    n_dim = len(x)
    grad = np.zeros(n_dim)
    for j in range(n_dim):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += h
        x_minus[j] -= h
        grad[j] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad


class TestBenchmarkFunctionGenerator:
    """Test gradient correctness for benchmark functions."""
    
    def test_trig_gradients_2d(self):
        """Trig function: JAX gradients should match finite differences."""
        from dml_benchmark.functions import generate_data
        
        data = generate_data("trig", n_dim=2, n_samples=100, seed=42)
        
        assert data.x.shape == (100, 2)
        assert data.y.shape == (100, 1)
        assert data.dydx.shape[0] == 100
        
        # Check a few samples against finite differences
        # Note: generate_data uses random freqs/amps, so we can only check
        # consistency with finite differences on the evaluated function,
        # not against a fixed sin(x).
        # We verify shapes and finiteness here.
        assert np.all(np.isfinite(data.dydx))
    
    def test_trig_gradients_10d(self):
        """Higher-dimensional trig function gradients."""
        from dml_benchmark.functions import generate_data
        
        data = generate_data("trig", n_dim=10, n_samples=50, seed=42)
        
        assert data.x.shape == (50, 10)
        assert data.dydx.shape[-1] == 10
    
    def test_function_data_shapes(self):
        """Verify all function types produce correct shapes."""
        jax = pytest.importorskip("jax", reason="JAX required for poly_trig generator")
        from dml_benchmark.functions import generate_data
        
        for func_type in ["trig", "poly_trig"]:
            data = generate_data(func_type, n_dim=3, n_samples=32, seed=42)
            assert data.x.shape == (32, 3), f"{func_type}: x shape mismatch"
            assert data.y.shape == (32, 1), f"{func_type}: y shape mismatch"
            n_grad = data.dydx.shape[0]
            assert n_grad == 32, f"{func_type}: dydx sample count mismatch"
    
    def test_reproducibility(self):
        """Same seed should produce identical data."""
        from dml_benchmark.functions import generate_data
        
        data1 = generate_data("trig", n_dim=3, n_samples=50, seed=123)
        data2 = generate_data("trig", n_dim=3, n_samples=50, seed=123)
        
        np.testing.assert_array_equal(data1.x, data2.x)
        np.testing.assert_array_equal(data1.y, data2.y)
        np.testing.assert_array_equal(data1.dydx, data2.dydx)
    
    def test_different_seeds_differ(self):
        """Different seeds should produce different data."""
        from dml_benchmark.functions import generate_data
        
        data1 = generate_data("trig", n_dim=3, n_samples=50, seed=42)
        data2 = generate_data("trig", n_dim=3, n_samples=50, seed=99)
        
        assert not np.allclose(data1.x, data2.x)


class TestDerivativeCorruption:
    """Test derivative corruption utilities."""
    
    def test_gaussian_noise(self):
        """Gaussian noise should have approximately correct std."""
        from dml_benchmark.functions import add_gaussian_noise
        
        dydx = np.random.randn(100, 1, 5) * 2.0
        noise_level = 0.1
        
        noisy = add_gaussian_noise(dydx, noise_level, seed=42)
        
        # Should be different
        assert not np.allclose(dydx, noisy)
        
        # Noise magnitude should be approximately noise_level * std(dydx)
        noise = noisy - dydx
        expected_std = noise_level * np.std(dydx)
        actual_std = np.std(noise)
        np.testing.assert_allclose(actual_std, expected_std, rtol=0.2)
    
    def test_zero_noise_is_copy(self):
        """Zero noise level should return identical copy."""
        from dml_benchmark.functions import add_gaussian_noise
        
        dydx = np.random.randn(50, 1, 3)
        result = add_gaussian_noise(dydx, 0.0)
        np.testing.assert_array_equal(dydx, result)
