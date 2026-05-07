"""
Tests for DataNormalizer — roundtrip consistency.
"""

import numpy as np
import pytest


class TestDataNormalizer:
    """Test normalization roundtrip properties."""
    
    def _make_normalizer(self, n_samples=200, n_dim=5):
        """Create a normalizer with random data."""
        from dml_benchmark.model import DataNormalizer
        
        x = np.random.randn(n_samples, n_dim) * 3 + 2  # Non-zero mean/std
        y = np.random.randn(n_samples, 1) * 10 + 5
        dydx = np.random.randn(n_samples, 1, n_dim) * 2
        
        norm = DataNormalizer()
        norm.initialize_with_data(x, y, dydx)
        return norm, x, y, dydx
    
    def test_roundtrip_x(self):
        """normalize_x → unscale_x should recover original."""
        norm, x, _, _ = self._make_normalizer()
        
        x_norm = norm.normalize_x(x)
        x_recovered = norm.unscale_x(x_norm)
        
        np.testing.assert_allclose(x_recovered, x, rtol=1e-6)
    
    def test_roundtrip_y(self):
        """normalize_y → unscale_y should recover original."""
        norm, _, y, _ = self._make_normalizer()
        
        y_norm = norm.normalize_y(y)
        y_recovered = norm.unscale_y(y_norm)
        
        np.testing.assert_allclose(y_recovered, y, rtol=1e-6)
    
    def test_roundtrip_dydx(self):
        """normalize_dydx → unscale_dydx should recover original."""
        norm, _, _, dydx = self._make_normalizer()
        
        dydx_norm = norm.normalize_dydx(dydx)
        dydx_recovered = norm.unscale_dydx(dydx_norm)
        
        np.testing.assert_allclose(dydx_recovered, dydx, rtol=1e-6)
    
    def test_normalized_mean_zero(self):
        """Normalized x and y should have approximately zero mean."""
        norm, x, y, _ = self._make_normalizer(n_samples=1000)
        
        x_norm = norm.normalize_x(x)
        y_norm = norm.normalize_y(y)
        
        np.testing.assert_allclose(np.mean(x_norm, axis=0), 0.0, atol=0.1)
        np.testing.assert_allclose(np.mean(y_norm), 0.0, atol=0.1)
    
    def test_normalized_std_one(self):
        """Normalized x and y should have approximately unit std."""
        norm, x, y, _ = self._make_normalizer(n_samples=1000)
        
        x_norm = norm.normalize_x(x)
        y_norm = norm.normalize_y(y)
        
        np.testing.assert_allclose(np.std(x_norm, axis=0), 1.0, atol=0.1)
        np.testing.assert_allclose(np.std(y_norm), 1.0, atol=0.1)
    
    def test_lambda_j_positive(self):
        """Lambda_j should be positive."""
        norm, _, _, _ = self._make_normalizer()
        assert np.all(norm.lambda_j > 0)
    
    def test_constant_feature_handling(self):
        """Constant feature (zero std) should be handled without NaN."""
        from dml_benchmark.model import DataNormalizer
        
        x = np.random.randn(100, 3)
        x[:, 1] = 5.0  # Constant feature
        y = np.random.randn(100, 1)
        dydx = np.random.randn(100, 1, 3)
        
        norm = DataNormalizer()
        norm.initialize_with_data(x, y, dydx)
        
        x_norm = norm.normalize_x(x)
        assert not np.any(np.isnan(x_norm))
        assert not np.any(np.isinf(x_norm))
