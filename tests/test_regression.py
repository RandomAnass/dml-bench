"""
Regression tests for fixed bugs and expanded coverage.

Tests cover:
  - Step-function noise bug (noise must be non-zero even when dydx ≡ 0)
  - GradNorm gradient contamination (model grads must be clean after GradNorm step)
  - Finance function generators (bachelier, black_scholes, heston)
  - Step function generator
  - Edge case: d=1 (single dimension)
  - Baseline model is_fitted guard
"""

import numpy as np
import torch
import pytest


# ============================================================================
# BUG REGRESSION: Step-function noise injection
# ============================================================================

class TestStepNoiseFix:
    """Verify that add_gaussian_noise works for zero-derivative functions."""

    def test_noise_applied_when_dydx_is_zero(self):
        """Step function has dydx=0 everywhere; noise must still be injected."""
        from dml_benchmark.functions import add_gaussian_noise

        n, d = 100, 5
        dydx = np.zeros((n, 1, d))  # Exactly zero derivatives (step-like)

        dydx_noisy = add_gaussian_noise(dydx, noise_level=0.1, seed=42)

        # dydx should be perturbed (was the bug: dydx stayed zero)
        assert not np.allclose(dydx, dydx_noisy), (
            "dydx noise was not applied — step noise bug still present"
        )

        # Noise should be proportional to noise_level (ref_std fallback = 1.0)
        dydx_noise_std = np.std(dydx_noisy)
        assert 0.01 < dydx_noise_std < 1.0, (
            f"Unexpected dydx noise magnitude: std={dydx_noise_std}"
        )

    def test_noise_level_zero_produces_no_noise(self):
        """noise_level=0 should produce identical outputs."""
        from dml_benchmark.functions import add_gaussian_noise

        dydx = np.zeros((50, 1, 3))

        dydx_noisy = add_gaussian_noise(dydx, noise_level=0.0, seed=42)

        np.testing.assert_array_equal(dydx, dydx_noisy)

    def test_noise_deterministic_with_seed(self):
        """Same seed should produce identical noise."""
        from dml_benchmark.functions import add_gaussian_noise

        dydx = np.ones((50, 1, 3)) * 0.5

        dydx1 = add_gaussian_noise(dydx, noise_level=0.1, seed=123)
        dydx2 = add_gaussian_noise(dydx, noise_level=0.1, seed=123)

        np.testing.assert_array_equal(dydx1, dydx2)


# ============================================================================
# BUG REGRESSION: GradNorm gradient contamination
# ============================================================================

class TestGradNormGradientContamination:
    """Verify that GradNorm's internal backward does not leak into model grads."""

    def test_model_grads_clean_after_gradnorm_forward(self):
        """After GradNormDmlLoss.forward(), model params should have NO gradients.

        The bug was: GradNorm calls .backward(retain_graph=True) internally,
        which accumulated gradients into model parameters. The fix calls
        model.zero_grad() before returning.
        """
        from dml_benchmark.model import DmlFeedForward
        from dml_benchmark.loss_balancing import GradNormDmlLoss

        model = DmlFeedForward(input_dim=3, output_dim=1, n_layers=2, hidden_size=32)
        loss_fn = GradNormDmlLoss(input_dim=3)

        x = torch.randn(16, 3, requires_grad=False)
        y_true = torch.randn(16, 1)
        dydx_true = torch.randn(16, 1, 3)

        # Forward pass (includes GradNorm's internal backward)
        y_pred, dydx_pred = model.forward_with_greek(x)
        result = loss_fn(y_pred, y_true, dydx_pred, dydx_true, model=model)

        # Model parameters should have NO gradients after GradNorm forward
        for name, param in model.named_parameters():
            if 'task_weights' in name:
                continue  # GradNorm's own weights may have grads
            assert param.grad is None or torch.all(param.grad == 0), (
                f"Gradient contamination in {name}: grad is non-zero after "
                f"GradNorm forward. The zero_grad() fix may be broken."
            )


# ============================================================================
# EXPANDED COVERAGE: Finance function generators
# ============================================================================

class TestFinanceFunctions:
    """Test finance function generators produce valid data."""

    def test_bachelier_generates_valid_data(self):
        """Bachelier model should produce finite prices and deltas."""
        from dml_benchmark.functions import generate_data

        data = generate_data("bachelier", n_dim=3, n_samples=64, seed=42)

        assert data.x.shape == (64, 3)
        assert data.y.shape == (64, 1)
        assert data.dydx.shape == (64, 1, 3)
        assert np.all(np.isfinite(data.x))
        assert np.all(np.isfinite(data.y))
        assert np.all(np.isfinite(data.dydx))
        # Bachelier prices should be non-negative
        assert np.all(data.y >= -1e-10), "Bachelier prices should be non-negative"

    def test_black_scholes_generates_valid_data(self):
        """Black-Scholes should produce finite prices with delta in [0, 1]."""
        from dml_benchmark.functions import generate_data

        data = generate_data("black_scholes", n_dim=1, n_samples=64, seed=42)

        assert data.x.shape == (64, 1)
        assert data.y.shape == (64, 1)
        assert data.dydx.shape == (64, 1, 1)
        assert np.all(np.isfinite(data.y))
        # BS call delta ∈ [0, 1]
        assert np.all(data.dydx >= -0.01) and np.all(data.dydx <= 1.01), (
            "Black-Scholes delta should be approximately in [0, 1]"
        )

    def test_heston_generates_valid_data(self):
        """Heston model should produce finite MC prices (may be noisy)."""
        from dml_benchmark.functions import generate_data

        data = generate_data("heston", n_dim=1, n_samples=32, seed=42)

        assert data.x.shape == (32, 2)  # Heston uses 2D input (S0, v0)
        assert data.y.shape == (32, 1)
        assert data.dydx.shape == (32, 1, 2)
        assert np.all(np.isfinite(data.x))
        assert np.all(np.isfinite(data.y))
        # Heston Greeks from FD can be noisy but should be finite
        assert np.all(np.isfinite(data.dydx))

    def test_heston_is_reproducible(self):
        """Same seed should give identical Heston MC results."""
        from dml_benchmark.functions import generate_data

        d1 = generate_data("heston", n_dim=1, n_samples=16, seed=99)
        d2 = generate_data("heston", n_dim=1, n_samples=16, seed=99)

        np.testing.assert_array_equal(d1.y, d2.y)
        np.testing.assert_array_equal(d1.dydx, d2.dydx)


# ============================================================================
# EXPANDED COVERAGE: Step function
# ============================================================================

class TestStepFunction:
    """Test step function generator."""

    def test_step_generates_valid_data(self):
        """Step function should produce piecewise-constant values."""
        from dml_benchmark.functions import generate_data

        data = generate_data("step", n_dim=3, n_samples=64, seed=42)

        assert data.x.shape == (64, 3)
        assert data.y.shape == (64, 1)
        assert data.dydx.shape == (64, 1, 3)
        assert np.all(np.isfinite(data.y))
        # Step derivatives should be identically zero (piecewise constant)
        np.testing.assert_array_equal(data.dydx, 0.0)


# ============================================================================
# EXPANDED COVERAGE: Edge case d=1
# ============================================================================

class TestEdgeCaseDim1:
    """Test that all generators work correctly at d=1."""

    @pytest.mark.parametrize("func_type", ["trig", "poly_trig", "step"])
    def test_dim1_generates_correct_shapes(self, func_type):
        """d=1 should produce (n,1) inputs and (n,1,1) derivatives."""
        if func_type == "poly_trig":
            pytest.importorskip("jax", reason="JAX required for poly_trig generator")
        from dml_benchmark.functions import generate_data

        data = generate_data(func_type, n_dim=1, n_samples=32, seed=42)

        assert data.x.shape == (32, 1), f"x shape wrong for {func_type} d=1"
        assert data.y.shape == (32, 1), f"y shape wrong for {func_type} d=1"
        assert data.dydx.shape == (32, 1, 1), f"dydx shape wrong for {func_type} d=1"
        assert np.all(np.isfinite(data.y))


# ============================================================================
# EXPANDED COVERAGE: Baseline is_fitted guard
# ============================================================================

class TestBaselineIsFittedGuard:
    """Test that calling predict before fit raises ValueError."""

    def test_krr_predict_before_fit_raises(self):
        """KRR.predict_gradients() before fit() should raise RuntimeError."""
        from dml_benchmark.baselines import KRRBaseline

        model = KRRBaseline()
        x = np.random.randn(10, 3)

        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict_gradients(x)

    def test_rf_predict_gradients_before_fit_raises(self):
        """RF.predict_gradients() before fit() should raise RuntimeError."""
        from dml_benchmark.baselines import RFBaseline

        model = RFBaseline()
        x = np.random.randn(10, 3)

        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict_gradients(x)
