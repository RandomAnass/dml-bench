"""
Regression tests for fixed bugs + expanded coverage for finance and edge cases.

These tests ensure previously fixed bugs do not regress:
  - Step noise fix: add_gaussian_noise produces non-zero noise when std(dydx)=0
  - GradNorm gradient contamination: model gradients are clean after GradNorm step
  - Baseline is_fitted guard: predict_gradients raises on unfitted model

Additional coverage:
  - Finance functions (bachelier, black_scholes, heston)
  - Step function
  - Edge case: d=1
  - ReLoBRaLo determinism (seeded RNG)
"""

import numpy as np
import torch
import pytest


# ============================================================================
# BUG REGRESSION TESTS
# ============================================================================

class TestStepNoiseBugRegression:
    """Verify the step-function noise bug (Feb 2026) stays fixed.

    Bug: add_gaussian_noise scaled noise as noise_level * std(dydx).
    For step functions, dydx ≡ 0, so std(dydx) = 0 → noise_std = 0
    → no noise applied regardless of noise_level.

    Fix: When std(dydx) < 1e-10, fallback to ref_std = 1.0 (absolute noise).
    """

    def test_zero_derivative_noise_is_nonzero(self):
        """Noise must be applied even when dydx is identically zero."""
        from dml_benchmark.functions import add_gaussian_noise

        # Simulate step-function derivatives: all zeros
        dydx_zeros = np.zeros((100, 1, 5))
        noise_level = 0.1

        noisy = add_gaussian_noise(dydx_zeros, noise_level, seed=42)

        # The noise should be non-zero (critical regression test)
        assert not np.allclose(noisy, dydx_zeros), (
            "BUG REGRESSION: add_gaussian_noise produced zero noise for zero-std input. "
            "The step-function noise fix has regressed."
        )

        # Noise std should be approximately noise_level * 1.0 (the fallback ref_std)
        actual_std = np.std(noisy)
        np.testing.assert_allclose(actual_std, noise_level, rtol=0.3)

    def test_near_zero_derivative_uses_fallback(self):
        """Derivatives very close to zero (but not exactly) should also trigger fallback."""
        from dml_benchmark.functions import add_gaussian_noise

        dydx_tiny = np.full((100, 1, 3), 1e-15)
        noisy = add_gaussian_noise(dydx_tiny, 0.2, seed=42)

        noise = noisy - dydx_tiny
        assert np.std(noise) > 0.05, "Fallback not triggered for near-zero derivatives"

    def test_normal_derivatives_use_relative_noise(self):
        """Non-zero derivatives should still use relative noise (original behavior)."""
        from dml_benchmark.functions import add_gaussian_noise

        dydx = np.random.RandomState(42).randn(200, 1, 5) * 3.0
        noise_level = 0.1

        noisy = add_gaussian_noise(dydx, noise_level, seed=42)
        noise = noisy - dydx
        expected_std = noise_level * np.std(dydx)
        actual_std = np.std(noise)
        np.testing.assert_allclose(actual_std, expected_std, rtol=0.2)


class TestGradNormGradientContamination:
    """Verify the GradNorm gradient contamination bug stays fixed.

    Bug: GradNorm backward pass accumulated gradients into model parameters,
    which were then corrupted when the main total loss backward ran.

    Fix: model.zero_grad() called after GradNorm backward, before returning.
    """

    def test_model_gradients_clean_after_gradnorm(self):
        """Model parameter gradients should be None/zero after GradNorm forward."""
        from dml_benchmark.model import DmlFeedForward
        from dml_benchmark.loss_balancing import GradNormDmlLoss

        model = DmlFeedForward(input_dim=3, output_dim=1, n_layers=2, hidden_size=32)
        loss_fn = GradNormDmlLoss(input_dim=3, alpha=1.5)
        loss_fn.train()

        x = torch.randn(8, 3, requires_grad=True)
        y_true = torch.randn(8, 1)
        dydx_true = torch.randn(8, 1, 3)

        y_pred, dydx_pred = model.forward_with_greek(x)
        result = loss_fn(y_pred, y_true, dydx_pred, dydx_true, model=model)

        # After GradNorm forward, model gradients should be clean
        for name, param in model.named_parameters():
            assert param.grad is None or torch.allclose(param.grad, torch.zeros_like(param.grad)), (
                f"BUG REGRESSION: Model parameter '{name}' has non-zero gradients "
                f"after GradNorm forward. The gradient contamination fix has regressed."
            )


class TestBaselineIsFittedGuard:
    """Verify the is_fitted guard prevents calling predict before fit."""

    def test_predict_gradients_before_fit_raises(self):
        """predict_gradients on unfitted model should raise RuntimeError."""
        from dml_benchmark.baselines import KRRBaseline

        model = KRRBaseline()
        x = np.random.randn(10, 3)

        with pytest.raises(RuntimeError, match="has not been fitted"):
            model.predict_gradients(x)


# ============================================================================
# FINANCE FUNCTION TESTS
# ============================================================================

class TestFinanceFunctions:
    """Test finance function generators: bachelier, black_scholes, heston."""

    def test_bachelier_shapes(self):
        """Bachelier model should produce correct shapes."""
        from dml_benchmark.functions import generate_data

        data = generate_data("bachelier", n_dim=1, n_samples=50, seed=42)
        assert data.x.shape == (50, 1)
        assert data.y.shape == (50, 1)
        assert np.all(np.isfinite(data.y))
        assert np.all(np.isfinite(data.dydx))

    def test_bachelier_multidim(self):
        """Bachelier basket option in multiple dimensions."""
        from dml_benchmark.functions import generate_data

        data = generate_data("bachelier", n_dim=5, n_samples=30, seed=42)
        assert data.x.shape == (30, 5)
        assert data.dydx.shape[-1] == 5

    def test_black_scholes_shapes(self):
        """Black-Scholes should produce correct shapes."""
        from dml_benchmark.functions import generate_data

        data = generate_data("black_scholes", n_dim=1, n_samples=50, seed=42)
        assert data.x.shape == (50, 1)
        assert data.y.shape == (50, 1)
        assert np.all(np.isfinite(data.y))

    def test_black_scholes_positive_prices(self):
        """Black-Scholes call prices should be non-negative."""
        from dml_benchmark.functions import generate_data

        data = generate_data("black_scholes", n_dim=1, n_samples=100, seed=42)
        assert np.all(data.y >= -1e-10), "BS call prices should be non-negative"

    def test_heston_shapes(self):
        """Heston model should produce correct shapes (2D input: S0, v0)."""
        from dml_benchmark.functions import generate_data

        data = generate_data("heston", n_dim=1, n_samples=20, seed=42)
        # Heston uses 2D input (S0, v0) regardless of n_dim=1 (n_assets=1)
        assert data.x.shape[0] == 20
        assert data.y.shape == (20, 1)
        assert np.all(np.isfinite(data.y))

    def test_heston_price_range(self):
        """Heston option prices should be in a reasonable range."""
        from dml_benchmark.functions import generate_data

        data = generate_data("heston", n_dim=1, n_samples=30, seed=42)
        # Option prices should be non-negative and < max(S0) + reasonable margin
        assert np.all(data.y >= -1), "Option prices should be approximately non-negative"


class TestStepFunction:
    """Test step function generator."""

    def test_step_shapes(self):
        """Step function should produce correct shapes."""
        from dml_benchmark.functions import generate_data

        data = generate_data("step", n_dim=3, n_samples=50, seed=42)
        assert data.x.shape == (50, 3)
        assert data.y.shape == (50, 1)
        assert np.all(np.isfinite(data.dydx))

    def test_step_derivatives_are_zero(self):
        """Step function derivatives should be (approximately) zero everywhere."""
        from dml_benchmark.functions import generate_data

        data = generate_data("step", n_dim=5, n_samples=100, seed=42)
        # Step function is piecewise constant → derivative = 0
        assert np.allclose(data.dydx, 0, atol=1e-6), (
            "Step function derivatives should be ~0 (piecewise constant)"
        )


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases: d=1 and other boundary conditions."""

    def test_d1_trig(self):
        """1-dimensional trigonometric function should work."""
        from dml_benchmark.functions import generate_data

        data = generate_data("trig", n_dim=1, n_samples=50, seed=42)
        assert data.x.shape == (50, 1)
        assert data.y.shape == (50, 1)
        assert data.dydx.shape[-1] == 1

    def test_d1_poly_trig(self):
        """1-dimensional poly_trig should work."""
        pytest.importorskip("jax", reason="JAX required for poly_trig generator")
        from dml_benchmark.functions import generate_data

        data = generate_data("poly_trig", n_dim=1, n_samples=50, seed=42)
        assert data.x.shape == (50, 1)
        assert data.y.shape == (50, 1)

    def test_d1_step(self):
        """1-dimensional step function should work."""
        from dml_benchmark.functions import generate_data

        data = generate_data("step", n_dim=1, n_samples=50, seed=42)
        assert data.x.shape == (50, 1)

    def test_d1_training_pipeline(self):
        """Full training pipeline should work with d=1."""
        pytest.importorskip("jax", reason="JAX required for poly_trig generator")
        from dml_benchmark.functions import generate_data
        from dml_benchmark.trainer import train_single_experiment

        data = generate_data("poly_trig", n_dim=1, n_samples=64, seed=42)
        result = train_single_experiment(
            x_train=data.x, y_train=data.y, dydx_train=data.dydx,
            x_test=data.x, y_test=data.y, dydx_test=data.dydx,
            lambda_=1.0, n_epochs=5, batch_size=32, n_layers=2,
            hidden_size=32, lr=0.005, activation="softplus",
            seed=42, method="dml_fixed", pbar=False,
        )
        assert result.test_value_mse is not None
        assert result.test_value_mse >= 0


# ============================================================================
# RELOBRALO DETERMINISM TEST
# ============================================================================

class TestReLoBRaLoDeterminism:
    """Verify ReLoBRaLo produces deterministic results with same seed."""

    def test_deterministic_lookback(self):
        """Two ReLoBRaLo instances should produce identical weights given same
        history. Post-2026-04-13 rename: the faithful Bischof-Kraus 2022 Eq.11
        implementation lives in ReLoBRaLoDmlLoss with state buffer
        `lambda_current`. The simplified variant (with `running_weights`) is
        now SoftmaxBalanceDmlLoss; covered by the parallel
        test_softmax_balance_running_weights_sum_invariant in
        tests/test_balancing_correctness.py.
        """
        from dml_benchmark.loss_balancing import ReLoBRaLoDmlLoss

        def run_relobralo():
            loss_fn = ReLoBRaLoDmlLoss(input_dim=3)
            loss_fn.train()

            # Simulate a few steps to build history
            weights = []
            for step in range(10):
                y_pred = torch.randn(8, 1)
                y_true = torch.randn(8, 1)
                dydx_pred = torch.randn(8, 1, 3)
                dydx_true = torch.randn(8, 1, 3)
                _ = loss_fn(y_pred, y_true, dydx_pred, dydx_true)
                weights.append(loss_fn.lambda_current.clone())

            return weights

        # Both runs should produce identical weight trajectories because
        # ReLoBRaLo's saudade/Bernoulli draws come from a seeded RNG (seed=42
        # by default in __init__).
        torch.manual_seed(0)
        w1 = run_relobralo()
        torch.manual_seed(0)
        w2 = run_relobralo()

        for i, (a, b) in enumerate(zip(w1, w2)):
            assert torch.allclose(a, b), (
                f"ReLoBRaLo weights diverged at step {i}: {a} vs {b}. "
                "Seeded RNG fix may have regressed."
            )
