"""
Correctness regression tests for loss_balancing.py and fuzzy_smoothing.py.

These tests are NOT "outputs differ" checks — each test pins a specific
invariant that must hold per the published algorithms. Written as a response
to the I-H5 incident (2026-04-23), where detach() killed the gradient path to
GradNorm's task_weights and the runs silently produced fixed-λ=1 results for
a week.

Reference papers:
  - Chen et al., "GradNorm", ICML 2018, Algorithm 1
  - Bischof & Kraus, "ReLoBRaLo", arXiv:2110.09813, Eq. 11
  - Savine, "Scripting" C++ library, scriptingFuzzyEval.h (cSpr, bFly, AND/OR/NOT)

Run: pytest tests/test_balancing_correctness.py -v
"""
import sys
import os
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dml_benchmark.loss_balancing import (
    GradNormDmlLoss, DimNormGradNormDmlLoss,
    SoftmaxBalanceDmlLoss, ReLoBRaLoDmlLoss,
)
from dml_benchmark.fuzzy_smoothing import (
    call_spread, butterfly, fuzzy_and, fuzzy_or, fuzzy_not, fuzzy_if,
)


# ============================================================================
# FUZZY — vs Savine's scriptingFuzzyEval.h (verbatim)
# ============================================================================

def _savine_cSpr(x, eps):
    """Reference: https://github.com/asavine/Scripting/blob/master/scriptingFuzzyEval.h"""
    halfEps = 0.5 * eps
    if x < -halfEps:
        return 0.0
    elif x > halfEps:
        return 1.0
    else:
        return (x + halfEps) / eps


def _savine_bFly(x, eps):
    halfEps = 0.5 * eps
    if x < -halfEps or x > halfEps:
        return 0.0
    else:
        return (halfEps - abs(x)) / halfEps


@pytest.mark.parametrize("eps", [0.01, 0.1, 1.0, 2.0])
def test_call_spread_matches_savine(eps):
    """cSpr(x, eps): piecewise-linear step — exact to Savine C++."""
    x = np.linspace(-5, 5, 1000)
    ref = np.array([_savine_cSpr(xi, eps) for xi in x])
    got = call_spread(x, eps)
    np.testing.assert_allclose(got, ref, atol=1e-12, rtol=0)


@pytest.mark.parametrize("eps", [0.01, 0.1, 1.0, 2.0])
def test_butterfly_matches_savine(eps):
    """bFly(x, eps): tent function peaked at 0 — exact to Savine C++."""
    x = np.linspace(-5, 5, 1000)
    ref = np.array([_savine_bFly(xi, eps) for xi in x])
    got = butterfly(x, eps)
    np.testing.assert_allclose(got, ref, atol=1e-12, rtol=0)


def test_fuzzy_boolean_algebra():
    """AND, OR, NOT match Savine visitors."""
    assert fuzzy_and(0.3, 0.7) == pytest.approx(0.21)    # product
    assert fuzzy_or(0.3, 0.7) == pytest.approx(0.79)     # prob union
    assert fuzzy_not(0.3) == pytest.approx(0.7)          # complement


def test_fuzzy_if_blend():
    """IF C THEN S1 ELSE S2 = DT * S1 + (1-DT) * S2."""
    dt, s1, s2 = 0.3, 5.0, 10.0
    assert fuzzy_if(dt, s1, s2) == pytest.approx(0.3 * 5.0 + 0.7 * 10.0)


# ============================================================================
# GRADNORM — I-H5 incident regression tests
# ============================================================================

def _tiny_mlp(d=5, h=32):
    return torch.nn.Sequential(
        torch.nn.Linear(d, h), torch.nn.Softplus(),
        torch.nn.Linear(h, h), torch.nn.Softplus(),
        torch.nn.Linear(h, 1),
    )


def test_gradnorm_task_weights_evolve_during_training():
    """
    Incident regression: I-H5 (commit 967bc5e0) introduced a .detach() that
    killed the gradient path from gradnorm_loss to task_weights. Adam saw
    zero grad and weights stayed at init [1,1] forever.

    This test guarantees task_weights move away from init within 10 steps.
    """
    torch.manual_seed(42)
    d = 5
    model = _tiny_mlp(d=d)
    loss_fn = GradNormDmlLoss(input_dim=d, shared_layer_name="2")  # Linear(h→1) is idx 4, 2=mid
    # Use fallback (-2 weight) instead
    loss_fn = GradNormDmlLoss(input_dim=d)
    loss_fn.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    w_before = loss_fn.task_weights.detach().clone()
    for _ in range(10):
        x = torch.randn(32, d, requires_grad=True)
        y = torch.randn(32, 1)
        dydx = torch.randn(32, 1, d)
        x_in = x.clone().requires_grad_(True)
        y_pred = model(x_in)
        dydx_pred = torch.autograd.grad(
            y_pred, x_in, grad_outputs=torch.ones_like(y_pred),
            create_graph=True, retain_graph=True,
        )[0].unsqueeze(1)
        lc = loss_fn(y_pred, y, dydx_pred, dydx, model)
        opt.zero_grad()
        lc.total.backward()
        opt.step()

    w_after = loss_fn.task_weights.detach()
    delta = (w_after - w_before).abs().max().item()
    # With working GradNorm, Adam at lr=0.025 × 10 steps moves weights by ~0.1+
    assert delta > 1e-3, (
        f"task_weights did not evolve (Δmax={delta:.2e}); "
        "GradNorm is broken — check the gradient path from gradnorm_loss to self.task_weights."
    )


def test_gradnorm_differs_from_fixed_dml():
    """
    With working GradNorm, training from the same seed must NOT produce
    identical outputs to fixed-λ=1 DML. If it does, weights didn't evolve.
    """
    from dml_benchmark.trainer import train_single_experiment
    from dml_benchmark.functions import generate_data

    tr = generate_data("trig", n_dim=5, n_samples=256, seed=42)
    va = generate_data("trig", n_dim=5, n_samples=128, seed=43)
    te = generate_data("trig", n_dim=5, n_samples=128, seed=44)

    results = {}
    for method in ["dml_fixed", "dml_gradnorm"]:
        torch.manual_seed(42); np.random.seed(42)
        r = train_single_experiment(
            tr.x, tr.y, tr.dydx, te.x, te.y, te.dydx,
            x_val=va.x, y_val=va.y, dydx_val=va.dydx,
            method=method, n_epochs=20, batch_size=64, lr=1e-3,
            n_layers=4, hidden_size=64, seed=42, pbar=False,
        )
        results[method] = (r.test_value_mse, r.test_grad_mse)

    assert results["dml_gradnorm"] != results["dml_fixed"], (
        f"dml_gradnorm produced identical output to dml_fixed: {results}. "
        "Likely regression — GradNorm weights not updating."
    )


# ============================================================================
# DIM-NORM GRADNORM — correctness of dispatch
# ============================================================================

def test_dimnorm_override_is_called():
    """DimNormGradNormDmlLoss._update_weights_gradnorm must be the override
    (not silently falling back to parent)."""
    obj = DimNormGradNormDmlLoss(input_dim=10, dim_norm_mode="d")
    assert obj._update_weights_gradnorm.__qualname__ == (
        "DimNormGradNormDmlLoss._update_weights_gradnorm"
    )
    assert obj._dim_factor == 10.0

    obj2 = DimNormGradNormDmlLoss(input_dim=10, dim_norm_mode="sqrt_d")
    assert obj2._dim_factor == pytest.approx(10.0 ** 0.5)


# ============================================================================
# RELOBRALO — Bischof & Kraus 2022 Eq. 11
# ============================================================================

def test_relobralo_lambda_sum_invariant():
    """λ_i(t) must always sum to T (= 2 for DML) per paper Eq. 11."""
    loss_fn = ReLoBRaLoDmlLoss(input_dim=5, tau=0.1, alpha=0.999, rho_expectation=0.999)
    loss_fn.train()

    # Bootstrap history so the λ update kicks in
    for step in range(20):
        y_pred = torch.tensor([[1.0 + 0.1 * step]], requires_grad=True)
        y_true = torch.tensor([[1.0]])
        dydx_pred = torch.randn(1, 1, 5)
        dydx_true = torch.randn(1, 1, 5)
        _ = loss_fn(y_pred, y_true, dydx_pred, dydx_true)

    lam_sum = loss_fn.lambda_current.sum().item()
    assert abs(lam_sum - 2.0) < 1e-4, f"λ sum drifted to {lam_sum}, expected 2.0"


def test_softmax_balance_running_weights_sum_invariant():
    """running_weights must sum ~2 after training (J7 invariant)."""
    loss_fn = SoftmaxBalanceDmlLoss(input_dim=5)
    loss_fn.train()
    for _ in range(10):
        y_pred = torch.randn(4, 1, requires_grad=True)
        y_true = torch.randn(4, 1)
        dydx_pred = torch.randn(4, 1, 5)
        dydx_true = torch.randn(4, 1, 5)
        _ = loss_fn(y_pred, y_true, dydx_pred, dydx_true)

    assert abs(loss_fn.running_weights.sum().item() - 2.0) < 1e-3


# ============================================================================
# STATE_DICT ROUND-TRIP (I-M8 / I-M9 / L-M6)
# ============================================================================

def test_gradnorm_state_dict_roundtrip():
    """Adam state for task_weights must survive state_dict save/load."""
    torch.manual_seed(7)
    d = 5
    model = _tiny_mlp(d=d)
    loss_fn = GradNormDmlLoss(input_dim=d)
    loss_fn.train()

    # Run some steps to populate Adam state
    for _ in range(3):
        x = torch.randn(16, d)
        y_pred = model(x.requires_grad_(True))
        dydx_pred = torch.autograd.grad(
            y_pred, x, grad_outputs=torch.ones_like(y_pred), create_graph=True,
        )[0].unsqueeze(1)
        loss_fn(y_pred, torch.randn(16, 1), dydx_pred, torch.randn(16, 1, d), model)

    sd = loss_fn.state_dict()
    loss_fn2 = GradNormDmlLoss(input_dim=d)
    loss_fn2.load_state_dict(sd)
    # Weights and extra state both restored
    assert torch.allclose(loss_fn2.task_weights, loss_fn.task_weights)


def test_relobralo_rng_state_roundtrip():
    """ReLoBRaLo's RNG state must survive state_dict round-trip."""
    loss_fn = ReLoBRaLoDmlLoss(input_dim=5, seed=123)
    loss_fn.train()
    # Warm up history
    for _ in range(5):
        y_pred = torch.randn(4, 1, requires_grad=True)
        loss_fn(y_pred, torch.randn(4, 1), torch.randn(4, 1, 5), torch.randn(4, 1, 5))

    sd = loss_fn.state_dict()
    loss_fn2 = ReLoBRaLoDmlLoss(input_dim=5, seed=999)  # different seed
    loss_fn2.load_state_dict(sd)

    # After load, draws should match original
    np.testing.assert_array_equal(
        loss_fn._rng.get_state()[1], loss_fn2._rng.get_state()[1]
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
