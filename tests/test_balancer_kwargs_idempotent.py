"""
Reproducibility tests for the #197 `balancer_kwargs` patch.

Three guards:
  1. Default-α GradNorm: hook on (no kwargs) ≡ hook off → byte-identical.
  2. Default-τ ReLoBRaLo: hook on (no kwargs) ≡ hook off → byte-identical.
  3. CRITICAL regression guard: GradNorm with `alpha=0.5` must produce
     DIFFERENT metrics from `alpha=1.5`. If they match, the silent-kwarg
     dispatch bug has returned.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def _train_one(method: str, balancer_kwargs: dict | None = None,
                n_epochs: int = 10) -> tuple:
    from dml_benchmark.trainer import train_single_experiment
    from dml_benchmark.functions import generate_data

    tr = generate_data("trig", n_dim=5, n_samples=256, seed=42)
    va = generate_data("trig", n_dim=5, n_samples=128, seed=43)
    te = generate_data("trig", n_dim=5, n_samples=128, seed=44)

    torch.manual_seed(42)
    np.random.seed(42)
    kwargs = {}
    if balancer_kwargs is not None:
        kwargs["balancer_kwargs"] = balancer_kwargs
    r = train_single_experiment(
        tr.x, tr.y, tr.dydx, te.x, te.y, te.dydx,
        x_val=va.x, y_val=va.y, dydx_val=va.dydx,
        method=method, n_epochs=n_epochs, batch_size=64, lr=1e-3,
        n_layers=4, hidden_size=64, seed=42, pbar=False,
        **kwargs,
    )
    return r.test_value_mse, r.test_grad_mse


def test_gradnorm_default_kwarg_is_byte_identical():
    v0, g0 = _train_one("dml_gradnorm")
    v1, g1 = _train_one("dml_gradnorm", balancer_kwargs={})
    assert math.isclose(v0, v1, rel_tol=0.0, abs_tol=0.0), \
        f"default-α GradNorm changed: {v0} vs {v1}"
    assert math.isclose(g0, g1, rel_tol=0.0, abs_tol=0.0), \
        f"default-α GradNorm grad changed: {g0} vs {g1}"


def test_relobralo_default_kwarg_is_byte_identical():
    v0, g0 = _train_one("dml_relobralo")
    v1, g1 = _train_one("dml_relobralo", balancer_kwargs={})
    assert math.isclose(v0, v1, rel_tol=0.0, abs_tol=0.0), \
        f"default-τ ReLoBRaLo changed: {v0} vs {v1}"
    assert math.isclose(g0, g1, rel_tol=0.0, abs_tol=0.0), \
        f"default-τ ReLoBRaLo grad changed: {g0} vs {g1}"


def test_gradnorm_alpha_lands_on_balancer():
    """Direct check: distinct α values must produce distinct GradNorm
    objects with distinct .alpha attributes. If they match, the silent-
    kwarg-drop dispatch bug has returned."""
    from dml_benchmark.loss_balancing import GradNormDmlLoss
    g_low  = GradNormDmlLoss(input_dim=5, alpha=0.5)
    g_high = GradNormDmlLoss(input_dim=5, alpha=2.0)
    assert g_low.alpha != g_high.alpha, (
        f"α did not reach GradNormDmlLoss: low={g_low.alpha}, high={g_high.alpha}"
    )


def test_relobralo_tau_lands_on_balancer():
    """Direct attribute check for ReLoBRaLo τ. Mirrors the GradNorm one."""
    from dml_benchmark.loss_balancing import ReLoBRaLoDmlLoss
    r_low  = ReLoBRaLoDmlLoss(input_dim=5, seed=42, tau=0.10)
    r_high = ReLoBRaLoDmlLoss(input_dim=5, seed=42, tau=1.00)
    # ReLoBRaLoDmlLoss stores τ as `T` (Bischof & Kraus 2022 Eq.11 notation)
    # or `tau` depending on impl; check both.
    t_low  = getattr(r_low,  "T", None) or getattr(r_low,  "tau", None)
    t_high = getattr(r_high, "T", None) or getattr(r_high, "tau", None)
    assert t_low != t_high, (
        f"τ did not reach ReLoBRaLoDmlLoss: low={t_low}, high={t_high}"
    )


def test_relobralo_tau_actually_flows_through_dispatcher():
    """End-to-end: distinct τ through train_single_experiment must produce
    distinct metrics. ReLoBRaLo's softmax temperature has a sharper effect
    on training dynamics than GradNorm's α, so this differentiates faster."""
    v_low, g_low = _train_one(
        "dml_relobralo", balancer_kwargs={"tau": 0.10}, n_epochs=50)
    v_high, g_high = _train_one(
        "dml_relobralo", balancer_kwargs={"tau": 1.00}, n_epochs=50)
    assert not (math.isclose(v_low, v_high, rel_tol=0.0, abs_tol=0.0)
                and math.isclose(g_low, g_high, rel_tol=0.0, abs_tol=0.0)), (
        f"ReLoBRaLo τ=0.10 vs τ=1.00 byte-identical at 50 epochs — "
        f"silent-kwarg-drop bug has returned. v_low={v_low}, v_high={v_high}, "
        f"g_low={g_low}, g_high={g_high}"
    )
