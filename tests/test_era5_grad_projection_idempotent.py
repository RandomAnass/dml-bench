"""
Reproducibility test for the gradient_projection_fn hook on DmlLoss.

Asserts that adding `gradient_projection_fn=None` (the default) is a pure
no-op: same RNG, same random batch, same lambda_, same input_dim → byte-
identical LossComponents (.total, .value_loss, .deriv_loss) compared to
constructing DmlLoss WITHOUT the kwarg at all.

This is the §19 reproducibility guard: existing Burgers / Darcy / SPY /
synthetic / rMD17 cells must be unaffected by the new hook.

Run:
    pytest tests/test_era5_grad_projection_idempotent.py -q
"""
from __future__ import annotations

import math

import numpy as np
import torch

from dml_benchmark.model import DmlLoss


def _make_batch(seed: int = 0, batch: int = 32, d: int = 5):
    g = torch.Generator().manual_seed(seed)
    y_pred = torch.randn(batch, 1, generator=g)
    y_true = torch.randn(batch, 1, generator=g)
    dydx_pred = torch.randn(batch, 1, d, generator=g)
    dydx_true = torch.randn(batch, 1, d, generator=g)
    return y_pred, y_true, dydx_pred, dydx_true


def test_gradient_projection_fn_default_is_byte_identical():
    """Default gradient_projection_fn=None ⇒ identical to legacy DmlLoss()."""
    for scheme in ("hs", "half"):
        loss_legacy = DmlLoss(lambda_=1.0, input_dim=5, weight_scheme=scheme)
        loss_new = DmlLoss(lambda_=1.0, input_dim=5, weight_scheme=scheme,
                           gradient_projection_fn=None)
        for seed in (0, 1, 7, 42):
            y_p, y_t, g_p, g_t = _make_batch(seed=seed)
            r0 = loss_legacy(y_p, y_t, g_p, g_t, model=None)
            r1 = loss_new(y_p, y_t, g_p, g_t, model=None)
            assert r0.total.item() == r1.total.item(), \
                f"total differs for scheme={scheme} seed={seed}: {r0.total} vs {r1.total}"
            assert r0.value_loss.item() == r1.value_loss.item()
            assert r0.deriv_loss.item() == r1.deriv_loss.item()


def test_gradient_projection_fn_default_works_with_mask():
    """Default None hook + dydx_mask is byte-identical to legacy mask path."""
    loss_legacy = DmlLoss(lambda_=1.0, input_dim=5)
    loss_new = DmlLoss(lambda_=1.0, input_dim=5, gradient_projection_fn=None)
    y_p, y_t, g_p, g_t = _make_batch(seed=11)
    mask = torch.ones_like(g_p)
    mask[:, :, 3:] = 0.0
    r0 = loss_legacy(y_p, y_t, g_p, g_t, model=None, dydx_mask=mask)
    r1 = loss_new(y_p, y_t, g_p, g_t, model=None, dydx_mask=mask)
    assert r0.total.item() == r1.total.item()
    assert r0.deriv_loss.item() == r1.deriv_loss.item()


def test_projection_hook_is_invoked_when_provided():
    """Sanity: when a hook is provided, the loss path uses it (not no-op)."""
    calls = {"n": 0}

    def proj(grad_pred, grad_target, x_query, dydx_mask):
        calls["n"] += 1
        return grad_pred[:, :, :2], grad_target[:, :, :2], (
            dydx_mask[:, :, :2] if dydx_mask is not None else torch.ones_like(grad_pred[:, :, :2])
        )

    loss = DmlLoss(lambda_=1.0, input_dim=2, gradient_projection_fn=proj)
    y_p, y_t, g_p, g_t = _make_batch(seed=3)
    _ = loss(y_p, y_t, g_p, g_t, model=None)
    assert calls["n"] == 1, "projection hook was not invoked"
