"""
Reproducibility test for the task-weights logging hook (F18 / F19).

Asserts that adding `task_weights_log_path=...` to train_single_experiment
is a pure observation: same seed must yield byte-identical
(test_value_mse, test_grad_mse) with the flag on vs off.

Run:
    pytest tests/test_task_weights_hook_idempotent.py -q
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch


def _train(extra_kwargs):
    from dml_benchmark.trainer import train_single_experiment
    from dml_benchmark.functions import generate_data

    tr = generate_data("trig", n_dim=5, n_samples=256, seed=42)
    va = generate_data("trig", n_dim=5, n_samples=128, seed=43)
    te = generate_data("trig", n_dim=5, n_samples=128, seed=44)

    torch.manual_seed(42)
    np.random.seed(42)
    r = train_single_experiment(
        tr.x, tr.y, tr.dydx, te.x, te.y, te.dydx,
        x_val=va.x, y_val=va.y, dydx_val=va.dydx,
        method="dml_gradnorm", n_epochs=15, batch_size=64, lr=1e-3,
        n_layers=4, hidden_size=64, seed=42, pbar=False,
        **extra_kwargs,
    )
    return r.test_value_mse, r.test_grad_mse


def test_hook_is_pure_observation():
    v0, g0 = _train({})

    with tempfile.TemporaryDirectory() as td:
        log_path = str(Path(td) / "tw.jsonl")
        v1, g1 = _train({"task_weights_log_path": log_path})

        assert Path(log_path).exists(), "JSONL was not written"
        with open(log_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) > 0, "JSONL is empty"
        for entry in lines[:3]:
            assert "epoch" in entry
            assert "task_weights" in entry
            assert isinstance(entry["task_weights"], list)
            assert len(entry["task_weights"]) == 2  # [w_value, w_deriv]

    # Same seed, same code path → byte-identical metrics. Hook must be a
    # pure observer (no extra RNG draws, no graph mutation).
    assert math.isclose(v0, v1, rel_tol=0.0, abs_tol=0.0), \
        f"value MSE differed with hook on/off: {v0} vs {v1}"
    assert math.isclose(g0, g1, rel_tol=0.0, abs_tol=0.0), \
        f"grad MSE differed with hook on/off: {g0} vs {g1}"
