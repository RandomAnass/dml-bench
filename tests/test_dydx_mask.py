"""dydx_mask propagation test — every DML loss class must

  (a) accept the dydx_mask kwarg without raising,
  (b) compute deriv_loss as the masked mean of the squared error,
  (c) leave value_loss unaffected by the mask, and
  (d) ignore the mask cleanly when it is None (back-compat).

VanillaLoss does not use the mask but must still accept the kwarg silently
(it absorbs into **kwargs) since the trainer always passes one through.
"""
import numpy as np
import pytest
import torch

from dml_benchmark.model import DmlLoss, VanillaLoss
from dml_benchmark.loss_balancing import (
    GradNormDmlLoss,
    DimNormGradNormDmlLoss,
    SoftmaxBalanceDmlLoss,
    ReLoBRaLoDmlLoss,
)


def _fixture(input_dim=3):
    """Two samples, 1 output, `input_dim` inputs. Exact numbers so the
    expected masked deriv_loss is computable by hand."""
    y_pred = torch.tensor([[1.0], [2.0]])
    y_true = torch.tensor([[1.0], [2.0]])
    dydx_pred = torch.tensor([[[0.5, 1.0, 1.5]], [[0.1, 0.2, 0.3]]])
    dydx_true = torch.tensor([[[0.5, 1.1, 1.0]], [[0.1, 0.2, 0.0]]])
    mask = torch.tensor([[[1.0, 1.0, 0.0]], [[1.0, 1.0, 0.0]]])
    return y_pred, y_true, dydx_pred, dydx_true, mask


# All DML losses we expect to honor the mask. Constructors take input_dim only.
DML_LOSSES = [
    ("DmlLoss",                lambda d: DmlLoss(lambda_=1.0, input_dim=d)),
    ("GradNormDmlLoss",        lambda d: GradNormDmlLoss(input_dim=d)),
    ("DimNormGradNormDmlLoss", lambda d: DimNormGradNormDmlLoss(input_dim=d)),
    ("SoftmaxBalanceDmlLoss",  lambda d: SoftmaxBalanceDmlLoss(input_dim=d)),
    ("ReLoBRaLoDmlLoss",       lambda d: ReLoBRaLoDmlLoss(input_dim=d)),
]


@pytest.mark.parametrize("name,ctor", DML_LOSSES)
def test_dydx_mask_honored(name, ctor):
    """Masked deriv_loss equals masked mean of squared error."""
    d = 3
    loss = ctor(d)
    loss.eval()  # avoid any training-time state mutation
    y_pred, y_true, dydx_pred, dydx_true, mask = _fixture(d)

    components = loss(
        y_pred, y_true, dydx_pred, dydx_true,
        model=None, dydx_mask=mask,
    )

    err = (dydx_pred - dydx_true) ** 2
    expected_deriv_loss = (err * mask).sum() / mask.sum()
    assert np.isclose(components.deriv_loss.item(), expected_deriv_loss.item()), (
        f"{name}: masked deriv_loss = {components.deriv_loss.item():.6e}, "
        f"expected {expected_deriv_loss.item():.6e}"
    )

    expected_value_loss = ((y_pred - y_true) ** 2).mean().item()
    assert np.isclose(components.value_loss.item(), expected_value_loss), (
        f"{name}: value_loss should be unaffected by mask"
    )


@pytest.mark.parametrize("name,ctor", DML_LOSSES)
def test_dydx_mask_none_is_unmasked_mean(name, ctor):
    """When dydx_mask=None, deriv_loss equals the plain mean of err — the
    pre-mask behavior. Guards against regressions on the no-mask path."""
    d = 3
    loss = ctor(d)
    loss.eval()
    y_pred, y_true, dydx_pred, dydx_true, _ = _fixture(d)

    components = loss(
        y_pred, y_true, dydx_pred, dydx_true,
        model=None, dydx_mask=None,
    )

    expected = ((dydx_pred - dydx_true) ** 2).mean().item()
    assert np.isclose(components.deriv_loss.item(), expected), (
        f"{name}: unmasked deriv_loss = {components.deriv_loss.item():.6e}, "
        f"expected {expected:.6e}"
    )


def test_vanilla_accepts_mask_kwarg():
    """VanillaLoss does not use the mask, but must accept the kwarg without
    raising — the trainer always threads dydx_mask through."""
    loss = VanillaLoss()
    y_pred, y_true, _, _, mask = _fixture(3)
    components = loss(y_pred, y_true, dydx_mask=mask)
    expected_value_loss = ((y_pred - y_true) ** 2).mean().item()
    assert np.isclose(components.value_loss.item(), expected_value_loss)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
