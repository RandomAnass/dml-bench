"""
DML Model and Loss implementations for benchmark.

Provides PyTorch implementation compatible with Colab and includes
loss decomposition for tracking value vs derivative loss components.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Callable
from dataclasses import dataclass


# ============================================================================
# DATASET
# ============================================================================

class DmlDataset(Dataset):
    """Dataset for DML training with values and derivatives."""
    
    def __init__(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        dydx: np.ndarray, 
        dydx_mask: Optional[np.ndarray] = None
    ):
        """
        Args:
            x: Inputs (n_samples, n_dim)
            y: Values (n_samples, 1)
            dydx: Gradients (n_samples, 1, n_dim) or (n_samples, n_dim)
            dydx_mask: Optional mask for gradients (n_samples, 1, n_dim). 
                       Zeros out loss penalty for specific partial derivatives.
        """
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
        # Ensure dydx has shape (n_samples, 1, n_dim)
        if len(dydx.shape) == 2:
            dydx = dydx.reshape(dydx.shape[0], 1, dydx.shape[1])
        self.dydx = torch.tensor(dydx, dtype=torch.float32)
        
        if dydx_mask is not None:
            if len(dydx_mask.shape) == 2:
                dydx_mask = dydx_mask.reshape(dydx_mask.shape[0], 1, dydx_mask.shape[1])
            self.dydx_mask = torch.tensor(dydx_mask, dtype=torch.float32)
        else:
            self.dydx_mask = torch.ones_like(self.dydx)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return {
            'x': self.x[idx],
            'y': self.y[idx],
            'dydx': self.dydx[idx],
            'dydx_mask': self.dydx_mask[idx]
        }


# ============================================================================
# DATA NORMALIZER
# ============================================================================

class DataNormalizer:
    """Normalize inputs and outputs for stable training."""
    
    def __init__(self):
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        self.lambda_j = None  # For derivative scaling
        
    def initialize_with_data(
        self,
        x_raw: np.ndarray,
        y_raw: np.ndarray,
        dydx_raw: np.ndarray
    ):
        """Compute normalization statistics from training data."""
        self.x_mean = np.mean(x_raw, axis=0)
        self.x_std = np.std(x_raw, axis=0)
        self.x_std = np.where(self.x_std < 1e-8, 1.0, self.x_std)
        
        self.y_mean = np.mean(y_raw)
        self.y_std = np.std(y_raw)
        if self.y_std < 1e-8:
            self.y_std = 1.0
        
        # Lambda_j for derivative scaling: ratio of x_std to y_std
        self.lambda_j = self.x_std / self.y_std
        
        self.input_dimension = x_raw.shape[1]
        self.output_dimension = y_raw.shape[1] if len(y_raw.shape) > 1 else 1
    
    def normalize_x(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean) / self.x_std
    
    def normalize_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_mean) / self.y_std
    
    def normalize_dydx(self, dydx: np.ndarray) -> np.ndarray:
        """Scale derivatives by lambda_j."""
        return dydx * self.lambda_j
    
    def normalize_all(
        self,
        x: np.ndarray,
        y: np.ndarray,
        dydx: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.normalize_x(x), self.normalize_y(y), self.normalize_dydx(dydx)
    
    def unscale_y(self, y_normalized: np.ndarray) -> np.ndarray:
        """Unscale normalized y back to original domain."""
        return y_normalized * self.y_std + self.y_mean
    
    def unscale_x(self, x_normalized: np.ndarray) -> np.ndarray:
        """Unscale normalized x back to original domain."""
        return x_normalized * self.x_std + self.x_mean
    
    def unscale_dydx(self, dydx_normalized: np.ndarray) -> np.ndarray:
        """
        Unscale normalized gradients back to original domain.
        
        The normalization is: dydx_norm = dydx_raw * lambda_j
        where lambda_j = x_std / y_std
        
        So: dydx_raw = dydx_norm / lambda_j
        """
        return dydx_normalized / self.lambda_j


# ============================================================================
# DML NETWORK
# ============================================================================

class DmlFeedForward(nn.Module):
    """
    Feedforward network for DML with analytical gradient computation.
    
    Uses softplus activation for smooth derivatives.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_layers: int,
        hidden_size: int,
        activation: str = "softplus"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_size))
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
        
        layers.append(nn.Linear(hidden_size, output_dim))
        self.layers = nn.ModuleList(layers)
        
        # Activation
        # C¹ audit 2026-04-13: softplus/sigmoid/tanh are C^∞ (safe for DML).
        # relu is C^0 only (not differentiable at 0). Code still allows relu
        # for backward-compat / vanilla-only experiments, but a warning is
        # printed in the trainer dispatch if relu is paired with a DML method.
        # See EVIDENCE/c1_activation_audit.md.
        if activation == "softplus":
            self.activation = nn.Softplus()
        elif activation == "relu":
            import warnings
            warnings.warn(
                "Activation 'relu' is C^0 only (not differentiable at 0). "
                "DML methods that rely on ∂E/∂x through the model may produce "
                "undefined gradients at zero-crossings. Use 'softplus' (default) "
                "for DML. See EVIDENCE/c1_activation_audit.md.",
                UserWarning, stacklevel=2,
            )
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.activation_name = activation
        
        # Initialize weights properly (He initialization) to prevent vanishing gradients
        # dy/dx depends on product of weights, so we need weights ~ O(1)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using activation-appropriate strategy.
        
        - Kaiming Normal (He init) for ReLU/Softplus: Var(w) = 2/fan_in
        - Xavier Normal (Glorot init) for Sigmoid/Tanh: Var(w) = 2/(fan_in+fan_out)
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.activation_name in ("relu", "softplus"):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif self.activation_name in ("sigmoid", "tanh"):
                    nn.init.xavier_normal_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning output only."""
        return self.forward_with_outputs(x)[-1]
    
    def forward_with_outputs(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass returning all intermediate outputs."""
        outputs = [x]
        h = x
        
        for i, layer in enumerate(self.layers[:-1]):
            h = layer(h)
            h = self.activation(h)
            outputs.append(h)
        
        # Output layer (no activation)
        h = self.layers[-1](h)
        outputs.append(h)
        
        return outputs
    
    def forward_with_greek(
        self,
        x: torch.Tensor,
        create_graph: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning output and gradients (Greeks).

        Computes gradients via autograd.

        Args:
            x: input tensor (batch, input_dim).
            create_graph: if True, build the second-order graph (needed for DML
                training where dydx_pred is used in the loss). If False, the
                returned dydx is detached from the autograd graph (faster +
                less memory; safe for evaluation only). Default = self.training
                (= True in train mode, False in eval mode). L2 fix 2026-04-13:
                previously hard-coded create_graph=True even in eval, wasting
                O(d * hidden) memory per batch.

        Returns:
            (y, dydx) where dydx has shape (batch, output_dim, input_dim)
        """
        if create_graph is None:
            create_graph = self.training

        # CRITICAL: Clone and detach to ensure fresh computation graph
        # This fixes issues when x comes from a dataloader or no_grad context
        x = x.detach().clone().requires_grad_(True)
        y = self.forward(x)

        # Compute gradient for each output dimension
        # Use list + stack instead of in-place assignment to preserve grad graph
        grad_list = []

        for k in range(self.output_dim):
            grad_outputs = torch.zeros_like(y)
            grad_outputs[:, k] = 1.0

            grads = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=grad_outputs,
                create_graph=create_graph,
                retain_graph=True
            )[0]

            grad_list.append(grads)

        # Stack preserves computation graph (unlike in-place assignment)
        dydx = torch.stack(grad_list, dim=1)  # (batch, output_dim, input_dim)

        return y, dydx


# ============================================================================
# DML LOSS WITH DECOMPOSITION
# ============================================================================

@dataclass
class LossComponents:
    """Container for loss components."""
    total: torch.Tensor
    value_loss: torch.Tensor
    deriv_loss: torch.Tensor
    reg_loss: torch.Tensor


class DmlLoss(nn.Module):
    """
    DML loss with tracking of value vs derivative components.

    Two weight schemes (V1/P7, 2026-04-13):
      - 'hs' (default, Huge & Savine 2018): w = λd/(1+λd), 1-w = 1/(1+λd).
        Faithful to the official H&S notebook (verified 2026-04-13, see
        EVIDENCE/hs2018_formula_check.md). At λ=1, d=1: 0.5/0.5. At higher d,
        the derivative term dominates (1/d weight on value).
      - 'half' (added 2026-04-13): w = 0.5, 1-w = 0.5 regardless of λ or d.
        Used for cross-architecture parity in the molecular pillar (GATv2/PaiNN
        natively use 0.5/0.5) and for V1/P7 A/B comparison vs the H&S formula.

    L = (1-w) * MSE(y) + w * MSE(dydx) + reg * ||weights||
    """

    def __init__(
        self,
        lambda_: float,
        input_dim: int,
        lambda_j: np.ndarray = None,
        regularization_scale: float = 0.0,
        weight_scheme: str = "hs",
        gradient_projection_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.lambda_ = lambda_
        self.input_dim = input_dim
        self.regularization_scale = regularization_scale
        self.weight_scheme = weight_scheme
        self.gradient_projection_fn = gradient_projection_fn

        # Derivative scaling factors
        if lambda_j is not None:
            self.register_buffer('lambda_j', torch.tensor(lambda_j, dtype=torch.float32))
        else:
            self.register_buffer('lambda_j', torch.ones(input_dim))

        # Compute loss weights per scheme
        if weight_scheme == "hs":
            self.ml_loss_scale = 1.0 / (1.0 + self.lambda_ * self.input_dim)
            self.dml_loss_scale = 1.0 - self.ml_loss_scale
        elif weight_scheme == "half":
            self.ml_loss_scale = 0.5
            self.dml_loss_scale = 0.5
        else:
            raise ValueError(
                f"Unknown weight_scheme: {weight_scheme!r}. Use 'hs' or 'half'."
            )
    
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        dydx_pred: torch.Tensor,
        dydx_true: torch.Tensor,
        model: nn.Module = None,
        dydx_mask: Optional[torch.Tensor] = None,
        x_query: Optional[torch.Tensor] = None,
    ) -> LossComponents:
        """
        Compute DML loss with components.

        Args:
            y_pred: Predicted values (batch, 1)
            y_true: True values (batch, 1)
            dydx_pred: Predicted gradients (batch, 1, input_dim)
            dydx_true: True gradients (batch, 1, input_dim)
            model: Optional model for regularization
            dydx_mask: Optional boolean/float mask (batch, 1, input_dim)
            x_query: Optional query inputs (batch, input_dim) — only used by
                gradient_projection_fn (default None ⇒ ignored).

        Returns:
            LossComponents with total and component losses
        """
        batch_size = y_pred.shape[0]

        # Value loss
        value_loss = torch.mean((y_pred - y_true) ** 2)

        # Optional directional / projected gradient hook (ERA5 sub-pillar).
        # Default None ⇒ byte-identical to legacy path.
        if self.gradient_projection_fn is not None:
            dydx_pred, dydx_true, dydx_mask = self.gradient_projection_fn(
                dydx_pred, dydx_true, x_query, dydx_mask,
            )

        # Derivative loss - NO SCALING HERE!
        # Derivatives are already scaled by DataNormalizer.normalize_dydx()
        # Scaling here would cause double-scaling (Issue #2 in audit)
        if dydx_mask is not None:
            err = (dydx_pred - dydx_true) ** 2
            deriv_loss = (err * dydx_mask).sum() / dydx_mask.sum().clamp(min=1.0)
        else:
            deriv_loss = torch.mean((dydx_pred - dydx_true) ** 2)
        
        # Regularization
        reg_loss = torch.tensor(0.0, device=y_pred.device)
        if model is not None and self.regularization_scale > 0:
            for param in model.parameters():
                reg_loss = reg_loss + torch.norm(param)
            reg_loss = self.regularization_scale * reg_loss
        
        # Combine
        total = (self.ml_loss_scale * value_loss + 
                 self.dml_loss_scale * deriv_loss + 
                 reg_loss)
        
        return LossComponents(
            total=total,
            value_loss=value_loss,
            deriv_loss=deriv_loss,
            reg_loss=reg_loss
        )


# ============================================================================
# VANILLA LOSS (lambda = 0)
# ============================================================================

class VanillaLoss(nn.Module):
    """Standard MSE loss without derivatives."""
    
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        **kwargs
    ) -> LossComponents:
        value_loss = torch.mean((y_pred - y_true) ** 2)
        return LossComponents(
            total=value_loss,
            value_loss=value_loss,
            deriv_loss=torch.tensor(0.0, device=y_pred.device),
            reg_loss=torch.tensor(0.0, device=y_pred.device)
        )


# ============================================================================
# DEVICE UTILITIES
# ============================================================================

def get_device() -> torch.device:
    """Get best available device (GPU if available)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def move_to_device(model: nn.Module, device: torch.device = None) -> nn.Module:
    """Move model to device."""
    if device is None:
        device = get_device()
    return model.to(device)
