"""
Adaptive loss balancing methods for DML.

Implements GradNorm (Chen et al., ICML 2018) and ReLoBRaLo (Bischof & Kraus, 2022)
for automatically balancing value loss vs derivative loss during training.

References:
    - GradNorm: https://arxiv.org/abs/1711.02257
    - ReLoBRaLo: https://arxiv.org/abs/2110.09813
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from dataclasses import dataclass

from .model import LossComponents

# I-M8 (2026-04-16): state_dict key for GradNorm's Adam over task_weights.
_GRADNORM_OPT_KEY = "_gradnorm_weight_optimizer_state"
_RNG_KEY = "_np_rng_state"


# ============================================================================
# GRADNORM LOSS (Chen et al., ICML 2018)
# ============================================================================

class GradNormDmlLoss(nn.Module):
    """
    DML loss with GradNorm adaptive weight balancing.
    
    GradNorm balances multi-task losses by dynamically adjusting task 
    weights so that gradient norms are similar across tasks. Tasks that
    train slower (higher loss ratio vs initial) get higher weight.
    
    Algorithm:
        1. Maintain learnable task weights w_i(t) 
        2. Each step, compute per-task gradient norms G_i = ||∇_W (w_i * L_i)||
        3. Compute loss ratios r_i = L_i(t) / L_i(0) (relative training rate)
        4. Compute target: G_target = mean(G_i)
        5. GradNorm loss: L_gn = Σ |G_i - G_target * r̃_i^α|
           where r̃_i = r_i / mean(r_i) (normalized inverse training rate)
        6. Update w_i by descent on L_gn (renormalize to sum to n_tasks)
    
    Args:
        input_dim: Input dimensionality (for compatibility)
        alpha: Asymmetry parameter controlling rebalancing strength. 
               α=0: equal rates. α=1: balanced. α>1: aggressive rebalancing.
        gradnorm_lr: Learning rate for weight updates
    """
    
    def __init__(
        self,
        input_dim: int,
        alpha: float = 1.5,
        gradnorm_lr: float = 0.025,
        shared_layer_name: Optional[str] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.alpha = alpha
        self.gradnorm_lr = gradnorm_lr

        # M2 (2026-04-13): Allow caller to pass an explicit "last shared layer"
        # parameter NAME (substring match against model.named_parameters()).
        # If None, fall back to weight_params[-2] heuristic (original behavior).
        # For PaiNN/GATv2: pass e.g. shared_layer_name="convs.3.lin" or whichever
        # is the last shared backbone layer per the GradNorm paper.
        self.shared_layer_name = shared_layer_name

        # Learnable task weights — initialized to 1.0 each
        # Using raw parameter (not log-space) following original paper
        self.task_weights = nn.Parameter(torch.ones(2))  # [value, deriv]

        # Track initial losses for relative training rate
        self.initial_losses: Optional[torch.Tensor] = None

        # Separate optimizer for task weights
        self._weight_optimizer = None

    def get_extra_state(self):
        """I-M8 / L-M6 (2026-04-16): use PyTorch's extra_state mechanism so
        the GradNorm-internal Adam state is included in parent state_dict
        with proper prefixing (no key collision at parent root). Previously
        a custom state_dict override leaked an unprefixed key."""
        state = {}
        if self._weight_optimizer is not None:
            state["weight_optimizer"] = self._weight_optimizer.state_dict()
        return state

    def set_extra_state(self, state):
        if not state:
            return
        opt_state = state.get("weight_optimizer")
        if opt_state is not None:
            _ = self.weight_optimizer  # lazy-init if needed
            self._weight_optimizer.load_state_dict(opt_state)
    
    @property
    def weight_optimizer(self):
        """Lazy init of weight optimizer (can't create in __init__ before parameters are registered)."""
        if self._weight_optimizer is None:
            self._weight_optimizer = torch.optim.Adam(
                [self.task_weights], lr=self.gradnorm_lr
            )
        return self._weight_optimizer
    
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        dydx_pred: torch.Tensor,
        dydx_true: torch.Tensor,
        model: nn.Module = None,
        dydx_mask: Optional[torch.Tensor] = None,
    ) -> LossComponents:
        """
        Compute weighted DML loss and update task weights via GradNorm.

        Args:
            y_pred: Predicted values (batch, 1)
            y_true: True values (batch, 1)
            dydx_pred: Predicted gradients (batch, 1, input_dim)
            dydx_true: True gradients (batch, 1, input_dim)
            model: The network model (needed for gradient norm computation)
            dydx_mask: Optional float mask (batch, 1, input_dim) zeroing out
                gradient channels with no ground truth (e.g. parameter-field
                slots in PDE pipelines, IC encoding slots).
        """
        # Compute per-task losses
        value_loss = torch.mean((y_pred - y_true) ** 2)
        if dydx_mask is not None:
            err = (dydx_pred - dydx_true) ** 2
            deriv_loss = (err * dydx_mask).sum() / dydx_mask.sum().clamp(min=1.0)
        else:
            deriv_loss = torch.mean((dydx_pred - dydx_true) ** 2)
        
        losses = torch.stack([value_loss, deriv_loss])

        # L-C-R2-1 (2026-04-14): gate on self.training so the reference is the
        # first TRAINING mini-batch per Chen et al. 2018, not a sanity-check
        # val batch (Lightning's num_sanity_val_steps defaults to 2).
        if self.initial_losses is None and self.training:
            self.initial_losses = losses.detach().clone()
        
        # Normalize weights to sum to n_tasks (2)
        with torch.no_grad():
            normalized_weights = 2.0 * self.task_weights / (self.task_weights.sum() + 1e-8)
        
        # Weighted total loss (use normalized weights but detach for main backward)
        total = normalized_weights[0] * value_loss + normalized_weights[1] * deriv_loss
        
        # GradNorm weight update (if model provided and in training mode)
        if model is not None and self.training:
            self._update_weights_gradnorm(model, losses)
        
        return LossComponents(
            total=total,
            value_loss=value_loss,
            deriv_loss=deriv_loss,
            reg_loss=torch.tensor(0.0, device=y_pred.device)
        )
    
    def _update_weights_gradnorm(
        self, 
        model: nn.Module, 
        losses: torch.Tensor
    ):
        """
        Update task weights using GradNorm algorithm.
        
        Adjusts weights so that tasks with higher relative loss 
        (slower training) get proportionally larger gradient norms.
        """
        # Resolve shared layer parameter
        # M2 (2026-04-13): explicit `shared_layer_name` overrides the
        # weight_params[-2] heuristic. L6: filter out LayerNorm and other
        # normalization-layer weights when applying the heuristic, since these
        # often appear at the end of named_parameters() and are not "shared
        # backbone" layers.
        if self.shared_layer_name is not None:
            shared_params = None
            for name, param in model.named_parameters():
                if self.shared_layer_name in name and 'weight' in name:
                    shared_params = param
                    break
            if shared_params is None:
                # Explicit name not found — fall back with a warning log
                # I-L4 (2026-04-16): explicit shared_layer_name supplied but
                # not found → hard-fail instead of silently falling back. If
                # user passed a name, they want that specific layer; silent
                # fallback could silently use the wrong layer.
                raise ValueError(
                    f"GradNorm shared_layer_name='{self.shared_layer_name}' not "
                    f"found in model.named_parameters(). Check the name. Set "
                    f"shared_layer_name=None to use the weight_params[-2] fallback."
                )
        else:
            shared_params = self._fallback_shared_params(model)
        if shared_params is None:
            return

        # M1 (2026-04-23): revert I-H5. The previous `normalized[i].detach()`
        # computation (intended to match paper's renormalized weights) killed
        # the gradient path from gradnorm_loss to self.task_weights, so Adam
        # saw zero grad and weights never updated — GradNorm silently ran as
        # fixed 1:1 DML since 2026-04-16. Chen et al. 2018 Alg.1 uses the
        # trainable w_i directly in G_W^(i) = ||∇_W w_i L_i||_2; that is what
        # allows gradnorm_loss.backward() to produce a non-zero gradient on w_i.
        # Renormalization to sum=T is done AFTER the Adam step (see below).
        grad_norms = []
        for i in range(2):
            weighted_loss_i = self.task_weights[i] * losses[i]
            grads = torch.autograd.grad(
                weighted_loss_i, shared_params,
                retain_graph=True, create_graph=True
            )[0]
            grad_norms.append(torch.norm(grads))

        grad_norms = torch.stack(grad_norms)

        # Compute relative inverse training rates
        loss_ratios = losses.detach() / (self.initial_losses + 1e-8)
        mean_ratio = loss_ratios.mean()
        relative_rates = loss_ratios / (mean_ratio + 1e-8)

        # Target gradient norm
        target_grad_norm = grad_norms.detach().mean()

        # GradNorm loss
        targets = target_grad_norm * (relative_rates ** self.alpha)
        gradnorm_loss = torch.sum(torch.abs(grad_norms - targets))

        # Update weights
        self.weight_optimizer.zero_grad()
        gradnorm_loss.backward(retain_graph=True)
        self.weight_optimizer.step()

        # CRITICAL FIX: Clear model gradients contaminated by GradNorm backward.
        # Without this, model.parameters() accumulate gradients from BOTH
        # gradnorm_loss.backward() AND the subsequent total.backward() in
        # train_epoch(), causing the model to receive corrupted gradients.
        model.zero_grad()

        # Re-normalize weights to sum to n_tasks
        with torch.no_grad():
            self.task_weights.data = (
                2.0 * self.task_weights.data / (self.task_weights.data.sum() + 1e-8)
            )
            # Clamp to prevent negative weights
            self.task_weights.data = torch.clamp(self.task_weights.data, min=0.01)

    @staticmethod
    def _fallback_shared_params(model: nn.Module):
        """L6 (2026-04-13): pick weight_params[-2] but exclude LayerNorm/norm/bn
        weights, which are not 'shared backbone' layers per the GradNorm paper."""
        weight_params = [
            param for name, param in model.named_parameters()
            if 'weight' in name
            and 'norm' not in name.lower()
            and 'bn' not in name.lower()
        ]
        if len(weight_params) < 2:
            return None
        return weight_params[-2]
    
    def get_weights(self) -> dict:
        """Return current task weights for logging."""
        w = self.task_weights.detach().cpu()
        return {"value_weight": w[0].item(), "deriv_weight": w[1].item()}


# ============================================================================
# DIMENSION-NORMALIZED GRADNORM (DimNormGradNorm)
# ============================================================================

class DimNormGradNormDmlLoss(GradNormDmlLoss):
    """
    GradNorm with dimension-aware gradient norm normalization.
    
    In DML, the derivative loss backpropagates through d Jacobian components,
    producing gradient norms that are O(d) larger than the value loss.
    Standard GradNorm misinterprets this as a training rate imbalance,
    incorrectly suppressing the derivative weight at high dimensions.
    
    Fix: Normalize the derivative gradient norm by a dimension factor before
    GradNorm's comparison step, so that both tasks produce comparable norms.
    
    Args:
        input_dim: Input dimensionality d
        alpha: GradNorm asymmetry parameter (default 1.5)
        gradnorm_lr: Learning rate for weight updates
        dim_norm_mode: Normalization mode:
            - "d": Divide derivative norm by d   (O(d) correction)
            - "sqrt_d": Divide by sqrt(d)        (O(√d) correction, softer)
    
    Reference: See GRADNORM_DIMENSION_FIX.md for full analysis.
    """
    
    def __init__(
        self,
        input_dim: int,
        alpha: float = 1.5,
        gradnorm_lr: float = 0.025,
        dim_norm_mode: str = "d"
    ):
        super().__init__(
            input_dim=input_dim,
            alpha=alpha,
            gradnorm_lr=gradnorm_lr
        )
        self.dim_norm_mode = dim_norm_mode
        
        # Pre-compute dimension factors: [1.0 for value, factor for deriv]
        if dim_norm_mode == "d":
            self._dim_factor = float(input_dim)
        elif dim_norm_mode == "sqrt_d":
            self._dim_factor = float(input_dim) ** 0.5
        else:
            raise ValueError(f"Unknown dim_norm_mode: {dim_norm_mode}. Use 'd' or 'sqrt_d'.")
    
    def _update_weights_gradnorm(
        self, 
        model: nn.Module, 
        losses: torch.Tensor
    ):
        """
        GradNorm weight update with dimension-normalized gradient norms.
        
        Same as standard GradNorm, but normalizes the derivative gradient norm
        by the dimension factor before computing the GradNorm target and loss.
        This prevents the O(d) scaling of derivative gradients from confusing
        the relative training rate estimation.
        """
        # I-H6 (2026-04-16): reuse parent's shared-layer resolution so that
        # (a) explicit `shared_layer_name` is honored per M2, and (b) the
        # fallback excludes norm/bn layers per L6. Previously this subclass
        # duplicated an older heuristic that silently picked the wrong layer
        # on any arch with LayerNorm (e.g. if someone dispatched this method
        # on GATv2 / PaiNN).
        if self.shared_layer_name is not None:
            shared_params = None
            for name, param in model.named_parameters():
                if self.shared_layer_name in name and 'weight' in name:
                    shared_params = param
                    break
            if shared_params is None:
                raise ValueError(
                    f"DimNormGradNorm shared_layer_name='{self.shared_layer_name}' not found."
                )
        else:
            shared_params = self._fallback_shared_params(model)
        if shared_params is None:
            return

        # M1 (2026-04-23): revert K-M2/I-H5. Using detached renormalized weights
        # killed the gradient path to self.task_weights — see parent class for
        # the same fix. Use trainable w_i directly.
        grad_norms = []
        for i in range(2):
            weighted_loss_i = self.task_weights[i] * losses[i]
            grads = torch.autograd.grad(
                weighted_loss_i, shared_params,
                retain_graph=True, create_graph=True
            )[0]
            grad_norms.append(torch.norm(grads))

        grad_norms = torch.stack(grad_norms)
        
        # ====================================================================
        # KEY FIX: Dimension-normalize derivative gradient norm
        # The derivative task has O(d) larger gradient norms because it
        # backpropagates through d Jacobian components. Normalize before
        # comparing.
        # ====================================================================
        dim_factors = torch.tensor(
            [1.0, self._dim_factor], 
            device=grad_norms.device, dtype=grad_norms.dtype
        )
        grad_norms_normalized = grad_norms / dim_factors
        
        # Compute relative inverse training rates (unchanged)
        loss_ratios = losses.detach() / (self.initial_losses + 1e-8)
        mean_ratio = loss_ratios.mean()
        relative_rates = loss_ratios / (mean_ratio + 1e-8)
        
        # Target gradient norm — computed on NORMALIZED norms
        target_grad_norm = grad_norms_normalized.detach().mean()
        
        # GradNorm loss — compare NORMALIZED norms to target
        targets = target_grad_norm * (relative_rates ** self.alpha)
        gradnorm_loss = torch.sum(torch.abs(grad_norms_normalized - targets))
        
        # Update weights
        self.weight_optimizer.zero_grad()
        gradnorm_loss.backward(retain_graph=True)
        self.weight_optimizer.step()
        
        # CRITICAL: Clear model gradients contaminated by GradNorm backward
        model.zero_grad()
        
        # Re-normalize weights to sum to n_tasks
        with torch.no_grad():
            self.task_weights.data = (
                2.0 * self.task_weights.data / (self.task_weights.data.sum() + 1e-8)
            )
            self.task_weights.data = torch.clamp(self.task_weights.data, min=0.01)


# ============================================================================
# SOFTMAX-BALANCED LOSS — was named ReLoBRaLoDmlLoss (Bischof & Kraus, 2022)
# but is a SIMPLIFIED variant of the paper's algorithm.
#
# 2026-04-13 RENAME: external review showed the implementation here is a
# substantial simplification of Bischof & Kraus 2022 Eq. 11 — paper has TWO
# smoothing parameters (Bernoulli ρ saudade lookback to t=0 + exponential α
# decay), this code has ONE EMA-style rho. Keeping the original code as
# `SoftmaxBalanceDmlLoss` (was misnamed); the FAITHFUL implementation lives
# in `ReLoBRaLoDmlLoss` below. Existing v2 result JSONs labeled
# method='dml_relobralo' refer to THIS simplified class (semantic rename
# applied 2026-04-13; no value rerun needed for v2 unless the dataset
# is intentionally re-running with the faithful version).
# ============================================================================

class SoftmaxBalanceDmlLoss(nn.Module):
    """
    DML loss with simplified softmax-of-loss-ratio rebalancing + EMA smoothing.

    This is a SIMPLIFIED variant of ReLoBRaLo (Bischof & Kraus 2022) — the paper's
    Eq. 11 has two smoothing parameters (Bernoulli ρ saudade + exponential α
    decay); this implementation collapses to a single EMA-style ρ. Use
    `ReLoBRaLoDmlLoss` for the full paper algorithm.

    L4 (2026-04-13): accepts a `seed` parameter for the internal random-lookback
    RNG. Default: derive from the first forward call to avoid tying all method
    runs to seed=42 regardless of the experiment seed.
    
    Algorithm:
        1. At each step, compute per-task losses L_i(t)
        2. Sample a random lookback step t_ref from history
        3. Compute weights: w_i = softmax(L_i(t) / (τ * L_i(t_ref)))
        4. Apply EMA smoothing: w̄_i = ρ * w̄_i + (1-ρ) * w_i
        5. Use w̄_i to weight the total loss
    
    Args:
        input_dim: Input dimensionality (for compatibility)
        tau: Temperature parameter controlling weight sharpness.
             Lower = sharper reweighting. Default 1.0.
        rho: EMA smoothing factor. Higher = smoother weight transitions.
             Default 0.999 (very smooth).
        max_history: Maximum number of loss snapshots to keep.
    """
    
    def __init__(
        self,
        input_dim: int,
        tau: float = 1.0,
        rho: float = 0.999,
        max_history: int = 1000,
        seed: int = 42,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.tau = tau
        self.rho = rho
        self.max_history = max_history
        self._seed = seed

        # Loss history for random lookback (stored as plain Python list, not parameters)
        self.loss_history: list = []

        # Seeded RNG for reproducible random lookback (avoids global np.random state).
        # L4 (2026-04-13): allow caller to override seed so the lookback pattern
        # differs across experiment seeds. Default preserves backward compat.
        self._rng = np.random.RandomState(seed)
        
        # J7 (2026-04-16): initialize to ones(2) (sum=2) so that the contract
        # "weights sum to T=2 at all times" holds from step 0. Prior init was
        # [0.5, 0.5] (sum=1), and the EMA slowly converged to sum=2 across
        # ~5000 steps, making the total-loss scale factor non-stationary early
        # in training. Gradient direction is unchanged, but fidelity improves.
        self.register_buffer('running_weights', torch.ones(2))
        
        # Step counter
        self._step = 0

    def get_extra_state(self):
        """I-M9 / L-M6 (2026-04-16): RNG + loss_history via extra_state (proper
        prefix handling). Prior state_dict override leaked unprefixed key."""
        return {
            "rng": self._rng.get_state(),
            # L-L... (2026-04-16): also serialize loss_history so EMA resumes.
            # M2 (2026-04-23): loss_history stores plain python lists of
            # floats (value.item(), deriv.item()); no tensor methods apply.
            "loss_history": [list(l) for l in self.loss_history],
        }

    def set_extra_state(self, state):
        if not state:
            return
        rng_state = state.get("rng")
        if rng_state is not None:
            self._rng.set_state(rng_state)
        hist = state.get("loss_history")
        if hist is not None:
            self.loss_history = list(hist)


    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        dydx_pred: torch.Tensor,
        dydx_true: torch.Tensor,
        model: nn.Module = None,
        dydx_mask: Optional[torch.Tensor] = None,
    ) -> LossComponents:
        """
        Compute ReLoBRaLo-weighted DML loss.

        dydx_mask: optional float mask (batch, 1, input_dim) zeroing out
            gradient channels with no ground truth.
        """
        # Compute per-task losses
        value_loss = torch.mean((y_pred - y_true) ** 2)
        if dydx_mask is not None:
            err = (dydx_pred - dydx_true) ** 2
            deriv_loss = (err * dydx_mask).sum() / dydx_mask.sum().clamp(min=1.0)
        else:
            deriv_loss = torch.mean((dydx_pred - dydx_true) ** 2)

        current_losses = [value_loss.item(), deriv_loss.item()]
        
        if self.training and len(self.loss_history) > 0:
            # Random lookback: sample a reference step (uses seeded RNG for reproducibility)
            ref_idx = self._rng.randint(0, len(self.loss_history))
            ref_losses = self.loss_history[ref_idx]
            
            # Softmax-style reweighting
            # w_i = exp(L_i(t) / (τ * L_i(t_ref))) 
            log_w_val = current_losses[0] / (self.tau * ref_losses[0] + 1e-8)
            log_w_der = current_losses[1] / (self.tau * ref_losses[1] + 1e-8)
            
            # Numerical stability: subtract max before exp
            max_log = max(log_w_val, log_w_der)
            w_val = np.exp(log_w_val - max_log)
            w_der = np.exp(log_w_der - max_log)
            
            # Normalize to sum to n_tasks (2)
            w_sum = w_val + w_der
            weights = torch.tensor(
                [2.0 * w_val / w_sum, 2.0 * w_der / w_sum],
                device=self.running_weights.device, dtype=torch.float32
            )
            
            # EMA smoothing
            # L5 (2026-04-13): use .copy_() on the registered buffer to preserve
            # state_dict()/load_state_dict() roundtripping. Plain assignment
            # `self.running_weights = ...` would replace the registered buffer
            # with a non-buffer Tensor and break checkpoint loading.
            self.running_weights.copy_(
                self.rho * self.running_weights + (1.0 - self.rho) * weights
            )
        
        # Store current losses in history
        if self.training:
            self.loss_history.append(current_losses)
            # Trim history to prevent unbounded memory growth
            if len(self.loss_history) > self.max_history:
                self.loss_history = self.loss_history[-self.max_history:]
            self._step += 1
        
        # Apply running weights (detached — no gradient through weights)
        w = self.running_weights.detach()
        total = w[0] * value_loss + w[1] * deriv_loss
        
        return LossComponents(
            total=total,
            value_loss=value_loss,
            deriv_loss=deriv_loss,
            reg_loss=torch.tensor(0.0, device=y_pred.device)
        )
    
    def get_weights(self) -> dict:
        """Return current task weights for logging."""
        w = self.running_weights.detach().cpu()
        return {"value_weight": w[0].item(), "deriv_weight": w[1].item()}
    
    def reset(self):
        """Reset state for a new training run."""
        self.loss_history = []
        # L5: preserve registered buffer
        self.running_weights.copy_(torch.ones(2, device=self.running_weights.device) / 2.0)
        self._step = 0


# ============================================================================
# RELOBRALO (Bischof & Kraus 2022, Eq. 11) — added 2026-04-13
#
# 2026-04-13 NAMING DECISION:
#   - `ReLoBRaLoDmlLoss` (this class) = canonical name, FAITHFUL Eq.11
#     implementation. New code should import this when they want
#     "ReLoBRaLo as published".
#   - `SoftmaxBalanceDmlLoss` (above) = the simplified variant from earlier
#     v2 code (was previously misnamed "ReLoBRaLoDmlLoss" — see D011).
# Backward-compat alias removed (v2 code that explicitly imports
# `ReLoBRaLoDmlLoss` will now get the faithful class with different
# behavior — that is intentional. v2 code that wants the simplified
# implementation must explicitly import `SoftmaxBalanceDmlLoss`).
# ============================================================================

class ReLoBRaLoDmlLoss(nn.Module):
    """
    Faithful implementation of ReLoBRaLo (Bischof & Kraus 2022, Eq. 11) for DML.

    The paper has TWO smoothing parameters:
      - Bernoulli `ρ` (saudade): probabilistic lookback to t'=0 (start of training)
      - Exponential `α` (decay): EMA between historical and current balancing weight

    Eq. 11 (paper notation, T=2 tasks here):
        λ_i^bal(t,t')  = T * exp(L_i(t) / (τ * L_i(t'))) / Σ_j exp(L_j(t) / (τ * L_j(t')))
        λ_i^hist(t)    = ρ * λ_i(t-1) + (1-ρ) * λ_i^bal(t, 0)        ← ρ ~ Bernoulli(E[ρ])
        λ_i(t)         = α * λ_i^hist(t) + (1-α) * λ_i^bal(t, t-1)

    Defaults drawn from paper Table VIII (PINN benchmarks):
      - τ = 0.1, α = 0.999 — from Burgers column of Table VIII
      - E[ρ] = 0.999 — chosen within the paper's BO range [0, 1];
        Table VIII reports 0.9999 (Burgers/Kirchhoff) and 0.9 (Helmholtz).
        0.999 is order-of-magnitude consistent with the Burgers column and
        within Fig. 3(d)'s validated configuration (α=0.999, τ=0.1, ρ=0.999).

    Args:
        input_dim: Input dimensionality (for compatibility).
        tau: Softmax temperature (paper τ); default 0.1 per paper Table VIII.
        alpha: Exponential decay weight; default 0.999 per paper Table VIII.
        rho_expectation: Bernoulli E[ρ] for saudade; default 0.999 per paper.
        max_history: Maximum loss history snapshots (must be >= 2).

    Reference: arXiv:2110.09813.
    """

    def __init__(
        self,
        input_dim: int,
        tau: float = 0.1,
        alpha: float = 0.999,
        rho_expectation: float = 0.999,
        max_history: int = 10000,
        seed: int = 42,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.tau = tau
        self.alpha = alpha
        self.rho_expectation = rho_expectation
        self.max_history = max_history
        self._seed = seed

        # Reproducible saudade RNG (Bernoulli draws). Independent from the global
        # torch RNG so the saudade pattern doesn't depend on optimizer state.
        # L4 (2026-04-13): allow caller to override seed.
        self._rng = np.random.RandomState(seed)

        # Loss history — index 0 is t=0 (start of training); last entry is t-1.
        self.loss_history: list = []

        # λ(t) — the running balancing weight, registered buffer for state_dict
        self.register_buffer('lambda_current', torch.ones(2))

        self._step = 0

    def get_extra_state(self):
        """I-M9 / L-M6 (2026-04-16): RNG state via extra_state (proper prefix)."""
        return {
            "rng": self._rng.get_state(),
            # M2 (2026-04-23): loss_history stores plain python lists of
            # floats (value.item(), deriv.item()); no tensor methods apply.
            "loss_history": [list(l) for l in self.loss_history],
        }

    def set_extra_state(self, state):
        if not state:
            return
        rng_state = state.get("rng")
        if rng_state is not None:
            self._rng.set_state(rng_state)
        hist = state.get("loss_history")
        if hist is not None:
            self.loss_history = list(hist)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        dydx_pred: torch.Tensor,
        dydx_true: torch.Tensor,
        model: nn.Module = None,
        dydx_mask: Optional[torch.Tensor] = None,
    ) -> LossComponents:
        """dydx_mask: optional float mask (batch, 1, input_dim)."""
        value_loss = torch.mean((y_pred - y_true) ** 2)
        if dydx_mask is not None:
            err = (dydx_pred - dydx_true) ** 2
            deriv_loss = (err * dydx_mask).sum() / dydx_mask.sum().clamp(min=1.0)
        else:
            deriv_loss = torch.mean((dydx_pred - dydx_true) ** 2)
        current_losses = [value_loss.item(), deriv_loss.item()]

        if self.training and len(self.loss_history) >= 2:
            # Compute λ^bal(t, t') for t' in {0, t-1}
            lam_bal_0 = self._compute_balancing(current_losses, self.loss_history[0])
            lam_bal_prev = self._compute_balancing(current_losses, self.loss_history[-1])

            # Bernoulli ρ saudade: with probability E[ρ], use last-step λ;
            # with prob (1 - E[ρ]), use λ^bal(t, 0)  (saudade lookback to start)
            rho_draw = self._rng.binomial(1, self.rho_expectation)
            if rho_draw == 1:
                lam_hist = self.lambda_current.clone()
            else:
                lam_hist = lam_bal_0

            # Exponential α decay between historical and current step-(t-1) balancing
            new_lambda = self.alpha * lam_hist + (1.0 - self.alpha) * lam_bal_prev

            # Re-normalize to sum to T (== 2)
            new_lambda = 2.0 * new_lambda / (new_lambda.sum() + 1e-8)
            self.lambda_current.copy_(new_lambda)

        if self.training:
            self.loss_history.append(current_losses)
            if len(self.loss_history) > self.max_history:
                # Always keep index 0 (saudade reference); trim middle.
                self.loss_history = [self.loss_history[0]] + self.loss_history[-(self.max_history - 1):]
            self._step += 1

        w = self.lambda_current.detach()
        total = w[0] * value_loss + w[1] * deriv_loss
        return LossComponents(
            total=total,
            value_loss=value_loss,
            deriv_loss=deriv_loss,
            reg_loss=torch.tensor(0.0, device=y_pred.device),
        )

    def _compute_balancing(self, current_losses, ref_losses):
        """λ^bal(t, t') = T * softmax(L_i(t) / (τ * L_i(t')))."""
        log_w = np.array([
            current_losses[0] / (self.tau * ref_losses[0] + 1e-8),
            current_losses[1] / (self.tau * ref_losses[1] + 1e-8),
        ])
        # Numerical-stable softmax
        max_log = log_w.max()
        w_unnorm = np.exp(log_w - max_log)
        w_sum = w_unnorm.sum()
        return torch.tensor(
            [2.0 * w_unnorm[0] / w_sum, 2.0 * w_unnorm[1] / w_sum],
            device=self.lambda_current.device, dtype=torch.float32,
        )

    def get_weights(self) -> dict:
        w = self.lambda_current.detach().cpu()
        return {"value_weight": w[0].item(), "deriv_weight": w[1].item()}

    def reset(self):
        self.loss_history = []
        self.lambda_current.copy_(torch.ones(2, device=self.lambda_current.device))
        self._step = 0


# ============================================================================
# (NO backward-compat alias — see naming-decision note above ReLoBRaLoDmlLoss.)
# Old code wanting the simplified variant must explicitly import
# SoftmaxBalanceDmlLoss.
# ============================================================================
