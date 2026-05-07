"""
Training utilities for DML benchmark.

Provides trainer class with comprehensive logging, checkpointing,
and support for both DML and vanilla training.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import platform
import subprocess
from tqdm import tqdm

from .model import (
    DmlFeedForward, DmlLoss, VanillaLoss, DmlDataset, 
    DataNormalizer, LossComponents, get_device
)


# ============================================================================
# REPRODUCIBILITY & METADATA
# ============================================================================

def set_deterministic(seed: int = 42):
    """Set all seeds for full reproducibility across CPU and GPU.

    J1/H-M7 (2026-04-16): strengthened per Reviewer H/F round-3 feedback.
    - PYTHONHASHSEED controls Python hash randomization (dict/set order).
    - CUBLAS_WORKSPACE_CONFIG is required for CUDA >= 10.2 deterministic
      matmul under torch.use_deterministic_algorithms(True).
    - torch.use_deterministic_algorithms(warn_only=True): some ops (scatter_add
      used by GATv2 message passing) have no deterministic CUDA kernel; warn
      rather than hard-fail so those experiments still run.
    """
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def get_run_metadata() -> Dict[str, Any]:
    """Collect hardware and software metadata for reproducibility logging."""
    meta = {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        meta["gpu_name"] = torch.cuda.get_device_name(0)
        meta["gpu_count"] = torch.cuda.device_count()
        meta["cuda_version"] = torch.version.cuda or "unknown"
    try:
        meta["git_hash"] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL
        ).strip().decode()
    except Exception:
        meta["git_hash"] = "unknown"
    # I-L3 (2026-04-16): record whether the tree had uncommitted changes at
    # run time. A NeurIPS D&B reviewer needs to know the state isn't pristine.
    try:
        porcelain = subprocess.check_output(
            ['git', 'status', '--porcelain'], stderr=subprocess.DEVNULL
        ).decode()
        meta["git_dirty"] = bool(porcelain.strip())
    except Exception:
        meta["git_dirty"] = None
    return meta


# ============================================================================
# TRAINING RESULTS
# ============================================================================

@dataclass
class TrainingLog:
    """Log entry for one epoch."""
    epoch: int
    train_loss: float
    val_loss: float
    train_value_loss: float
    train_deriv_loss: float
    val_value_loss: float
    val_deriv_loss: float
    lr: float
    time_s: float


@dataclass
class TrainingResult:
    """Complete training result."""
    config: Dict[str, Any]
    final_train_loss: float
    final_val_loss: float
    test_value_mse: float
    test_grad_mse: float
    training_logs: List[Dict[str, float]] = field(default_factory=list)
    total_time_s: float = 0.0
    best_epoch: int = 0
    # I-H1 (2026-04-16): expose best model state so callers (e.g. MLP-pairwise
    # runner) can rerun the trained model on custom test metrics like a
    # reconstructed Cartesian force MSE. Not serialized to JSON.
    best_model_state: Optional[Any] = field(default=None, repr=False)
    # I-L6 (2026-04-16): explicit early-stopping status, so analyzers don't
    # need to infer via len(training_logs) < n_epochs.
    early_stopped: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "metrics": {
                "final_train_loss": self.final_train_loss,
                "final_val_loss": self.final_val_loss,
                "test_value_mse": self.test_value_mse,
                "test_grad_mse": self.test_grad_mse,
                "total_time_s": self.total_time_s,
                "best_epoch": self.best_epoch
            },
            "training_logs": self.training_logs
        }
    
    def save(self, path: Path):
        """Save result to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# ============================================================================
# TRAINER
# ============================================================================

class DmlTrainer:
    """
    Trainer for DML benchmark with logging and checkpointing.
    """
    
    def __init__(
        self,
        model: DmlFeedForward,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        normalizer: 'DataNormalizer' = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = None,
        use_dml: bool = True,
        max_grad_norm: float = 1.0,
        task_weights_log_path: Optional[str] = None,
    ):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        if isinstance(self.loss_fn, nn.Module):
            self.loss_fn = self.loss_fn.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler  # LR scheduler (e.g. ReduceLROnPlateau)
        self.normalizer = normalizer  # Required for proper MSE evaluation
        self.use_dml = use_dml
        self.max_grad_norm = max_grad_norm  # Gradient clipping threshold

        # Optional per-step task-weights logging for F18/F19 figures. Default
        # off → byte-identical to pre-hook behaviour. When a path is provided,
        # we accumulate a buffer of {epoch, batch, task_weights, losses} and
        # flush as JSONL at end of training. Read-only observation; no RNG draws,
        # no autograd interaction.
        self.task_weights_log_path = task_weights_log_path
        self._task_weights_buffer: List[Dict] = []

        self.training_logs: List[TrainingLog] = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.best_epoch = 0  # Track best epoch explicitly (Issue #10)
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        pbar: bool = True
    ) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_value_loss = 0.0
        total_deriv_loss = 0.0
        n_batches = 0
        
        iterator = tqdm(dataloader, disable=not pbar, desc=f"Epoch {epoch}")
        
        for batch in iterator:
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)
            dydx = batch['dydx'].to(self.device)
            dydx_mask = batch.get('dydx_mask', None)
            if dydx_mask is not None:
                dydx_mask = dydx_mask.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_dml:
                y_pred, dydx_pred = self.model.forward_with_greek(x)
                loss_components = self.loss_fn(y_pred, y, dydx_pred, dydx, self.model, dydx_mask=dydx_mask)
            else:
                y_pred = self.model(x)
                loss_components = self.loss_fn(y_pred, y)
            
            loss_components.total.backward()
            
            # Gradient clipping to prevent NaN explosions (Task 0.7)
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.max_grad_norm
                )
            
            self.optimizer.step()

            total_loss += loss_components.total.item()
            total_value_loss += loss_components.value_loss.item()
            total_deriv_loss += loss_components.deriv_loss.item()
            n_batches += 1

            if self.task_weights_log_path is not None:
                tw = getattr(self.loss_fn, "task_weights", None)
                if tw is not None:
                    self._task_weights_buffer.append({
                        "epoch": int(epoch),
                        "batch_in_epoch": int(n_batches - 1),
                        "task_weights": tw.detach().cpu().tolist(),
                        "value_loss": float(loss_components.value_loss.item()),
                        "deriv_loss": float(loss_components.deriv_loss.item()),
                        "total_loss": float(loss_components.total.item()),
                    })

            iterator.set_postfix(loss=loss_components.total.item())
        
        return {
            'loss': total_loss / n_batches,
            'value_loss': total_value_loss / n_batches,
            'deriv_loss': total_deriv_loss / n_batches
        }
    
    def validate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """Validate on dataset."""
        # K-L3 (2026-04-16): save prior mode and restore on exit instead of
        # unconditionally calling .train(). If the caller expected eval mode
        # to persist (e.g. during a test-time evaluation pipeline), the old
        # code would silently switch back to train mode.
        model_was_training = self.model.training
        loss_fn_was_training = getattr(self.loss_fn, 'training', None)
        self.model.eval()
        if isinstance(self.loss_fn, nn.Module):
            self.loss_fn.eval()
        
        total_loss = 0.0
        total_value_loss = 0.0
        total_deriv_loss = 0.0
        n_batches = 0
        
        for batch in dataloader:
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)
            dydx = batch['dydx'].to(self.device)
            dydx_mask = batch.get('dydx_mask', None)
            if dydx_mask is not None:
                dydx_mask = dydx_mask.to(self.device)
            
            if self.use_dml:
                # DML requires gradient computation for forward_with_greek
                # Use torch.enable_grad() to ensure graph is built
                with torch.enable_grad():
                    y_pred, dydx_pred = self.model.forward_with_greek(x)
                    loss_components = self.loss_fn(y_pred, y, dydx_pred, dydx, self.model, dydx_mask=dydx_mask)
            else:
                with torch.no_grad():
                    y_pred = self.model(x)
                    loss_components = self.loss_fn(y_pred, y)
            
            total_loss += loss_components.total.item()
            total_value_loss += loss_components.value_loss.item()
            total_deriv_loss += loss_components.deriv_loss.item()
            n_batches += 1
        
        # K-L3 (2026-04-16): restore prior mode (not unconditionally train).
        if model_was_training:
            self.model.train()
        if isinstance(self.loss_fn, nn.Module) and loss_fn_was_training:
            self.loss_fn.train()

        return {
            'loss': total_loss / n_batches,
            'value_loss': total_value_loss / n_batches,
            'deriv_loss': total_deriv_loss / n_batches
        }
    
    def evaluate(
        self,
        dataloader: DataLoader,
        unscale: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        CRITICAL: Computes MSE in ORIGINAL (unscaled) domain for scientifically
        valid results. If normalizer is not available, falls back to normalized
        MSE with a warning.
        
        Args:
            dataloader: Test data loader
            unscale: If True and normalizer available, unscale to original domain
        
        Returns:
            Dictionary with value_mse and grad_mse (in original scale)
        """
        self.model.eval()
        # I-L2/I-M2 (2026-04-16): put loss_fn in eval mode during evaluate so
        # stateful balancing classes (GradNorm initial_losses gating, ReLoBRaLo
        # history append) don't mutate during test inference. Restore on exit.
        loss_fn_was_training = getattr(self.loss_fn, 'training', None)
        if hasattr(self.loss_fn, 'eval'):
            self.loss_fn.eval()

        all_y_pred = []
        all_y_true = []
        all_dydx_pred = []
        all_dydx_true = []

        for batch in dataloader:
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)
            dydx = batch['dydx'].to(self.device)
            
            # Compute value and gradient predictions
            # For both DML and vanilla, we use forward_with_greek to get
            # autodiff gradients of the learned function. This gives a fair
            # gradient comparison: vanilla learns gradients implicitly through
            # its value approximation, and autodiff extracts them.
            with torch.enable_grad():
                y_pred, dydx_pred = self.model.forward_with_greek(x)
            
            all_y_pred.append(y_pred.detach().cpu().numpy())
            all_y_true.append(y.cpu().numpy())
            all_dydx_pred.append(dydx_pred.detach().cpu().numpy())
            all_dydx_true.append(dydx.cpu().numpy())
        
        y_pred = np.concatenate(all_y_pred, axis=0)
        y_true = np.concatenate(all_y_true, axis=0)
        dydx_pred = np.concatenate(all_dydx_pred, axis=0)
        dydx_true = np.concatenate(all_dydx_true, axis=0)
        
        # CRITICAL FIX: Unscale to original domain before computing MSE
        if unscale and self.normalizer is not None:
            y_pred = self.normalizer.unscale_y(y_pred)
            y_true = self.normalizer.unscale_y(y_true)
            dydx_pred = self.normalizer.unscale_dydx(dydx_pred)
            dydx_true = self.normalizer.unscale_dydx(dydx_true)
        elif unscale and self.normalizer is None:
            import warnings
            warnings.warn(
                "Normalizer not provided - MSE computed in NORMALIZED space! "
                "Results may not be comparable across different scales. "
                "Pass normalizer to DmlTrainer for valid metrics.",
                UserWarning
            )
        
        value_mse = float(np.mean((y_pred - y_true) ** 2))
        grad_mse = float(np.mean((dydx_pred - dydx_true) ** 2))

        # I-L2/I-M2: restore loss_fn prior mode.
        if hasattr(self.loss_fn, 'train') and loss_fn_was_training:
            self.loss_fn.train()

        return {
            'value_mse': value_mse,
            'grad_mse': grad_mse
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
        config: Dict[str, Any] = None,
        pbar: bool = True,
        early_stopping_patience: int = 50
    ) -> TrainingResult:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Number of epochs
            config: Configuration dict for logging
            pbar: Show progress bar
            early_stopping_patience: Epochs without improvement before stopping
            
        Returns:
            TrainingResult with logs and metrics
        """
        config = config or {}
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(n_epochs):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch, pbar=pbar)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Get learning rate
            lr = self.optimizer.param_groups[0]['lr']
            
            # Log
            log = TrainingLog(
                epoch=epoch,
                train_loss=train_metrics['loss'],
                val_loss=val_metrics['loss'],
                train_value_loss=train_metrics['value_loss'],
                train_deriv_loss=train_metrics['deriv_loss'],
                val_value_loss=val_metrics['value_loss'],
                val_deriv_loss=val_metrics['deriv_loss'],
                lr=lr,
                time_s=time.time() - epoch_start
            )
            self.training_logs.append(log)
            
            # Track best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                self.best_epoch = epoch  # Track best epoch explicitly (Issue #10)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Step LR scheduler (Task 0.2)
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        total_time = time.time() - start_time
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.model = self.model.to(self.device)
        
        # Create result
        result = TrainingResult(
            config=config,
            final_train_loss=self.training_logs[-1].train_loss,
            final_val_loss=self.best_val_loss,
            test_value_mse=0.0,  # Will be filled by evaluate()
            test_grad_mse=0.0,
            training_logs=[
                {
                    'epoch': log.epoch,
                    'train_loss': log.train_loss,
                    'val_loss': log.val_loss,
                    'train_value_loss': log.train_value_loss,
                    'train_deriv_loss': log.train_deriv_loss,
                    'val_value_loss': log.val_value_loss,
                    'val_deriv_loss': log.val_deriv_loss,
                    'lr': log.lr,
                    'time_s': log.time_s
                }
                for log in self.training_logs
            ],
            total_time_s=total_time,
            best_epoch=self.best_epoch,  # Use tracked value (Issue #10)
            # I-H1 / I-L6 (2026-04-16): expose best_model_state + ES flag.
            best_model_state=self.best_model_state,
            early_stopped=(patience_counter >= early_stopping_patience),
        )

        # Flush optional task-weights JSONL (F18/F19). No-op when path is None.
        if self.task_weights_log_path is not None and self._task_weights_buffer:
            from pathlib import Path as _P
            p = _P(self.task_weights_log_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w") as fh:
                for entry in self._task_weights_buffer:
                    fh.write(json.dumps(entry) + "\n")

        return result
    
    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict[str, float]):
        """Save training checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_val_loss': self.best_val_loss
        }, path)
    
    def load_checkpoint(self, path: Path) -> Dict[str, Any]:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        return checkpoint


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def create_data_loaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    dydx_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    dydx_test: np.ndarray,
    batch_size: int = 256,
    val_split: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    x_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    dydx_val: Optional[np.ndarray] = None,
    dydx_train_mask: Optional[np.ndarray] = None,
    dydx_val_mask: Optional[np.ndarray] = None,
    dydx_test_mask: Optional[np.ndarray] = None,
) -> tuple:
    """
    Create train/val/test data loaders.

    Args:
        num_workers: DataLoader workers. 0 = auto-detect based on CUDA availability.
        pin_memory: Pin memory for GPU transfer. None = auto-detect.
        x_val, y_val, dydx_val: V3 (2026-04-13) optional EXPLICIT validation set.
            If provided, `val_split` is ignored and `x_train` is used in full as
            training. If None, fall back to `random_split` of the train data
            using `val_split` (legacy 80/20 behavior).

    Returns:
        (train_loader, val_loader, test_loader, normalizer)
    """
    # Auto-detect optimal settings for GPU (Task 0.3)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if num_workers == 0 and torch.cuda.is_available():
        import os
        num_workers = min(4, os.cpu_count() or 1)

    # Normalize on train data
    normalizer = DataNormalizer()
    normalizer.initialize_with_data(x_train, y_train, dydx_train)

    x_train_norm, y_train_norm, dydx_train_norm = normalizer.normalize_all(
        x_train, y_train, dydx_train
    )
    x_test_norm, y_test_norm, dydx_test_norm = normalizer.normalize_all(
        x_test, y_test, dydx_test
    )

    train_dataset_full = DmlDataset(x_train_norm, y_train_norm, dydx_train_norm, dydx_mask=dydx_train_mask)
    test_dataset = DmlDataset(x_test_norm, y_test_norm, dydx_test_norm, dydx_mask=dydx_test_mask)

    if x_val is not None and y_val is not None and dydx_val is not None:
        # V3 (2026-04-13): explicit val set provided. Use train in full.
        x_val_norm, y_val_norm, dydx_val_norm = normalizer.normalize_all(
            x_val, y_val, dydx_val
        )
        train_dataset = train_dataset_full
        val_dataset = DmlDataset(x_val_norm, y_val_norm, dydx_val_norm, dydx_mask=dydx_val_mask)
    else:
        # J-L6 (2026-04-16): warn if caller set a non-default val_split while
        # NOT passing explicit val args — this used to silently use the default.
        # Explicit val args take precedence (block above); this branch runs
        # only when val arrays are absent.
        # Legacy: split train into train/val by val_split (default 80/20)
        n_train = int(len(train_dataset_full) * (1 - val_split))
        n_val = len(train_dataset_full) - n_train
        # J6 (2026-04-16): local Generator so we don't overwrite the global
        # torch RNG set by set_deterministic() upstream.
        g = torch.Generator()
        g.manual_seed(seed)
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset_full, [n_train, n_val], generator=g
        )
    
    # Create loaders with optimized settings
    loader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs
    )
    
    return train_loader, val_loader, test_loader, normalizer


def train_single_experiment(
    x_train: np.ndarray,
    y_train: np.ndarray,
    dydx_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    dydx_test: np.ndarray,
    lambda_: float = 1.0,
    n_epochs: int = 200,
    batch_size: int = 256,
    n_layers: int = 4,
    hidden_size: int = 256,
    lr: float = 0.005,
    activation: str = "softplus",
    seed: int = 42,
    pbar: bool = True,
    method: str = "dml_fixed",
    max_grad_norm: float = 1.0,
    scheduler_patience: int = 20,
    scheduler_factor: float = 0.5,
    x_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    dydx_val: Optional[np.ndarray] = None,
    dydx_train_mask: Optional[np.ndarray] = None,
    dydx_val_mask: Optional[np.ndarray] = None,
    dydx_test_mask: Optional[np.ndarray] = None,
    weight_scheme: str = "hs",
    warmup_fraction: float = 0.5,
    task_weights_log_path: Optional[str] = None,
    balancer_kwargs: Optional[dict] = None,
    early_stopping_patience: int = 50,
    **kwargs,
) -> 'TrainingResult':
    """
    Run a single training experiment.
    
    Args:
        x_train, y_train, dydx_train: Training data
        x_test, y_test, dydx_test: Test data
        lambda_: DML weight (only used for method='dml_fixed')
        n_epochs: Number of training epochs
        batch_size: Batch size
        n_layers: Number of hidden layers
        hidden_size: Hidden layer dimension
        lr: Learning rate
        activation: Activation function
        seed: Random seed
        pbar: Show progress bar
        method: One of 'vanilla', 'dml_fixed', 'dml_gradnorm', 'dml_relobralo'
        max_grad_norm: Max gradient norm for clipping (0 = disabled)
        scheduler_patience: ReduceLROnPlateau patience 
        scheduler_factor: ReduceLROnPlateau factor
        
    Returns:
        TrainingResult with all metrics
    """
    # Deterministic seeding (Task 0.4)
    set_deterministic(seed)
    
    input_dim = x_train.shape[1]
    
    # Create data loaders
    train_loader, val_loader, test_loader, normalizer = create_data_loaders(
        x_train, y_train, dydx_train,
        x_test, y_test, dydx_test,
        batch_size=batch_size,
        seed=seed,
        x_val=x_val, y_val=y_val, dydx_val=dydx_val,
        dydx_train_mask=dydx_train_mask, dydx_val_mask=dydx_val_mask, 
        dydx_test_mask=dydx_test_mask,
    )
    
    # Create model
    model = DmlFeedForward(
        input_dim=input_dim,
        output_dim=1,
        n_layers=n_layers,
        hidden_size=hidden_size,
        activation=activation
    )
    
    # Create loss based on method (Task 0.14)
    if method == "vanilla":
        loss_fn = VanillaLoss()
        use_dml = False
    elif method == "dml_fixed":
        # H&S 2018 weights: 1/(1+λd), λd/(1+λd) — see EVIDENCE/hs2018_formula_check.md
        loss_fn = DmlLoss(
            lambda_=lambda_,
            input_dim=input_dim,
            lambda_j=normalizer.lambda_j,
            weight_scheme=weight_scheme,
        )
        use_dml = True
    elif method == "dml_fixed_half":
        # V1/P7 (2026-04-13): force-balanced 0.5/0.5 convex weights regardless of d.
        # For molecular cross-arch parity with GATv2/PaiNN; for synthetic A/B vs HS.
        loss_fn = DmlLoss(
            lambda_=lambda_,
            input_dim=input_dim,
            lambda_j=normalizer.lambda_j,
            weight_scheme="half",
        )
        use_dml = True
    elif method == "dml_gradnorm":
        # J4 (2026-04-16): pass explicit shared_layer_name for MLP. The
        # fallback heuristic (weight_params[-2]) happens to be correct for
        # a flat MLP (last hidden linear) but is silently fragile if a
        # future MLP variant adds normalization or a second head.
        # #197 (2026-05-03): wire balancer_kwargs (alpha, gradnorm_lr) through
        # with whitelist filtering. Default empty dict → byte-identical.
        from .loss_balancing import GradNormDmlLoss
        bk = dict(balancer_kwargs or {})
        loss_fn = GradNormDmlLoss(
            input_dim=input_dim,
            shared_layer_name=f"layers.{n_layers - 1}.weight",
            **{k: v for k, v in bk.items() if k in {"alpha", "gradnorm_lr"}},
        )
        use_dml = True
    elif method == "dml_dimnorm_gradnorm":
        from .loss_balancing import DimNormGradNormDmlLoss
        loss_fn = DimNormGradNormDmlLoss(input_dim=input_dim, dim_norm_mode="d")
        use_dml = True
    elif method == "dml_sqrtdimnorm_gradnorm":
        from .loss_balancing import DimNormGradNormDmlLoss
        loss_fn = DimNormGradNormDmlLoss(input_dim=input_dim, dim_norm_mode="sqrt_d")
        use_dml = True
    elif method == "dml_softmax_balance":
        # Was misnamed `dml_relobralo` in v2; 2026-04-13 rename. Simplified
        # softmax-of-loss-ratio with EMA. See EVIDENCE/DEVIATIONS_FROM_CANONICAL.md.
        # M-C-R2-1 (2026-04-14): seed threaded from outer call so stochastic
        # reference-step draw varies across experimental seeds.
        # #197 (2026-05-03): wire balancer_kwargs (tau, rho, max_history).
        from .loss_balancing import SoftmaxBalanceDmlLoss
        bk = dict(balancer_kwargs or {})
        loss_fn = SoftmaxBalanceDmlLoss(
            input_dim=input_dim, seed=seed,
            **{k: v for k, v in bk.items() if k in {"tau", "rho", "max_history"}},
        )
        use_dml = True
    elif method == "dml_relobralo":
        # 2026-04-13: now refers to the FAITHFUL Bischof & Kraus 2022 Eq.11
        # (Bernoulli ρ saudade + exponential α decay). For the SIMPLIFIED v2
        # version, use 'dml_softmax_balance'.
        # M-C-R2-1 (2026-04-14): seed threaded so saudade Bernoulli RNG varies.
        # #197 (2026-05-03): wire balancer_kwargs (tau, rho, max_history).
        from .loss_balancing import ReLoBRaLoDmlLoss
        bk = dict(balancer_kwargs or {})
        loss_fn = ReLoBRaLoDmlLoss(
            input_dim=input_dim, seed=seed,
            **{k: v for k, v in bk.items() if k in {"tau", "rho", "max_history"}},
        )
        use_dml = True
    elif method == "dml_warmup":
        # L-H1 proper fix (2026-04-16): dml_warmup is a two-phase method that
        # requires the train_warmup wrapper. Delegate from here so that ANY
        # caller of train_single_experiment (including run_full_benchmark.py)
        # can use method="dml_warmup" without special-casing.
        from experiments.unified_comparison.run_unified_experiment import train_warmup
        return train_warmup(
            x_train=x_train, y_train=y_train, dydx_train=dydx_train,
            x_test=x_test, y_test=y_test, dydx_test=dydx_test,
            x_val=x_val, y_val=y_val, dydx_val=dydx_val,
            warmup_fraction=warmup_fraction,
            seed=seed, pbar=pbar,
            n_epochs=n_epochs, batch_size=batch_size, lr=lr,
            n_layers=n_layers, hidden_size=hidden_size, activation=activation,
            max_grad_norm=max_grad_norm,
            scheduler_patience=scheduler_patience,
            scheduler_factor=scheduler_factor,
        )
    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Use 'vanilla', 'dml_fixed', 'dml_fixed_half', 'dml_gradnorm', "
            f"'dml_dimnorm_gradnorm', 'dml_sqrtdimnorm_gradnorm', "
            f"'dml_softmax_balance', 'dml_relobralo', or 'dml_warmup'."
        )
    
    # Create optimizer
    # V4 (2026-04-13): switched from Adam → AdamW for cross-architecture
    # consistency with GATv2/PaiNN molecular runners. weight_decay=0 matches
    # canonical SchNetPack v2.2.0 (configs/task/optimizer/adam.yaml).
    # Behavior with weight_decay=0 is equivalent to plain Adam, so v2 results
    # are not affected numerically; this is for protocol consistency.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    
    # Create LR scheduler (Task 0.2)
    # TODO: After Tier 2 completes, implement LR warmup for high-dim experiments.
    # Options: (a) Linear warmup: lr=1e-4 → 5e-3 over 50 epochs, then ReduceLROnPlateau
    #          (b) CosineAnnealingWarmRestarts with T_0=100
    #          (c) OneCycleLR with max_lr=0.01
    # High-dim (d=50,100) may benefit most. Run ablation to confirm.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=scheduler_factor,
        patience=scheduler_patience, min_lr=1e-6
    )
    
    # Create trainer with scheduler and gradient clipping
    trainer = DmlTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        normalizer=normalizer,
        scheduler=scheduler,
        use_dml=use_dml,
        max_grad_norm=max_grad_norm,
        task_weights_log_path=task_weights_log_path,
    )
    
    # Config for logging — includes hardware metadata (Task 0.5)
    config = {
        'method': method,
        'lambda': lambda_,
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'n_layers': n_layers,
        'hidden_size': hidden_size,
        'lr': lr,
        'activation': activation,
        'seed': seed,
        'n_train': len(x_train),
        'n_test': len(x_test),
        'input_dim': input_dim,
        'max_grad_norm': max_grad_norm,
        'scheduler_patience': scheduler_patience,
        'scheduler_factor': scheduler_factor,
        'run_metadata': get_run_metadata()
    }
    
    # Train
    # early_stopping_patience: backward-compat default 50 matches previous behaviour
    # exactly. Existing callers that don't pass it get identical results to before
    # this kwarg was added (2026-05-04).
    result = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
        config=config,
        pbar=pbar,
        early_stopping_patience=early_stopping_patience,
    )
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(test_loader)
    result.test_value_mse = test_metrics['value_mse']
    result.test_grad_mse = test_metrics['grad_mse']
    
    return result
