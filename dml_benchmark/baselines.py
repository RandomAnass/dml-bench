"""
Non-neural-network baselines for DML benchmark.

Provides GP, KRR, and RF baselines with a unified interface.
Gradient predictions for baselines use finite differences on the fitted model.

These baselines answer the question: "When do you even need a neural network?"
"""

import numpy as np
from typing import Tuple, Dict, Optional
from abc import ABC, abstractmethod
import time
import warnings


class BaselineModel(ABC):
    """
    Abstract base class for non-NN baseline models.
    
    All baselines implement the same fit/predict interface
    so they can be plugged into the benchmark runner.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.fit_time_s = 0.0
    
    @abstractmethod
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        dydx_train: Optional[np.ndarray] = None
    ) -> None:
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict_values(self, x: np.ndarray) -> np.ndarray:
        """Predict function values. Returns (n_samples, 1)."""
        pass
    
    def predict_gradients(
        self,
        x: np.ndarray,
        h: float = 1e-5
    ) -> np.ndarray:
        """
        Predict gradients via central finite differences on the fitted model.
        
        Args:
            x: Input points (n_samples, n_dim)
            h: Finite difference step size
            
        Returns:
            Gradient predictions (n_samples, 1, n_dim)
            
        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.name} model has not been fitted. Call fit() before predict_gradients()."
            )
        n_samples, n_dim = x.shape
        dydx = np.zeros((n_samples, 1, n_dim))
        
        for j in range(n_dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[:, j] += h
            x_minus[:, j] -= h
            
            y_plus = self.predict_values(x_plus)
            y_minus = self.predict_values(x_minus)
            
            dydx[:, 0, j] = (y_plus[:, 0] - y_minus[:, 0]) / (2 * h)
        
        return dydx
    
    def predict(
        self, 
        x: np.ndarray,
        h: float = 1e-5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict values and gradients.
        
        Returns:
            (y_pred, dydx_pred) where shapes are (n, 1) and (n, 1, d)
        """
        y_pred = self.predict_values(x)
        dydx_pred = self.predict_gradients(x, h=h)
        return y_pred, dydx_pred


# ============================================================================
# GAUSSIAN PROCESS BASELINE
# ============================================================================

class GPBaseline(BaselineModel):
    """
    Gaussian Process Regression baseline with RBF kernel.
    
    Strengths: 
        - Natural uncertainty quantification
        - Optimal for smooth functions with small data
    Limitations:
        - O(n³) training, O(n²) prediction → infeasible for n > ~5000
        - Curse of dimensionality with RBF kernel
        
    Use for: dim ≤ 20, n_samples ≤ 5000
    """
    
    def __init__(self, alpha: float = 1e-6, n_restarts: int = 3):
        super().__init__(name="GP-RBF")
        self.alpha = alpha
        self.n_restarts = n_restarts
        self._model = None
    
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        dydx_train: Optional[np.ndarray] = None
    ) -> None:
        """Fit GP. Ignores derivative information (standard GP doesn't use it)."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        
        # Check feasibility
        n_samples = x_train.shape[0]
        if n_samples > 5000:
            warnings.warn(
                f"GP with {n_samples} samples will be very slow (O(n³)). "
                f"Consider using KRR instead.",
                UserWarning
            )
        
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=self.alpha)
        
        self._model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts,
            normalize_y=True
        )
        
        y = y_train.ravel()
        
        start = time.time()
        self._model.fit(x_train, y)
        self.fit_time_s = time.time() - start
        self.is_fitted = True
    
    def predict_values(self, x: np.ndarray) -> np.ndarray:
        """Predict values. Returns (n_samples, 1)."""
        y = self._model.predict(x)
        return y.reshape(-1, 1)
    
    def predict_with_uncertainty(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict values with uncertainty (std)."""
        y, std = self._model.predict(x, return_std=True)
        return y.reshape(-1, 1), std.reshape(-1, 1)


# ============================================================================
# KERNEL RIDGE REGRESSION BASELINE
# ============================================================================

class KRRBaseline(BaselineModel):
    """
    Kernel Ridge Regression baseline with RBF kernel.
    
    Strengths:
        - O(n³) training but faster constant than GP (no hyperparameter optimization)
        - No uncertainty but often similar accuracy to GP
    Limitations:
        - Same cubic scaling as GP
        - Hyperparameter (alpha, gamma) needs cross-validation
        
    Use for: dim ≤ 50, n_samples ≤ 5000
    """
    
    def __init__(self, alpha: float = 1.0, gamma: Optional[float] = None):
        super().__init__(name="KRR-RBF")
        self.alpha = alpha
        self.gamma = gamma  # None = 1 / (n_features * X.var())
        self._model = None
    
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        dydx_train: Optional[np.ndarray] = None
    ) -> None:
        """Fit KRR. Ignores derivative information."""
        from sklearn.kernel_ridge import KernelRidge
        
        self._model = KernelRidge(
            alpha=self.alpha,
            kernel='rbf',
            gamma=self.gamma
        )
        
        y = y_train.ravel()
        
        start = time.time()
        self._model.fit(x_train, y)
        self.fit_time_s = time.time() - start
        self.is_fitted = True
    
    def predict_values(self, x: np.ndarray) -> np.ndarray:
        y = self._model.predict(x)
        return y.reshape(-1, 1)


# ============================================================================
# RANDOM FOREST BASELINE
# ============================================================================

class RFBaseline(BaselineModel):
    """
    Random Forest Regression baseline.
    
    Strengths:
        - Fast training and prediction
        - Handles high dimensions well
        - No hyperparameter sensitivity
    Limitations:
        - Cannot extrapolate beyond training range
        - Piecewise constant predictions → poor gradient estimation
        
    Use for: Any dimension, any sample size. Poor gradient predictor.
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 5,
        random_state: int = 42
    ):
        super().__init__(name="RF")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self._model = None
    
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        dydx_train: Optional[np.ndarray] = None
    ) -> None:
        """Fit Random Forest. Ignores derivative information."""
        from sklearn.ensemble import RandomForestRegressor
        
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        y = y_train.ravel()
        
        start = time.time()
        self._model.fit(x_train, y)
        self.fit_time_s = time.time() - start
        self.is_fitted = True
    
    def predict_values(self, x: np.ndarray) -> np.ndarray:
        y = self._model.predict(x)
        return y.reshape(-1, 1)


# ============================================================================
# BASELINE RUNNER
# ============================================================================

def get_baseline(name: str, **kwargs) -> BaselineModel:
    """
    Factory function to create a baseline model by name.
    
    Args:
        name: One of 'gp', 'krr', 'rf'
        **kwargs: Passed to the baseline constructor
        
    Returns:
        BaselineModel instance
    """
    baselines = {
        'gp': GPBaseline,
        'krr': KRRBaseline,
        'rf': RFBaseline,
    }
    
    if name not in baselines:
        raise ValueError(
            f"Unknown baseline: {name}. Available: {list(baselines.keys())}"
        )
    
    return baselines[name](**kwargs)


def run_baseline_experiment(
    baseline_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    dydx_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    dydx_test: np.ndarray,
    **kwargs
) -> Dict[str, float]:
    """
    Run a single baseline experiment and return metrics.
    
    Returns:
        Dict with value_mse, grad_mse, training_time_s
    """
    baseline = get_baseline(baseline_name, **kwargs)
    
    # Fit 
    baseline.fit(x_train, y_train, dydx_train)
    
    # Predict
    y_pred, dydx_pred = baseline.predict(x_test)
    
    # Ensure correct shapes for comparison
    y_true = y_test.reshape(-1, 1) if y_test.ndim == 1 else y_test
    if dydx_test.ndim == 2:
        dydx_true = dydx_test.reshape(dydx_test.shape[0], 1, dydx_test.shape[1])
    else:
        dydx_true = dydx_test
    
    # Compute MSE
    value_mse = float(np.mean((y_pred - y_true) ** 2))
    grad_mse = float(np.mean((dydx_pred - dydx_true) ** 2))
    
    return {
        'method': f'baseline_{baseline_name}',
        'value_mse': value_mse,
        'grad_mse': grad_mse,
        'training_time_s': baseline.fit_time_s,
        'baseline_name': baseline.name
    }
