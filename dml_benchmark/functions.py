"""
Function generators for DML Benchmark Suite.

Provides synthetic test functions with exact gradients using JAX autodiff.
Includes derivative corruption utilities for Gap C (imperfect derivatives).
"""

import numpy as np
from typing import Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass
import warnings

try:
    import jax
    import jax.numpy as jnp
    from jax import random, grad, vmap, jacrev
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    warnings.warn(
        "JAX not available. NumPy fallbacks will be used for trig functions. "
        "JAX is preferred for guaranteed exact gradients.",
        UserWarning
    )

# Import scipy at module level for better error handling (Issue #11)
try:
    from scipy.stats import norm as scipy_norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn(
        "Scipy not available. Finance functions (Bachelier, Black-Scholes, Heston) will not work.",
        UserWarning
    )


# ============================================================================
# DATA TYPES
# ============================================================================

@dataclass
class FunctionData:
    """Container for function evaluation data."""
    x: np.ndarray          # Input points (n_samples, n_dim)
    y: np.ndarray          # Function values (n_samples, 1)
    dydx: np.ndarray       # Gradients (n_samples, 1, n_dim) or (n_samples, n_dim)
    function_type: str     # Identifier for this function
    config: Dict[str, Any] # Function-specific parameters


# ============================================================================
# DERIVATIVE CORRUPTION UTILITIES (GAP C)
# ============================================================================

def add_gaussian_noise(dydx: np.ndarray, noise_level: float, seed: int = 42) -> np.ndarray:
    """
    Add Gaussian noise to derivatives.
    
    Noise is scaled relative to std(dydx). When derivatives are near-zero
    (e.g. step functions where dydx=0 a.e.), a fallback scale of 1.0 is
    used so that noise_level still injects meaningful perturbation.
    
    Args:
        dydx: Original gradients
        noise_level: Noise as fraction of std(dydx)  (or absolute if std≈0)
        seed: Random seed for reproducibility
        
    Returns:
        Noisy gradients
    """
    if noise_level <= 0:
        return dydx.copy()
    
    rng = np.random.RandomState(seed)  # Use RandomState for isolation
    ref_std = np.std(dydx)
    if ref_std < 1e-10:
        # Derivatives are effectively zero (e.g. step function).
        # Fall back to absolute noise so noise_level still has an effect.
        ref_std = 1.0
    noise_std = noise_level * ref_std
    noise = rng.normal(0, noise_std, dydx.shape)
    return dydx + noise


def add_bias(dydx: np.ndarray, scale: float = 1.0, offset: float = 0.0) -> np.ndarray:
    """
    Add systematic bias to derivatives.
    
    Args:
        dydx: Original gradients
        scale: Multiplicative bias
        offset: Additive bias
        
    Returns:
        Biased gradients
    """
    return dydx * scale + offset


def finite_difference_derivatives(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    h: float = 1e-5
) -> np.ndarray:
    """
    Compute derivatives via central finite differences.
    
    Args:
        f: Function to differentiate (vectorized)
        x: Input points (n_samples, n_dim)
        h: Step size
        
    Returns:
        Approximate gradients (n_samples, n_dim)
    """
    n_samples, n_dim = x.shape
    dydx = np.zeros((n_samples, n_dim))
    
    for j in range(n_dim):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[:, j] += h
        x_minus[:, j] -= h
        dydx[:, j] = (f(x_plus) - f(x_minus)).flatten() / (2 * h)
    
    return dydx


def corrupt_derivatives(
    dydx: np.ndarray,
    noise_level: float = 0.0,
    bias_scale: float = 1.0,
    bias_offset: float = 0.0,
    seed: int = 42
) -> np.ndarray:
    """
    Apply corruption to derivatives (combined noise + bias).
    
    Args:
        dydx: Original gradients
        noise_level: Gaussian noise as fraction of std
        bias_scale: Multiplicative bias
        bias_offset: Additive bias
        seed: Random seed
        
    Returns:
        Corrupted gradients
    """
    result = add_bias(dydx, bias_scale, bias_offset)
    result = add_gaussian_noise(result, noise_level, seed)
    return result


# ============================================================================
# FUNCTION GENERATORS (JAX-BASED)
# ============================================================================

class BenchmarkFunctionGenerator:
    """
    Unified function generator for DML benchmark.
    
    Generates synthetic functions with exact gradients.
    """
    
    def __init__(self, n_dim: int, seed: int = 42):
        """
        Initialize generator.
        
        Args:
            n_dim: Input dimension
            seed: Random seed
        """
        self.n_dim = n_dim
        self.seed = seed
        if JAX_AVAILABLE:
            self.key = random.PRNGKey(seed)
        
    def _split_key(self):
        """Split JAX random key."""
        if JAX_AVAILABLE:
            self.key, subkey = random.split(self.key)
            return subkey
        return None
    
    def _generate_random_x(self, n_samples: int, low: float = -1.0, high: float = 1.0) -> np.ndarray:
        """Generate random input points."""
        if JAX_AVAILABLE:
            return np.asarray(random.uniform(
                self._split_key(),
                shape=(n_samples, self.n_dim),
                minval=low,
                maxval=high
            ))
        else:
            # J5 (2026-04-16): use RandomState to avoid polluting the global
            # numpy RNG. Prior `np.random.seed(self.seed)` stomped caller state.
            rng = np.random.RandomState(self.seed)
            return rng.uniform(low, high, (n_samples, self.n_dim))
    
    # -------------------------------------------------------------------------
    # TRIGONOMETRIC FUNCTION
    # -------------------------------------------------------------------------
    
    def generate_trigonometric(
        self,
        n_samples: int,
        frequencies: Optional[np.ndarray] = None,
        amplitudes: Optional[np.ndarray] = None
    ) -> FunctionData:
        """
        Generate trigonometric function: y = sum_i(a_i * sin(w_i * x_i))
        
        Uses JAX if available for exact autodiff gradients.
        Falls back to analytical NumPy implementation if JAX unavailable.
        
        Args:
            n_samples: Number of samples
            frequencies: Per-dimension frequencies (default: random)
            amplitudes: Per-dimension amplitudes (default: random)
            
        Returns:
            FunctionData with x, y, dydx
        """
        # Use seeded RandomState for reproducibility (Issue #9)
        rng = np.random.RandomState(self.seed)
        
        # Generate random parameters if not provided
        if frequencies is None:
            if JAX_AVAILABLE:
                frequencies = np.asarray(random.uniform(
                    self._split_key(), shape=(self.n_dim,), minval=0.5, maxval=5.0
                ))
            else:
                frequencies = rng.uniform(0.5, 5.0, size=self.n_dim)
        if amplitudes is None:
            if JAX_AVAILABLE:
                amplitudes = np.asarray(random.uniform(
                    self._split_key(), shape=(self.n_dim,), minval=0.5, maxval=2.0
                ))
            else:
                amplitudes = rng.uniform(0.5, 2.0, size=self.n_dim)
        
        frequencies = np.asarray(frequencies)
        amplitudes = np.asarray(amplitudes)
        
        # Generate input data
        x = self._generate_random_x(n_samples, low=-np.pi, high=np.pi)
        
        if JAX_AVAILABLE:
            # Use JAX for exact autodiff
            freq_jax = jnp.array(frequencies)
            amp_jax = jnp.array(amplitudes)
            
            def f(x):
                return jnp.sum(amp_jax * jnp.sin(freq_jax * x))
            
            f_vec = vmap(f)
            grad_f = vmap(grad(f))
            
            x_jax = jnp.array(x)
            y = np.asarray(f_vec(x_jax)).reshape(-1, 1)
            dydx = np.asarray(grad_f(x_jax))
        else:
            # NumPy fallback with ANALYTICAL gradients (Issue #4)
            # y = sum_i(a_i * sin(w_i * x_i))
            # dy/dx_i = a_i * w_i * cos(w_i * x_i)
            warnings.warn(
                "Using NumPy fallback for trigonometric function. "
                "JAX is preferred for guaranteed gradient correctness.",
                UserWarning
            )
            
            # Compute y: (n_samples,)
            y = np.sum(amplitudes * np.sin(frequencies * x), axis=1).reshape(-1, 1)
            
            # Compute analytical gradients: dy/dx_i = a_i * w_i * cos(w_i * x_i)
            dydx = amplitudes * frequencies * np.cos(frequencies * x)
        
        # Reshape for DML format: (n_samples, 1, n_dim)
        if len(dydx.shape) == 2:
            dydx = dydx.reshape(n_samples, 1, self.n_dim)
        
        return FunctionData(
            x=x, y=y, dydx=dydx,
            function_type="trig",
            config={"frequencies": frequencies.tolist(), "amplitudes": amplitudes.tolist()}
        )
    
    # -------------------------------------------------------------------------
    # POLYNOMIAL + TRIGONOMETRIC
    # -------------------------------------------------------------------------
    
    def generate_poly_trig(
        self,
        n_samples: int,
        poly_degree: int = 3,
        alpha: float = 0.5,
        frequency: float = 2.0
    ) -> FunctionData:
        """
        Generate polynomial + trigonometric function.
        
        y = sum_i(poly(x_i)) + alpha * sum_i(sin(frequency * x_i))
        
        Args:
            n_samples: Number of samples
            poly_degree: Polynomial degree
            alpha: Weight of trigonometric term
            frequency: Sin frequency
            
        Returns:
            FunctionData with x, y, dydx
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX required for poly_trig generator")
        
        # Random polynomial coefficients
        coeffs = np.asarray(random.uniform(
            self._split_key(),
            shape=(self.n_dim, poly_degree + 1),
            minval=-1, maxval=1
        # Decay coefficients geometrically (0.9^k) to prevent high-degree polynomial
        # terms from dominating — keeps the function well-conditioned for learning.
        )) * np.array([0.9 ** i for i in range(poly_degree + 1)])
        coeffs = jnp.array(coeffs)
        
        def f(x):
            # Polynomial term
            poly = 0.0
            for j in range(self.n_dim):
                for k in range(poly_degree + 1):
                    poly = poly + coeffs[j, k] * (x[j] ** k)
            # Trig term
            trig = alpha * jnp.sum(jnp.sin(frequency * x))
            return poly + trig
        
        f_vec = vmap(f)
        grad_f = vmap(grad(f))
        
        x = self._generate_random_x(n_samples, low=-1.0, high=1.0)
        x_jax = jnp.array(x)
        
        y = np.asarray(f_vec(x_jax)).reshape(-1, 1)
        dydx = np.asarray(grad_f(x_jax)).reshape(n_samples, 1, self.n_dim)
        
        return FunctionData(
            x=x, y=y, dydx=dydx,
            function_type="poly_trig",
            config={"poly_degree": poly_degree, "alpha": alpha, "frequency": frequency}
        )
    
    # -------------------------------------------------------------------------
    # STEP FUNCTION (DISCONTINUOUS - FOR GAP C)
    # -------------------------------------------------------------------------
    
    def generate_step(
        self,
        n_samples: int,
        n_steps: int = 5
    ) -> FunctionData:
        """
        Generate step function (piecewise constant).
        
        Note: Derivatives are zero almost everywhere (not useful for DML).
        This is for testing failure cases.
        
        Args:
            n_samples: Number of samples
            n_steps: Number of step discontinuities per dimension
            
        Returns:
            FunctionData with x, y, dydx (dydx = 0)
        """
        rng = np.random.RandomState(self.seed)  # Use RandomState for isolation
        
        x = self._generate_random_x(n_samples, low=0.0, high=1.0)
        
        # Random step locations and values
        step_points = np.sort(rng.uniform(0, 1, (self.n_dim, n_steps)), axis=1)
        step_values = rng.uniform(0, 1, (self.n_dim, n_steps))
        
        # Compute y as sum of step contributions from each dimension
        y = np.zeros(n_samples)
        for i in range(n_samples):
            for j in range(self.n_dim):
                # Find which step bin this point falls into
                bin_idx = np.searchsorted(step_points[j], x[i, j])
                if bin_idx > 0:
                    y[i] += step_values[j, bin_idx - 1]
        
        y = y.reshape(-1, 1)
        dydx = np.zeros((n_samples, 1, self.n_dim))  # Zero derivatives
        
        return FunctionData(
            x=x, y=y, dydx=dydx,
            function_type="step",
            config={"n_steps": n_steps}
        )


# ============================================================================
# FINANCE FUNCTIONS (SEPARATE GROUP)
# ============================================================================

class FinanceFunctionGenerator:
    """
    Finance-specific function generators.
    
    Implements Bachelier, Black-Scholes, and Heston model payoffs.
    """
    
    def __init__(self, n_assets: int = 1, seed: int = 42):
        """
        Initialize generator.
        
        Args:
            n_assets: Number of underlying assets (dimension)
            seed: Random seed
        """
        self.n_assets = n_assets
        self.seed = seed
    
    def generate_bachelier(
        self,
        n_samples: int,
        strike: float = 1.0,
        vol: float = 0.2,
        T: float = 1.0
    ) -> FunctionData:
        """
        Generate Bachelier basket option data.
        
        Bachelier model: dS = sigma * dW (arithmetic Brownian motion)
        Payoff: max(sum(S_i)/n - K, 0)
        
        Args:
            n_samples: Number of samples
            strike: Strike price
            vol: Volatility
            T: Time to maturity
            
        Returns:
            FunctionData with spots, prices, deltas
        """
        if not SCIPY_AVAILABLE:
            raise RuntimeError("Scipy required for Bachelier generator")
        
        # Use seeded RandomState for reproducibility (Issue #9)
        rng = np.random.RandomState(self.seed)
        
        # Generate spot prices in [0.5K, 1.5K] — covers deep ITM to deep OTM,
        # spanning the moneyness region where derivative (delta) is most informative.
        spots = rng.uniform(
            strike * 0.5, strike * 1.5, 
            (n_samples, self.n_assets)
        )
        
        # Basket average
        basket_avg = np.mean(spots, axis=1)
        
        # Bachelier call price and delta
        d = (basket_avg - strike) / (vol * np.sqrt(T))
        prices = (basket_avg - strike) * scipy_norm.cdf(d) + vol * np.sqrt(T) * scipy_norm.pdf(d)
        prices = np.maximum(prices, 0).reshape(-1, 1)
        
        # Delta: d(price)/d(spot_i) = (1/n) * N(d) for each asset
        delta_unit = scipy_norm.cdf(d) / self.n_assets
        deltas = np.tile(delta_unit.reshape(-1, 1), (1, self.n_assets))
        deltas = deltas.reshape(n_samples, 1, self.n_assets)
        
        return FunctionData(
            x=spots, y=prices, dydx=deltas,
            function_type="bachelier",
            config={"strike": strike, "vol": vol, "T": T, "n_assets": self.n_assets}
        )
    
    def generate_black_scholes(
        self,
        n_samples: int,
        strike: float = 100.0,
        vol: float = 0.2,
        r: float = 0.05,
        T: float = 1.0
    ) -> FunctionData:
        """
        Generate Black-Scholes call option data (single asset).
        
        Args:
            n_samples: Number of samples
            strike: Strike price
            vol: Implied volatility
            r: Risk-free rate
            T: Time to maturity
            
        Returns:
            FunctionData with spots, prices, deltas
        """
        if self.n_assets != 1:
            raise ValueError("Black-Scholes is single-asset. Use n_assets=1.")
        if not SCIPY_AVAILABLE:
            raise RuntimeError("Scipy required for Black-Scholes generator")
        
        # Use seeded RandomState for reproducibility (Issue #9)
        rng = np.random.RandomState(self.seed)
        
        # Spot prices in [0.5K, 1.5K] — same moneyness range as Bachelier.
        spots = rng.uniform(
            strike * 0.5, strike * 1.5,
            (n_samples, 1)
        )
        
        S = spots.flatten()
        K = strike
        
        # Black-Scholes formulas
        d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        
        prices = S * scipy_norm.cdf(d1) - K * np.exp(-r * T) * scipy_norm.cdf(d2)
        deltas = scipy_norm.cdf(d1)
        
        return FunctionData(
            x=spots,
            y=prices.reshape(-1, 1),
            dydx=deltas.reshape(n_samples, 1, 1),
            function_type="black_scholes",
            config={"strike": strike, "vol": vol, "r": r, "T": T}
        )
    
    def generate_heston(
        self,
        n_samples: int,
        strike: float = 100.0,
        v0: float = 0.04,
        kappa: float = 2.0,
        theta: float = 0.04,
        sigma_v: float = 0.3,
        rho: float = -0.7,
        r: float = 0.05,
        T: float = 1.0
    ) -> FunctionData:
        """
        Generate Heston stochastic volatility option data.
        
        Heston SDE:
            dS_t = r*S_t*dt + sqrt(v_t)*S_t*dW_1
            dv_t = kappa*(theta - v_t)*dt + sigma_v*sqrt(v_t)*dW_2
            Corr(dW_1, dW_2) = rho
        
        Uses Monte Carlo for pricing and finite-difference for Greeks.
        This is computationally more expensive than BS but provides
        realistic stochastic volatility dynamics.
        
        Args:
            n_samples: Number of input points (S0, v0 pairs)
            strike: Strike price
            v0: Initial variance
            kappa: Mean reversion speed
            theta: Long-term variance
            sigma_v: Volatility of volatility
            rho: Spot-vol correlation
            r: Risk-free rate
            T: Time to maturity
            
        Returns:
            FunctionData with (S0, v0) inputs, prices, and deltas
        """
        if self.n_assets != 1:
            raise ValueError("Heston is single-asset. Use n_assets=1.")
        
        # Use seeded RandomState for reproducibility
        rng = np.random.RandomState(self.seed)
        
        # Generate (S0, v_initial) pairs as 2D input.
        # S0 in [0.7K, 1.3K]: narrower range than BS/Bachelier because MC is expensive.
        # v in [0.5*v0, 1.5*v0]: samples stochastic volatility around nominal.
        S0_samples = rng.uniform(strike * 0.7, strike * 1.3, n_samples)
        v_samples = rng.uniform(v0 * 0.5, v0 * 1.5, n_samples)
        
        inputs = np.column_stack([S0_samples, v_samples])  # (n_samples, 2)
        
        # Monte Carlo settings — reduced for speed vs. production accuracy.
        # n_paths=1000 gives ~3% MC standard error on vanilla put prices.
        # n_steps=50 is adequate for Euler-Maruyama with dt≈0.02.
        # For publication-quality pricing, use n_paths≥10,000, n_steps≥200.
        n_paths = 1000
        n_steps = 50
        dt = T / n_steps
        
        prices = np.zeros(n_samples)
        deltas = np.zeros((n_samples, 2))  # delta_S and delta_v (vega-like)
        
        # Finite difference step for MC Greeks (bump-and-reprice).
        # h=0.01 ≈ 1% of typical S0=100 scale; balances truncation vs MC noise.
        #
        # NOTE: We intentionally do NOT use Common Random Numbers (CRN) for the
        # bump-and-reprice derivative estimates. Each call to _heston_mc_price
        # draws fresh random paths from `rng`, so base, bump-up, and bump-down
        # evaluations use independent path sets. This represents the worst-case
        # scenario for finite-difference derivative estimation, where MC path
        # noise is not shared between evaluations and inflates Greek variance.
        # For variance-reduced Greeks, one would fix the RNG state before each
        # call (CRN) — see Glasserman (2003), Chapter 7. # TODO: check this
        h = 0.01
        
        for i in range(n_samples):
            S0 = S0_samples[i]
            v_init = v_samples[i]
            
            # Price at (S0, v_init)
            prices[i] = self._heston_mc_price(
                S0, v_init, strike, kappa, theta, sigma_v, rho, r, T,
                n_paths, n_steps, dt, rng
            )
            
            # Delta_S via finite difference
            price_up = self._heston_mc_price(
                S0 * (1 + h), v_init, strike, kappa, theta, sigma_v, rho, r, T,
                n_paths, n_steps, dt, rng
            )
            price_down = self._heston_mc_price(
                S0 * (1 - h), v_init, strike, kappa, theta, sigma_v, rho, r, T,
                n_paths, n_steps, dt, rng
            )
            deltas[i, 0] = (price_up - price_down) / (2 * h * S0)
            
            # Delta_v (sensitivity to initial variance)
            price_v_up = self._heston_mc_price(
                S0, v_init * (1 + h), strike, kappa, theta, sigma_v, rho, r, T,
                n_paths, n_steps, dt, rng
            )
            price_v_down = self._heston_mc_price(
                S0, v_init * (1 - h), strike, kappa, theta, sigma_v, rho, r, T,
                n_paths, n_steps, dt, rng
            )
            deltas[i, 1] = (price_v_up - price_v_down) / (2 * h * v_init)
        
        return FunctionData(
            x=inputs,
            y=prices.reshape(-1, 1),
            dydx=deltas.reshape(n_samples, 1, 2),
            function_type="heston",
            config={
                "strike": strike, "v0": v0, "kappa": kappa, "theta": theta,
                "sigma_v": sigma_v, "rho": rho, "r": r, "T": T
            }
        )
    
    def _heston_mc_price(
        self,
        S0: float,
        v0: float,
        K: float,
        kappa: float,
        theta: float,
        sigma_v: float,
        rho: float,
        r: float,
        T: float,
        n_paths: int,
        n_steps: int,
        dt: float,
        rng
    ) -> float:
        """
        Monte Carlo price using Euler-Maruyama discretization of Heston model.
        Uses variance reflection (floor at zero) for stability.
        """
        sqrt_dt = np.sqrt(dt)
        
        S = np.ones(n_paths) * S0
        v = np.ones(n_paths) * v0
        
        for _ in range(n_steps):
            # Correlated Brownian motions
            Z1 = rng.normal(0, 1, n_paths)
            Z2 = rng.normal(0, 1, n_paths)
            W1 = Z1
            W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2
            
            # Truncated variance (floor at 0)
            v_plus = np.maximum(v, 0)
            sqrt_v = np.sqrt(v_plus)
            
            # Euler step for S
            S = S * np.exp((r - 0.5 * v_plus) * dt + sqrt_v * sqrt_dt * W1)
            
            # Euler step for v (with reflection)
            v = v + kappa * (theta - v_plus) * dt + sigma_v * sqrt_v * sqrt_dt * W2
            v = np.maximum(v, 0)  # Reflection scheme
        
        # European call payoff
        payoffs = np.maximum(S - K, 0)
        price = np.exp(-r * T) * np.mean(payoffs)
        
        return price


# ============================================================================
# UNIFIED GENERATOR INTERFACE
# ============================================================================

def generate_data(
    function_type: str,
    n_dim: int,
    n_samples: int,
    seed: int = 42,
    **kwargs
) -> FunctionData:
    """
    Unified interface for generating benchmark data.
    
    Args:
        function_type: One of trig, poly_trig, step, bachelier, black_scholes, heston
        n_dim: Input dimension
        n_samples: Number of samples
        seed: Random seed
        **kwargs: Function-specific parameters
        
    Returns:
        FunctionData object
    """
    if function_type == "trig":
        gen = BenchmarkFunctionGenerator(n_dim, seed)
        return gen.generate_trigonometric(n_samples, **kwargs)
    
    elif function_type == "poly_trig":
        gen = BenchmarkFunctionGenerator(n_dim, seed)
        return gen.generate_poly_trig(n_samples, **kwargs)
    
    elif function_type == "step":
        gen = BenchmarkFunctionGenerator(n_dim, seed)
        return gen.generate_step(n_samples, **kwargs)
    
    elif function_type == "bachelier":
        gen = FinanceFunctionGenerator(n_dim, seed)
        return gen.generate_bachelier(n_samples, **kwargs)
    
    elif function_type == "black_scholes":
        if n_dim != 1:
            raise ValueError("Black-Scholes requires n_dim=1")
        gen = FinanceFunctionGenerator(n_dim, seed)
        return gen.generate_black_scholes(n_samples, **kwargs)
    
    elif function_type == "heston":
        if n_dim != 1:
            raise ValueError("Heston requires n_dim=1 (2D input generated)")
        gen = FinanceFunctionGenerator(1, seed)
        return gen.generate_heston(n_samples, **kwargs)
    
    else:
        raise ValueError(
            f"Unknown function type: {function_type}. "
            "Valid: trig, poly_trig, step, bachelier, black_scholes, heston"
        )


# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================

def train_test_split(
    data: FunctionData,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[FunctionData, FunctionData]:
    """
    Split FunctionData into train and test sets.
    
    Args:
        data: FunctionData to split
        train_ratio: Fraction for training
        seed: Random seed for shuffling
        
    Returns:
        (train_data, test_data)
    """
    rng = np.random.RandomState(seed)  # Use RandomState for isolation
    n = len(data.x)
    indices = rng.permutation(n)
    
    split_idx = int(n * train_ratio)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    train_data = FunctionData(
        x=data.x[train_idx],
        y=data.y[train_idx],
        dydx=data.dydx[train_idx],
        function_type=data.function_type,
        config=data.config
    )
    
    test_data = FunctionData(
        x=data.x[test_idx],
        y=data.y[test_idx],
        dydx=data.dydx[test_idx],
        function_type=data.function_type,
        config=data.config
    )

    return train_data, test_data


# ============================================================================
# EXTRAPOLATION SPLIT (Appendix H — periodic extrapolation experiments)
# ============================================================================

def extrapolation_split(
    data: FunctionData,
    mode: str = "halfspace",
    halfspace_axis: int = 0,
    halfspace_threshold: float = 0.0,
    radial_threshold: Optional[float] = None,
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
    seed: int = 42,
) -> Tuple[FunctionData, FunctionData]:
    """
    Spatially split FunctionData into train (in-support) and test (out-of-support).

    Distinct from train_test_split (random permutation = interpolation) — this
    forces test points outside the training-data convex region.

    Modes:
      - "halfspace": train where x[:, halfspace_axis] >= halfspace_threshold,
                     test where x[:, halfspace_axis] < halfspace_threshold.
                     Volume-symmetric for any d when threshold is 0 and the
                     domain is symmetric.
      - "radial":    train where ‖x‖_∞ ≤ radial_threshold,
                     test where ‖x‖_∞ > radial_threshold.
                     If radial_threshold is None, picks
                       r = x_max · 0.5^(1/d)
                     so that the train region has 50% of cube volume — making
                     halfspace and radial directly comparable.

    Caller is responsible for sample density: at high dimensions with radial
    mode, generate 2-3× more samples than n_train + n_test to handle MC noise
    in the partition fraction.

    Returns: (train_data, test_data) — same FunctionData schema as train_test_split.
    """
    rng = np.random.RandomState(seed)
    x = data.x
    n, d = x.shape

    if mode == "halfspace":
        in_train_mask = x[:, halfspace_axis] >= halfspace_threshold
    elif mode == "radial":
        if radial_threshold is None:
            x_max = float(np.max(np.abs(x)))
            radial_threshold = x_max * (0.5 ** (1.0 / d))
        in_train_mask = np.max(np.abs(x), axis=1) <= radial_threshold
    else:
        raise ValueError(f"Unknown extrapolation_split mode: {mode}. "
                         "Use 'halfspace' or 'radial'.")

    train_idx = np.where(in_train_mask)[0]
    test_idx = np.where(~in_train_mask)[0]

    # shuffle within each region for sample selection
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    if n_train is not None:
        if len(train_idx) < n_train:
            raise RuntimeError(
                f"extrapolation_split mode={mode}: only {len(train_idx)} samples "
                f"in train region but n_train={n_train} requested. "
                f"Caller should oversample (try n_samples >= 3*(n_train+n_test))."
            )
        train_idx = train_idx[:n_train]
    if n_test is not None:
        if len(test_idx) < n_test:
            raise RuntimeError(
                f"extrapolation_split mode={mode}: only {len(test_idx)} samples "
                f"in test region but n_test={n_test} requested."
            )
        test_idx = test_idx[:n_test]

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError(
            f"extrapolation_split with mode={mode} produced empty region: "
            f"train={len(train_idx)}, test={len(test_idx)}."
        )

    split_meta = {
        "split_kind": "extrapolation",
        "split_mode": mode,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }
    if mode == "halfspace":
        split_meta["halfspace_axis"] = int(halfspace_axis)
        split_meta["halfspace_threshold"] = float(halfspace_threshold)
    elif mode == "radial":
        split_meta["radial_threshold"] = float(radial_threshold)

    train_data = FunctionData(
        x=x[train_idx],
        y=data.y[train_idx],
        dydx=data.dydx[train_idx],
        function_type=data.function_type,
        config={**data.config, **split_meta, "split_role": "train"},
    )
    test_data = FunctionData(
        x=x[test_idx],
        y=data.y[test_idx],
        dydx=data.dydx[test_idx],
        function_type=data.function_type,
        config={**data.config, **split_meta, "split_role": "test"},
    )
    return train_data, test_data


def nearest_neighbor_distances(x_test: np.ndarray, x_train: np.ndarray) -> np.ndarray:
    """Per-test-point Euclidean distance to the closest train point.

    Used by the M2 / Appendix-H runner to bin test MSE by extrapolation distance.
    """
    # n_test × n_train pairwise distance matrix; argmin per row.
    diffs = x_test[:, None, :] - x_train[None, :, :]
    d2 = np.sum(diffs ** 2, axis=2)
    return np.sqrt(np.min(d2, axis=1))
