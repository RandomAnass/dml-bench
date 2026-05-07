"""
Hedging backtest for finance appendix.

Compares delta hedging performance using:
- Analytical Black-Scholes delta
- DML-predicted delta  
- Vanilla NN-predicted delta

Metric: std(hedging P&L) as percentage of option notional.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class HedgingResult:
    """Container for hedging backtest results."""
    model_name: str
    mean_pnl: float            # Mean hedging P&L
    std_pnl: float             # Std of hedging P&L (lower = better hedge)
    hedging_error_pct: float   # std(P&L) / option_value * 100
    n_paths: int
    n_rebalances: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "mean_pnl": self.mean_pnl,
            "std_pnl": self.std_pnl,
            "hedging_error_pct": self.hedging_error_pct,
            "n_paths": self.n_paths,
            "n_rebalances": self.n_rebalances
        }


class HedgingBacktest:
    """
    Black-Scholes delta hedging P&L comparison.
    
    Simulates daily delta hedging of a European call option and 
    measures the hedging error (residual P&L variance).
    
    Args:
        spot: Initial spot price
        strike: Option strike price
        vol: Black-Scholes volatility
        r: Risk-free rate
        T: Time to maturity (years)
        n_rebalances: Number of rebalancing dates (252 = daily)
        n_paths: Number of Monte Carlo paths
        seed: Random seed
    """
    
    def __init__(
        self,
        spot: float = 100.0,
        strike: float = 100.0,
        vol: float = 0.20,
        r: float = 0.05,
        T: float = 1.0,
        n_rebalances: int = 252,
        n_paths: int = 10000,
        seed: int = 42
    ):
        self.spot = spot
        self.strike = strike
        self.vol = vol
        self.r = r
        self.T = T
        self.n_rebalances = n_rebalances
        self.n_paths = n_paths
        self.seed = seed
    
    def _simulate_paths(self) -> np.ndarray:
        """
        Simulate GBM stock price paths.
        
        Returns:
            paths: (n_paths, n_rebalances + 1) array of spot prices
        """
        rng = np.random.RandomState(self.seed)
        dt = self.T / self.n_rebalances
        
        paths = np.zeros((self.n_paths, self.n_rebalances + 1))
        paths[:, 0] = self.spot
        
        for t in range(self.n_rebalances):
            z = rng.standard_normal(self.n_paths)
            paths[:, t + 1] = paths[:, t] * np.exp(
                (self.r - 0.5 * self.vol**2) * dt + self.vol * np.sqrt(dt) * z
            )
        
        return paths
    
    @staticmethod
    def _bs_delta(S: np.ndarray, K: float, vol: float, r: float, T: float) -> np.ndarray:
        """Analytical Black-Scholes delta for a European call."""
        from scipy.stats import norm
        
        T = np.maximum(T, 1e-8)  # Prevent division by zero at expiry
        d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        return norm.cdf(d1)
    
    @staticmethod
    def _bs_price(S: float, K: float, vol: float, r: float, T: float) -> float:
        """Analytical Black-Scholes price for a European call."""
        from scipy.stats import norm
        
        if T < 1e-8:
            return max(S - K, 0.0)
        d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    def run_analytical(self) -> HedgingResult:
        """
        Run hedging backtest using analytical BS delta.
        
        This is the best possible hedge under BS assumptions and 
        serves as the lower bound on hedging error.
        """
        paths = self._simulate_paths()
        dt = self.T / self.n_rebalances
        
        pnl = np.zeros(self.n_paths)
        
        for t in range(self.n_rebalances):
            tau = self.T - t * dt  # Time to maturity
            S = paths[:, t]
            
            # Analytical delta
            delta = self._bs_delta(S, self.strike, self.vol, self.r, tau)
            
            # P&L from delta hedge over this period
            dS = paths[:, t + 1] - paths[:, t]
            pnl += delta * dS
        
        # Subtract option payoff
        payoff = np.maximum(paths[:, -1] - self.strike, 0.0)
        hedging_pnl = pnl - payoff
        
        # Discount
        option_value = self._bs_price(self.spot, self.strike, self.vol, self.r, self.T)
        
        return HedgingResult(
            model_name="BS_analytical",
            mean_pnl=float(np.mean(hedging_pnl)),
            std_pnl=float(np.std(hedging_pnl)),
            hedging_error_pct=float(np.std(hedging_pnl) / (option_value + 1e-8) * 100),
            n_paths=self.n_paths,
            n_rebalances=self.n_rebalances
        )
    
    def run_with_model(
        self,
        predict_delta_fn,
        model_name: str = "DML"
    ) -> HedgingResult:
        """
        Run hedging backtest using a model-predicted delta.
        
        Args:
            predict_delta_fn: Function (S, tau) -> delta, where
                S is (n_paths,) spot prices and tau is scalar time-to-maturity.
                Should return (n_paths,) delta predictions.
            model_name: Name for result labeling.
            
        Returns:
            HedgingResult
        """
        paths = self._simulate_paths()
        dt = self.T / self.n_rebalances
        
        pnl = np.zeros(self.n_paths)
        
        for t in range(self.n_rebalances):
            tau = self.T - t * dt
            S = paths[:, t]
            
            # Model-predicted delta
            delta = predict_delta_fn(S, tau)
            
            # P&L from delta hedge
            dS = paths[:, t + 1] - paths[:, t]
            pnl += delta * dS
        
        # Subtract option payoff
        payoff = np.maximum(paths[:, -1] - self.strike, 0.0)
        hedging_pnl = pnl - payoff
        
        option_value = self._bs_price(self.spot, self.strike, self.vol, self.r, self.T)
        
        return HedgingResult(
            model_name=model_name,
            mean_pnl=float(np.mean(hedging_pnl)),
            std_pnl=float(np.std(hedging_pnl)),
            hedging_error_pct=float(np.std(hedging_pnl) / (option_value + 1e-8) * 100),
            n_paths=self.n_paths,
            n_rebalances=self.n_rebalances
        )
    
    def compare_models(
        self,
        model_delta_fns: Dict[str, Any]
    ) -> Dict[str, HedgingResult]:
        """
        Compare multiple models' hedging performance.
        
        Args:
            model_delta_fns: {model_name: predict_delta_fn}
            
        Returns:
            {model_name: HedgingResult}, including analytical baseline
        """
        results = {}
        
        # Analytical baseline
        results["BS_analytical"] = self.run_analytical()
        
        # Each model
        for name, fn in model_delta_fns.items():
            results[name] = self.run_with_model(fn, model_name=name)
        
        return results
