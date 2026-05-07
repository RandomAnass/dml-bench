"""
SPY options data loader with Black-Scholes Greek derivative labels.

Loads real-world SPY EOD options data (2020–2022) and computes analytical
Black-Scholes Greeks as derivative labels for DML training.

Data source: Kaggle SPY EOD options, CC0 license.
Preprocessed into spy_processed.npz with 5D features:
    moneyness (S/K), T (years), r (risk-free), iv (implied vol), log_volume
    + dates (YYYY-MM-DD) for temporal splitting.

The key innovation: real market mid-prices paired with exact analytical
derivatives — best of both worlds for DML experiments. No simulation needed.

Split modes:
    - "temporal": Train on dates before cutoff, test on dates after.
      Default cutoff: 2021-07-01 (~50/50 split). Includes optional embargo
      gap to avoid information leakage from overlapping option contracts.
      This is the standard approach for financial ML (Lopez de Prado, 2018).
    - "random": Legacy random split (deprecated — causes temporal leakage).
      Preserved for backward compatibility and comparison only.
    - "purged_walkforward": Walk-forward CV with temporal purging and embargo.
      Returns multiple (train_idx, test_idx) folds for cross-validation.

Integration:
    Returns data compatible with train_single_experiment():
        x_train:    (n_train, d)       with d = 4 or 5
        y_train:    (n_train, 1)       mid_price
        dydx_train: (n_train, 1, d)    BS Greeks
"""

import warnings
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from scipy.stats import norm as scipy_norm


# ============================================================================
# BLACK-SCHOLES GREEKS (ANALYTICAL)
# ============================================================================

def compute_bs_greeks(
    moneyness: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    iv: np.ndarray,
    include_volume: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Compute Black-Scholes Greeks analytically for call options.

    Given input features (moneyness = S/K, T, r, iv), computes:
        delta = ∂C/∂(S/K) = Φ(d₁)
        vega  = ∂C/∂σ     = (S/K) · φ(d₁) · √T
        theta = ∂C/∂T     = (S/K)·φ(d₁)·σ/(2√T) + r·e^{-rT}·Φ(d₂)
        rho   = ∂C/∂r     = T·e^{-rT}·Φ(d₂)

    Note: prices in spy_processed.npz are normalized (divided by K),
    so the option value is C/K = (S/K)Φ(d₁) − e^{-rT}Φ(d₂).
    Greeks are with respect to the normalized inputs.

    Args:
        moneyness: S/K ratios, shape (n,).
        T: Time to expiry in years, shape (n,).
        r: Risk-free rate, shape (n,).
        iv: Implied volatility, shape (n,).
        include_volume: If True, add a zero column for ∂C/∂log_volume.

    Returns:
        Dictionary with 'delta', 'vega', 'theta', 'rho', and 'greeks' (stacked).
    """
    # Clip to avoid numerical issues
    T_safe = np.maximum(T, 1e-6)
    iv_safe = np.maximum(iv, 1e-6)
    m_safe = np.maximum(moneyness, 1e-8)

    sqrt_T = np.sqrt(T_safe)

    d1 = (np.log(m_safe) + (r + 0.5 * iv_safe ** 2) * T_safe) / (iv_safe * sqrt_T)
    d2 = d1 - iv_safe * sqrt_T

    # Standard normal CDF and PDF
    Nd1 = scipy_norm.cdf(d1)
    nd1 = scipy_norm.pdf(d1)
    Nd2 = scipy_norm.cdf(d2)

    discount = np.exp(-r * T_safe)

    # Greeks w.r.t. normalized inputs
    delta = Nd1                                            # ∂C/∂(S/K)
    vega = m_safe * nd1 * sqrt_T                           # ∂C/∂σ
    theta = m_safe * nd1 * iv_safe / (2.0 * sqrt_T) + r * discount * Nd2  # ∂C/∂T (positive for time value)
    rho = T_safe * discount * Nd2                          # ∂C/∂r

    greeks = {"delta": delta, "vega": vega, "theta": theta, "rho": rho}

    # Stack into (n, d) array — column order matches feature order
    # Features: [moneyness, T, r, iv, (log_volume)]
    # Greeks:   [delta,     theta, rho, vega, (0)]
    greek_cols = [delta, theta, rho, vega]
    if include_volume:
        greek_cols.append(np.zeros_like(delta))  # ∂C/∂log_volume = 0

    greeks["stacked"] = np.column_stack(greek_cols)  # (n, 4 or 5)

    return greeks


def compute_bs_price_normalized(
    moneyness: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    iv: np.ndarray,
) -> np.ndarray:
    """BS-formula call price, normalised by K: C/K = (S/K)·Φ(d1) − e^{-rT}·Φ(d2).

    Required by target_mode="svi" to recompute the price target at the
    smile-fitted IV (rather than at the raw market IV).
    """
    T_safe = np.maximum(T, 1e-6)
    iv_safe = np.maximum(iv, 1e-6)
    m_safe = np.maximum(moneyness, 1e-8)
    sqrt_T = np.sqrt(T_safe)
    d1 = (np.log(m_safe) + (r + 0.5 * iv_safe ** 2) * T_safe) / (iv_safe * sqrt_T)
    d2 = d1 - iv_safe * sqrt_T
    return m_safe * scipy_norm.cdf(d1) - np.exp(-r * T_safe) * scipy_norm.cdf(d2)


# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================

# Default temporal cutoff: 2021-07-01 gives ~50/50 train/test on the full
# 1.57M records (780K train / 796K test). The train period covers
# 2020-01-02 to 2021-06-30 (COVID crash → recovery) and the test period
# covers 2021-07-01 to 2022-12-30 (rate hikes regime), providing a genuine
# out-of-sample regime-change test.
DEFAULT_TEMPORAL_CUTOFF = "2021-07-01"

# Default embargo: 5 trading days (~1 week) between train and test periods.
# This removes any train samples within the embargo window after the last
# train date, preventing information leakage from overlapping option
# contracts that span the cutoff boundary.
DEFAULT_EMBARGO_DAYS = 5


def _stratified_subsample(
    indices: np.ndarray,
    moneyness: np.ndarray,
    n_select: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Stratified subsampling by moneyness bins within given indices.

    Ensures coverage of ITM/ATM/OTM regions when subsampling from a
    large pool to a smaller training/test set.
    """
    if n_select >= len(indices):
        return indices

    m_vals = moneyness[indices]
    bins = np.array([0.85, 0.93, 0.97, 1.03, 1.07, 1.15])
    bin_indices = np.digitize(m_vals, bins) - 1
    n_bins = len(bins) - 1
    per_bin = n_select // n_bins

    selected = []
    for b in range(n_bins):
        available = np.where(bin_indices == b)[0]
        n_take = min(per_bin, len(available))
        if n_take > 0:
            chosen = rng.choice(available, size=n_take, replace=False)
            selected.append(indices[chosen])

    selected = np.concatenate(selected) if selected else np.array([], dtype=int)

    # Fill remainder
    if len(selected) < n_select:
        remaining = np.setdiff1d(indices, selected)
        n_extra = n_select - len(selected)
        if n_extra > 0 and len(remaining) > 0:
            extra = rng.choice(remaining, size=min(n_extra, len(remaining)), replace=False)
            selected = np.concatenate([selected, extra])

    rng.shuffle(selected)
    return selected[:n_select]


def _build_output(
    X: np.ndarray,
    y: np.ndarray,
    bs_price: np.ndarray,
    train_global_idx: np.ndarray,
    test_global_idx: np.ndarray,
    include_volume: bool,
    feature_names: list,
    extra_metadata: Dict[str, Any],
    target_mode: str = "bs_price",
    iv_svi: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Build the standardized output dict from global indices.

    target_mode: which scalar to use as the regression target.
        - "bs_price" (default): BS-formula price computed from the *raw market*
          (moneyness, T, r, IV). The BS Greeks below are then exact analytical
          derivatives of this target — H&S 2020 ground-truth training mode
          (Risk Mag 2020 §"Training with derivatives").
        - "svi":     BS-formula price recomputed at the SVI-fitted IV
          (Gatheral & Jacquier 2014 raw SVI per (date, maturity) slice;
          see `experiments/real_data_spy/svi_calibration.py`). The IV
          column in the input features is replaced by σ_SVI as well, so
          target and gradient labels both come from the same calibrated
          smile model. Requires `iv_svi` (full-length array) to be passed.
        - "market":  observed market mid price. The BS Greeks are NOT exact
          derivatives of this target; this is the v3 mode, retained only for
          reproducing prior numbers, not for paper-quality experiments.
    """
    if target_mode not in ("bs_price", "market", "svi"):
        raise ValueError(f"unknown target_mode: {target_mode!r}; "
                          "use 'bs_price' (default), 'svi' or 'market'.")
    if target_mode == "svi" and iv_svi is None:
        raise ValueError("target_mode='svi' requires iv_svi (full-length array). "
                          "Run scripts/calibrate_svi.py to generate the cache.")

    if include_volume:
        x_train = X[train_global_idx].copy()
        x_test = X[test_global_idx].copy()
        used_features = list(feature_names)
    else:
        x_train = X[train_global_idx, :4].copy()
        x_test = X[test_global_idx, :4].copy()
        used_features = list(feature_names[:4])

    if target_mode == "svi":
        # Replace the iv column with σ_SVI so the network sees the smile-fitted
        # IV and the Greeks below (computed at iv_svi) are exact derivatives of
        # the recomputed bs_price_svi target.
        used_features[3] = "iv_svi"
        x_train[:, 3] = iv_svi[train_global_idx]
        x_test[:, 3] = iv_svi[test_global_idx]
        bs_price_svi = compute_bs_price_normalized(
            moneyness=X[:, 0], T=X[:, 1], r=X[:, 2], iv=iv_svi,
        )
        y_used = bs_price_svi
    elif target_mode == "bs_price":
        y_used = bs_price
    else:                                     # "market"
        y_used = y
    y_train = y_used[train_global_idx].reshape(-1, 1)
    y_test = y_used[test_global_idx].reshape(-1, 1)
    bs_train = bs_price[train_global_idx]
    bs_test = bs_price[test_global_idx]

    # Compute BS Greeks. For target_mode="svi" the Greeks are evaluated at
    # σ_SVI (so they are the exact derivatives of bs_price_svi); for the other
    # two modes they remain at market IV.
    iv_train = x_train[:, 3]   # already σ_SVI in svi mode (set above)
    iv_test  = x_test[:, 3]
    greeks_train = compute_bs_greeks(
        moneyness=X[train_global_idx, 0],
        T=X[train_global_idx, 1],
        r=X[train_global_idx, 2],
        iv=iv_train,
        include_volume=include_volume,
    )
    greeks_test = compute_bs_greeks(
        moneyness=X[test_global_idx, 0],
        T=X[test_global_idx, 1],
        r=X[test_global_idx, 2],
        iv=iv_test,
        include_volume=include_volume,
    )

    d = x_train.shape[1]
    dydx_train = greeks_train["stacked"].reshape(-1, 1, d)
    dydx_test = greeks_test["stacked"].reshape(-1, 1, d)

    n_train = len(train_global_idx)
    n_test = len(test_global_idx)
    all_idx = np.concatenate([train_global_idx, test_global_idx])

    metadata = {
        "dataset": "spy_options_2020_2022",
        "source": "Kaggle SPY EOD, CC0 license",
        "target_mode": target_mode,
        "n_total_available": X.shape[0],
        "n_train": n_train,
        "n_test": n_test,
        "d": d,
        "features": used_features,
        "include_volume": include_volume,
        "moneyness_range": [float(X[all_idx, 0].min()), float(X[all_idx, 0].max())],
        "T_range": [float(X[all_idx, 1].min()), float(X[all_idx, 1].max())],
        "iv_range": [float(X[all_idx, 3].min()), float(X[all_idx, 3].max())],
        "y_range": [float(np.concatenate([y_train, y_test]).min()),
                     float(np.concatenate([y_train, y_test]).max())],
        # bs_vs_mid_rmse always reports the market-vs-BS gap, regardless of which
        # one is the training target — this characterises the data, not the
        # target choice.
        "bs_vs_mid_rmse": float(np.sqrt(
            np.mean((y[np.concatenate([train_global_idx, test_global_idx])]
                      - bs_price[np.concatenate([train_global_idx, test_global_idx])]) ** 2)
        )),
        **extra_metadata,
    }

    return {
        "x_train": x_train,
        "y_train": y_train,
        "dydx_train": dydx_train,
        "x_test": x_test,
        "y_test": y_test,
        "dydx_test": dydx_test,
        "metadata": metadata,
        "greeks_train": {k: v for k, v in greeks_train.items() if k != "stacked"},
        "greeks_test": {k: v for k, v in greeks_test.items() if k != "stacked"},
    }


def load_spy_data(
    data_path: str = "data/spy_options/spy_processed.npz",
    n_train: int = 50000,
    n_test: int = 10000,
    include_volume: bool = False,
    stratify_by_moneyness: bool = True,
    seed: int = 42,
    split_mode: str = "temporal",
    temporal_cutoff: str = DEFAULT_TEMPORAL_CUTOFF,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
    target_mode: str = "bs_price",
) -> Dict[str, Any]:
    """
    Load SPY options data and compute BS Greek derivative labels.

    Args:
        data_path: Path to spy_processed.npz (must include 'dates' array).
        n_train: Number of training samples (subsampled from available pool).
        n_test: Number of test samples.
        include_volume: Include log_volume as 5th input dimension.
            If False, uses 4D input (moneyness, T, r, iv) — all features
            for which we have analytical derivatives.
        stratify_by_moneyness: If True, stratified sampling by moneyness bins
            to ensure coverage of ITM/ATM/OTM regions.
        seed: Random seed for subsampling.
        split_mode: How to split train/test:
            - "temporal" (default): Train on dates < cutoff, test on dates >= cutoff.
              Applies embargo gap. Standard for financial ML.
            - "random": Legacy random split (deprecated — temporal leakage).
        temporal_cutoff: Date string (YYYY-MM-DD) for temporal split.
            Default: "2021-07-01" (~50/50 split).
        embargo_days: Number of trading days to exclude before cutoff
            in temporal mode (prevents information leakage from overlapping
            option contracts). Default: 5.

    Returns:
        Dictionary with:
            x_train, y_train, dydx_train: Training data
            x_test, y_test, dydx_test: Test data
            metadata: dict with dataset info including split_mode
            greeks_train, greeks_test: individual Greek values
    """
    assert split_mode in ("temporal", "random"), \
        f"Unknown split_mode: {split_mode!r}. Use 'temporal' or 'random'."

    if split_mode == "random":
        warnings.warn(
            "split_mode='random' is deprecated due to temporal leakage. "
            "Use split_mode='temporal' (default) for proper financial ML evaluation.",
            DeprecationWarning,
            stacklevel=2,
        )

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"SPY data not found at {data_path}. "
            f"Copy from o_nn/main_code/big_experiments/data/spy_processed.npz"
        )

    raw = np.load(str(data_path), allow_pickle=True)
    X = raw["X"]                # (1576419, 5)
    y = raw["y"]                # (1576419,)
    bs_price = raw["bs_price"]  # (1576419,)
    feature_names = list(raw["feature_names"])

    # Load the SVI-fitted IV cache when target_mode='svi'. Built once by
    # `experiments/real_data_spy/calibrate_svi.py`.
    iv_svi = None
    if target_mode == "svi":
        svi_iv_path = data_path.parent / "svi_iv.npy"
        if not svi_iv_path.exists():
            raise FileNotFoundError(
                f"SVI cache not found: {svi_iv_path}. Run "
                f"`python experiments/real_data_spy/calibrate_svi.py` first."
            )
        iv_svi = np.load(svi_iv_path)
        if iv_svi.shape[0] != X.shape[0]:
            raise ValueError(
                f"SVI cache length mismatch: {iv_svi.shape[0]} vs {X.shape[0]}"
            )

    n_total = X.shape[0]
    rng = np.random.RandomState(seed)
    moneyness = X[:, 0]

    if split_mode == "temporal":
        # --- Temporal split ---
        if "dates" not in raw:
            raise ValueError(
                "spy_processed.npz does not contain 'dates' array. "
                "Rebuild with the preprocessing script to include dates."
            )
        dates = raw["dates"]  # (n,) array of 'YYYY-MM-DD' strings

        # Find embargo boundary: last train date is embargo_days before cutoff
        unique_dates = np.sort(np.unique(dates))
        cutoff_idx = np.searchsorted(unique_dates, temporal_cutoff)
        embargo_end_idx = max(0, cutoff_idx - embargo_days)

        if embargo_end_idx == 0:
            raise ValueError(
                f"Embargo of {embargo_days} days leaves no training data "
                f"before cutoff {temporal_cutoff}"
            )

        train_end_date = unique_dates[embargo_end_idx - 1]  # last date in train
        test_start_date = unique_dates[cutoff_idx] if cutoff_idx < len(unique_dates) else temporal_cutoff

        # Build train and test pools
        train_pool = np.where(dates <= train_end_date)[0]
        test_pool = np.where(dates >= test_start_date)[0]
        embargo_pool = np.where(
            (dates > train_end_date) & (dates < test_start_date)
        )[0]

        if len(train_pool) < n_train:
            raise ValueError(
                f"Only {len(train_pool)} samples before embargo boundary "
                f"({train_end_date}), but n_train={n_train} requested."
            )
        if len(test_pool) < n_test:
            raise ValueError(
                f"Only {len(test_pool)} samples after cutoff "
                f"({test_start_date}), but n_test={n_test} requested."
            )

        # Subsample within each pool
        if stratify_by_moneyness:
            train_idx = _stratified_subsample(train_pool, moneyness, n_train, rng)
            test_idx = _stratified_subsample(test_pool, moneyness, n_test, rng)
        else:
            train_idx = rng.choice(train_pool, size=n_train, replace=False)
            test_idx = rng.choice(test_pool, size=n_test, replace=False)

        extra_meta = {
            "split_mode": "temporal",
            "temporal_cutoff": temporal_cutoff,
            "embargo_days": embargo_days,
            "train_end_date": str(train_end_date),
            "test_start_date": str(test_start_date),
            "n_embargo_excluded": int(len(embargo_pool)),
            "train_pool_size": int(len(train_pool)),
            "test_pool_size": int(len(test_pool)),
            "stratify_by_moneyness": stratify_by_moneyness,
            "seed": seed,
        }

    else:
        # --- Legacy random split (deprecated) ---
        n_needed = n_train + n_test
        if n_needed > n_total:
            raise ValueError(
                f"Requested {n_needed} samples but only {n_total} available"
            )

        if stratify_by_moneyness:
            all_idx = np.arange(n_total)
            selected = _stratified_subsample(all_idx, moneyness, n_needed, rng)
        else:
            selected = rng.choice(n_total, size=n_needed, replace=False)

        rng.shuffle(selected)
        train_idx = selected[:n_train]
        test_idx = selected[n_train:n_train + n_test]

        extra_meta = {
            "split_mode": "random",
            "stratify_by_moneyness": stratify_by_moneyness,
            "seed": seed,
            "WARNING": "Random split has temporal leakage. Use split_mode='temporal'.",
        }

    return _build_output(
        X, y, bs_price, train_idx, test_idx,
        include_volume, feature_names, extra_meta,
        target_mode=target_mode,
        iv_svi=iv_svi,
    )


def load_spy_data_purged_walkforward(
    data_path: str = "data/spy_options/spy_processed.npz",
    n_train: int = 50000,
    n_test: int = 10000,
    include_volume: bool = False,
    stratify_by_moneyness: bool = True,
    seed: int = 42,
    n_folds: int = 5,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
    target_mode: str = "bs_price",
) -> List[Dict[str, Any]]:
    """
    Load SPY data with purged walk-forward cross-validation.

    Implements temporal walk-forward CV following Lopez de Prado (2018),
    "Advances in Financial Machine Learning," Chapter 7. Each fold trains
    on all data before a cutoff date and tests on the next temporal segment,
    with an embargo gap between train and test to prevent leakage from
    overlapping option contracts.

    Uses expanding-window walk-forward: fold k trains on all data up to
    segment boundary k (with embargo), tests on segment k+1. Training
    pool grows with each fold but is capped by n_train via subsampling.

    The date range (2020-01-02 to 2022-12-30, 758 trading days) is divided
    into n_folds+1 equal-length temporal segments. Fold k trains on
    segments [0..k] (with embargo), tests on segment [k+1].

    Args:
        data_path: Path to spy_processed.npz.
        n_train: Max training samples per fold (subsampled if pool is larger).
        n_test: Max test samples per fold.
        include_volume: Include log_volume as 5th input feature.
        stratify_by_moneyness: Stratified sampling within each fold.
        seed: Random seed.
        n_folds: Number of CV folds. Default: 5.
        embargo_days: Trading days to exclude between train and test.

    Returns:
        List of n_folds dicts, each with:
            x_train, y_train, dydx_train, x_test, y_test, dydx_test,
            metadata (includes fold_idx, cutoff dates, etc.)
    """
    data_path = Path(data_path)
    raw = np.load(str(data_path), allow_pickle=True)
    X, y, bs_price = raw["X"], raw["y"], raw["bs_price"]
    feature_names = list(raw["feature_names"])
    dates = raw["dates"]
    moneyness = X[:, 0]

    iv_svi = None
    if target_mode == "svi":
        svi_iv_path = data_path.parent / "svi_iv.npy"
        if not svi_iv_path.exists():
            raise FileNotFoundError(
                f"SVI cache not found: {svi_iv_path}. Run "
                f"`python experiments/real_data_spy/calibrate_svi.py` first."
            )
        iv_svi = np.load(svi_iv_path)
        if iv_svi.shape[0] != X.shape[0]:
            raise ValueError(
                f"SVI cache length mismatch: {iv_svi.shape[0]} vs {X.shape[0]}"
            )

    unique_dates = np.sort(np.unique(dates))
    n_dates = len(unique_dates)

    # Divide dates into n_folds + 1 segments
    # Folds: train on segments [0..k], test on segment [k+1]
    # (fold 0 trains on seg 0, tests on seg 1; fold n-1 trains on segs 0..n-1, tests on seg n)
    segment_size = n_dates // (n_folds + 1)
    if segment_size < embargo_days + 10:
        raise ValueError(
            f"Not enough dates ({n_dates}) for {n_folds} folds with "
            f"embargo={embargo_days}. Need >= {(embargo_days + 10) * (n_folds + 1)} dates."
        )

    folds = []
    rng = np.random.RandomState(seed)

    for fold_idx in range(n_folds):
        # Training: segments 0 through fold_idx
        train_end_date_idx = (fold_idx + 1) * segment_size - 1
        # Test: segment fold_idx + 1
        test_start_date_idx = (fold_idx + 1) * segment_size
        test_end_date_idx = min((fold_idx + 2) * segment_size - 1, n_dates - 1)

        # Apply embargo: remove last `embargo_days` from training
        train_end_date_idx_embargoed = max(0, train_end_date_idx - embargo_days)

        train_end_date = unique_dates[train_end_date_idx_embargoed]
        test_start_date = unique_dates[test_start_date_idx]
        test_end_date = unique_dates[test_end_date_idx]

        train_pool = np.where(dates <= train_end_date)[0]
        test_pool = np.where(
            (dates >= test_start_date) & (dates <= test_end_date)
        )[0]

        # Subsample
        actual_n_train = min(n_train, len(train_pool))
        actual_n_test = min(n_test, len(test_pool))

        if stratify_by_moneyness:
            train_idx = _stratified_subsample(train_pool, moneyness, actual_n_train, rng)
            test_idx = _stratified_subsample(test_pool, moneyness, actual_n_test, rng)
        else:
            train_idx = rng.choice(train_pool, size=actual_n_train, replace=False)
            test_idx = rng.choice(test_pool, size=actual_n_test, replace=False)

        extra_meta = {
            "split_mode": "purged_walkforward",
            "fold_idx": fold_idx,
            "n_folds": n_folds,
            "embargo_days": embargo_days,
            "train_end_date": str(train_end_date),
            "test_start_date": str(test_start_date),
            "test_end_date": str(test_end_date),
            "train_pool_size": int(len(train_pool)),
            "test_pool_size": int(len(test_pool)),
            "stratify_by_moneyness": stratify_by_moneyness,
            "seed": seed,
        }

        fold_data = _build_output(
            X, y, bs_price, train_idx, test_idx,
            include_volume, feature_names, extra_meta,
            target_mode=target_mode,
            iv_svi=iv_svi,
        )
        folds.append(fold_data)

    return folds
