"""
Statistical testing module for DML benchmark.

Provides publication-grade statistical tests for comparing methods:
- Wilcoxon signed-rank test (paired non-parametric)
- Bootstrap confidence intervals 
- Cohen's d effect size
- Holm-Bonferroni correction for multiple comparisons
- Full comparison report generation

NOTE: This module is structured now but will be populated with results
after experiments are run. The functions are complete and tested independently.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class PairwiseTestResult:
    """Result of a pairwise statistical test.

    C2 (2026-04-30, codebase audit): the historic field name was
    `effect_size`. The value stored is Cohen's d_z (paired); the
    `effect_size_label` thresholds {0.2, 0.5, 0.8} are calibrated for
    Cohen's d_s (independent-samples). Reading `effect_size` and
    interpreting via the small/medium/large labels mixed two
    conventions, which is the source of the historic d ≈ 47 vs +318
    disagreement in the SPY analysis. The field is renamed
    `effect_size_d_z` to force callers to opt into the specific
    convention. A read-only property `effect_size` is kept for
    backward-compat in this release; its use is deprecated.
    """
    method_a: str
    method_b: str
    statistic: float
    p_value: float
    effect_size_d_z: float  # Cohen's d_z (paired); see class docstring.
    significant_005: bool
    significant_001: bool
    ci_diff_lower: float  # 95% CI on difference
    ci_diff_upper: float

    @property
    def effect_size(self) -> float:
        """Deprecated alias for effect_size_d_z (C2; remove next release)."""
        import warnings
        warnings.warn(
            "PairwiseTestResult.effect_size is deprecated; "
            "use .effect_size_d_z (paired Cohen's d_z) explicitly. "
            "See dml_benchmark.stats docstring on the d_z vs d_s distinction.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.effect_size_d_z


@dataclass
class MethodSummary:
    """Statistical summary for one method across seeds."""
    method: str
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    n_seeds: int


# ============================================================================
# CORE STATISTICAL TESTS
# ============================================================================

def paired_wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alternative: str = "two-sided"
) -> Dict[str, Any]:
    """
    Wilcoxon signed-rank test for paired comparisons across seeds.
    
    Non-parametric alternative to paired t-test. Appropriate when 
    we can't assume normality of score differences.
    
    Args:
        scores_a: MSE values for method A across seeds (n_seeds,)
        scores_b: MSE values for method B across seeds (n_seeds,)
        alternative: 'two-sided', 'less', or 'greater'
        
    Returns:
        Dict with statistic, p_value, and significance flags
    """
    from scipy import stats as sp_stats
    
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    
    # Need at least 6 paired observations for Wilcoxon
    if len(scores_a) < 6:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "significant_005": False,
            "significant_001": False,
            "warning": f"Too few samples ({len(scores_a)}) for Wilcoxon test. Need ≥ 6."
        }
    
    # Check if differences are all zero
    diffs = scores_a - scores_b
    if np.all(np.abs(diffs) < 1e-12):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant_005": False,
            "significant_001": False,
            "warning": "All differences are zero."
        }
    
    stat, p = sp_stats.wilcoxon(scores_a, scores_b, alternative=alternative)
    
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "significant_005": p < 0.05,
        "significant_001": p < 0.01
    }


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
    method: str = "BCa"
) -> Dict[str, float]:
    """
    Bootstrap confidence interval for the mean.
    
    Uses scipy.stats.bootstrap with BCa (bias-corrected and accelerated)
    by default.  BCa corrects for both bias and skewness in the bootstrap
    distribution, giving better coverage than the percentile method,
    especially with small samples (n ≈ 10).  Falls back to the percentile
    method when n < 3 (BCa needs the jackknife).
    
    Reference: Efron & Tibshirani (1993), Chapter 14.
    
    Args:
        values: Array of measurements (e.g., MSE across seeds)
        n_bootstrap: Number of bootstrap resamples
        alpha: Significance level (0.05 = 95% CI)
        seed: Random seed for reproducibility
        method: 'BCa' (default, recommended) or 'percentile'
        
    Returns:
        Dict with mean, ci_lower, ci_upper, std, method
    """
    from scipy.stats import bootstrap as sp_bootstrap

    values = np.asarray(values, dtype=float)

    # BCa requires n ≥ 3 for the jackknife acceleration estimate
    if len(values) < 3 or method.lower() == "percentile":
        rng = np.random.RandomState(seed)
        bootstrap_means = np.array([
            np.mean(rng.choice(values, size=len(values), replace=True))
            for _ in range(n_bootstrap)
        ])
        ci_lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
        ci_upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))
        used_method = "percentile"
    else:
        res = sp_bootstrap(
            (values,),
            statistic=np.mean,
            n_resamples=n_bootstrap,
            confidence_level=1 - alpha,
            method="BCa",
            random_state=seed,
        )
        ci_lower = float(res.confidence_interval.low)
        ci_upper = float(res.confidence_interval.high)
        used_method = "BCa"

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n": len(values),
        "n_bootstrap": n_bootstrap,
        "alpha": alpha,
        "method": used_method,
    }


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d_z effect size for PAIRED samples (d_z = mean(diff)/std(diff)).

    Note (J3 / H-L4, 2026-04-16): this is the PAIRED d_z formula, not the
    independent-sample Cohen's d (which uses pooled SD). The standard labels
    {negligible, small, medium, large} at {0.2, 0.5, 0.8} are calibrated for
    the independent-sample d and may LOOK large under d_z when paired variance
    is small. Prefer `cohens_ds` (pooled-SD form) for label interpretation;
    retain this name for backwards compatibility with existing callers.

    Interpretation (d_z-calibrated, informal):
        Small paired variance + moderate mean shift → d_z can be very large
        (e.g. d_z > 2 common in paired DML experiments) without implying a
        correspondingly large standardized mean difference.

    Args:
        group1: Scores for method 1
        group2: Scores for method 2

    Returns:
        Cohen's d_z (positive means group2 is better/lower)
    """
    group1 = np.asarray(group1, dtype=float)
    group2 = np.asarray(group2, dtype=float)

    diff = group1 - group2
    d = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))

    return d


def cohens_ds(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d_s for INDEPENDENT samples (mean diff / pooled SD).

    This is the effect-size measure whose thresholds 0.2/0.5/0.8 correspond
    to the labels small/medium/large.
    """
    group1 = np.asarray(group1, dtype=float)
    group2 = np.asarray(group2, dtype=float)
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return float((np.mean(group1) - np.mean(group2)) / (pooled + 1e-12))


def cohens_d_ci(
    group1: np.ndarray,
    group2: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
    method: str = "BCa"
) -> Dict[str, float]:
    """
    Cohen's d for paired samples with bootstrap confidence interval.
    
    Uses BCa bootstrap (Efron & Tibshirani 1993) to compute a CI on the
    effect size.  Reporting effect-size CIs is recommended by the APA
    Publication Manual (7th ed., §6.44) and strengthens claims about
    practical significance.
    
    Bootstraps the paired differences directly (1-D), which avoids
    degenerate jackknife issues that arise with a 2-D paired array.
    Falls back to percentile method when BCa encounters numerical
    issues (e.g., degenerate distributions).
    
    Args:
        group1: Scores for method/condition 1 (n,)
        group2: Scores for method/condition 2 (n,)
        n_bootstrap: Number of bootstrap resamples
        alpha: Significance level (0.05 → 95% CI)
        seed: Random seed for reproducibility
        method: 'BCa' (default) or 'percentile'
        
    Returns:
        Dict with d, ci_lower, ci_upper, label, n, method
    """
    import warnings
    from scipy.stats import bootstrap as sp_bootstrap

    group1 = np.asarray(group1, dtype=float)
    group2 = np.asarray(group2, dtype=float)
    assert len(group1) == len(group2), "Paired samples must have equal length"

    d = cohens_d(group1, group2)
    diffs = group1 - group2  # 1-D array

    def _d_from_diffs(x, axis):
        """Cohen's d from paired differences, vectorised along axis."""
        return np.mean(x, axis=axis) / (np.std(x, ddof=1, axis=axis) + 1e-12)

    def _percentile_fallback():
        rng = np.random.RandomState(seed)
        boot_d = np.array([
            _d_from_diffs(diffs[rng.choice(len(diffs), size=len(diffs), replace=True)], axis=0)
            for _ in range(n_bootstrap)
        ])
        lo = float(np.percentile(boot_d, 100 * alpha / 2))
        hi = float(np.percentile(boot_d, 100 * (1 - alpha / 2)))
        return lo, hi, "percentile"

    if len(group1) < 3 or method.lower() == "percentile":
        ci_lower, ci_upper, used_method = _percentile_fallback()
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = sp_bootstrap(
                    (diffs,),
                    statistic=_d_from_diffs,
                    n_resamples=n_bootstrap,
                    confidence_level=1 - alpha,
                    method="BCa",
                    random_state=seed,
                )
                ci_lower = float(res.confidence_interval.low)
                ci_upper = float(res.confidence_interval.high)
                used_method = "BCa"
                # Fall back if BCa produced NaN (degenerate distribution)
                if np.isnan(ci_lower) or np.isnan(ci_upper):
                    ci_lower, ci_upper, used_method = _percentile_fallback()
            except Exception:
                ci_lower, ci_upper, used_method = _percentile_fallback()

    return {
        "d": d,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "label": effect_size_label(d),
        "n": len(group1),
        "n_bootstrap": n_bootstrap,
        "alpha": alpha,
        "method": used_method,
    }


def effect_size_label(d: float) -> str:
    """Human-readable label for Cohen's d magnitude."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


# ============================================================================
# MULTIPLE COMPARISONS CORRECTION
# ============================================================================

def holm_bonferroni(
    p_values: List[float],
    alpha: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Holm-Bonferroni step-down correction for multiple comparisons.
    
    More powerful than Bonferroni while still controlling FWER.
    
    Args:
        p_values: List of unadjusted p-values from multiple tests
        alpha: Family-wise significance level
        
    Returns:
        List of dicts with original_p, adjusted_p, significant, rank
    """
    n = len(p_values)
    
    # Sort p-values and track original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    
    results = [None] * n
    
    for rank, (orig_idx, p) in enumerate(zip(sorted_indices, sorted_p)):
        # Holm-Bonferroni threshold: α / (n - rank)
        threshold = alpha / (n - rank)
        adjusted_p = min(p * (n - rank), 1.0)
        
        results[orig_idx] = {
            "original_p": float(p),
            "adjusted_p": float(adjusted_p),
            "significant": adjusted_p < alpha,
            "rank": rank + 1
        }
    
    return results


# ============================================================================
# FULL COMPARISON REPORT
# ============================================================================

def full_comparison_report(
    results_dict: Dict[str, List[float]],
    metric_name: str = "value_mse"
) -> Dict[str, Any]:
    """
    Generate a complete statistical comparison between all methods.
    
    Args:
        results_dict: {method_name: [score_seed1, score_seed2, ...]}
        metric_name: Name of the metric being compared (for reporting)
        
    Returns:
        Dict with:
            - summaries: per-method summary statistics
            - pairwise_tests: all pairwise Wilcoxon tests
            - corrected_p_values: Holm-Bonferroni corrected p-values
    """
    methods = sorted(results_dict.keys())
    n_methods = len(methods)
    
    # 1. Per-method summaries with bootstrap CI
    summaries = {}
    for method in methods:
        values = np.array(results_dict[method])
        ci = bootstrap_ci(values)
        summaries[method] = MethodSummary(
            method=method,
            mean=ci["mean"],
            std=ci["std"],
            ci_lower=ci["ci_lower"],
            ci_upper=ci["ci_upper"],
            n_seeds=len(values)
        )
    
    # 2. All pairwise comparisons
    pairwise_tests = []
    p_values = []
    
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            scores_a = np.array(results_dict[methods[i]])
            scores_b = np.array(results_dict[methods[j]])
            
            test_result = paired_wilcoxon_test(scores_a, scores_b)
            # J-L3 (2026-04-16): report both d_z (paired) and d_s (independent-
            # sample pooled-SD). d_s is the one whose thresholds map to the
            # labels small/medium/large. d_z is reported separately because
            # the report JSON previously mixed d_z values with d_s thresholds.
            d = cohens_d(scores_a, scores_b)       # paired d_z
            d_s = cohens_ds(scores_a, scores_b)    # independent-samples d_s
            
            # Bootstrap CI on difference
            diffs = scores_a - scores_b
            diff_ci = bootstrap_ci(diffs)
            
            pairwise = PairwiseTestResult(
                method_a=methods[i],
                method_b=methods[j],
                statistic=test_result.get("statistic", np.nan),
                p_value=test_result.get("p_value", np.nan),
                effect_size_d_z=d,                 # paired d_z
                significant_005=test_result.get("significant_005", False),
                significant_001=test_result.get("significant_001", False),
                ci_diff_lower=diff_ci["ci_lower"],
                ci_diff_upper=diff_ci["ci_upper"]
            )
            pairwise_tests.append(pairwise)
            p_values.append(test_result.get("p_value", 1.0))
    
    # 3. Multiple comparison correction
    corrected = holm_bonferroni(p_values) if p_values else []
    
    return {
        "metric": metric_name,
        "summaries": {m: s.__dict__ for m, s in summaries.items()},
        "pairwise_tests": [
            {
                "comparison": f"{t.method_a} vs {t.method_b}",
                "p_value": t.p_value,
                # C2 (2026-04-30): explicit d_z naming. Old "effect_size" key
                # is dropped; downstream readers must use "effect_size_d_z"
                # and "effect_label_d_z" together.
                "effect_size_d_z": t.effect_size_d_z,
                "effect_label_d_z": effect_size_label(t.effect_size_d_z),
                "note": "effect_label_d_z is d_z-based (paired); for "
                        "d_s-based small/medium/large labels use "
                        "cohens_ds(scores_a, scores_b) directly.",
                "significant_005": t.significant_005,
                "ci_diff": [t.ci_diff_lower, t.ci_diff_upper],
                "corrected": corrected[i] if i < len(corrected) else None
            }
            for i, t in enumerate(pairwise_tests)
        ],
        "n_comparisons": len(pairwise_tests)
    }


def format_results_table(
    report: Dict[str, Any],
    latex: bool = False
) -> str:
    """
    Format comparison report as a readable table.
    
    Args:
        report: Output of full_comparison_report()
        latex: If True, format as LaTeX table
        
    Returns:
        Formatted string table
    """
    summaries = report["summaries"]
    
    if latex:
        lines = [
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Method & Mean & Std & 95\% CI \\",
            r"\midrule"
        ]
        for method, s in summaries.items():
            lines.append(
                f"{method} & {s['mean']:.6f} & {s['std']:.6f} & "
                f"[{s['ci_lower']:.6f}, {s['ci_upper']:.6f}] \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        return "\n".join(lines)
    else:
        lines = [
            f"{'Method':<20} {'Mean':>12} {'Std':>12} {'95% CI':>24}",
            "-" * 70
        ]
        for method, s in summaries.items():
            ci_str = f"[{s['ci_lower']:.6f}, {s['ci_upper']:.6f}]"
            lines.append(
                f"{method:<20} {s['mean']:>12.6f} {s['std']:>12.6f} {ci_str:>24}"
            )
        return "\n".join(lines)
