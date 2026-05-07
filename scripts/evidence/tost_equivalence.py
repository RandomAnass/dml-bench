import numpy as np
from scipy import stats

def tost_equivalence(log_ratios, margin_log10=np.log10(1.05), alpha=0.05):
    """
    Two One-Sided Tests (TOST) for equivalence of paired log-ratios.
    Null hypothesis: true mean log-ratio <= -margin or >= margin.
    Alternative: -margin < true mean log-ratio < margin.
    """
    n = len(log_ratios)
    mean_diff = np.mean(log_ratios)
    se = np.std(log_ratios, ddof=1) / np.sqrt(n)
    
    # t-statistics for lower and upper bounds
    t_lower = (mean_diff - (-margin_log10)) / se
    t_upper = (mean_diff - margin_log10) / se
    
    # Critical value (one-sided)
    t_crit = stats.t.ppf(1 - alpha, n - 1)
    
    # Reject null if both conditions are met
    reject_lower = t_lower > t_crit
    reject_upper = t_upper < -t_crit
    
    is_equivalent = reject_lower and reject_upper
    
    return {
        "mean_log_ratio": mean_diff,
        "se": se,
        "t_lower": t_lower,
        "t_upper": t_upper,
        "t_crit": t_crit,
        "is_equivalent": bool(is_equivalent),
        "reject_lower": bool(reject_lower),
        "reject_upper": bool(reject_upper),
        "n": n
    }

if __name__ == "__main__":
    # Test it
    np.random.seed(42)
    # Log ratios very close to zero (e.g. 1.01x ratio)
    approx_equivalent = np.random.normal(0.004, 0.01, 20)
    res = tost_equivalence(approx_equivalent)
    print("Equivalent expected:", res)
    
    # Log ratios far from zero (e.g. 1.1x ratio)
    not_equivalent = np.random.normal(np.log10(1.1), 0.01, 20)
    res = tost_equivalence(not_equivalent)
    print("Not equivalent expected:", res)

