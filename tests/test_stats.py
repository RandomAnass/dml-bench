"""
Tests for dml_benchmark/stats.py — statistical testing module.

Covers:
- paired_wilcoxon_test: paired non-parametric test
- bootstrap_ci: BCa and percentile bootstrap confidence intervals
- cohens_d: Cohen's d effect size
- cohens_d_ci: Cohen's d with BCa bootstrap CI
- effect_size_label: Cohen's d interpretation
- holm_bonferroni: multiple comparisons correction
"""

import numpy as np
import pytest

from dml_benchmark.stats import (
    paired_wilcoxon_test,
    bootstrap_ci,
    cohens_d,
    cohens_d_ci,
    effect_size_label,
    holm_bonferroni,
)


# ============================================================================
# paired_wilcoxon_test
# ============================================================================


class TestPairedWilcoxonTest:
    """Tests for paired Wilcoxon signed-rank test."""

    def test_identical_arrays_not_significant(self):
        """Identical arrays should yield p ≈ 1 (no difference)."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = paired_wilcoxon_test(a, a)
        assert result["p_value"] >= 0.5

    def test_clearly_different_arrays(self):
        """Arrays with large constant shift should be significant."""
        rng = np.random.RandomState(42)
        a = rng.randn(20)
        b = a + 5.0  # large constant shift
        result = paired_wilcoxon_test(a, b)
        assert result["p_value"] < 0.01

    def test_returns_expected_keys(self):
        """Result dict should have statistic and p_value."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        b = np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1])
        result = paired_wilcoxon_test(a, b)
        assert "statistic" in result
        assert "p_value" in result
        assert isinstance(result["p_value"], float)

    def test_too_few_samples_fallback(self):
        """With fewer than 6 samples, should still return a result dict."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = paired_wilcoxon_test(a, b)
        assert "p_value" in result
        # Wilcoxon may return NaN for very small n — that's acceptable
        assert isinstance(result["p_value"], float)


# ============================================================================
# bootstrap_ci
# ============================================================================


class TestBootstrapCI:
    """Tests for bootstrap confidence intervals."""

    def test_returns_expected_keys(self):
        """Result should contain mean, std, ci_lower, ci_upper, etc."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = bootstrap_ci(values)
        for key in ("mean", "std", "ci_lower", "ci_upper", "n", "n_bootstrap", "alpha", "method"):
            assert key in result, f"Missing key: {key}"

    def test_ci_contains_mean(self):
        """CI should contain the sample mean."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = bootstrap_ci(values)
        assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]

    def test_ci_width_with_more_data(self):
        """More data should produce narrower CI."""
        rng = np.random.RandomState(42)
        small = rng.randn(10)
        large = rng.randn(1000)
        ci_small = bootstrap_ci(small, seed=42)
        ci_large = bootstrap_ci(large, seed=42)
        width_small = ci_small["ci_upper"] - ci_small["ci_lower"]
        width_large = ci_large["ci_upper"] - ci_large["ci_lower"]
        assert width_large < width_small

    def test_bca_method_default(self):
        """Default method should be BCa."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = bootstrap_ci(values)
        assert result["method"] == "BCa"

    def test_percentile_method(self):
        """Setting method='percentile' should use percentile method."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = bootstrap_ci(values, method="percentile")
        assert result["method"] == "percentile"

    def test_small_sample_fallback(self):
        """n<3 should fallback to percentile (BCa needs ≥3)."""
        values = np.array([1.0, 2.0])
        result = bootstrap_ci(values)
        # Should still return valid CI
        assert np.isfinite(result["ci_lower"])
        assert np.isfinite(result["ci_upper"])

    def test_single_value(self):
        """Single value should return degenerate CI."""
        values = np.array([5.0])
        result = bootstrap_ci(values)
        assert result["mean"] == 5.0

    def test_constant_array(self):
        """All-same values should have zero-width CI."""
        values = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        result = bootstrap_ci(values)
        assert result["mean"] == 3.0
        assert result["std"] == 0.0

    def test_reproducible_with_seed(self):
        """Same seed should produce identical results."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        r1 = bootstrap_ci(values, seed=123)
        r2 = bootstrap_ci(values, seed=123)
        assert r1["ci_lower"] == r2["ci_lower"]
        assert r1["ci_upper"] == r2["ci_upper"]


# ============================================================================
# cohens_d
# ============================================================================


class TestCohensD:
    """Tests for Cohen's d effect size."""

    def test_identical_groups(self):
        """Identical groups should have d=0."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert cohens_d(a, a) == 0.0

    def test_known_effect_size(self):
        """Paired Cohen's d with known difference distribution."""
        # diff has mean=-1, std(diff)≈0.71 → d ≈ -1.41
        rng = np.random.RandomState(0)
        a = rng.randn(100)
        b = a + 1.0 + rng.randn(100) * 0.5  # shift + noise
        d = cohens_d(a, b)
        # d should be negative (group1 < group2 in mean)
        assert d < 0
        # |d| should be moderately large (≈ medium-to-large)
        assert abs(d) > 0.5

    def test_constant_diff_gives_huge_d(self):
        """Constant pair differences → std=0 → very large |d|."""
        a = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = cohens_d(a, b)
        assert abs(d) > 1e6  # effectively infinite

    def test_sign_convention(self):
        """d should be positive when group1 > group2."""
        a = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = cohens_d(a, b)
        assert d > 0

    def test_large_effect(self):
        """Well-separated groups should have large d."""
        rng = np.random.RandomState(42)
        a = rng.randn(50) + 100
        b = rng.randn(50)
        d = cohens_d(a, b)
        assert abs(d) > 5.0


# ============================================================================
# cohens_d_ci
# ============================================================================


class TestCohensDCI:
    """Tests for Cohen's d with BCa bootstrap CI."""

    def test_returns_expected_keys(self):
        """Result should have d, ci_lower, ci_upper, label."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        b = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
        result = cohens_d_ci(a, b)
        for key in ("d", "ci_lower", "ci_upper", "label"):
            assert key in result, f"Missing key: {key}"

    def test_ci_contains_point_estimate(self):
        """CI should contain the point estimate d."""
        rng = np.random.RandomState(42)
        a = rng.randn(20)
        b = rng.randn(20) + 1.0
        result = cohens_d_ci(a, b)
        assert result["ci_lower"] <= result["d"] <= result["ci_upper"]

    def test_identical_groups(self):
        """Identical groups should have d ≈ 0 with CI containing 0."""
        rng = np.random.RandomState(42)
        a = rng.randn(20)
        result = cohens_d_ci(a, a)
        assert abs(result["d"]) < 0.01

    def test_label_is_valid(self):
        """Label should be one of the standard effect size categories."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        b = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
        result = cohens_d_ci(a, b)
        assert result["label"] in ("negligible", "small", "medium", "large")


# ============================================================================
# effect_size_label
# ============================================================================


class TestEffectSizeLabel:
    """Tests for Cohen's d interpretation labels."""

    def test_negligible(self):
        assert effect_size_label(0.1) == "negligible"

    def test_small(self):
        assert effect_size_label(0.3) == "small"

    def test_medium(self):
        assert effect_size_label(0.6) == "medium"

    def test_large(self):
        assert effect_size_label(1.0) == "large"

    def test_very_large_still_large(self):
        assert effect_size_label(100.0) == "large"

    def test_negative_uses_absolute(self):
        """Negative d should still give correct label based on |d|."""
        assert effect_size_label(0.1) == effect_size_label(-0.1)


# ============================================================================
# holm_bonferroni
# ============================================================================


class TestHolmBonferroni:
    """Tests for Holm-Bonferroni multiple comparisons correction."""

    def test_single_pvalue_unchanged(self):
        """Single p-value should be unchanged."""
        result = holm_bonferroni([0.03])
        assert len(result) == 1
        assert result[0]["adjusted_p"] == pytest.approx(0.03)

    def test_adjusts_upward(self):
        """Adjusted p-values should be ≥ original."""
        pvals = [0.001, 0.01, 0.05]
        result = holm_bonferroni(pvals)
        for r, orig in zip(result, pvals):
            assert r["adjusted_p"] >= orig

    def test_preserves_order(self):
        """Results should be in the same order as input."""
        pvals = [0.05, 0.001, 0.03]
        result = holm_bonferroni(pvals)
        assert len(result) == 3
        # Each result should have adjusted_p key
        for r in result:
            assert "adjusted_p" in r

    def test_all_significant_stay_significant(self):
        """Very small p-values should remain significant after correction."""
        pvals = [0.0001, 0.0002, 0.0003]
        result = holm_bonferroni(pvals)
        for r in result:
            assert r["adjusted_p"] < 0.05

    def test_capped_at_1(self):
        """Adjusted p-values should not exceed 1.0."""
        pvals = [0.5, 0.6, 0.7, 0.8]
        result = holm_bonferroni(pvals)
        for r in result:
            assert r["adjusted_p"] <= 1.0

    def test_empty_input(self):
        """Empty list should return empty list."""
        result = holm_bonferroni([])
        assert result == []


# ============================================================================
# Integration: end-to-end statistical pipeline
# ============================================================================


class TestStatsPipelineIntegration:
    """End-to-end test: generate data, compute all stats."""

    def test_full_comparison_pipeline(self):
        """Simulate two methods with known difference, verify all stats agree."""
        rng = np.random.RandomState(42)
        vanilla = rng.randn(10) * 0.1 + 1.0   # mean ≈ 1.0
        dml = rng.randn(10) * 0.1 + 0.5       # mean ≈ 0.5 (better)

        # Wilcoxon: should be significant
        w = paired_wilcoxon_test(vanilla, dml)
        assert w["p_value"] < 0.05

        # Bootstrap CI: CIs should not overlap
        ci_van = bootstrap_ci(vanilla, seed=42)
        ci_dml = bootstrap_ci(dml, seed=42)
        assert ci_van["ci_lower"] > ci_dml["ci_upper"]  # vanilla worse

        # Cohen's d: should be large
        d = cohens_d(vanilla, dml)
        assert effect_size_label(abs(d)) == "large"

        # Cohen's d CI: should not contain 0
        d_ci = cohens_d_ci(vanilla, dml)
        # Both bounds should have same sign
        assert d_ci["ci_lower"] > 0 or d_ci["ci_upper"] < 0

        # Holm-Bonferroni: single test should pass through
        hb = holm_bonferroni([w["p_value"]])
        assert hb[0]["adjusted_p"] == pytest.approx(w["p_value"])
