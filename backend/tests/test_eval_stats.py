"""tests/test_eval_stats.py — synthetic-data tests for eval_stats helpers."""

import math

import numpy as np
import pytest

from tests.eval_stats import bootstrap_ci, paired_diff_ci


class TestPairedDiffCi:
    def test_zero_difference_interval_brackets_zero(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0.0, 1.0, size=200).tolist()
        # Identical samples ⇒ mean diff is exactly 0 and the CI is degenerate
        # (sd = 0, se = 0). The interval must contain 0 either way.
        mean, lo, hi = paired_diff_ci(x, list(x))
        assert mean == 0.0
        assert lo <= 0.0 <= hi

    def test_known_shift_interval_excludes_zero(self):
        rng = np.random.default_rng(7)
        a = rng.normal(0.5, 0.1, size=400).tolist()
        b = rng.normal(0.0, 0.1, size=400).tolist()
        mean, lo, hi = paired_diff_ci(a, b)
        assert lo > 0.0
        assert hi > lo
        assert math.isclose(mean, 0.5, abs_tol=0.05)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            paired_diff_ci([1.0, 2.0], [1.0])

    def test_singleton_raises(self):
        with pytest.raises(ValueError):
            paired_diff_ci([1.0], [0.0])


class TestBootstrapCi:
    def test_constant_sample_collapses_to_point(self):
        mean, lo, hi = bootstrap_ci([2.5] * 50, n_resamples=200)
        assert mean == 2.5
        assert lo == 2.5
        assert hi == 2.5

    def test_synthetic_sample_brackets_population_mean(self):
        rng = np.random.default_rng(123)
        # Small n is exactly the regime bootstrap_ci is for (Bowyer 2025).
        samples = rng.normal(1.0, 0.5, size=40).tolist()
        mean, lo, hi = bootstrap_ci(samples, n_resamples=2000)
        assert lo < 1.0 < hi
        assert math.isclose(mean, float(np.mean(samples)))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci([], n_resamples=100)
