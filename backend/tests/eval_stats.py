"""tests/eval_stats.py — confidence-interval helpers for offline eval work.

Two pure functions backed by numpy only (already in requirements.txt). Opt-in:
no existing test runner imports this module; future eval scripts can.

Per Bowyer (2025), bootstrap CIs are preferred for small samples (n < 200) where
the CLT-based t-interval becomes unreliable.
"""

from __future__ import annotations

import math

import numpy as np


def paired_diff_ci(
    a: list[float], b: list[float], alpha: float = 0.05,
) -> tuple[float, float, float]:
    """CLT-based t-interval on paired differences ``a[i] - b[i]``.

    Returns ``(mean_diff, ci_low, ci_high)`` for the (1 - alpha) interval.
    Raises ``ValueError`` if ``a`` and ``b`` are not the same non-empty length.
    """
    if len(a) != len(b):
        raise ValueError("paired_diff_ci requires equal-length samples")
    n = len(a)
    if n < 2:
        raise ValueError("paired_diff_ci requires at least 2 paired observations")

    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    mean = float(diffs.mean())
    # ddof=1 → unbiased sample variance; the t-interval needs s, not sigma.
    sd = float(diffs.std(ddof=1))
    se = sd / math.sqrt(n)

    # Wilson-Hilferty approximation of the inverse Student-t CDF, accurate
    # to ~3 decimals for df ≥ 2 and well within the resolution of any eval
    # CI we'd report. Avoids a scipy dependency.
    df = n - 1
    p = 1.0 - alpha / 2.0
    z = _normal_quantile(p)
    t = z * math.sqrt(df / (df - 2.0)) if df > 2 else z * math.sqrt(df + 1)
    half = t * se
    return (mean, mean - half, mean + half)


def bootstrap_ci(
    samples: list[float], n_resamples: int = 10000, alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI on the mean of ``samples``.

    Returns ``(mean, ci_low, ci_high)`` for the (1 - alpha) interval. Default
    n_resamples=10_000 is the Bowyer 2025 default for n < 200.
    """
    if not samples:
        raise ValueError("bootstrap_ci requires at least 1 observation")
    arr = np.asarray(samples, dtype=float)
    mean = float(arr.mean())
    rng = np.random.default_rng(seed=0)
    # Vectorised resampling: one (n_resamples, n) matrix of indices.
    idx = rng.integers(0, arr.size, size=(int(n_resamples), arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return (mean, lo, hi)


def _normal_quantile(p: float) -> float:
    """Inverse normal CDF via Beasley-Springer-Moro. numpy-only."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = (
        -3.969683028665376e01, 2.209460984245205e02,
        -2.759285104469687e02, 1.383577518672690e02,
        -3.066479806614716e01, 2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01, 1.615858368580409e02,
        -1.556989798598866e02, 6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e00, -2.549732539343734e00,
        4.374664141464968e00, 2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e00, 3.754408661907416e00,
    )
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (
            ((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]
        ) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )
