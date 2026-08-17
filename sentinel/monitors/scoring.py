"""Shared piecewise score mapping used by monitors that threshold a
continuous statistic (Mahalanobis distance, z-score, ...) into PASS/WARN/
FAIL.

Design rationale: score must equal 1.0 (full nominal trust) for the ENTIRE
PASS region, not decay continuously from the moment a statistic is
nonzero. A raw `1 - value/fail_threshold` mapping makes ordinary nominal
noise (which is never exactly 0) register as partial risk, which then
leaks into the trust coordinator's fused_risk and causes false
interventions under nominal operation — exactly the failure this function
exists to prevent. Below `warn_threshold`, nominal statistical
fluctuation is treated as zero risk; between warn and fail, score
interpolates linearly to zero; at/above fail, score is 0.
"""

from __future__ import annotations


def piecewise_score(value: float, warn_threshold: float, fail_threshold: float) -> float:
    if fail_threshold <= warn_threshold:
        raise ValueError("fail_threshold must be > warn_threshold")
    if value <= warn_threshold:
        return 1.0
    if value >= fail_threshold:
        return 0.0
    frac = (value - warn_threshold) / (fail_threshold - warn_threshold)
    return round(1.0 - frac, 4)
