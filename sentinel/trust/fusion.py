"""Severity-weighted evidence fusion.

fused_risk = sum(w_i * risk_i) / sum(w_i_active)

where risk_i = 1 - monitor_i.score, EXCEPT verdict FAIL or STALE forces
risk_i = 1.0 regardless of the numeric score — a monitor can report FAIL
with any score, and that verdict always dominates. Only monitors that
reported this tick (present in `results`) contribute; a monitor that
hasn't reported yet is excluded from the weighted average rather than
assumed nominal (excluding, not zero-filling, avoids diluting risk when
some monitors simply haven't produced their first result yet at startup).

This is intentionally NOT a max() over monitors (spec explicitly rules
that out) — a max() throws away the fact that several monitors mildly
elevated together is a different, and often earlier, signal than one
monitor screaming alone. It is also intentionally not a Bayesian/DS
fusion — see the design note in the top-level response for the trade-off.
"""

from __future__ import annotations

from dataclasses import dataclass

from sentinel.schemas.monitor_result import MonitorResult, Verdict


@dataclass(frozen=True)
class FusionWeights:
    weights: dict[str, float]  # monitor_id -> weight, must be > 0


def _risk_for_result(result: MonitorResult) -> float:
    if result.verdict in (Verdict.FAIL, Verdict.STALE):
        return 1.0
    return max(0.0, min(1.0, 1.0 - result.score))


@dataclass(frozen=True)
class FusionOutput:
    fused_risk: float
    per_monitor_risk: dict[str, float]
    active_weight_total: float
    missing_monitors: tuple[str, ...]  # monitors in the weight table with no result this tick


def fuse(results: dict[str, MonitorResult], weights: FusionWeights) -> FusionOutput:
    per_monitor_risk: dict[str, float] = {}
    weighted_sum = 0.0
    weight_total = 0.0
    missing: list[str] = []

    # Sorted iteration for determinism (spec: "sorted dict iteration").
    for monitor_id in sorted(weights.weights.keys()):
        w = weights.weights[monitor_id]
        result = results.get(monitor_id)
        if result is None:
            missing.append(monitor_id)
            continue
        risk = _risk_for_result(result)
        per_monitor_risk[monitor_id] = risk
        weighted_sum += w * risk
        weight_total += w

    fused_risk = (weighted_sum / weight_total) if weight_total > 0 else 1.0
    return FusionOutput(
        fused_risk=round(fused_risk, 6),
        per_monitor_risk=per_monitor_risk,
        active_weight_total=weight_total,
        missing_monitors=tuple(missing),
    )
