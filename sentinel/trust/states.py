"""TrustState: the four authority levels Sentinel can grant, ordered from
most to least permissive. Ordering matters — `sentinel.policy.gate` and
the hysteresis logic both compare states by rank, not by name.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TrustState(str, Enum):
    FULL_AUTONOMY = "FULL_AUTONOMY"
    DEGRADED = "DEGRADED"
    MINIMAL_RISK = "MINIMAL_RISK"
    HANDOFF = "HANDOFF"


_RANK = {
    TrustState.FULL_AUTONOMY: 0,
    TrustState.DEGRADED: 1,
    TrustState.MINIMAL_RISK: 2,
    TrustState.HANDOFF: 3,
}


def rank(state: TrustState) -> int:
    """Higher rank = less authority. Used to enforce monotonicity: applying
    evidence can only move rank up (or leave it unchanged), never down."""
    return _RANK[state]


def more_permissive(a: TrustState, b: TrustState) -> bool:
    """True if `a` grants strictly more authority than `b`."""
    return rank(a) < rank(b)


@dataclass(frozen=True)
class RiskThresholds:
    full_autonomy_max_risk: float
    degraded_max_risk: float
    minimal_risk_max_risk: float


def state_for_risk(fused_risk: float, thresholds: RiskThresholds) -> TrustState:
    if fused_risk <= thresholds.full_autonomy_max_risk:
        return TrustState.FULL_AUTONOMY
    if fused_risk <= thresholds.degraded_max_risk:
        return TrustState.DEGRADED
    if fused_risk <= thresholds.minimal_risk_max_risk:
        return TrustState.MINIMAL_RISK
    return TrustState.HANDOFF
