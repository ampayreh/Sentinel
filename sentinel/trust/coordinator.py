"""TrustCoordinator: fuses monitor results into a TrustState every tick.

Pipeline per tick:
  1. fuse() -> instantaneous fused_risk from severity-weighted scoring.
  2. meta-monitoring -> detect correlated failure across monitors that
     should be independent; bump fused_risk if found.
  3. state_for_risk() -> instantaneous target TrustState for this tick.
  4. hysteresis.step() -> actual (possibly unchanged) TrustState, with
     escalate-fast/de-escalate-slow dynamics applied.
  5. Build a structured reason string citing which monitors and values
     drove the decision, for the evidence record.

Pure with respect to wall-clock and I/O: this class holds only in-memory
state (`HysteresisState`) and is fully driven by the `results` dict passed
into `step()`. Two coordinators fed the identical sequence of results
produce identical state sequences — this is what makes trust decisions
replayable.
"""

from __future__ import annotations

from dataclasses import dataclass

from sentinel.schemas.monitor_result import MonitorResult, Verdict
from sentinel.trust.fusion import FusionOutput, FusionWeights, fuse
from sentinel.trust.hysteresis import HysteresisConfig, HysteresisState, step
from sentinel.trust.states import RiskThresholds, TrustState, state_for_risk


@dataclass(frozen=True)
class MetaConfig:
    stale_monitor_risk: float
    correlated_failure_min_monitors: int
    correlated_failure_window_ticks: int
    correlated_failure_risk_bonus: float


@dataclass(frozen=True)
class TrustCoordinatorConfig:
    weights: FusionWeights
    risk_thresholds: RiskThresholds
    hysteresis: HysteresisConfig
    meta: MetaConfig


@dataclass
class TrustStepResult:
    trust_state: TrustState
    transitioned: bool
    rule_id: str
    reason: str
    fused_risk: float
    per_monitor_risk: dict[str, float]
    contributing_monitors: tuple[str, ...]
    correlated_failure_detected: bool


class TrustCoordinator:
    def __init__(self, config: TrustCoordinatorConfig) -> None:
        self.config = config
        self._hysteresis_state = HysteresisState()
        # Ring of recent (tick_index, set of monitor_ids that FAILed) for
        # correlated-failure detection over a sliding window.
        self._recent_failures: list[tuple[int, frozenset[str]]] = []
        self._tick_index = 0

    def _detect_correlated_failure(self, results: dict[str, MonitorResult]) -> tuple[bool, tuple[str, ...]]:
        cfg = self.config.meta
        failing_now = frozenset(
            mid for mid, r in sorted(results.items()) if r.verdict in (Verdict.FAIL, Verdict.STALE)
        )
        self._recent_failures.append((self._tick_index, failing_now))
        cutoff = self._tick_index - cfg.correlated_failure_window_ticks
        self._recent_failures = [(t, s) for t, s in self._recent_failures if t >= cutoff]

        union_recent: set[str] = set()
        for _, s in self._recent_failures:
            union_recent |= s
        if len(union_recent) >= cfg.correlated_failure_min_monitors:
            return True, tuple(sorted(union_recent))
        return False, ()

    def step(self, results: dict[str, MonitorResult]) -> TrustStepResult:
        cfg = self.config
        fusion: FusionOutput = fuse(results, cfg.weights)

        correlated, correlated_monitors = self._detect_correlated_failure(results)
        fused_risk = fusion.fused_risk
        if correlated:
            fused_risk = min(1.0, fused_risk + cfg.meta.correlated_failure_risk_bonus)

        target_state = state_for_risk(fused_risk, cfg.risk_thresholds)

        new_hyst, transitioned, rule_id = step(self._hysteresis_state, target_state, cfg.hysteresis)
        self._hysteresis_state = new_hyst
        self._tick_index += 1

        contributing = tuple(
            sorted(
                mid
                for mid, risk in fusion.per_monitor_risk.items()
                if risk >= 0.5  # meaningfully elevated, cited in the reason string
            )
        )

        reason_parts = [f"fused_risk={fused_risk:.3f}", f"target={target_state.value}", f"rule={rule_id}"]
        if contributing:
            reason_parts.append("contributing=" + ",".join(contributing))
        if fusion.missing_monitors:
            reason_parts.append("missing=" + ",".join(fusion.missing_monitors))
        if correlated:
            reason_parts.append("correlated_failure=" + ",".join(correlated_monitors))
        reason = "; ".join(reason_parts)

        return TrustStepResult(
            trust_state=new_hyst.current,
            transitioned=transitioned,
            rule_id=rule_id,
            reason=reason,
            fused_risk=fused_risk,
            per_monitor_risk=fusion.per_monitor_risk,
            contributing_monitors=contributing,
            correlated_failure_detected=correlated,
        )
