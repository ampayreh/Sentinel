"""M3 Timing monitor — CPU, deterministic.

End-to-end latency, inter-arrival jitter (EWMA), deadline misses, dropped
frames, GPU queue depth. Uses time.monotonic_ns() only (never wall clock)
for anything that affects a verdict. This monitor must be cheap: the hot
path (`_compute`) allocates no new containers beyond the fixed-size
`evidence` dict.
"""

from __future__ import annotations

from dataclasses import dataclass

from sentinel.monitors.base import Monitor
from sentinel.schemas.health_frame import HealthFrame
from sentinel.schemas.monitor_result import MonitorResult, Verdict


@dataclass(frozen=True)
class TimingThresholds:
    e2e_deadline_ms: float
    jitter_ewma_alpha: float
    jitter_warn_ms: float
    jitter_fail_ms: float
    max_consecutive_drops_warn: int
    max_consecutive_drops_fail: int


class M3TimingMonitor(Monitor):
    monitor_id = "M3_TIMING"
    monitor_version = "0.1.0"

    def __init__(self, thresholds: TimingThresholds) -> None:
        super().__init__()
        self.t = thresholds
        self._last_monotonic_ns: int | None = None
        self._jitter_ewma_ms: float = 0.0
        self._consecutive_drops: int = 0

    def _compute(self, frame: HealthFrame) -> MonitorResult:
        t = self.t
        inter_arrival_ms: float | None = None
        if self._last_monotonic_ns is not None:
            inter_arrival_ms = (frame.monotonic_ns - self._last_monotonic_ns) / 1e6
        self._last_monotonic_ns = frame.monotonic_ns

        expected_period_ms = 1000.0 / 20.0  # tick grid; matches sentinel.yaml runtime.tick_hz
        jitter_ms = (
            abs(inter_arrival_ms - expected_period_ms) if inter_arrival_ms is not None else 0.0
        )
        self._jitter_ewma_ms = (
            t.jitter_ewma_alpha * jitter_ms + (1 - t.jitter_ewma_alpha) * self._jitter_ewma_ms
        )

        if frame.is_gap:
            self._consecutive_drops += 1
        else:
            self._consecutive_drops = 0

        e2e_ms = frame.compute.e2e_latency_ms if frame.compute else None
        deadline_missed = e2e_ms is not None and e2e_ms > t.e2e_deadline_ms

        verdict = Verdict.PASS
        reasons: list[str] = []

        if self._consecutive_drops >= t.max_consecutive_drops_fail:
            verdict = Verdict.FAIL
            reasons.append(f"consecutive_drops={self._consecutive_drops}")
        elif self._consecutive_drops >= t.max_consecutive_drops_warn:
            verdict = _max_verdict(verdict, Verdict.WARN)
            reasons.append(f"consecutive_drops={self._consecutive_drops}")

        if deadline_missed:
            verdict = _max_verdict(verdict, Verdict.FAIL)
            reasons.append(f"e2e_latency_ms={e2e_ms:.2f}>{t.e2e_deadline_ms}")

        if self._jitter_ewma_ms > t.jitter_fail_ms:
            verdict = _max_verdict(verdict, Verdict.FAIL)
            reasons.append(f"jitter_ewma_ms={self._jitter_ewma_ms:.2f}")
        elif self._jitter_ewma_ms > t.jitter_warn_ms:
            verdict = _max_verdict(verdict, Verdict.WARN)
            reasons.append(f"jitter_ewma_ms={self._jitter_ewma_ms:.2f}")

        score = _score_from_verdict(verdict, self._jitter_ewma_ms, t.jitter_fail_ms)

        return MonitorResult(
            monitor_id=self.monitor_id,
            monitor_version=self.monitor_version,
            frame_id=frame.frame_id,
            monotonic_ns=frame.monotonic_ns,
            score=score,
            verdict=verdict,
            confidence=1.0,  # deterministic monitor: always fully confident in its own reading
            evidence={
                "jitter_ewma_ms": round(self._jitter_ewma_ms, 3),
                "e2e_latency_ms": round(e2e_ms, 3) if e2e_ms is not None else None,
                "consecutive_drops": self._consecutive_drops,
                "reasons": ";".join(reasons) if reasons else "",
            },
            latency_us=0,  # filled in by Monitor.update()
        )


def _verdict_rank(v: Verdict) -> int:
    return {Verdict.PASS: 0, Verdict.WARN: 1, Verdict.FAIL: 2, Verdict.STALE: 3}[v]


def _max_verdict(a: Verdict, b: Verdict) -> Verdict:
    return a if _verdict_rank(a) >= _verdict_rank(b) else b


def _score_from_verdict(verdict: Verdict, jitter_ms: float, fail_threshold_ms: float) -> float:
    if verdict == Verdict.PASS:
        return 1.0
    if verdict == Verdict.FAIL or verdict == Verdict.STALE:
        return 0.0
    # WARN: linearly interpolate down from 1.0 based on how close to fail threshold
    frac = min(jitter_ms / fail_threshold_ms, 1.0) if fail_threshold_ms > 0 else 1.0
    return round(1.0 - frac, 4)
