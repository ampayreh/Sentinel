"""Monitor ABC. Every monitor (statistical, ML, deterministic, rule-based)
implements this same narrow interface so the trust coordinator can treat
them uniformly, and so heterogeneity (the point of having four of them)
lives in the implementation, not the contract.

Staleness contract: a monitor that has not produced a fresh result within
`staleness_budget_ticks` ticks must self-report `Verdict.STALE` via
`health_check()`, rather than the coordinator having to infer silence.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from sentinel.schemas.health_frame import HealthFrame
from sentinel.schemas.monitor_result import MonitorResult, Verdict


class Monitor(ABC):
    monitor_id: str
    monitor_version: str
    staleness_budget_ticks: int = 5  # default: 5 ticks @ 20Hz = 250ms without output -> STALE

    def __init__(self) -> None:
        self._last_result_tick: int | None = None
        self._current_tick: int = 0

    @abstractmethod
    def _compute(self, frame: HealthFrame) -> MonitorResult:
        """Subclasses implement the actual monitor logic here. Must not
        raise on bad/missing input — degrade to a low-confidence WARN/FAIL
        result instead, since a crashing monitor is worse than a
        conservative one."""
        raise NotImplementedError

    def update(self, frame: HealthFrame) -> MonitorResult:
        start = time.monotonic_ns()
        self._current_tick += 1
        try:
            result = self._compute(frame)
        except Exception as exc:  # noqa: BLE001 - monitors must never crash the loop
            result = MonitorResult(
                monitor_id=self.monitor_id,
                monitor_version=self.monitor_version,
                frame_id=frame.frame_id,
                monotonic_ns=frame.monotonic_ns,
                score=0.0,
                verdict=Verdict.FAIL,
                confidence=1.0,
                evidence={"error": str(exc)[:200]},
                latency_us=(time.monotonic_ns() - start) // 1000,
            )
        self._last_result_tick = self._current_tick
        return result

    def health_check(self, frame: HealthFrame) -> MonitorResult | None:
        """Called by the coordinator every tick regardless of whether
        `update` was invoked this tick. Returns a STALE result if the
        monitor has gone quiet, else None (coordinator uses the last
        `update` result)."""
        if self._last_result_tick is None:
            return None
        ticks_since = self._current_tick - self._last_result_tick
        if ticks_since > self.staleness_budget_ticks:
            return MonitorResult(
                monitor_id=self.monitor_id,
                monitor_version=self.monitor_version,
                frame_id=frame.frame_id,
                monotonic_ns=frame.monotonic_ns,
                score=0.0,
                verdict=Verdict.STALE,
                confidence=1.0,
                evidence={"ticks_since_last_result": ticks_since},
                latency_us=0,
            )
        return None
