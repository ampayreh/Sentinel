"""M0 stub monitor: always PASS, used only to prove the frame -> monitor ->
result path works end to end before any real monitor exists. Not wired
into the trust coordinator's default monitor set after M2 lands.
"""

from __future__ import annotations

from sentinel.monitors.base import Monitor
from sentinel.schemas.health_frame import HealthFrame
from sentinel.schemas.monitor_result import MonitorResult, Verdict


class StubMonitor(Monitor):
    monitor_id = "M0_STUB"
    monitor_version = "0.0.1"

    def _compute(self, frame: HealthFrame) -> MonitorResult:
        return MonitorResult(
            monitor_id=self.monitor_id,
            monitor_version=self.monitor_version,
            frame_id=frame.frame_id,
            monotonic_ns=frame.monotonic_ns,
            score=1.0,
            verdict=Verdict.PASS,
            confidence=1.0,
            evidence={"is_gap": frame.is_gap},
            latency_us=1,
        )
