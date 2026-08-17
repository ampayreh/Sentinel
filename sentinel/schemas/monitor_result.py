"""MonitorResult: the common output contract every monitor produces.

`sentinel.trust.coordinator` fuses a dict of these into a TrustState. No
monitor is allowed to skip fields — a monitor that cannot compute a score
this tick must say so via `verdict=STALE` rather than omitting output,
because "monitor stopped producing" is itself a safety signal (see
`Monitor.health_check` in sentinel.monitors.base).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, field_validator


class Verdict(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    STALE = "STALE"  # monitor itself is unhealthy / not producing fresh output


class MonitorResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    monitor_id: str  # "M1_PERCEPTION" | "M2_SENSOR_AGREEMENT" | "M3_TIMING" | "M4_ODD"
    monitor_version: str
    frame_id: int
    monotonic_ns: int

    score: float  # 0.0 = worst, 1.0 = best (nominal), normalized across all monitors
    verdict: Verdict
    confidence: float  # 0..1, monitor's confidence in its own score
    evidence: dict[str, float | int | str | bool | None]
    latency_us: int

    @field_validator("score", "confidence")
    @classmethod
    def _unit_interval(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"expected value in [0,1], got {v}")
        return v
