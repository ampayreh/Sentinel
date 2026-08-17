"""Named demo/run scenarios, including the spec's scripted 3-minute demo:
sensor agreement degrades -> Sentinel drops to DEGRADED -> an overtake
request is denied with a cited rule -> the condition clears -> authority
is restored only after the dwell time -> replayed from the log to prove
bit-identical decisions.
"""

from __future__ import annotations

from collections.abc import Iterator

from sentinel.eval.fault_injection import FaultSpec, FaultType, inject_faults
from sentinel.pipeline.readers.synthetic import ScenarioSpec, SyntheticReader
from sentinel.pipeline.stream import health_stream
from sentinel.schemas.health_frame import HealthFrame

TICK_HZ = 20.0


def nominal_urban_loop(duration_s: float = 180.0, seed: int = 1) -> Iterator[HealthFrame]:
    reader = SyntheticReader(ScenarioSpec(duration_s=duration_s, name="nominal_urban_loop"), seed=seed)
    yield from health_stream(reader)


def scripted_demo(seed: int = 1) -> Iterator[tuple[HealthFrame, FaultSpec | None]]:
    """The exact 3-minute (3600-tick @ 20Hz) scripted scenario from the
    spec: nominal driving, a calibration-drift fault starting at t=60s
    lasting 30s (long enough for CUSUM to trip and for the trust
    coordinator to escalate + dwell), then clean recovery."""
    duration_s = 180.0
    reader = SyntheticReader(ScenarioSpec(duration_s=duration_s, name="scripted_demo"), seed=seed)
    fault = FaultSpec(
        fault_type=FaultType.CALIBRATION_DRIFT,
        start_tick=int(60 * TICK_HZ),
        duration_ticks=int(30 * TICK_HZ),
        magnitude=1.5,
    )
    yield from inject_faults(health_stream(reader), [fault])


SCENARIOS = {
    "nominal_urban_loop": nominal_urban_loop,
    "scripted_demo": scripted_demo,
}
