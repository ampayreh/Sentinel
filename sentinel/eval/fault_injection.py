"""Fault injection: reproduce DMV/NHTSA-taxonomy failure classes on top of
the synthetic nominal generator.

Each fault type below maps to a `FailureClass` from
`sentinel.pipeline.readers.dmv_taxonomy` (see `FAULT_TO_FAILURE_CLASS`),
so eval results can be reported per-failure-class against the same
taxonomy the monitor set was designed from.

Design: a `FaultSpec` describes WHEN (start/duration, in ticks) and HOW
STRONG a fault is; `inject_faults` wraps a nominal frame iterator and
returns both the mutated frames and the ground-truth fault intervals (tick
ranges), which `sentinel.eval.metrics` needs to compute detection lead
time and per-class detection rate against.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum

from sentinel.pipeline.readers.dmv_taxonomy import FailureClass
from sentinel.schemas.health_frame import GnssSample, HealthFrame


class FaultType(str, Enum):
    SENSOR_DROPOUT = "sensor_dropout"
    CALIBRATION_DRIFT = "calibration_drift"
    PERCEPTION_DEGRADATION = "perception_degradation"
    LATENCY_SPIKE = "latency_spike"
    CLOCK_SKEW = "clock_skew"
    GNSS_MULTIPATH = "gnss_multipath"
    STALE_FRAME = "stale_frame"
    ODD_EXIT = "odd_exit"


FAULT_TO_FAILURE_CLASS: dict[FaultType, FailureClass] = {
    FaultType.SENSOR_DROPOUT: FailureClass.HARDWARE_SENSOR_FAULT,
    FaultType.CALIBRATION_DRIFT: FailureClass.HARDWARE_SENSOR_FAULT,
    FaultType.PERCEPTION_DEGRADATION: FailureClass.PERCEPTION_FAILURE,
    FaultType.LATENCY_SPIKE: FailureClass.COMMUNICATION_LATENCY_FAULT,
    FaultType.CLOCK_SKEW: FailureClass.COMMUNICATION_LATENCY_FAULT,
    FaultType.GNSS_MULTIPATH: FailureClass.LOCALIZATION_FAULT,
    FaultType.STALE_FRAME: FailureClass.SOFTWARE_FAULT,
    FaultType.ODD_EXIT: FailureClass.LOCALIZATION_FAULT,  # ODD exit modeled here via GNSS/geofence
}


@dataclass(frozen=True)
class FaultSpec:
    fault_type: FaultType
    start_tick: int
    duration_ticks: int
    magnitude: float = 1.0  # fault-specific scale, documented per branch below

    @property
    def end_tick(self) -> int:
        return self.start_tick + self.duration_ticks


def _apply_fault(frame: HealthFrame, spec: FaultSpec, ticks_into_fault: int) -> HealthFrame:
    ft = spec.fault_type

    if ft == FaultType.SENSOR_DROPOUT:
        # A genuine sensor dropout means the sensor produced nothing this
        # tick, not just "this frame arrived late" — null the GNSS and
        # range-sensor payload (modeling a GNSS/LiDAR outage) in addition
        # to marking is_gap, so M2 (missing cross-modal data) and M3
        # (is_gap) both have a chance to see it, matching the real-world
        # failure mode ("hardware discrepancy: sensor fault") this fault
        # class is meant to reproduce.
        return frame.model_copy(update={"is_gap": True, "gap_reason": "dropped", "gnss": None, "range_sensors": []})

    if ft == FaultType.CALIBRATION_DRIFT:
        # Linearly growing GNSS speed bias — the "slow drift thresholds
        # miss" failure mode CUSUM specifically targets.
        drift = spec.magnitude * (ticks_into_fault / max(spec.duration_ticks, 1)) * 5.0
        if frame.gnss is None:
            return frame
        return frame.model_copy(
            update={"gnss": frame.gnss.model_copy(update={"speed_mps": (frame.gnss.speed_mps or 0) + drift})}
        )

    if ft == FaultType.PERCEPTION_DEGRADATION:
        # Simulates fog/glare/occlusion: confidence craters, jitter spikes.
        if frame.perception is None:
            return frame
        p = frame.perception
        return frame.model_copy(
            update={
                "perception": p.model_copy(
                    update={
                        "mean_confidence": max(0.0, (p.mean_confidence or 0.9) - 0.5 * spec.magnitude),
                        "min_confidence": max(0.0, (p.min_confidence or 0.8) - 0.6 * spec.magnitude),
                        "mean_box_jitter_px": (p.mean_box_jitter_px or 1.5) + 8.0 * spec.magnitude,
                        "class_flip_count": p.class_flip_count + int(3 * spec.magnitude),
                    }
                )
            }
        )

    if ft == FaultType.LATENCY_SPIKE:
        from sentinel.schemas.health_frame import ComputeTelemetry
        c = frame.compute or ComputeTelemetry(e2e_latency_ms=20.0, gpu_queue_depth=1, frame_dropped=False)
        return frame.model_copy(
            update={
                "compute": c.model_copy(
                    update={"e2e_latency_ms": (c.e2e_latency_ms or 20.0) + 100.0 * spec.magnitude}
                )
            }
        )

    if ft == FaultType.CLOCK_SKEW:
        # Wall-clock diverges from monotonic — decision logic must not
        # care (it never reads wall_time_ns), but this is recorded as a
        # known, currently-undetected fault class (see eval results.md
        # gap notes: no monitor currently compares wall vs monotonic
        # drift, since decision logic is required to ignore wall-clock).
        skew_ns = int(spec.magnitude * 1_000_000_000 * ticks_into_fault)
        return frame.model_copy(update={"wall_time_ns": frame.wall_time_ns + skew_ns})

    if ft == FaultType.GNSS_MULTIPATH:
        if frame.gnss is None:
            return frame
        return frame.model_copy(
            update={
                "gnss": frame.gnss.model_copy(
                    update={
                        "hdop": (frame.gnss.hdop or 0.9) + 5.0 * spec.magnitude,
                        "num_satellites": max(0, (frame.gnss.num_satellites or 10) - 6),
                        "speed_mps": (frame.gnss.speed_mps or 0) + 15.0 * spec.magnitude,
                    }
                )
            }
        )

    if ft == FaultType.STALE_FRAME:
        # Caller handles this one specially (needs access to the previous
        # frame) — see inject_faults below.
        return frame

    if ft == FaultType.ODD_EXIT:
        gnss_update = (
            frame.gnss.model_copy(update={"fix_valid": False}) if frame.gnss else GnssSample(fix_valid=False)
        )
        return frame.model_copy(update={"geofence_ok": False, "gnss": gnss_update})

    raise ValueError(f"unhandled fault type {ft}")


def inject_faults(
    frames: Iterator[HealthFrame], faults: list[FaultSpec]
) -> Iterator[tuple[HealthFrame, FaultSpec | None]]:
    """Yields (frame, active_fault_or_None) so callers can build ground
    truth fault intervals without re-deriving them from mutated frames."""
    last_frame: HealthFrame | None = None
    for tick, frame in enumerate(frames):
        active = next((f for f in faults if f.start_tick <= tick < f.end_tick), None)
        if active is None:
            yield frame, None
            last_frame = frame
            continue

        if active.fault_type == FaultType.STALE_FRAME and last_frame is not None:
            # Freeze perception/sensor payload at the pre-fault frame's
            # values while timestamps keep advancing — models a stalled
            # perception process that keeps publishing its last output.
            frozen = last_frame.model_copy(
                update={
                    "frame_id": frame.frame_id,
                    "monotonic_ns": frame.monotonic_ns,
                    "wall_time_ns": frame.wall_time_ns,
                    "tick_seq": frame.tick_seq,
                }
            )
            yield frozen, active
            continue

        mutated = _apply_fault(frame, active, tick - active.start_tick)
        yield mutated, active
        last_frame = frame
