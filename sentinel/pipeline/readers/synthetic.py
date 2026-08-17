"""Deterministic synthetic nominal-driving generator.

This is the "replace CARLA" simulator called for in the spec: CARLA has no
official arm64 Linux build, so instead of depending on an x86 host over the
network, Sentinel ships a lightweight, fully deterministic scenario
generator. It is not a physics/rendering simulator — it does not stand in
for perception model quality — it stands in for *sensor health signal
shapes* so the monitors, trust coordinator, policy gate, and replay engine
can be developed, tested, and demoed without any external dataset or GPU.

Determinism: given the same `seed` and `ScenarioSpec`, `SyntheticReader`
yields byte-identical HealthFrame sequences. All randomness flows through
a single `numpy.random.Generator` seeded once at construction.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from sentinel.pipeline.readers.base import Reader
from sentinel.schemas.health_frame import (
    CanSample,
    ComputeTelemetry,
    EgoPose,
    GnssSample,
    HealthFrame,
    ImuSample,
    LightingCondition,
    PerceptionSummary,
    RangeSensorSummary,
    RoadClass,
    WeatherCondition,
)

_NS_PER_S = 1_000_000_000


@dataclass(frozen=True)
class ScenarioSpec:
    """Declarative description of a synthetic drive. Fault injection
    (sentinel.eval.fault_injection) composes with this by wrapping a
    nominal ScenarioSpec's output stream, rather than this class knowing
    about faults itself — keeps the nominal generator honest."""

    name: str = "nominal_urban_loop"
    duration_s: float = 180.0
    source_hz: float = 20.0
    nominal_speed_mps: float = 12.0
    road_class: RoadClass = RoadClass.ARTERIAL
    lighting: LightingCondition = LightingCondition.DAYLIGHT
    weather: WeatherCondition = WeatherCondition.CLEAR
    model_version: str = "sentinel-sim-0.1.0"
    config_hash: str = "synthetic"
    drop_probability: float = 0.0  # fraction of frames dropped entirely (gap)
    late_probability: float = 0.0  # fraction of frames delivered out of order
    extra: dict[str, float] = field(default_factory=dict)


class SyntheticReader(Reader):
    source_name = "synthetic"

    def __init__(self, spec: ScenarioSpec, seed: int) -> None:
        self.spec = spec
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[HealthFrame]:
        spec = self.spec
        n_frames = int(spec.duration_s * spec.source_hz)
        dt_ns = int(_NS_PER_S / spec.source_hz)
        x, y, heading = 0.0, 0.0, 0.0

        # Pre-draw all randomness up front, in a fixed order, so the
        # sequence of frames does not depend on control flow (e.g. whether
        # a frame gets dropped) perturbing the RNG stream differently
        # across runs.
        confidence_noise = self._rng.normal(0.93, 0.02, n_frames)
        imu_noise = self._rng.normal(0.0, 0.05, (n_frames, 6))
        gnss_noise = self._rng.normal(0.0, 0.5, (n_frames, 2))
        # Vision/pose-derived speed estimate carries its own small
        # independent noise relative to true (CAN) speed, as a real
        # perception-derived odometry estimate would — this is what gives
        # M2's cross-modal residual channels a non-degenerate covariance
        # to calibrate against instead of an exact-zero residual.
        vision_speed_noise = self._rng.normal(0.0, 0.15, n_frames)
        gnss_speed_noise = self._rng.normal(0.0, 0.25, n_frames)
        drop_draws = self._rng.random(n_frames)
        late_draws = self._rng.random(n_frames)

        pending_late: HealthFrame | None = None

        for i in range(n_frames):
            monotonic_ns = i * dt_ns
            wall_ns = 1_700_000_000 * _NS_PER_S + monotonic_ns

            if drop_draws[i] < spec.drop_probability:
                continue  # simulates a dropped sample; align.py will mark the gap

            speed = spec.nominal_speed_mps + 0.5 * np.sin(i / 97.0)
            heading += 0.001 * np.sin(i / 233.0)
            x += float(speed * np.cos(heading) / spec.source_hz)
            y += float(speed * np.sin(heading) / spec.source_hz)

            frame = HealthFrame(
                source=self.source_name,
                frame_id=i,
                monotonic_ns=monotonic_ns,
                wall_time_ns=wall_ns,
                perception=PerceptionSummary(
                    num_detections=int(self._rng.integers(3, 12)),
                    mean_confidence=float(np.clip(confidence_noise[i], 0.0, 1.0)),
                    min_confidence=float(np.clip(confidence_noise[i] - 0.15, 0.0, 1.0)),
                    track_births=int(self._rng.integers(0, 2)),
                    track_deaths=int(self._rng.integers(0, 2)),
                    track_id_churn=0,
                    class_flip_count=0,
                    mean_box_jitter_px=float(abs(self._rng.normal(1.5, 0.5))),
                ),
                ego_pose=EgoPose(
                    x_m=x, y_m=y, heading_rad=heading, speed_mps=speed + float(vision_speed_noise[i])
                ),
                imu=ImuSample(
                    accel_x=float(imu_noise[i, 0]),
                    accel_y=float(imu_noise[i, 1]),
                    accel_z=9.81 + float(imu_noise[i, 2]),
                    gyro_x=float(imu_noise[i, 3]),
                    gyro_y=float(imu_noise[i, 4]),
                    gyro_z=float(imu_noise[i, 5]),
                ),
                gnss=GnssSample(
                    lat=37.7749 + (x + gnss_noise[i, 0]) / 111_000.0,
                    lon=-122.4194 + (y + gnss_noise[i, 1]) / 111_000.0,
                    alt_m=10.0,
                    speed_mps=speed + float(gnss_speed_noise[i]),
                    num_satellites=int(self._rng.integers(8, 14)),
                    hdop=float(abs(self._rng.normal(0.9, 0.1))),
                    fix_valid=True,
                ),
                can=CanSample(
                    speed_mps=speed,
                    steering_angle_rad=heading * 0.1,
                    brake_active=False,
                    throttle_pct=float(np.clip(40 + 5 * np.sin(i / 50.0), 0, 100)),
                ),
                range_sensors=[
                    RangeSensorSummary(
                        modality="lidar",
                        num_returns=int(self._rng.integers(800, 1200)),
                        nearest_object_range_m=float(abs(self._rng.normal(25, 8))),
                        mean_range_residual_vs_vision_m=float(abs(self._rng.normal(0.1, 0.05))),
                    ),
                    RangeSensorSummary(
                        modality="radar",
                        num_returns=int(self._rng.integers(10, 40)),
                        nearest_object_range_m=float(abs(self._rng.normal(25, 9))),
                        mean_range_residual_vs_vision_m=float(abs(self._rng.normal(0.15, 0.07))),
                    ),
                ],
                compute=ComputeTelemetry(
                    e2e_latency_ms=float(abs(self._rng.normal(22, 3))),
                    gpu_queue_depth=int(self._rng.integers(0, 3)),
                    frame_dropped=False,
                    inference_latency_ms=float(abs(self._rng.normal(8, 1.5))),
                ),
                road_class=spec.road_class,
                lighting=spec.lighting,
                weather=spec.weather,
                geofence_ok=True,
                model_version=spec.model_version,
                config_hash=spec.config_hash,
            )

            if late_draws[i] < spec.late_probability and pending_late is None:
                pending_late = frame
                continue
            if pending_late is not None:
                yield frame
                yield pending_late
                pending_late = None
                continue

            yield frame

        if pending_late is not None:
            yield pending_late
