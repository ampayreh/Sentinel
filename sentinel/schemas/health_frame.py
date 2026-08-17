"""HealthFrame: the single normalized schema every data source is mapped into.

Every reader in `sentinel.pipeline.readers`, real or synthetic, ultimately
yields a stream of these. Downstream code (monitors, trust coordinator,
evidence writer) never sees source-specific fields again after this point.

Design notes:
- All fields that a given source cannot populate are left as `None`. A
  `None` is a documented absence, not an assumed zero — monitors must
  treat missing fields as evidence of degraded observability, not as
  "value is 0.0".
- `tick_seq` and `is_gap` are set by `sentinel.pipeline.align`, not by
  readers. Readers only ever emit raw, source-timestamped frames.
- Every field that participates in a decision-boundary comparison
  downstream is a plain float/int, never a numpy scalar, so hashing and
  equality behave predictably across replay.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class RoadClass(str, Enum):
    UNKNOWN = "unknown"
    HIGHWAY = "highway"
    ARTERIAL = "arterial"
    RESIDENTIAL = "residential"
    PARKING = "parking"


class LightingCondition(str, Enum):
    UNKNOWN = "unknown"
    DAYLIGHT = "daylight"
    DUSK_DAWN = "dusk_dawn"
    NIGHT = "night"
    NIGHT_UNLIT = "night_unlit"


class WeatherCondition(str, Enum):
    UNKNOWN = "unknown"
    CLEAR = "clear"
    RAIN = "rain"
    FOG = "fog"
    SNOW = "snow"
    GLARE = "glare"


class PerceptionSummary(BaseModel):
    """Per-tick aggregate of the perception stack's detections."""

    model_config = ConfigDict(frozen=True)

    num_detections: int = 0
    mean_confidence: float | None = None
    min_confidence: float | None = None
    track_births: int = 0
    track_deaths: int = 0
    track_id_churn: int = 0  # ID switches this tick
    class_flip_count: int = 0  # detections whose class changed vs. last tick
    mean_box_jitter_px: float | None = None


class EgoPose(BaseModel):
    model_config = ConfigDict(frozen=True)

    x_m: float | None = None
    y_m: float | None = None
    heading_rad: float | None = None
    speed_mps: float | None = None  # vision/pose-derived speed estimate


class ImuSample(BaseModel):
    model_config = ConfigDict(frozen=True)

    accel_x: float | None = None
    accel_y: float | None = None
    accel_z: float | None = None
    gyro_x: float | None = None
    gyro_y: float | None = None
    gyro_z: float | None = None


class GnssSample(BaseModel):
    model_config = ConfigDict(frozen=True)

    lat: float | None = None
    lon: float | None = None
    alt_m: float | None = None
    speed_mps: float | None = None
    num_satellites: int | None = None
    hdop: float | None = None  # horizontal dilution of precision -> localization quality proxy
    fix_valid: bool | None = None


class CanSample(BaseModel):
    model_config = ConfigDict(frozen=True)

    speed_mps: float | None = None
    steering_angle_rad: float | None = None
    brake_active: bool | None = None
    throttle_pct: float | None = None


class RangeSensorSummary(BaseModel):
    """Summary stats for radar or LiDAR, kept coarse on purpose: monitors
    consume statistics, not raw point clouds."""

    model_config = ConfigDict(frozen=True)

    modality: str  # "radar" | "lidar"
    num_returns: int | None = None
    nearest_object_range_m: float | None = None
    mean_range_residual_vs_vision_m: float | None = None


class ComputeTelemetry(BaseModel):
    model_config = ConfigDict(frozen=True)

    e2e_latency_ms: float | None = None
    gpu_queue_depth: int | None = None
    frame_dropped: bool = False
    inference_latency_ms: float | None = None


class HealthFrame(BaseModel):
    """One time-aligned tick of the normalized health stream."""

    model_config = ConfigDict(frozen=True)

    # Identity / provenance
    source: str
    frame_id: int
    monotonic_ns: int
    wall_time_ns: int  # epoch ns, for human-readable reporting only; never used in decisions

    # Alignment metadata (set by sentinel.pipeline.align, not by readers)
    tick_seq: int | None = None
    is_gap: bool = False
    gap_reason: str | None = None  # "dropped" | "late" | "out_of_order" | None

    # Payload
    perception: PerceptionSummary | None = None
    ego_pose: EgoPose | None = None
    imu: ImuSample | None = None
    gnss: GnssSample | None = None
    can: CanSample | None = None
    range_sensors: list[RangeSensorSummary] = Field(default_factory=list)
    compute: ComputeTelemetry | None = None

    # Operating context
    road_class: RoadClass = RoadClass.UNKNOWN
    lighting: LightingCondition = LightingCondition.UNKNOWN
    weather: WeatherCondition = WeatherCondition.UNKNOWN
    geofence_ok: bool | None = None

    # Model/config provenance for this tick
    model_version: str | None = None
    config_hash: str | None = None
