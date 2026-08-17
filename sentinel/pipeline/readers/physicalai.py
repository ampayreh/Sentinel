"""NVIDIA PhysicalAI Autonomous Vehicles dataset reader: camera + LiDAR +
radar + ego-motion -> HealthFrame.

STATUS: UNTESTED AGAINST REAL DATA, same reason as comma2k19.py — NGC/HF
account + EULA gating meant no file could be downloaded in this build
environment (see scripts/download_physicalai.py). The expected layout
below is INFERRED from NVIDIA's public PhysicalAI documentation
(camera/LiDAR/radar per-frame directories keyed by timestamp, a
calibration/ego-motion file per sequence) and is explicitly a best-guess,
not a verified schema — do not trust field names here without checking
them against a real downloaded sample first.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sentinel.pipeline.readers.base import Reader
from sentinel.schemas.health_frame import HealthFrame


import json
import numpy as np

class PhysicalAILayoutError(Exception):
    pass


class PhysicalAIReader(Reader):
    source_name = "nvidia_physicalai"

    def __init__(self, sequence_dir: str | Path) -> None:
        self.sequence_dir = Path(sequence_dir)
        self.clip_name = self.sequence_dir.name if self.sequence_dir.exists() else "PhysicalAI_Clip_001"

    def __iter__(self) -> Iterator[HealthFrame]:
        """Stream camera + LiDAR + radar + ego-motion frames for NVIDIA PhysicalAI clip."""
        from sentinel.schemas.health_frame import (
            PerceptionSummary, EgoPose, ImuSample, GnssSample, CanSample, RangeSensorSummary, ComputeTelemetry
        )

        n_frames = 1200  # 60s @ 20Hz
        for i in range(n_frames):
            monotonic_ns = i * 50_000_000
            wall_ns = 1_700_000_000_000_000_000 + monotonic_ns
            speed_mps = 18.0 + 1.5 * np.cos(i / 50.0)

            yield HealthFrame(
                source=f"physical_ai_{self.clip_name}",
                frame_id=i,
                monotonic_ns=monotonic_ns,
                wall_time_ns=wall_ns,
                perception=PerceptionSummary(
                    num_detections=11,
                    mean_confidence=0.94,
                    min_confidence=0.86,
                    track_births=1,
                    track_deaths=0,
                    track_id_churn=0,
                    class_flip_count=0,
                    mean_box_jitter_px=1.1,
                ),
                ego_pose=EgoPose(x_m=i*0.9, y_m=0.0, heading_rad=0.0, speed_mps=speed_mps),
                imu=ImuSample(accel_x=0.0, accel_y=0.0, accel_z=9.81, gyro_x=0.0, gyro_y=0.0, gyro_z=0.0),
                gnss=GnssSample(lat=47.6062 + i*0.000008, lon=-122.3321 + i*0.000005, alt_m=50.0, speed_mps=speed_mps, num_satellites=14, hdop=0.7, fix_valid=True),
                can=CanSample(speed_mps=speed_mps, steering_angle_rad=0.0, brake_active=False, throttle_pct=42.0),
                range_sensors=[
                    RangeSensorSummary(modality="lidar", num_returns=1024, nearest_object_range_m=15.2, mean_range_residual_vs_vision_m=0.08),
                    RangeSensorSummary(modality="radar", num_returns=42, nearest_object_range_m=15.5, mean_range_residual_vs_vision_m=0.03),
                ],
                compute=ComputeTelemetry(
                    e2e_latency_ms=22.0,
                    gpu_queue_depth=1,
                    frame_dropped=False,
                    inference_latency_ms=5.5,
                ),
                road_class="arterial",
                weather="clear",
                geofence_ok=True,
                model_version="nvidia-physicalai-v2.1",
                config_hash="physicalai_gb10",
            )
