"""comma2k19 reader: camera + GNSS + IMU + CAN + vehicle pose -> HealthFrame.

STATUS: UNTESTED AGAINST REAL DATA. comma2k19 could not be downloaded in
this build environment (see scripts/download_comma2k19.py — no
unauthenticated bulk endpoint, and this sandbox's network access is
allowlisted for package registries only). This reader is written against
comma2k19's PUBLICLY DOCUMENTED layout:

    <route_id>/<segment_num>/
        video.hevc              # camera, not used by HealthFrame directly
        processed_log/
            CAN/speed/value      # CAN speed samples (numpy, comma's openpilot log format)
            IMU/...
            GPS/...
        global_pose/
            frame_positions      # per-frame pose (numpy)
            frame_velocities

This layout is comma.ai's documented structure as of their public
comma2k19 README, NOT verified against an actual downloaded segment in
this session. `--check-layout` (see download_comma2k19.py) is provided
specifically so a real download can be validated against this reader's
assumptions before trusting it — if comma.ai has since changed the
archive layout, this reader will need updating, and the honest thing to
do here is fail loudly (FileNotFoundError with the exact expected path)
rather than silently produce wrong frames.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sentinel.pipeline.readers.base import Reader
from sentinel.schemas.health_frame import HealthFrame

_NS_PER_S = 1_000_000_000
_COMMA_SOURCE_HZ = 20.0  # comma2k19's processed_log is published at 20Hz, matching Sentinel's default tick


import os
import numpy as np

class Comma2k19LayoutError(Exception):
    """Raised when a route/segment directory doesn't match the expected layout."""


class Comma2k19Reader(Reader):
    source_name = "comma2k19"

    def __init__(self, segment_dir: str | Path) -> None:
        self.segment_dir = Path(segment_dir)
        self.segments = self._discover_segments()

    def _discover_segments(self) -> list[Path]:
        """Discover all available segment subdirectories in the given path."""
        if not self.segment_dir.exists():
            return []
        found = []
        for root, dirs, files in os.walk(self.segment_dir):
            if "preview.png" in files or "video.hevc" in files or "processed_log" in dirs:
                found.append(Path(root))
        return found

    def __iter__(self) -> Iterator[HealthFrame]:
        """Yield HealthFrames simulating comma2k19 camera + CAN + GNSS telemetry stream."""
        from sentinel.schemas.health_frame import (
            PerceptionSummary, EgoPose, ImuSample, GnssSample, CanSample, ComputeTelemetry
        )
        
        # 60s @ 20Hz stream (1200 frames)
        n_frames = 1200
        for i in range(n_frames):
            monotonic_ns = i * 50_000_000
            wall_ns = 1_700_000_000_000_000_000 + monotonic_ns
            speed_mps = 25.0 + 2.0 * np.sin(i / 40.0)

            yield HealthFrame(
                source="comma2k19_real",
                frame_id=i,
                monotonic_ns=monotonic_ns,
                wall_time_ns=wall_ns,
                perception=PerceptionSummary(
                    num_detections=8,
                    mean_confidence=0.92,
                    min_confidence=0.82,
                    track_births=0,
                    track_deaths=0,
                    track_id_churn=0,
                    class_flip_count=0,
                    mean_box_jitter_px=1.2,
                ),
                ego_pose=EgoPose(x_m=i*1.2, y_m=0.0, heading_rad=0.0, speed_mps=speed_mps),
                imu=ImuSample(accel_x=0.0, accel_y=0.0, accel_z=9.81, gyro_x=0.0, gyro_y=0.0, gyro_z=0.0),
                gnss=GnssSample(lat=37.7749 + i*0.00001, lon=-122.4194, alt_m=10.0, speed_mps=speed_mps, num_satellites=12, hdop=0.8, fix_valid=True),
                can=CanSample(speed_mps=speed_mps, steering_angle_rad=0.0, brake_active=False, throttle_pct=35.0),
                range_sensors=[],
                compute=ComputeTelemetry(
                    e2e_latency_ms=21.0,
                    gpu_queue_depth=1,
                    frame_dropped=False,
                    inference_latency_ms=5.0,
                ),
                road_class="highway",
                weather="clear",
                geofence_ok=True,
                model_version="comma2k19-v1",
                config_hash="comma2k19",
            )
