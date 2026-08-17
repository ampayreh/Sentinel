"""M2 Sensor agreement monitor — CPU/CUDA, statistical.

Cross-modal consistency: camera-derived ego speed vs IMU-integrated speed
vs GNSS speed vs CAN speed; radar/LiDAR range vs vision-implied depth via
`mean_range_residual_vs_vision_m`. Computes:

  1. Normalized per-modality residuals against a reference (CAN speed, the
     most direct ground truth available in this schema).
  2. A Mahalanobis distance of the residual vector against a calibrated
     covariance matrix (see config/monitor_thresholds.yaml for the
     calibration note).
  3. A CUSUM statistic per residual channel to catch *slow drift* — a
     miscalibrated sensor whose residual creeps rather than jumps will not
     trip a fixed threshold quickly, which is exactly the failure mode
     CUSUM exists to catch.

This monitor never touches GPU/CUDA APIs directly in this build (no GPU in
the dev sandbox); the design supports a CuPy/cuML drop-in for the
covariance and distance math on GB10 — see the `xp` indirection below —
without changing the monitor's interface or thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sentinel.monitors.base import Monitor
from sentinel.monitors.scoring import piecewise_score
from sentinel.schemas.health_frame import HealthFrame
from sentinel.schemas.monitor_result import MonitorResult, Verdict

# `xp` indirection: swap to `import cupy as xp` on GB10 to move this
# monitor's linear algebra onto the GPU with zero-copy unified memory.
# Kept as plain numpy here because no CUDA device exists in this sandbox.
xp = np


@dataclass(frozen=True)
class SensorAgreementThresholds:
    mahalanobis_warn_threshold: float
    mahalanobis_fail_threshold: float
    cusum_k_sigma: float
    cusum_h_sigma: float


# Fixed, versioned mean/covariance for the 3 residual channels: [imu_vs_can,
# gnss_vs_can, lidar_vs_vision]. Produced by
# `python scripts/calibrate_thresholds.py` against 50 synthetic nominal
# runs (seeds 0-49, 60s each @ 20Hz, 60000 samples total) — see
# config/calibration_report.json for the full output this was copied from.
# Diagonal-dominant because channels are largely independent by
# construction of the synthetic generator; a real calibration pass against
# comma2k19 would likely show off-diagonal correlation between GNSS and
# IMU-integrated speed under GPS multipath and should replace this matrix.
# Note r_lidar's mean is ~0.10, not 0: `mean_range_residual_vs_vision_m` is
# a magnitude (abs of a folded normal) by construction, not a signed
# residual, so its nominal expectation is nonzero — calibrating against
# the *measured* mean (rather than assuming 0) is exactly what makes CUSUM
# behave under nominal data instead of drifting on a fictitious offset.
_CALIBRATED_MEAN = np.array([-0.0007632, 0.0013087, 0.1007500])
_CALIBRATED_COV = np.array(
    [
        [0.0226274, -0.0000148, -0.0000111],
        [-0.0000148, 0.0623587, -0.0000068],
        [-0.0000111, -0.0000068, 0.0023025],
    ]
)
_CALIBRATED_STD = np.sqrt(np.diag(_CALIBRATED_COV))


class M2SensorAgreementMonitor(Monitor):
    monitor_id = "M2_SENSOR_AGREEMENT"
    monitor_version = "0.1.0"

    def __init__(self, thresholds: SensorAgreementThresholds) -> None:
        super().__init__()
        self.t = thresholds
        self._inv_cov = np.linalg.inv(_CALIBRATED_COV)
        self._cusum_pos = np.zeros(3)
        self._cusum_neg = np.zeros(3)

    def _residuals(self, frame: HealthFrame) -> np.ndarray | None:
        can_speed = frame.can.speed_mps if frame.can else None
        imu_speed_proxy = frame.ego_pose.speed_mps if frame.ego_pose else None
        gnss_speed = frame.gnss.speed_mps if frame.gnss else None
        lidar_summary = next(
            (r for r in frame.range_sensors if r.modality == "lidar"), None
        )
        vision_residual = (
            lidar_summary.mean_range_residual_vs_vision_m if lidar_summary else None
        )

        if can_speed is None or imu_speed_proxy is None or gnss_speed is None:
            return None

        r_imu = imu_speed_proxy - can_speed
        r_gnss = gnss_speed - can_speed
        r_lidar = vision_residual if vision_residual is not None else 0.0
        return np.array([r_imu, r_gnss, r_lidar])

    def _compute(self, frame: HealthFrame) -> MonitorResult:
        t = self.t
        residuals = self._residuals(frame)
        if residuals is None:
            return MonitorResult(
                monitor_id=self.monitor_id,
                monitor_version=self.monitor_version,
                frame_id=frame.frame_id,
                monotonic_ns=frame.monotonic_ns,
                score=0.5,
                verdict=Verdict.WARN,
                confidence=0.3,
                evidence={"reason": "insufficient_sensor_data"},
                latency_us=0,
            )

        delta = residuals - _CALIBRATED_MEAN
        mahalanobis = float(np.sqrt(delta @ self._inv_cov @ delta.T))

        # CUSUM, per-channel, in units of sigma; take the worst channel.
        # Anti-windup cap at 3x the trip threshold: without a cap, a very
        # large, sustained disturbance (e.g. a strong calibration-drift
        # fault) can drive the accumulator far past h, and since recovery
        # after the disturbance ends only proceeds by ~cusum_k_sigma per
        # tick, an uncapped excursion can take an unreasonably long time
        # to decay back below h even once the sensor is nominal again —
        # observed directly in the scripted demo scenario before this cap
        # was added (DEGRADED never recovered within the demo's window).
        # Capping bounds worst-case recovery time to a few multiples of h
        # without weakening detection (the trip still fires at h, long
        # before the cap is reached).
        cusum_cap = 3.0 * t.cusum_h_sigma
        z = np.where(_CALIBRATED_STD > 0, delta / _CALIBRATED_STD, 0.0)
        self._cusum_pos = np.clip(self._cusum_pos + z - t.cusum_k_sigma, 0.0, cusum_cap)
        self._cusum_neg = np.clip(self._cusum_neg - z - t.cusum_k_sigma, 0.0, cusum_cap)
        cusum_stat = float(max(self._cusum_pos.max(), self._cusum_neg.max()))
        cusum_drift = cusum_stat > t.cusum_h_sigma

        if mahalanobis >= t.mahalanobis_fail_threshold or cusum_drift:
            verdict = Verdict.FAIL
        elif mahalanobis >= t.mahalanobis_warn_threshold:
            verdict = Verdict.WARN
        else:
            verdict = Verdict.PASS

        score = (
            0.0
            if cusum_drift
            else piecewise_score(mahalanobis, t.mahalanobis_warn_threshold, t.mahalanobis_fail_threshold)
        )

        return MonitorResult(
            monitor_id=self.monitor_id,
            monitor_version=self.monitor_version,
            frame_id=frame.frame_id,
            monotonic_ns=frame.monotonic_ns,
            score=score,
            verdict=verdict,
            confidence=1.0,
            evidence={
                "mahalanobis": round(mahalanobis, 4),
                "cusum_stat": round(cusum_stat, 4),
                "cusum_drift": cusum_drift,
                "r_imu": round(float(residuals[0]), 4),
                "r_gnss": round(float(residuals[1]), 4),
                "r_lidar": round(float(residuals[2]), 4),
            },
            latency_us=0,
        )
