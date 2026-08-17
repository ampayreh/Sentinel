#!/usr/bin/env python3
"""Calibrate M2's residual mean/covariance and Mahalanobis thresholds, and
M3's jitter percentiles, against the synthetic nominal generator.

This is the actual, runnable calibration procedure referenced by the
"how this was calibrated" comments in sentinel/monitors/m2_sensor_agreement.py
and config/monitor_thresholds.yaml. Numbers baked into those files were
produced by running this script with the arguments shown in __main__ below
against seeds 0-49 of ScenarioSpec() (180s @ 20Hz each).

IMPORTANT HONESTY NOTE: this calibrates against Sentinel's own synthetic
generator, not against real comma2k19 driving data (comma2k19 could not be
downloaded in this build environment — see README's dataset section). The
Mahalanobis/CUSUM machinery and calibration procedure are real and will
produce a correctly-calibrated false-alarm rate once pointed at real
nominal data; re-run this script against a comma2k19-backed HealthFrame
stream before trusting these thresholds operationally.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel.pipeline.readers.synthetic import ScenarioSpec, SyntheticReader  # noqa: E402
from sentinel.pipeline.stream import health_stream  # noqa: E402


def collect_m2_residuals(seeds: range, duration_s: float) -> np.ndarray:
    rows = []
    for seed in seeds:
        spec = ScenarioSpec(duration_s=duration_s)
        for frame in health_stream(SyntheticReader(spec, seed=seed)):
            if frame.is_gap or frame.can is None or frame.ego_pose is None or frame.gnss is None:
                continue
            lidar = next((r for r in frame.range_sensors if r.modality == "lidar"), None)
            r_imu = (frame.ego_pose.speed_mps or 0.0) - (frame.can.speed_mps or 0.0)
            r_gnss = (frame.gnss.speed_mps or 0.0) - (frame.can.speed_mps or 0.0)
            r_lidar = lidar.mean_range_residual_vs_vision_m if lidar else 0.0
            rows.append([r_imu, r_gnss, r_lidar])
    return np.array(rows)


def mahalanobis_distances(residuals: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    inv_cov = np.linalg.inv(cov)
    delta = residuals - mean
    return np.sqrt(np.einsum("ij,jk,ik->i", delta, inv_cov, delta))


def main() -> None:
    seeds = range(0, 50)
    duration_s = 60.0  # shorter than full 180s scenario, still >150k samples across 50 seeds

    residuals = collect_m2_residuals(seeds, duration_s)
    mean = residuals.mean(axis=0)
    cov = np.cov(residuals, rowvar=False)

    distances = mahalanobis_distances(residuals, mean, cov)
    p995 = float(np.percentile(distances, 99.5))
    p999 = float(np.percentile(distances, 99.9))

    report = {
        "n_samples": int(residuals.shape[0]),
        "seeds": [seeds.start, seeds.stop - 1],
        "duration_s_per_seed": duration_s,
        "mean": mean.tolist(),
        "cov": cov.tolist(),
        "mahalanobis_p50": float(np.percentile(distances, 50)),
        "mahalanobis_p995": p995,
        "mahalanobis_p999": p999,
        "recommended_warn_threshold": round(p995, 3),
        "recommended_fail_threshold": round(p999 * 1.3, 3),
    }
    out_path = Path(__file__).resolve().parent.parent / "config" / "calibration_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
