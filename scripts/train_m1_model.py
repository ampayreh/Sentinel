#!/usr/bin/env python3
"""Train the M1 perception-health autoencoder on nominal data and
calibrate its conformal thresholds.

HONESTY NOTE: trains on the synthetic nominal generator (seeds 0-79 for
training, 80-99 held out for conformal calibration, 100-119 held out
again for reporting empirical coverage), not on comma2k19, because
comma2k19 could not be downloaded in this build environment (see
README's dataset section and scripts/download_comma2k19.py's manual-step
fallback). The training/calibration CODE is real and correct; re-point
`_collect_features` at a comma2k19-backed HealthFrame stream (swap
SyntheticReader for Comma2k19Reader) before trusting the resulting model
operationally.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel.monitors.m1_learned_model import (  # noqa: E402
    LearnedThresholds,
    export_to_onnx,
    frame_to_feature,
    make_windows,
    reconstruction_errors,
    split_conformal_threshold,
    train_autoencoder,
)
from sentinel.pipeline.readers.synthetic import ScenarioSpec, SyntheticReader  # noqa: E402
from sentinel.pipeline.stream import health_stream  # noqa: E402

WINDOW_LEN = 20
WARN_ALPHA = 0.01  # target: at most 1% of nominal windows WARN
FAIL_ALPHA = 0.002  # target: at most 0.2% of nominal windows FAIL


def _collect_features(seeds: range, duration_s: float) -> np.ndarray:
    rows = []
    for seed in seeds:
        for frame in health_stream(SyntheticReader(ScenarioSpec(duration_s=duration_s), seed=seed)):
            if frame.is_gap:
                continue
            feat = frame_to_feature(frame)
            if feat is not None:
                rows.append(feat)
    return np.array(rows, dtype=np.float32)


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "artifacts" / "m1_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting training features (seeds 0-79)...")
    train_features = _collect_features(range(0, 80), duration_s=30.0)
    print(f"  {train_features.shape[0]} frames")

    print("Training autoencoder...")
    model, stats = train_autoencoder(train_features, window_len=WINDOW_LEN, epochs=15)
    print(f"  final train MSE: {stats.final_train_mse:.6f} over {stats.n_windows} windows")

    print("Collecting calibration features (seeds 80-99, held out from training)...")
    calib_features = _collect_features(range(80, 100), duration_s=30.0)
    calib_windows = make_windows(calib_features, WINDOW_LEN)
    calib_errors = reconstruction_errors(model, calib_windows)

    warn_threshold = split_conformal_threshold(calib_errors, WARN_ALPHA)
    fail_threshold = split_conformal_threshold(calib_errors, FAIL_ALPHA)
    thresholds = LearnedThresholds(
        window_len=WINDOW_LEN,
        warn_alpha=WARN_ALPHA,
        fail_alpha=FAIL_ALPHA,
        warn_threshold=warn_threshold,
        fail_threshold=fail_threshold,
    )
    print(f"  warn_threshold(alpha={WARN_ALPHA})={warn_threshold:.6f}")
    print(f"  fail_threshold(alpha={FAIL_ALPHA})={fail_threshold:.6f}")

    print("Collecting held-out report features (seeds 100-119, never seen during calibration)...")
    report_features = _collect_features(range(100, 120), duration_s=30.0)
    report_windows = make_windows(report_features, WINDOW_LEN)
    report_errors = reconstruction_errors(model, report_windows)
    empirical_warn_rate = float((report_errors >= warn_threshold).mean())
    empirical_fail_rate = float((report_errors >= fail_threshold).mean())
    print(f"  empirical WARN+ rate on fresh nominal data: {empirical_warn_rate:.4%} (target <= {WARN_ALPHA:.2%})")
    print(f"  empirical FAIL rate on fresh nominal data: {empirical_fail_rate:.4%} (target <= {FAIL_ALPHA:.2%})")

    torch.save(model.state_dict(), out_dir / "model.pt")
    export_to_onnx(model, WINDOW_LEN, out_dir / "model.onnx")
    (out_dir / "thresholds.json").write_text(
        json.dumps(
            {
                "window_len": thresholds.window_len,
                "warn_alpha": thresholds.warn_alpha,
                "fail_alpha": thresholds.fail_alpha,
                "warn_threshold": thresholds.warn_threshold,
                "fail_threshold": thresholds.fail_threshold,
                "train_stats": {
                    "n_windows": stats.n_windows,
                    "epochs": stats.epochs,
                    "final_train_mse": stats.final_train_mse,
                },
                "empirical_held_out_warn_rate": empirical_warn_rate,
                "empirical_held_out_fail_rate": empirical_fail_rate,
                "trained_on": "synthetic_nominal_seeds_0-79",
                "calibrated_on": "synthetic_nominal_seeds_80-99",
                "reported_on": "synthetic_nominal_seeds_100-119",
                "model_version": "m1-learned-0.1.0-synthetic",
            },
            indent=2,
        )
    )
    print(f"\nWrote model.pt, model.onnx, thresholds.json to {out_dir}")


if __name__ == "__main__":
    main()
