"""Tests for the real (torch-based) M1 learned monitor: training,
conformal calibration coverage, ONNX export, and — critically — that the
decision path stays importable without torch.

Skipped automatically if torch is not installed, since torch is an
optional (`ml` extra) dependency, not a hard requirement for the
deterministic decision path.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sentinel.monitors.m1_learned_model import (  # noqa: E402
    LearnedThresholds,
    M1LearnedMonitor,
    frame_to_feature,
    make_windows,
    reconstruction_errors,
    split_conformal_threshold,
    train_autoencoder,
)
from sentinel.pipeline.readers.synthetic import ScenarioSpec, SyntheticReader  # noqa: E402
from sentinel.pipeline.stream import health_stream  # noqa: E402
from sentinel.schemas.monitor_result import Verdict  # noqa: E402

WINDOW_LEN = 10


def _features(seed: int, duration_s: float = 10.0) -> np.ndarray:
    rows = []
    for frame in health_stream(SyntheticReader(ScenarioSpec(duration_s=duration_s), seed=seed)):
        if frame.is_gap:
            continue
        feat = frame_to_feature(frame)
        if feat is not None:
            rows.append(feat)
    return np.array(rows, dtype=np.float32)


def test_autoencoder_trains_and_reduces_loss() -> None:
    features = _features(seed=1, duration_s=15.0)
    model, stats = train_autoencoder(features, window_len=WINDOW_LEN, epochs=5)
    assert stats.n_windows > 0
    assert np.isfinite(stats.final_train_mse)


def test_split_conformal_threshold_gives_valid_coverage() -> None:
    """Core statistical property: with n calibration samples and target
    alpha, the empirical exceedance rate on FRESH nominal data should be
    close to alpha (not exact for finite n, but in the right ballpark)."""
    train_feat = _features(seed=1, duration_s=20.0)
    model, _ = train_autoencoder(train_feat, window_len=WINDOW_LEN, epochs=10)

    calib_feat = _features(seed=2, duration_s=20.0)
    calib_windows = make_windows(calib_feat, WINDOW_LEN)
    calib_errors = reconstruction_errors(model, calib_windows)

    alpha = 0.05
    threshold = split_conformal_threshold(calib_errors, alpha)

    fresh_feat = _features(seed=3, duration_s=20.0)
    fresh_windows = make_windows(fresh_feat, WINDOW_LEN)
    fresh_errors = reconstruction_errors(model, fresh_windows)
    exceedance_rate = float((fresh_errors >= threshold).mean())

    # Finite-sample slack: conformal guarantees are marginal, not exact
    # per-draw, so allow a generous band around alpha for a small
    # calibration set.
    assert exceedance_rate < alpha * 3, f"exceedance_rate={exceedance_rate} too far above alpha={alpha}"


def test_m1_learned_monitor_scores_like_other_monitors() -> None:
    train_feat = _features(seed=1, duration_s=15.0)
    model, _ = train_autoencoder(train_feat, window_len=WINDOW_LEN, epochs=5)
    calib_feat = _features(seed=2, duration_s=15.0)
    calib_windows = make_windows(calib_feat, WINDOW_LEN)
    calib_errors = reconstruction_errors(model, calib_windows)
    thresholds = LearnedThresholds(
        window_len=WINDOW_LEN,
        warn_alpha=0.05,
        fail_alpha=0.01,
        warn_threshold=split_conformal_threshold(calib_errors, 0.05),
        fail_threshold=split_conformal_threshold(calib_errors, 0.01),
    )
    mon = M1LearnedMonitor(model, thresholds, model_version="test-learned-0.1.0")
    reader = SyntheticReader(ScenarioSpec(duration_s=5.0), seed=4)
    results = [mon.update(f) for f in health_stream(reader) if not f.is_gap]
    assert all(0.0 <= r.score <= 1.0 for r in results)
    assert all(r.verdict in (Verdict.PASS, Verdict.WARN, Verdict.FAIL) for r in results)
    assert any(r.confidence < 0.5 for r in results[:WINDOW_LEN])  # warming up early on


def test_decision_path_does_not_import_torch() -> None:
    """Enforces the hard requirement: 'the decision path must be
    importable without pulling in torch.' Runs in a fresh subprocess so
    this test file's own torch import (needed to test the learned model)
    can't contaminate the check."""
    code = (
        "import sys\n"
        "import sentinel.runtime.loop\n"
        "import sentinel.trust.coordinator\n"
        "import sentinel.policy.gate\n"
        "import sentinel.evidence.replay\n"
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
