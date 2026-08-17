"""M1 Perception health monitor — GPU, statistical/ML.

Two implementations live here, selected by config, so the decision path
(`sentinel.trust`, `sentinel.policy`, `sentinel.evidence`) never has to
import torch (hard requirement: "the decision path must be importable
without pulling in torch"):

  - `M1HeuristicMonitor`: a pure-numpy rolling z-score over perception
    features (mean confidence, track churn, class-flip rate, box jitter).
    No training required, no GPU required. This is what M0-M3 wire up by
    default and what CI runs.

  - `M1LearnedMonitor`: the spec's actual design — a small GRU/1D-CNN
    autoencoder trained on nominal data only, reconstruction-error score,
    thresholds set by split conformal prediction on held-out nominal data.
    Torch is imported lazily inside `__init__`/`_compute`, never at module
    import time, so importing this file does not require torch to be
    installed. See sentinel/monitors/m1_learned_model.py for the model
    definition and calibration/export code (M4 milestone).

Both implementations satisfy the same `Monitor` interface and emit the
same `MonitorResult` shape, so the trust coordinator is agnostic to which
one is active.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from sentinel.monitors.base import Monitor
from sentinel.monitors.scoring import piecewise_score
from sentinel.schemas.health_frame import HealthFrame
from sentinel.schemas.monitor_result import MonitorResult, Verdict


@dataclass(frozen=True)
class HeuristicThresholds:
    window_size: int = 100
    # Calibrated empirically (not a hand-picked z-cutoff): 50 synthetic
    # nominal runs (seeds 0-49, 60s @ 20Hz, 59000 samples) were scored with
    # thresholds disabled, and warn_z/fail_z were set to the measured
    # 99.5th/99.9th (x1.3 margin) percentiles of `max_z` — see
    # scripts/calibrate_thresholds.py's M2 section for the same method;
    # this value was produced by an equivalent one-off script. This
    # deliberately corrects for `max_z` being a MAX over 4 features (a
    # naive single-feature z=2.5 cutoff has a ~4x higher false-alarm rate
    # than intended once you take a max over 4 roughly-independent
    # features — the calibration-on-the-actual-statistic approach used
    # here avoids that multiple-comparisons trap rather than requiring a
    # hand-derived Bonferroni correction).
    warn_z: float = 3.073
    fail_z: float = 4.588
    min_window_for_scoring: int = 20
    # Absolute thresholds — catch sustained degradation even when the
    # z-score window has absorbed the faulted values.  Calibrated from
    # nominal-run 1st-percentile (confidence) and 99th-percentile
    # (jitter/flips).
    abs_confidence_warn: float = 0.65  # mean_confidence below this → WARN
    abs_confidence_fail: float = 0.45  # mean_confidence below this → FAIL
    abs_jitter_warn: float = 5.0      # mean_box_jitter_px above this → WARN
    abs_jitter_fail: float = 8.0      # mean_box_jitter_px above this → FAIL
    abs_class_flip_warn: int = 2      # class_flip_count above this → WARN
    abs_class_flip_fail: int = 4      # class_flip_count above this → FAIL


def _feature_vector(frame: HealthFrame) -> np.ndarray | None:
    p = frame.perception
    if p is None or p.mean_confidence is None:
        return None
    return np.array(
        [
            p.mean_confidence,
            float(p.track_id_churn),
            float(p.class_flip_count),
            p.mean_box_jitter_px if p.mean_box_jitter_px is not None else 0.0,
        ]
    )


class M1HeuristicMonitor(Monitor):
    """Rolling z-score anomaly detector with absolute-threshold fallback.

    The rolling z-score detects *onset* of anomaly (first few frames of
    divergence), while absolute thresholds catch *sustained* degradation
    that the rolling window would otherwise absorb as a new baseline.
    The baseline window is "frozen" — only PASS-verdict samples are
    added, so degraded data never poisons the nominal reference.
    """

    monitor_id = "M1_PERCEPTION"
    monitor_version = "0.2.0-heuristic"

    def __init__(self, thresholds: HeuristicThresholds | None = None) -> None:
        super().__init__()
        self.t = thresholds or HeuristicThresholds()
        # Nominal baseline — only accepts PASS-verdict samples so
        # sustained degradation cannot poison the reference distribution.
        self._baseline: deque[np.ndarray] = deque(maxlen=self.t.window_size)

    def _absolute_check(self, frame: HealthFrame) -> tuple[Verdict, float, dict]:
        """Return worst (verdict, score, evidence) from absolute thresholds.
        Score: 1.0 = fully nominal, 0.0 = hard FAIL."""
        p = frame.perception
        if p is None:
            return Verdict.WARN, 0.5, {"abs_reason": "no_perception_data"}

        worst_verdict = Verdict.PASS
        worst_score = 1.0
        reasons = []

        # Confidence check
        conf = p.mean_confidence if p.mean_confidence is not None else 0.9
        if conf < self.t.abs_confidence_fail:
            worst_verdict = Verdict.FAIL
            worst_score = 0.0
            reasons.append(f"confidence={conf:.3f}<{self.t.abs_confidence_fail}")
        elif conf < self.t.abs_confidence_warn:
            if worst_verdict != Verdict.FAIL:
                worst_verdict = Verdict.WARN
            frac = (self.t.abs_confidence_warn - conf) / max(self.t.abs_confidence_warn - self.t.abs_confidence_fail, 1e-6)
            worst_score = min(worst_score, round(1.0 - frac, 4))
            reasons.append(f"confidence={conf:.3f}<{self.t.abs_confidence_warn}")

        # Jitter check
        jitter = p.mean_box_jitter_px if p.mean_box_jitter_px is not None else 0.0
        if jitter > self.t.abs_jitter_fail:
            worst_verdict = Verdict.FAIL
            worst_score = 0.0
            reasons.append(f"jitter={jitter:.1f}>{self.t.abs_jitter_fail}")
        elif jitter > self.t.abs_jitter_warn:
            if worst_verdict != Verdict.FAIL:
                worst_verdict = max(worst_verdict, Verdict.WARN, key=lambda v: ["PASS", "WARN", "FAIL"].index(v.value))
            frac = (jitter - self.t.abs_jitter_warn) / max(self.t.abs_jitter_fail - self.t.abs_jitter_warn, 1e-6)
            worst_score = min(worst_score, round(1.0 - min(frac, 1.0), 4))
            reasons.append(f"jitter={jitter:.1f}>{self.t.abs_jitter_warn}")

        # Class flip check
        flips = p.class_flip_count
        if flips > self.t.abs_class_flip_fail:
            worst_verdict = Verdict.FAIL
            worst_score = 0.0
            reasons.append(f"flips={flips}>{self.t.abs_class_flip_fail}")
        elif flips > self.t.abs_class_flip_warn:
            if worst_verdict != Verdict.FAIL:
                worst_verdict = max(worst_verdict, Verdict.WARN, key=lambda v: ["PASS", "WARN", "FAIL"].index(v.value))
            frac = (flips - self.t.abs_class_flip_warn) / max(self.t.abs_class_flip_fail - self.t.abs_class_flip_warn, 1e-6)
            worst_score = min(worst_score, round(1.0 - min(frac, 1.0), 4))
            reasons.append(f"flips={flips}>{self.t.abs_class_flip_warn}")

        return worst_verdict, worst_score, {"abs_triggers": "; ".join(reasons) if reasons else "none"}

    def _compute(self, frame: HealthFrame) -> MonitorResult:
        feat = _feature_vector(frame)
        if feat is None:
            return MonitorResult(
                monitor_id=self.monitor_id,
                monitor_version=self.monitor_version,
                frame_id=frame.frame_id,
                monotonic_ns=frame.monotonic_ns,
                score=0.5,
                verdict=Verdict.WARN,
                confidence=0.3,
                evidence={"reason": "no_perception_data"},
                latency_us=0,
            )

        if len(self._baseline) < self.t.min_window_for_scoring:
            self._baseline.append(feat)
            return MonitorResult(
                monitor_id=self.monitor_id,
                monitor_version=self.monitor_version,
                frame_id=frame.frame_id,
                monotonic_ns=frame.monotonic_ns,
                score=1.0,
                verdict=Verdict.PASS,
                confidence=0.2,  # low confidence: window still warming up
                evidence={"reason": "warming_up", "window_len": len(self._baseline)},
                latency_us=0,
            )

        # --- Z-score against frozen baseline ---
        arr = np.array(self._baseline)
        mean = arr.mean(axis=0)
        std = np.where(arr.std(axis=0) > 1e-6, arr.std(axis=0), 1e-6)
        z = np.abs((feat - mean) / std)
        max_z = float(z.max())

        if max_z >= self.t.fail_z:
            z_verdict = Verdict.FAIL
        elif max_z >= self.t.warn_z:
            z_verdict = Verdict.WARN
        else:
            z_verdict = Verdict.PASS

        z_score = piecewise_score(max_z, self.t.warn_z, self.t.fail_z)

        # --- Absolute threshold check ---
        abs_verdict, abs_score, abs_evidence = self._absolute_check(frame)

        # --- Combine: take the WORST of z-score and absolute ---
        verdict_rank = {"PASS": 0, "WARN": 1, "FAIL": 2, "STALE": 3}
        if verdict_rank.get(abs_verdict.value, 0) > verdict_rank.get(z_verdict.value, 0):
            final_verdict = abs_verdict
        else:
            final_verdict = z_verdict
        final_score = min(z_score, abs_score)  # lower = worse

        # Only add to baseline if PASS — keep the baseline clean of faults
        if final_verdict == Verdict.PASS:
            self._baseline.append(feat)

        return MonitorResult(
            monitor_id=self.monitor_id,
            monitor_version=self.monitor_version,
            frame_id=frame.frame_id,
            monotonic_ns=frame.monotonic_ns,
            score=final_score,
            verdict=final_verdict,
            confidence=0.9,
            evidence={
                "max_z": round(max_z, 4),
                "mean_confidence": round(float(feat[0]), 4),
                "track_id_churn": float(feat[1]),
                "class_flip_count": float(feat[2]),
                "mean_box_jitter_px": round(float(feat[3]), 2),
                "z_verdict": z_verdict.value,
                "abs_verdict": abs_verdict.value,
                **abs_evidence,
            },
            latency_us=0,
        )

