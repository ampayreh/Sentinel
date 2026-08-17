"""The actual M1 design from the spec: a small GRU autoencoder trained on
nominal perception-feature windows, with reconstruction-error scoring and
split conformal calibration for a chosen false-alarm rate — plus the
TensorRT export path for GB10.

Import discipline: this module imports torch at MODULE level, which is
exactly why nothing in `sentinel.trust`, `sentinel.policy`, or
`sentinel.evidence` (or `sentinel.runtime.loop`'s default construction
path) imports this module. It is only imported by
`sentinel.monitors.m1_perception` callers who explicitly opt into the
learned monitor, and by training/calibration/export scripts. See
`tests/unit/test_m1_learned_model.py::test_decision_path_does_not_import_torch`
for the enforcement test.

GB10 note: TensorRT itself (and a CUDA-capable torch build) cannot be
exercised in this development sandbox (x86_64, no GPU, no TensorRT
package available). `export_to_onnx` is real and tested (produces a valid
ONNX graph from the trained module); `onnx_to_tensorrt_export_command`
below documents, but does not run, the `trtexec` invocation for the GB10
target — running and benchmarking that step is flagged as not done in
this environment rather than faked.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sentinel.monitors.base import Monitor
from sentinel.monitors.scoring import piecewise_score
from sentinel.schemas.health_frame import HealthFrame
from sentinel.schemas.monitor_result import MonitorResult, Verdict

FEATURE_DIM = 5  # [mean_confidence, min_confidence, track_id_churn, class_flip_count, box_jitter]


def frame_to_feature(frame: HealthFrame) -> np.ndarray | None:
    p = frame.perception
    if p is None or p.mean_confidence is None:
        return None
    return np.array(
        [
            p.mean_confidence,
            p.min_confidence if p.min_confidence is not None else p.mean_confidence,
            float(p.track_id_churn),
            float(p.class_flip_count),
            p.mean_box_jitter_px if p.mean_box_jitter_px is not None else 0.0,
        ],
        dtype=np.float32,
    )


class PerceptionAutoencoder(nn.Module):
    """GRU encoder -> latent -> GRU decoder, reconstructing a window of
    perception feature vectors. Reconstruction error (MSE over the window)
    is the anomaly signal: a perception stack behaving like nominal
    training data reconstructs cleanly; drift, occlusion, or model
    confusion the network never saw shows up as elevated error.

    Deliberately small (hackathon/edge-deployment sized): this is meant to
    run comfortably on GB10's Blackwell tensor cores in FP16/FP4, not to
    be a research-scale model.
    """

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_dim: int = 16) -> None:
        super().__init__()
        self.encoder = nn.GRU(feature_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, feature_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window_len, feature_dim)
        _, h_n = self.encoder(x)  # h_n: (1, batch, hidden_dim)
        window_len = x.shape[1]
        latent_seq = h_n.repeat(window_len, 1, 1).permute(1, 0, 2)  # (batch, window_len, hidden)
        decoded, _ = self.decoder(latent_seq)
        return self.output_proj(decoded)  # (batch, window_len, feature_dim)


@dataclass(frozen=True)
class TrainingStats:
    n_windows: int
    epochs: int
    final_train_mse: float


def make_windows(features: np.ndarray, window_len: int) -> np.ndarray:
    """Non-overlapping-stride-1 sliding windows: (n_windows, window_len, feature_dim)."""
    n = features.shape[0] - window_len + 1
    if n <= 0:
        return np.empty((0, window_len, features.shape[1]), dtype=np.float32)
    return np.stack([features[i : i + window_len] for i in range(n)]).astype(np.float32)


def train_autoencoder(
    nominal_features: np.ndarray,
    window_len: int = 20,
    hidden_dim: int = 16,
    epochs: int = 15,
    lr: float = 1e-3,
    seed: int = 0,
) -> tuple[PerceptionAutoencoder, TrainingStats]:
    """Trains ONLY on nominal data (spec requirement: no online training,
    no adapting at runtime — this is an offline, versioned artifact)."""
    torch.manual_seed(seed)
    windows = make_windows(nominal_features, window_len)
    if windows.shape[0] == 0:
        raise ValueError("not enough nominal frames to form even one window")

    x = torch.from_numpy(windows)
    model = PerceptionAutoencoder(feature_dim=windows.shape[2], hidden_dim=hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    final_loss = float("nan")
    for _epoch in range(epochs):
        opt.zero_grad()
        recon = model(x)
        loss = loss_fn(recon, x)
        loss.backward()
        opt.step()
        final_loss = float(loss.item())

    model.eval()
    return model, TrainingStats(n_windows=windows.shape[0], epochs=epochs, final_train_mse=final_loss)


def reconstruction_errors(model: PerceptionAutoencoder, windows: np.ndarray) -> np.ndarray:
    """Per-window mean squared reconstruction error, no gradient tracking."""
    if windows.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    with torch.no_grad():
        x = torch.from_numpy(windows.astype(np.float32))
        recon = model(x)
        err = ((recon - x) ** 2).mean(dim=(1, 2))
        return err.numpy()


def split_conformal_threshold(calibration_scores: np.ndarray, alpha: float) -> float:
    """Split conformal prediction: threshold = the
    ceil((n+1)*(1-alpha))-th smallest calibration score (1-indexed), the
    standard finite-sample-valid conformal quantile. This is what makes
    the false-alarm rate a CHOSEN number (`alpha`) rather than a
    percentile picked by eyeballing a histogram — for n held-out nominal
    calibration windows, P(a fresh nominal window's score exceeds the
    threshold) <= alpha, exactly, regardless of the score distribution's
    shape (Vovk et al., conformal prediction).
    """
    n = len(calibration_scores)
    if n == 0:
        raise ValueError("calibration_scores must be non-empty")
    sorted_scores = np.sort(calibration_scores)
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    rank = min(max(rank, 1), n)  # clamp into valid 1-indexed range
    return float(sorted_scores[rank - 1])


@dataclass(frozen=True)
class LearnedThresholds:
    window_len: int
    warn_alpha: float  # conformal target false-alarm rate for WARN
    fail_alpha: float  # conformal target false-alarm rate for FAIL (< warn_alpha)
    warn_threshold: float  # computed by split_conformal_threshold, stored for provenance
    fail_threshold: float


class M1LearnedMonitor(Monitor):
    """The real M1 design. Same `Monitor` interface as
    `M1HeuristicMonitor` — swap-in compatible, distinguished by
    monitor_version in evidence records for auditability."""

    monitor_id = "M1_PERCEPTION"

    def __init__(
        self,
        model: PerceptionAutoencoder,
        thresholds: LearnedThresholds,
        model_version: str,
    ) -> None:
        super().__init__()
        self.model = model
        self.model.eval()
        self.t = thresholds
        self.monitor_version = model_version
        self._buffer: list[np.ndarray] = []

    def _compute(self, frame: HealthFrame) -> MonitorResult:
        feat = frame_to_feature(frame)
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

        self._buffer.append(feat)
        if len(self._buffer) > self.t.window_len:
            self._buffer.pop(0)

        if len(self._buffer) < self.t.window_len:
            return MonitorResult(
                monitor_id=self.monitor_id,
                monitor_version=self.monitor_version,
                frame_id=frame.frame_id,
                monotonic_ns=frame.monotonic_ns,
                score=1.0,
                verdict=Verdict.PASS,
                confidence=0.2,
                evidence={"reason": "warming_up", "buffer_len": len(self._buffer)},
                latency_us=0,
            )

        window = np.stack(self._buffer)[None, :, :]  # (1, window_len, feature_dim)
        err = float(reconstruction_errors(self.model, window)[0])

        if err >= self.t.fail_threshold:
            verdict = Verdict.FAIL
        elif err >= self.t.warn_threshold:
            verdict = Verdict.WARN
        else:
            verdict = Verdict.PASS

        score = piecewise_score(err, self.t.warn_threshold, self.t.fail_threshold)

        return MonitorResult(
            monitor_id=self.monitor_id,
            monitor_version=self.monitor_version,
            frame_id=frame.frame_id,
            monotonic_ns=frame.monotonic_ns,
            score=score,
            verdict=verdict,
            confidence=0.9,
            evidence={
                "reconstruction_error": round(err, 6),
                "warn_threshold": self.t.warn_threshold,
                "fail_threshold": self.t.fail_threshold,
            },
            latency_us=0,
        )


def export_to_onnx(model: PerceptionAutoencoder, window_len: int, out_path: str | Path) -> Path:
    """Real, runnable ONNX export — verified in
    tests/unit/test_m1_learned_model.py. This is the artifact
    `onnx_to_tensorrt_export_command` below turns into a TensorRT engine
    on the actual GB10 target."""
    out_path = Path(out_path)
    dummy = torch.randn(1, window_len, model.output_proj.out_features)
    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        input_names=["perception_window"],
        output_names=["reconstruction"],
        dynamic_axes={"perception_window": {0: "batch"}, "reconstruction": {0: "batch"}},
        opset_version=17,
    )
    return out_path


def onnx_to_tensorrt_export_command(onnx_path: str, engine_path: str) -> str:
    """Documents (does not execute — no TensorRT/GPU in this sandbox) the
    GB10 deployment step: build a TensorRT engine from the ONNX graph,
    targeting FP16 (Blackwell 5th-gen tensor cores support FP4, but FP16
    is the safer first target for a reconstruction-error model where
    quantization error directly perturbs the anomaly score; FP4 is worth
    evaluating once real accuracy-vs-speed numbers exist on-device).
    Run this on the GB10 unit itself, where `trtexec` and a matching
    TensorRT/CUDA install are actually present:
    """
    return (
        f"trtexec --onnx={onnx_path} --saveEngine={engine_path} "
        f"--fp16 --memPoolSize=workspace:512 --verbose"
    )
