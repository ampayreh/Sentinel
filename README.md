# Sentinel

A runtime assurance layer for AI-native autonomous systems. Sentinel sits
**outside** the autonomy stack and continuously answers: *should we trust
this autonomous system right now, under this operating condition, based
on independent evidence — and can we prove why we made that decision?*

Sentinel does not decide what the vehicle should do. It decides how much
authority the vehicle's autonomy stack is currently allowed to exercise.

Core loop: **Monitor → Decide → Constrain → Act → Record → Replay**

No LLM/VLM is ever a safety authority. Learned models produce typed
`Evidence`; a deterministic, versioned, unit-tested policy gate produces
authorization. Evidence can only ever *tighten* authority, never expand
it — enforced structurally (see `sentinel/policy/gate.py`) and proven by
a Hypothesis property test (`tests/property/test_vlm_monotonicity.py`).

## Honesty notes (read this before trusting any number in this repo)

- **Development platform vs. target hardware.** This repo was built and
  tested in an x86_64 Linux sandbox with no GPU. The production target is
  the Acer Veriton GN100 (NVIDIA GB10 Grace Blackwell, aarch64, unified
  CPU/GPU memory). Every dependency here is checked for aarch64
  compatibility and the code has no x86-only assumptions, but **nothing
  in this repo has been run on GB10 hardware** — TensorRT export,
  FP4/FP16 inference timing, and unified-memory zero-copy behavior are
  documented/designed for GB10 but not benchmarked on it.
- **Datasets.** comma2k19 and the NVIDIA PhysicalAI AV dataset could not
  be downloaded in the build environment (gated behind accounts/EULAs,
  and this sandbox's network access is allowlisted for package
  registries only — no general web access). CA DMV's disengagement
  report site and NHTSA's SGO portal both returned connection failures
  when actually tested (`curl -I` returned HTTP 403 for dmv.ca.gov).
  Every `scripts/download_*.py` script documents this honestly and gives
  a real manual-download fallback. **All monitor calibration, the demo
  scenario, and the eval report run against Sentinel's own deterministic
  synthetic generator** (`sentinel/pipeline/readers/synthetic.py`), not
  real driving data. This is stated again at the top of every generated
  `results.md`.
- **The DMV failure taxonomy** (`sentinel/pipeline/readers/dmv_taxonomy.py`)
  encodes real, well-documented AV disengagement failure categories and a
  real, tested keyword classifier — but was never run against an actual
  downloaded CA DMV file (see above), so treat the column-name assumptions
  as provisional until validated against a real file.
- **M1's learned monitor is real** (GRU autoencoder, trained, split
  conformal calibration verified to hold its target false-alarm rate —
  see `tests/unit/test_m1_learned_model.py`) but trained on synthetic
  nominal data, not comma2k19. TensorRT export (`onnx_to_tensorrt_export_command`
  in `sentinel/monitors/m1_learned_model.py`) is documented but not run —
  no TensorRT/GPU in this sandbox.
- **Every number in `eval_results/results.md`** comes from an actual run
  of the real decision-loop code against the synthetic generator. Nothing
  is hand-picked or fabricated. Read the caveats printed inside that file
  (especially around the false-intervention-rate confidence interval)
  before citing any of them.

## 10-minute quickstart

```bash
git clone <this repo> && cd sentinel
make setup                    # creates .venv, installs deps via uv
source .venv/bin/activate

make test                     # 47 tests: unit, property (hypothesis), e2e replay-determinism
make lint                     # ruff
make typecheck                # mypy — decision path importable without torch

sentinel run --scenario nominal_urban_loop --duration 60
sentinel verify evidence_logs/nominal_urban_loop_1.jsonl
sentinel replay evidence_logs/nominal_urban_loop_1.jsonl

sentinel demo                 # the spec's scripted 3-minute demo end to end
```

`sentinel demo` runs the exact scripted scenario the spec asks for:
nominal driving → a calibration-drift fault → Sentinel drops to
`DEGRADED` → an overtake request is denied with a cited rule
(`POL-BASE-DEGRADED`) → the fault clears → authority is restored only
after the hysteresis dwell time → the log is verified (hash chain) and
replayed (bit-identical decisions).

For the full evaluation report (fault injection across 7 failure classes,
ablation, latency percentiles — takes ~2-3 minutes):

```bash
sentinel eval --all
open eval_results/results.md
```

Live dashboard (terminal UI, `rich`-based):

```bash
sentinel run --scenario nominal_urban_loop --duration 60 --dashboard
```

## Optional extras

```bash
pip install -e ".[ml]"        # torch, onnx, onnxscript — for the learned M1 monitor
python scripts/train_m1_model.py     # trains + calibrates + exports M1 to ONNX
pip install -e ".[eval]"      # matplotlib — for eval charts (installed by default via [dev])
```

## Repo layout

```
sentinel/
├── schemas/          HealthFrame, MonitorResult — the two schemas everything else builds on
├── config.py         pydantic-settings + versioned-YAML loader with content hashing
├── pipeline/          readers (synthetic + comma2k19/PhysicalAI/DMV stubs) + time alignment
├── monitors/          M1 (perception, heuristic + learned), M2 (sensor agreement), M3 (timing), M4 (ODD)
├── trust/             severity-weighted fusion + escalate-fast/de-escalate-slow hysteresis + coordinator
├── policy/             pure-function authority gate + VLM evidence adapter (the only VLM door)
├── evidence/           hash-chained JSONL writer + deterministic replay engine
├── runtime/            wires monitors+trust+policy+evidence into the actual decision loop
├── eval/               fault injection (8 fault classes) + metrics + results.md/chart generator
└── demo/               CLI, incident report formatter, scripted demo, live dashboard
```

Import boundary: `sentinel/trust`, `sentinel/policy`, `sentinel/evidence`,
and `sentinel/runtime/loop.py`'s default construction path never import
torch — enforced by a subprocess-level test
(`tests/unit/test_m1_learned_model.py::test_decision_path_does_not_import_torch`).

## Architecture decisions (see also the full design note delivered with this build)

1. **Fusion is severity-weighted linear scoring, not max() and not
   Bayesian/DS fusion** — auditable ("which term dominated" is always
   answerable) but doesn't natively model monitor independence, so a
   separate correlated-failure detector compensates.
2. **Calibration is empirical (split conformal for M1, measured
   percentiles for M2/M3), never a hand-picked constant** — every
   threshold in `config/monitor_thresholds.yaml` cites how it was
   produced and by which script.
3. **Determinism is scoped explicitly**: M2/M3/M4 + trust + policy replay
   bit-identically from source; M1's learned (GPU) evidence, when used,
   replays from the recorded score rather than re-inferring, because GPU
   floating-point inference is not guaranteed bit-reproducible in
   general. `sentinel/evidence/replay.py`'s docstring states this scope
   explicitly.

## What's stubbed / documented gaps

- `Comma2k19Reader` / `PhysicalAIReader`: layout validation is real;
  frame-extraction is `NotImplementedError` pending an actual downloaded
  sample to develop against (see honesty notes above).
- Failure classes with no monitor coverage (documented, not hidden — see
  `FAILURE_CLASS_TO_MONITOR` in `sentinel/pipeline/readers/dmv_taxonomy.py`):
  planner/behavior discrepancies (Sentinel observes, it doesn't watch the
  planner's internal state by design), unexpected-road-user-behavior as a
  first-class signal, and precautionary test-driver disengagements (which
  arguably shouldn't be "caught" by a health-signal system at all).
- `clock_skew` fault class: no monitor currently compares wall-clock to
  monotonic time, since decision logic is required to ignore wall-clock
  for replay determinism. Currently undetectable; flagged in every
  `results.md`.
- TensorRT export path is documented (`onnx_to_tensorrt_export_command`)
  but not executable/verified in this build (no TensorRT/GPU).
- **Web dashboard (`sentinel run --web`) telemetry broadcast was unwired
  from the live run loop, found by actually running it, not by review.**
  `broadcast_state()` in `sentinel/dashboard/server.py` called an
  undefined `diagnose_failure()` (would raise `NameError` on the first
  connected client) and read `MonitorResult` fields (`m.name`, `m.status`)
  that don't exist on that schema (real fields: `monitor_id`, `verdict`) —
  both are now fixed and verified: `pytest -q` still passes 47/47, mypy no
  longer flags either error, and `diagnose_failure_from_monitors()` is a
  real implementation built from the same `FAILURE_CLASS_TO_MONITOR`
  reverse-map the DMV CSV clustering path already uses, so the two "which
  monitor covers this failure class" views can't drift apart. **Not
  fixed, and flagged here rather than silently patched:** the state keys
  `broadcast_state()` reads (`latest_trust`, `latest_llm`,
  `latest_snapshot`) don't match the keys `sentinel/demo/cli.py`'s run
  loop actually writes (`latest_trust_state`, `latest_frame`; `latest_llm`
  is never written at all; `latest_policy` is written as a list of
  `PolicyDecisionRecord`, not the single object with `.actions`/
  `.rule_trace` the broadcaster expects). Trust/LLM/snapshot silently
  degrade to placeholder defaults rather than crash; policy would raise
  `AttributeError` if a client ever connects while a real decision loop is
  driving `current_state`. This needs the state_dict contract reconciled
  between the run loop and the broadcaster — a small integration task, not
  a one-line fix — and reconciling it by guessing at the intended shape
  risked introducing new unverified behavior of exactly the kind this
  audit pass exists to catch, so it's documented here instead.
