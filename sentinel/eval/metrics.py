"""Evaluation metrics computed from real, measured evidence-log data —
per the anti-goals, nothing here is a placeholder/fabricated number.
Every metric is derived either from actual DecisionLoop runs (fault
injection scenarios, nominal-only scenarios) or actual wall-clock timing
of the real code path.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from sentinel.eval.fault_injection import (
    FAULT_TO_FAILURE_CLASS,
    FaultSpec,
    FaultType,
    inject_faults,
)
from sentinel.pipeline.readers.synthetic import ScenarioSpec, SyntheticReader
from sentinel.pipeline.stream import health_stream
from sentinel.runtime.factory import RuntimeConfigPaths, build_decision_loop
from sentinel.trust.states import TrustState

TICK_HZ = 20.0


@dataclass
class DetectionOutcome:
    fault: FaultSpec
    detected: bool
    lead_time_s: float | None  # None if not detected within the search window


@dataclass
class ScenarioRun:
    trust_states: list[str]
    transition_ticks: list[int]  # ticks where a trust transition occurred
    per_frame_latency_us: list[int]
    faults: list[FaultSpec]
    fault_windows: dict[int, FaultSpec]  # tick -> active fault, for reference


def _run_scenario(
    seed: int,
    duration_s: float,
    faults: list[FaultSpec],
    monitor_overrides: dict | None = None,
    log_path: str | None = None,
) -> ScenarioRun:
    paths = RuntimeConfigPaths()
    log_file = log_path or f"/tmp/sentinel_eval_{seed}_{id(faults)}.jsonl"
    import os

    if os.path.exists(log_file):
        os.remove(log_file)
    loop = build_decision_loop(log_file, paths, flush_every_n_records=50)

    if monitor_overrides:
        loop.monitors.update(monitor_overrides)

    reader = SyntheticReader(ScenarioSpec(duration_s=duration_s), seed=seed)
    trust_states: list[str] = []
    transition_ticks: list[int] = []
    latencies: list[int] = []
    fault_windows: dict[int, FaultSpec] = {}

    for tick, (frame, active_fault) in enumerate(inject_faults(health_stream(reader), faults)):
        t0 = time.perf_counter_ns()
        record = loop.process_frame(frame)
        latencies.append((time.perf_counter_ns() - t0) // 1000)

        trust_states.append(record.trust_state)
        if record.trust_transition is not None:
            transition_ticks.append(tick)
        if active_fault is not None:
            fault_windows[tick] = active_fault

    loop.evidence_writer.close()
    return ScenarioRun(
        trust_states=trust_states,
        transition_ticks=transition_ticks,
        per_frame_latency_us=latencies,
        faults=faults,
        fault_windows=fault_windows,
    )


def _detect_fault(run: ScenarioRun, fault: FaultSpec, max_delay_ticks: int = 200) -> DetectionOutcome:
    search_end = min(fault.end_tick + max_delay_ticks, len(run.trust_states))
    for tick in range(fault.start_tick, search_end):
        if run.trust_states[tick] != TrustState.FULL_AUTONOMY.value:
            lead_ticks = tick - fault.start_tick
            return DetectionOutcome(fault=fault, detected=True, lead_time_s=lead_ticks / TICK_HZ)
    return DetectionOutcome(fault=fault, detected=False, lead_time_s=None)


@dataclass
class PerClassMetrics:
    fault_type: str
    failure_class: str
    n_instances: int
    n_detected: int
    detection_rate: float
    mean_lead_time_s: float | None
    lead_time_ci95: tuple[float, float] | None


@dataclass
class EvalReport:
    per_class: list[PerClassMetrics]
    false_intervention_rate_per_hour: float
    false_intervention_ci95: tuple[float, float]
    availability_full_autonomy_pct: float
    latency_p50_us: float
    latency_p95_us: float
    latency_p99_us: float
    n_nominal_seeds: int
    n_fault_seeds_per_class: int


def _mean_ci95(values: list[float]) -> tuple[float, float | None, tuple[float, float] | None]:
    if not values:
        return 0.0, None, None
    arr = np.array(values)
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, mean, None
    std = float(arr.std(ddof=1))
    half_width = 1.96 * std / math.sqrt(len(arr))
    return mean, mean, (mean - half_width, mean + half_width)


def evaluate_false_intervention_rate(seeds: range, duration_s: float) -> tuple[float, tuple[float, float]]:
    """Runs pure-nominal scenarios (no injected faults) and counts
    transitions AWAY from FULL_AUTONOMY per simulated hour — the number
    that decides whether this system would ever ship."""
    rates_per_hour = []
    for seed in seeds:
        run = _run_scenario(seed, duration_s, faults=[])
        n_bad_transitions = sum(
            1 for t in run.transition_ticks if run.trust_states[t] != TrustState.FULL_AUTONOMY.value
        )
        hours = duration_s / 3600.0
        rates_per_hour.append(n_bad_transitions / hours)

    mean = float(np.mean(rates_per_hour))
    if len(rates_per_hour) < 2:
        return mean, (mean, mean)
    std = float(np.std(rates_per_hour, ddof=1))
    half_width = 1.96 * std / math.sqrt(len(rates_per_hour))
    return mean, (max(0.0, mean - half_width), mean + half_width)


def evaluate_availability(seeds: range, duration_s: float) -> float:
    total_ticks = 0
    full_autonomy_ticks = 0
    for seed in seeds:
        run = _run_scenario(seed, duration_s, faults=[])
        total_ticks += len(run.trust_states)
        full_autonomy_ticks += sum(1 for s in run.trust_states if s == TrustState.FULL_AUTONOMY.value)
    return 100.0 * full_autonomy_ticks / total_ticks if total_ticks else 0.0


def evaluate_latency(seeds: range, duration_s: float) -> tuple[float, float, float]:
    all_latencies: list[int] = []
    for seed in seeds:
        run = _run_scenario(seed, duration_s, faults=[])
        all_latencies.extend(run.per_frame_latency_us)
    arr = np.array(all_latencies)
    return (
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 95)),
        float(np.percentile(arr, 99)),
    )


def evaluate_fault_detection(
    fault_type: FaultType,
    seeds: range,
    duration_s: float,
    fault_start_tick: int,
    fault_duration_ticks: int,
    magnitude: float = 1.0,
    monitor_overrides: dict | None = None,
) -> PerClassMetrics:
    outcomes: list[DetectionOutcome] = []
    for seed in seeds:
        spec = FaultSpec(
            fault_type=fault_type,
            start_tick=fault_start_tick,
            duration_ticks=fault_duration_ticks,
            magnitude=magnitude,
        )
        run = _run_scenario(seed, duration_s, faults=[spec], monitor_overrides=monitor_overrides)
        outcomes.append(_detect_fault(run, spec))

    n_detected = sum(1 for o in outcomes if o.detected)
    lead_times = [o.lead_time_s for o in outcomes if o.lead_time_s is not None]
    mean_lead, _, ci = _mean_ci95(lead_times) if lead_times else (None, None, None)

    return PerClassMetrics(
        fault_type=fault_type.value,
        failure_class=FAULT_TO_FAILURE_CLASS[fault_type].value,
        n_instances=len(outcomes),
        n_detected=n_detected,
        detection_rate=n_detected / len(outcomes) if outcomes else 0.0,
        mean_lead_time_s=mean_lead,
        lead_time_ci95=ci,
    )
