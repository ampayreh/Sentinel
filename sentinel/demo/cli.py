"""Sentinel CLI: `sentinel run|replay|verify|eval|demo`."""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator

import click

from sentinel.demo.incident_report import format_incident_report
from sentinel.demo.scenarios import scripted_demo
from sentinel.evidence.hash_chain import HashChainError, verify_chain
from sentinel.evidence.record import EvidenceRecord
from sentinel.evidence.replay import ReplayMismatchError, replay_log
from sentinel.evidence.writer import read_log
from sentinel.logging import configure_logging, get_logger
from sentinel.pipeline.readers.synthetic import ScenarioSpec, SyntheticReader
from sentinel.pipeline.stream import health_stream
from sentinel.policy.schema import ActionClass
from sentinel.runtime.factory import build_decision_loop

logger = get_logger(__name__)


@click.group()
@click.option("--log-level", default="INFO")
def main(log_level: str) -> None:
    """Sentinel — runtime assurance layer for AI-native autonomous systems."""
    configure_logging(level=log_level)


@main.command()
@click.option("--scenario", default="nominal_urban_loop")
@click.option("--seed", default=1, type=int)
@click.option("--duration", default=180.0, type=float)
@click.option("--out", default=None, help="evidence log path (default: evidence_logs/<scenario>_<seed>.jsonl)")
@click.option("--dashboard", is_flag=True, default=False, help="show the live terminal dashboard while running")
@click.option("--web", is_flag=True, default=False, help="start live FastAPI + WebSocket web dashboard")
@click.option("--port", default=8765, type=int, help="port for live web dashboard")
def run(scenario: str, seed: int, duration: float, out: str | None, dashboard: bool, web: bool, port: int) -> None:
    """Run a scenario through the full decision loop, writing an evidence log."""
    log_path = out or f"evidence_logs/{scenario}_{seed}.jsonl"
    loop = build_decision_loop(log_path)
    reader = SyntheticReader(ScenarioSpec(duration_s=duration, name=scenario), seed=seed)

    if web:
        import threading
        import uvicorn
        from sentinel.dashboard.server import app, current_state

        def _bg_loop():
            import asyncio
            async def _run():
                for frame in health_stream(reader):
                    record = loop.process_frame(frame)
                    current_state["snap_count"] += 1
                    current_state["latest_record"] = record
                    current_state["latest_trust_state"] = record.trust_state
                    current_state["latest_monitors"] = record.monitor_results
                    current_state["latest_policy"] = record.policy_decisions
                    current_state["latest_frame"] = frame
                    await asyncio.sleep(0.05)
            asyncio.run(_run())

        threading.Thread(target=_bg_loop, daemon=True).start()
        click.echo(f"Starting Glassmorphic Web Dashboard on http://localhost:{port}...")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
        return

    if dashboard:
        from sentinel.demo.dashboard import run_dashboard

        def _records() -> Iterator[EvidenceRecord]:
            for frame in health_stream(reader):
                yield loop.process_frame(frame)

        run_dashboard(_records())
        loop.evidence_writer.close()
        click.echo(f"\nWrote records to {log_path}.")
        return

    n_frames = 0
    n_transitions = 0
    for frame in health_stream(reader):
        record = loop.process_frame(frame)
        n_frames += 1
        if record.trust_transition is not None:
            n_transitions += 1
            click.echo(f"[t={n_frames / 20.0:6.1f}s] TRANSITION: {record.trust_transition.from_state} -> {record.trust_transition.to_state}")
            click.echo(f"           reason: {record.trust_transition.reason}")
    loop.evidence_writer.close()

    click.echo(f"\nWrote {n_frames} records to {log_path} ({n_transitions} trust transitions).")


@main.command()
@click.argument("log_path")
@click.option("--speed", default=4.0, type=float, help="playback speed multiplier vs. real time")
def dashboard(log_path: str, speed: float) -> None:
    """Replay a recorded evidence log through the live dashboard (offline viewing)."""
    from sentinel.demo.dashboard import run_dashboard

    records = read_log(log_path)

    def _paced() -> Iterator[EvidenceRecord]:
        last_ns = None
        for r in records:
            if last_ns is not None:
                dt = max(0.0, (r.monotonic_ns - last_ns) / 1e9 / speed)
                time.sleep(min(dt, 0.2))
            last_ns = r.monotonic_ns
            yield r

    run_dashboard(_paced())


@main.command()
@click.argument("log_path")
def replay(log_path: str) -> None:
    """Re-run a recorded log through a fresh decision loop and assert bit-identical decisions."""
    try:
        result = replay_log(log_path, strict=True)
    except ReplayMismatchError as exc:
        click.echo(f"REPLAY MISMATCH:\n{exc}", err=True)
        sys.exit(1)
    click.echo(f"Replay OK: {result.total_records} records bit-identical to the recorded log.")


@main.command()
@click.argument("log_path")
def verify(log_path: str) -> None:
    """Verify a log's tamper-evident hash chain."""
    records = read_log(log_path)
    try:
        verify_chain(records)
    except HashChainError as exc:
        click.echo(f"HASH CHAIN BROKEN: {exc}", err=True)
        sys.exit(1)
    click.echo(f"Hash chain OK: {len(records)} records, untampered.")


@main.command(name="eval")
@click.option("--all", "run_all", is_flag=True, default=True)
@click.option("--nominal-seeds", default=10, type=int)
@click.option("--fault-seeds", default=8, type=int)
@click.option("--nominal-duration", default=60.0, type=float)
@click.option("--fault-duration", default=40.0, type=float)
@click.option("--out-dir", default="eval_results")
def eval_cmd(
    run_all: bool, nominal_seeds: int, fault_seeds: int, nominal_duration: float, fault_duration: float, out_dir: str
) -> None:
    """Run the full fault-injection evaluation and write results.md + charts."""
    from sentinel.eval.report import run_full_eval

    path = run_full_eval(
        out_dir=out_dir,
        n_nominal_seeds=nominal_seeds,
        n_fault_seeds=fault_seeds,
        nominal_duration_s=nominal_duration,
        fault_duration_s=fault_duration,
    )
    click.echo(f"Wrote {path}")


@main.command()
@click.option("--seed", default=1, type=int)
@click.option("--out", default="evidence_logs/scripted_demo.jsonl")
def demo(seed: int, out: str) -> None:
    """Run the spec's scripted 3-minute demo: sensor agreement degrades,
    Sentinel drops to DEGRADED, an overtake request is denied with a cited
    rule, the condition clears, authority is restored only after the
    dwell time — then replay from the log proves bit-identical decisions.
    """
    loop = build_decision_loop(out)
    printed_degrade_incident = False
    printed_recovery = False

    for frame, _active_fault in scripted_demo(seed=seed):
        record = loop.process_frame(frame)

        if record.trust_transition is not None:
            click.echo(
                f"\n[t={record.monotonic_ns / 1e9:6.1f}s] TRANSITION: "
                f"{record.trust_transition.from_state} -> {record.trust_transition.to_state}"
            )
            click.echo(f"  reason: {record.trust_transition.reason}")

            if record.trust_state == "DEGRADED" and not printed_degrade_incident:
                click.echo("\n--- INCIDENT REPORT (first DEGRADE) ---")
                click.echo(format_incident_report(record))
                overtake_decision = next(
                    d for d in record.policy_decisions if d.action_class == ActionClass.OVERTAKE.value
                )
                click.echo(f"\nOVERTAKE REQUEST: {overtake_decision.decision} [{overtake_decision.rule_id}]")
                click.echo(f"  {overtake_decision.justification}")
                printed_degrade_incident = True

            if record.trust_state == "FULL_AUTONOMY" and printed_degrade_incident and not printed_recovery:
                click.echo("\n--- AUTHORITY RESTORED ---")
                click.echo(format_incident_report(record))
                printed_recovery = True

    loop.evidence_writer.close()
    click.echo(f"\nWrote {out}. Verifying hash chain and replay determinism...")

    records = read_log(out)
    verify_chain(records)
    click.echo("Hash chain: OK")

    result = replay_log(out, strict=True)
    click.echo(f"Replay: OK ({result.total_records} records bit-identical)")


if __name__ == "__main__":
    main()
