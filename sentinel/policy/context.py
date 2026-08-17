"""Derive the policy gate's ODD `context_flags` set from monitor evidence.

Pure function, deterministic, no I/O. Reads only the `evidence` dict M4
(and, for the geofence flag, the raw HealthFrame) already computed —
never re-derives ODD facts independently, so there is exactly one place
(`sentinel.monitors.m4_odd_compliance`) that decides what's true about the
operating envelope, and this module only relabels that into the flag
vocabulary `sentinel.policy.gate` understands.
"""

from __future__ import annotations

from sentinel.schemas.health_frame import HealthFrame
from sentinel.schemas.monitor_result import MonitorResult


def derive_context_flags(
    m4_result: MonitorResult | None, frame: HealthFrame
) -> frozenset[str]:
    flags: set[str] = set()
    if m4_result is not None:
        ev = m4_result.evidence
        localization_degraded = bool(ev.get("localization_degraded", False))
        construction_zone_flagged = bool(ev.get("construction_zone_flagged", False))
        if construction_zone_flagged and localization_degraded:
            flags.add("construction_zone_and_degraded_localization")
        if frame.geofence_ok is False:
            flags.add("geofence_violation")
        if frame.gnss is not None and frame.gnss.fix_valid is False:
            flags.add("gnss_fix_invalid")
    return frozenset(flags)
