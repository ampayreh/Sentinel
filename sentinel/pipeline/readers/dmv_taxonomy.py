"""CA DMV AV Disengagement Report parsing + failure-class taxonomy.

HONESTY NOTE (read before trusting anything below operationally): the CSV
parser here is real and was NOT run against an actual downloaded DMV file
in this build — dmv.ca.gov returned HTTP 403 to this sandbox's network
access (verified: `curl -I` was actually attempted, see
scripts/download_dmv_disengagement.py). `_COLUMN_ALIASES` reflects CA
DMV's disengagement report schema as documented publicly in general
knowledge, not a schema inspected from a real file this session. The
classifier and monitor-mapping table below encode the well-documented,
industry-standard AV disengagement failure taxonomy (perception failure,
planner/behavior discrepancy, hardware/sensor fault, software fault,
communication/latency fault, adverse weather, unexpected road-user
behavior, precautionary/proactive test-driver disengagement, localization
fault, construction/road-work zone) — this is real domain knowledge, but
it has not been validated against actual free-text disengagement
descriptions from a downloaded CA DMV file. Re-run `cluster_directory`
against a real download and inspect misclassifications before trusting
the resulting failure-class distribution as ground truth.

Design choice: deterministic keyword-based classification, NOT an ML
clusterer. A DMV taxonomy that feeds directly into "which monitor catches
this failure class" needs to be auditable line-by-line (a reviewer must
be able to see exactly why record N was classified as X) — an unsupervised
clustering algorithm would produce classes that drift with the input data
and can't be inspected rule-by-rule the way this can.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class FailureClass(str, Enum):
    PERCEPTION_FAILURE = "perception_failure"
    PLANNER_BEHAVIOR_DISCREPANCY = "planner_behavior_discrepancy"
    HARDWARE_SENSOR_FAULT = "hardware_sensor_fault"
    SOFTWARE_FAULT = "software_fault"
    COMMUNICATION_LATENCY_FAULT = "communication_latency_fault"
    LOCALIZATION_FAULT = "localization_fault"
    ADVERSE_WEATHER = "adverse_weather"
    UNEXPECTED_ROAD_USER_BEHAVIOR = "unexpected_road_user_behavior"
    CONSTRUCTION_ROAD_WORK = "construction_road_work"
    PRECAUTIONARY_TEST_DRIVER = "precautionary_test_driver"
    UNKNOWN_OTHER = "unknown_other"


# Ordered: first matching class wins, so ordering encodes precedence for
# descriptions that could match multiple keyword sets (e.g. a perception
# failure ROOT-CAUSED by weather is filed as ADVERSE_WEATHER, checked
# before the generic PERCEPTION_FAILURE bucket).
_KEYWORD_RULES: list[tuple[FailureClass, tuple[str, ...]]] = [
    (
        FailureClass.ADVERSE_WEATHER,
        ("rain", "fog", "glare", "sun glare", "snow", "wet road", "weather"),
    ),
    (
        FailureClass.CONSTRUCTION_ROAD_WORK,
        ("construction", "road work", "cone", "lane closure", "work zone"),
    ),
    (
        FailureClass.UNEXPECTED_ROAD_USER_BEHAVIOR,
        ("pedestrian", "cyclist", "jaywalk", "erratic", "cut in", "unpredictable", "other vehicle"),
    ),
    (
        FailureClass.LOCALIZATION_FAULT,
        ("gps", "localization", "position error", "map mismatch", "pose"),
    ),
    (
        FailureClass.HARDWARE_SENSOR_FAULT,
        ("hardware discrepancy", "sensor fault", "camera fault", "lidar fault", "radar fault", "occlusion"),
    ),
    (
        FailureClass.COMMUNICATION_LATENCY_FAULT,
        ("communication", "latency", "delay", "connection", "dropped frame"),
    ),
    (
        FailureClass.SOFTWARE_FAULT,
        ("software discrepancy", "software fault", "system fault", "control fault", "crash", "reboot"),
    ),
    (
        FailureClass.PLANNER_BEHAVIOR_DISCREPANCY,
        ("incorrect behavior", "unwanted maneuver", "hard brake", "planner", "trajectory"),
    ),
    (
        FailureClass.PERCEPTION_FAILURE,
        ("perception discrepancy", "false positive", "false negative", "misclassif", "detection"),
    ),
    (
        FailureClass.PRECAUTIONARY_TEST_DRIVER,
        ("precautionary", "proactive", "safe operation of the vehicle", "test driver discretion"),
    ),
]


def classify_description(description: str) -> FailureClass:
    text = description.lower()
    for failure_class, keywords in _KEYWORD_RULES:
        if any(kw in text for kw in keywords):
            return failure_class
    return FailureClass.UNKNOWN_OTHER


# The deliverable the spec asks for: failure class -> which Sentinel
# monitor catches it, with gaps documented rather than hidden.
FAILURE_CLASS_TO_MONITOR: dict[FailureClass, list[str]] = {
    FailureClass.PERCEPTION_FAILURE: ["M1_PERCEPTION"],
    FailureClass.PLANNER_BEHAVIOR_DISCREPANCY: [],  # GAP: see note below
    FailureClass.HARDWARE_SENSOR_FAULT: ["M2_SENSOR_AGREEMENT", "M3_TIMING"],
    FailureClass.SOFTWARE_FAULT: ["M3_TIMING"],
    FailureClass.COMMUNICATION_LATENCY_FAULT: ["M3_TIMING"],
    FailureClass.LOCALIZATION_FAULT: ["M4_ODD_COMPLIANCE", "M2_SENSOR_AGREEMENT"],
    FailureClass.ADVERSE_WEATHER: ["M4_ODD_COMPLIANCE", "M1_PERCEPTION"],
    FailureClass.UNEXPECTED_ROAD_USER_BEHAVIOR: [],  # GAP: see note below
    FailureClass.CONSTRUCTION_ROAD_WORK: ["M4_ODD_COMPLIANCE"],  # via VLM advisory evidence, POL-014
    FailureClass.PRECAUTIONARY_TEST_DRIVER: [],  # GAP: see note below
    FailureClass.UNKNOWN_OTHER: [],
}

# Documented gaps (spec: "Any class no monitor catches is a documented
# gap, not a hidden one"):
#   - PLANNER_BEHAVIOR_DISCREPANCY: Sentinel has no monitor over the
#     autonomy stack's planning/decision layer itself (by design — it
#     observes sensing/timing/ODD, not the planner's internal state). A
#     "planner disagreement" monitor would need access to the planner's
#     candidate trajectories, which is out of scope for a runtime
#     assurance layer that sits OUTSIDE the autonomy stack.
#   - UNEXPECTED_ROAD_USER_BEHAVIOR: partially covered indirectly (a
#     sudden pedestrian/cyclist event often also perturbs M1's perception
#     features and could trip M1), but there is no dedicated monitor for
#     "another road user did something unpredictable" as a first-class
#     signal — this would need a scene-level behavior-prediction monitor,
#     which is not built in this milestone.
#   - PRECAUTIONARY_TEST_DRIVER: by definition these are cases where
#     nothing measurably went wrong and a human intervened out of
#     caution. No monitor SHOULD catch these — they are not a Sentinel
#     gap, they are outside what a health-signal-based system can or
#     should flag; noted here for completeness of the taxonomy table.


# --- CSV parsing (untested against a real file — see module docstring) ---

_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "manufacturer": ("manufacturer", "company"),
    "date": ("date", "date of disengagement"),
    "description": (
        "description of facts causing disengagement",
        "description of facts",
        "narrative",
        "description",
    ),
}


@dataclass(frozen=True)
class DisengagementRecord:
    manufacturer: str
    date: str
    description: str
    failure_class: FailureClass


def _resolve_columns(fieldnames: list[str]) -> dict[str, str]:
    lower_map = {fn.strip().lower(): fn for fn in fieldnames}
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for canonical, aliases in _COLUMN_ALIASES.items():
        found = next((lower_map[a] for a in aliases if a in lower_map), None)
        if found is None:
            missing.append(canonical)
        else:
            resolved[canonical] = found
    if missing:
        raise ValueError(
            f"Could not find columns for {missing} in CSV header {fieldnames}. "
            "The DMV report schema may differ from what this parser expects "
            "(see module docstring's honesty note) — update _COLUMN_ALIASES "
            "after inspecting the real file."
        )
    return resolved


def parse_csv(path: Path) -> list[DisengagementRecord]:
    with path.open(encoding="utf-8-sig", errors="replace") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return []
        cols = _resolve_columns(list(reader.fieldnames))
        records = []
        for row in reader:
            desc = row.get(cols["description"], "") or ""
            records.append(
                DisengagementRecord(
                    manufacturer=row.get(cols["manufacturer"], "") or "",
                    date=row.get(cols["date"], "") or "",
                    description=desc,
                    failure_class=classify_description(desc),
                )
            )
        return records


@dataclass
class ClusterResult:
    records: list[DisengagementRecord] = field(default_factory=list)

    def counts(self) -> Counter[FailureClass]:
        return Counter(r.failure_class for r in self.records)

    def summary_table(self) -> str:
        counts = self.counts()
        total = sum(counts.values()) or 1
        lines = ["failure_class,count,pct,monitors_covering,gap"]
        for fc in FailureClass:
            n = counts.get(fc, 0)
            monitors = FAILURE_CLASS_TO_MONITOR.get(fc, [])
            gap = "GAP" if not monitors else ""
            lines.append(f"{fc.value},{n},{n / total:.1%},{'|'.join(monitors)},{gap}")
        return "\n".join(lines)


def cluster_directory(csv_paths: list[Path]) -> ClusterResult:
    result = ClusterResult()
    for p in csv_paths:
        result.records.extend(parse_csv(p))
    return result
