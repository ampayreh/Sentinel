"""Tests for the DMV disengagement taxonomy classifier.

The example descriptions below are ILLUSTRATIVE, written to match the
style of real CA DMV disengagement report narratives, but are NOT copied
from an actual downloaded file (none was available in this build
environment — see sentinel/pipeline/readers/dmv_taxonomy.py's module
docstring). These tests validate the deterministic keyword-classification
logic itself, not real-world classification accuracy.
"""

from __future__ import annotations

import csv
from pathlib import Path

from sentinel.pipeline.readers.dmv_taxonomy import (
    FAILURE_CLASS_TO_MONITOR,
    FailureClass,
    classify_description,
    cluster_directory,
    parse_csv,
)


def test_classify_perception_failure() -> None:
    fc = classify_description("Perception discrepancy: false positive detection of an object in lane")
    assert fc == FailureClass.PERCEPTION_FAILURE


def test_classify_weather_takes_precedence_over_perception() -> None:
    fc = classify_description(
        "Heavy rain reduced camera perception discrepancy in object detection confidence"
    )
    assert fc == FailureClass.ADVERSE_WEATHER


def test_classify_construction_zone() -> None:
    fc = classify_description("Vehicle approached a construction zone with lane closure and cones")
    assert fc == FailureClass.CONSTRUCTION_ROAD_WORK


def test_classify_hardware_fault() -> None:
    fc = classify_description("Hardware discrepancy: LiDAR fault detected during test run")
    assert fc == FailureClass.HARDWARE_SENSOR_FAULT


def test_classify_unknown_falls_back_to_unknown_other() -> None:
    fc = classify_description("Unrelated administrative note about permit renewal")
    assert fc == FailureClass.UNKNOWN_OTHER


def test_every_failure_class_has_a_documented_monitor_mapping_entry() -> None:
    for fc in FailureClass:
        assert fc in FAILURE_CLASS_TO_MONITOR  # even if the value is [] (a documented gap)


def test_csv_round_trip_with_illustrative_data(tmp_path: Path) -> None:
    csv_path = tmp_path / "illustrative_dmv_2024.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Manufacturer", "Date", "Description of Facts Causing Disengagement"])
        writer.writerow(["ExampleCo", "2024-01-15", "Perception discrepancy: false negative object detection"])
        writer.writerow(["ExampleCo", "2024-02-03", "GPS localization error near tunnel"])
        writer.writerow(["ExampleCo", "2024-03-11", "Precautionary disengagement, proactive test driver"])

    records = parse_csv(csv_path)
    assert len(records) == 3
    assert records[0].failure_class == FailureClass.PERCEPTION_FAILURE
    assert records[1].failure_class == FailureClass.LOCALIZATION_FAULT
    assert records[2].failure_class == FailureClass.PRECAUTIONARY_TEST_DRIVER

    result = cluster_directory([csv_path])
    table = result.summary_table()
    assert "perception_failure" in table
    assert "GAP" in table  # planner_behavior_discrepancy etc. are documented gaps
