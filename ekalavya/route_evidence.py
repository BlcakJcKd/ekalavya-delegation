"""Validated, packaged frozen evidence for route recommendations.

The route engine consumes this compact registry only.  It deliberately never
parses reports, retained runs, or benchmark prose at recommendation time.
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any


REQUIRED = {
    "evidence_id", "target", "identity", "task_tags", "study", "date",
    "harness", "outcome", "sample_size", "provenance", "comparison_group_id",
    "comparability",
}


def load_registry() -> dict[str, Any]:
    value = json.loads(files("ekalavya").joinpath("data/route_evidence.v1.json").read_text())
    if not isinstance(value, dict) or value.get("schema_version") != 1 or not isinstance(value.get("records"), list):
        raise ValueError("invalid route evidence registry")
    for record in value["records"]:
        if not isinstance(record, dict) or REQUIRED - set(record):
            raise ValueError("route evidence record is incomplete")
        if not isinstance(record["target"], str) or not isinstance(record["identity"], dict):
            raise ValueError("route evidence record identity is invalid")
        if not isinstance(record["task_tags"], list) or not all(isinstance(tag, str) for tag in record["task_tags"]):
            raise ValueError("route evidence task tags are invalid")
        if not isinstance(record["comparison_group_id"], str):
            raise ValueError("route evidence comparison group is invalid")
        comparability = record["comparability"]
        if not isinstance(comparability, dict) or not all(isinstance(comparability.get(key), bool) for key in ("correctness_comparable", "scope_comparable", "operational_time_comparable")):
            raise ValueError("route evidence comparability is invalid")
    return value


def applicable_records(task: str, target: str, *, registry: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    records = (registry or load_registry())["records"]
    return [record for record in records if record["target"] == target and task in record["task_tags"]]

