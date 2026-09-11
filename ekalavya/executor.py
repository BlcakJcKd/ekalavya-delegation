"""Narrow execution bridge for explicitly configured Ekalavya routes."""

from __future__ import annotations

from pathlib import Path

from delegation.core import run_consultation
from .readiness import binding_preflight


def execute(resolution: dict, prompt_file: Path, workspace: Path, *, primary: str | None = None, timeout_seconds: int | None = None) -> dict[str, object]:
    """Execute only a route named in the resolved catalogue entry.

    Ekalavya V1 does not invent a provider adapter. A catalogue entry may carry
    execution route (for example a configured adapter name); that route is passed to the
    existing safety-hardened wrapper, which enforces scope, depth, stdin,
    timeout, process cleanup, and response retention. Missing route metadata
    is a deterministic harness-unavailable result.
    """
    candidate = resolution.get("resolved") or {}
    route = candidate.get("execution_route") or candidate.get("legacy_route")
    resolved_model = candidate.get("provider_model_id")
    resolved_reasoning = resolution.get("resolved_reasoning")
    checked = binding_preflight(
        candidate, route, candidate.get("harness"),
        model=resolved_model, reasoning=resolved_reasoning,
    )
    if not checked["ok"]:
        return {"state": "harness-unavailable", "reason": checked["reason"]}
    task = prompt_file.read_text(encoding="utf-8")
    timeout = timeout_seconds or 300
    if route.startswith("vllm:"):
        from delegation.vllm import run_vllm_consultation
        route_name = route.split(":", 1)[1]
        outcome = run_vllm_consultation(route_name, workspace, task, timeout_seconds=timeout)
        code, evidence = outcome.exit_code, outcome.record_dir
    else:
        code, evidence = run_consultation(
            route, workspace, task, timeout_seconds=timeout, primary=primary,
            caller="ekalavya", model=resolved_model, effort=resolved_reasoning,
        )
    result: dict[str, object] = {"state": "completed" if code == 0 else "failed", "exit_code": code, "evidence": str(evidence), "retries": 0}
    metadata = evidence / "execution.json"
    if metadata.is_file():
        import json
        captured = json.loads(metadata.read_text(encoding="utf-8"))
        for key in ("response_status", "response_recorded", "response_file", "request_count", "wall_seconds", "timed_out", "provider", "requested_model", "provider_reported_model_id", "provider_reported_usage", "provider_reported_usage_by_model", "usage_provenance", "telemetry_parse_warning", "requested_effort", "transport", "error_category"):
            if key in captured:
                result[key] = captured[key]
    return result
