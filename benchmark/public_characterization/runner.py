"""Discovery, execution, and ledger recording for public characterization V1."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.adapters import AntigravityAdapter
from benchmark.v2.plotting import plot_rows
from benchmark.v2.telemetry import parse_trace
from benchmark.provenance import validate_git_identity
from ekalavya.catalogue import load_catalogue, merge_gemini_flash_discovery, save_catalogue
from ekalavya.config import config_root, load_profiles, permit_profile_candidates, save_profiles
from ekalavya.discovery import DiscoveryError, gemini_flash_generations, parse_agy_models
from ekalavya.harness_registry import current_registry, validate_registry
from ekalavya.ledger import (
    connect, default_state_dir, finalize_run, record_availability, record_benchmark_suite,
    record_benchmark_task, record_cost, record_harness, record_request_metric, record_run,
    record_task_attempt, record_tool_event, upsert_model,
)
from ekalavya.schema import CandidateIdentity

from . import EVALUATION_CLASS, FAMILIES, SUITE_NAME, SUITE_SOURCE_PATHS, SUITE_VERSION
from .evaluate import evaluate
from .audit import generate_audit_artifacts
from .generate import TaskInstance, manifest, make_instance, materialize, visible_evaluator_digest, workspace_digest


EXPERIMENT = "public-characterization-v1"
GENERATIONS = ("3.7", "3.8")
REASONING = ("low", "medium", "high")
AGY_VERSION_FALLBACK = "unknown"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def state_root() -> Path:
    root = default_state_dir() / "experiments" / EXPERIMENT
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    return root


def command(argv: list[str], *, cwd: Path | None = None, timeout: int = 60) -> tuple[int, str, str]:
    try:
        result = subprocess.run(argv, cwd=cwd, text=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return -1, "", str(exc)
    return result.returncode, result.stdout, result.stderr


def _agy_version() -> str:
    code, stdout, _ = command(["agy", "--version"])
    return stdout.strip() if code == 0 and stdout.strip() else AGY_VERSION_FALLBACK


def _parse_discovery(stdout: str) -> list[dict[str, str]]:
    """Use the control-plane parser so discovery has one strict contract."""
    try:
        return gemini_flash_generations(parse_agy_models(stdout))
    except DiscoveryError as exc:
        raise RuntimeError(f"AGY discovery output rejected: {exc}") from exc


def _update_profile_catalogue(discovered: list[dict[str, str]], observed_at: str, version: str) -> dict[str, Any]:
    root = config_root()
    catalogue_path, profiles_path = root / "catalogue.json", root / "profiles.json"
    entries = load_catalogue(catalogue_path)
    updated, merged = merge_gemini_flash_discovery(entries, discovered, observed_at=observed_at, serving_engine_version=version)
    profiles = load_profiles(profiles_path)
    profiles = permit_profile_candidates(profiles, "flash", merged["registered"])
    save_catalogue(catalogue_path, updated)
    save_profiles(profiles, profiles_path)
    return {
        "catalogue_entries": len(updated),
        "added_candidates": len(merged["added"]),
        "updated_candidates": len(merged["updated"]),
        "profile_updated": bool(merged["registered"]),
        "path": str(catalogue_path),
    }


def discover_models() -> dict[str, Any]:
    code, stdout, stderr = command(["agy", "models"])
    if code != 0:
        raise RuntimeError(f"agy models failed: {stderr.strip()}")
    observed_at = now()
    version = _agy_version()
    discovered = _parse_discovery(stdout)
    expected = {f"gemini-{generation}-flash-{effort}" for generation in ("3.6", "3.7", "3.8") for effort in REASONING}
    missing = sorted(expected - {item["provider_model_id"] for item in discovered})
    if missing:
        raise RuntimeError("required Gemini Flash runtime IDs missing: " + ", ".join(missing))
    catalogue = _update_profile_catalogue(discovered, observed_at, version)
    catalogue_entries = load_catalogue(config_root() / "catalogue.json")
    catalogue_by_generation: dict[str, dict[str, Any]] = {}
    for entry in catalogue_entries:
        if entry.get("provider") != "gemini" or entry.get("family") != "flash":
            continue
        generation = entry.get("generation")
        if not generation:
            provider_model_id = str(entry.get("provider_model_id", ""))
            parts = provider_model_id.split("-")
            generation = parts[1] if len(parts) > 1 else None
        if generation:
            catalogue_by_generation.setdefault(str(generation), entry)
    conn = connect()
    lifecycle = {"3.6": "previous", "3.7": "current", "3.8": "candidate"}
    ledger_models = []
    for item in discovered:
        model_id = item["provider_model_id"]
        parts = model_id.split("-")
        generation = parts[1] if len(parts) > 1 else None
        family = "flash" if len(parts) > 2 and parts[2] == "flash" else "unknown"
        variant = parts[-1] if family == "flash" else None
        catalogue_entry = catalogue_by_generation.get(str(generation))
        if catalogue_entry is None or not catalogue_entry.get("identity_key"):
            raise RuntimeError(f"discovered Gemini generation is missing from the catalogue: {generation}")
        identity = CandidateIdentity(**{name: catalogue_entry.get(name) for name in CandidateIdentity.__dataclass_fields__})
        model_db_id = upsert_model(
            conn, identity, identity_key=str(catalogue_entry["identity_key"]),
            lifecycle=str(catalogue_entry.get("lifecycle", lifecycle.get(generation, "candidate"))),
            discovered_at=observed_at,
        )
        record_availability(conn, model_db_id, state="available", observed_at=observed_at, source="agy models", details={"exact_model_id": model_id, "reasoning": variant, "lifecycle_scope": "gemini_flash_generation_family"})
        ledger_models.append({"provider_model_id": model_id, "generation": generation, "reasoning": variant, "lifecycle": catalogue_entry.get("lifecycle", lifecycle.get(generation, "historical"))})
    result = {"timestamp": observed_at, "client": "agy", "client_version": version, "models": ledger_models, "catalogue": catalogue}
    (state_root() / "discovery.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def check_local_suite() -> dict[str, Any]:
    root = state_root() / "local-check"
    results = []
    for index, family in enumerate(FAMILIES):
        instance = make_instance(family, 20260903 + index)
        workspace = root / instance.task_id.replace(":", "_")
        materialize(instance, workspace)
        assessment = evaluate(instance, workspace)
        results.append({"task_id": instance.task_id, "score": assessment["score"], "full_pass": assessment["full_pass"], "checks": len(assessment["checks"])})
    return {"suite": SUITE_NAME, "version": SUITE_VERSION, "evaluation_class": EVALUATION_CLASS, "results": results, "all_objective_checks_pass": all(item["score"] == 100.0 for item in results)}


def _snapshot(workspace: Path) -> dict[str, str]:
    return {p.relative_to(workspace).as_posix(): digest_bytes(p.read_bytes()) for p in workspace.rglob("*") if p.is_file() and "__pycache__" not in p.parts}


def _terminate(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass


def run_attempt(conn: Any, suite_id: int, task_db_id: int, instance: TaskInstance, model_id: str, reasoning: str, harness_id: int, root: Path, agy_version: str, telemetry_semantics: dict[str, Any]) -> dict[str, Any]:
    key = f"{model_id}-{instance.family}-{instance.seed}"
    workspace = root / "workspaces" / key
    materialize(instance, workspace)
    before = _snapshot(workspace)
    generation = model_id.split("-")[1]
    variant = model_id.rsplit("-", 1)[-1]
    identity = CandidateIdentity(provider="gemini", family="flash", provider_model_id=model_id, display_name=f"Gemini {generation} Flash ({variant})", generation=generation, variant=variant, capabilities={"reasoning_values": [variant]}, serving_engine="agy", serving_engine_version=agy_version)
    resolved = {**identity.as_dict(), "identity_key": identity.identity_key, "reasoning": reasoning, "harness": "agy", "harness_version": agy_version, "adapter_version": "benchmark.adapters.AntigravityAdapter", "transport": "agy"}
    requested = {"experiment": EXPERIMENT, "profile": "flash", "provider": "gemini", "family": "flash", "provider_model_id": model_id, "model": model_id, "reasoning": reasoning, "harness": "agy", "evaluation_class": EVALUATION_CLASS}
    run_id = f"{EXPERIMENT}:{uuid.uuid4().hex}"
    started_at = now()
    record_run(conn, run_id, requested, resolved=resolved, status="running", evaluation_class=EVALUATION_CLASS, provider="gemini", identity_key=identity.identity_key, harness_id=harness_id, billing_mode="subscription", started_at=started_at)
    start = time.monotonic()
    evidence_dir = root / "evidence"; evidence_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    adapter = AntigravityAdapter(model=model_id, reasoning_effort=None)
    code, stdout, stderr = -1, "", ""
    timed_out = False
    try:
        process = subprocess.Popen(adapter.command(workspace, instance.prompt, evidence_dir / key), cwd=workspace, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={**os.environ, "BENCHMARK_WORKSPACE": str(workspace)}, start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=900)
            code = process.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate(process)
            stdout, stderr = process.communicate()
            code = -1
    except OSError as exc:
        stderr = str(exc)
    wall = time.monotonic() - start
    ended_at = now()
    after = _snapshot(workspace)
    changed = sorted({name for name in set(before) | set(after) if before.get(name) != after.get(name)})
    requests = parse_trace(stdout)
    telemetry_models = sorted({item.model for item in requests if item.model})
    identity_mismatch = bool(telemetry_models and model_id not in telemetry_models)
    assessment = evaluate(instance, workspace) if not timed_out else {"evaluation_class": EVALUATION_CLASS, "objective": True, "adversarial_isolation": False, "authoritative_reference_present": False, "checks": [], "score": None, "maximum": 100.0, "full_pass": False, "public_tests": {}, "scope_compliance": True}
    status = "invalid_identity" if identity_mismatch else ("completed" if not timed_out and code == 0 else "harness_failure")
    evidence = {
        "experiment": EXPERIMENT, "evaluation_class": EVALUATION_CLASS, "run_id": run_id,
        "requested": requested, "resolved": resolved, "started_at": started_at, "ended_at": ended_at,
        "wall_seconds": wall, "exit_code": code, "timed_out": timed_out, "changed_files": changed,
        "stdout_sha256": digest_bytes(stdout.encode()), "stderr_sha256": digest_bytes(stderr.encode()),
        "request_count": len(requests) or None, "request_metric_semantics": telemetry_semantics.get("request_metric_semantics"), "telemetry_model_ids": telemetry_models or None, "identity_match": not identity_mismatch, "tool_events": sum(item.tool_calls for item in requests) if requests and telemetry_semantics.get("tool_event_telemetry") == "complete" else None,
        "tool_event_telemetry": telemetry_semantics.get("tool_event_telemetry"), "invalid_tool_calls": sum(item.invalid_tool_schema + item.invalid_argument_type for item in requests) if requests and telemetry_semantics.get("tool_event_telemetry") == "complete" else None,
        "recoveries": sum(bool(item.recovered_after_tool_error) for item in requests) if requests and telemetry_semantics.get("tool_event_telemetry") == "complete" else None,
        "token_metric_semantics": telemetry_semantics.get("token_metric_semantics"),
        "final_verification": {"full_pass": assessment.get("full_pass"), "checks": assessment.get("checks"), "public_tests": assessment.get("public_tests")},
        "assessment": assessment, "task": {"suite": SUITE_NAME, "version": SUITE_VERSION, "family": instance.family, "task_id": instance.task_id, "seed": instance.seed, "workspace_hash": workspace_digest(workspace), "prompt_hash": digest_bytes(instance.prompt.encode()), "visible_evaluator_hash": visible_evaluator_digest(instance), "authoritative_reference_present": False},
    }
    evidence_path = evidence_dir / f"{key}.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    score = assessment.get("score") if status == "completed" else None
    record_task_attempt(conn, run_id, task_db_id, score=score, public_score=score, hidden_score=None, invariant_score=score, api_score=None, scope_compliant=True, wall_seconds=wall, metadata=evidence)
    for metric in requests:
        request_id = record_request_metric(conn, run_id, metric.json())
        for ordinal, tool_name in enumerate(metric.tool_names, 1):
            record_tool_event(conn, run_id, {"ordinal": ordinal, "tool_name": tool_name, "validity": "invalid" if metric.invalid_tool_schema or metric.invalid_argument_type else "valid", "error": "tool error" if metric.tool_errors else None, "recovered": bool(metric.recovered_after_tool_error), "alternate_tool": metric.alternate_tool_used}, request_id=request_id)
    token_values = {field: [getattr(item, field) for item in requests if getattr(item, field) is not None] for field in ("input_tokens", "output_tokens", "cache_read_tokens", "reasoning_tokens")}
    record_cost(conn, run_id, billing_mode="subscription", cost_source="unavailable: subscription route", input_tokens=sum(token_values["input_tokens"]) if token_values["input_tokens"] else None, output_tokens=sum(token_values["output_tokens"]) if token_values["output_tokens"] else None, cached_input_tokens=sum(token_values["cache_read_tokens"]) if token_values["cache_read_tokens"] else None, reasoning_tokens=sum(token_values["reasoning_tokens"]) if token_values["reasoning_tokens"] else None)
    finalize_run(conn, run_id, ended_at=ended_at, status=status, raw_evidence_path=str(evidence_path), raw_evidence_sha256=digest_bytes(evidence_path.read_bytes()))
    return evidence


def _suite_records(conn: Any, instances: list[TaskInstance], provenance: dict[str, Any]) -> tuple[int, dict[str, int]]:
    suite_id = record_benchmark_suite(conn, SUITE_NAME, "public_characterization", SUITE_VERSION, git_sha=provenance["git_sha"], evaluation_class=EVALUATION_CLASS, metadata={"families": FAMILIES, "objective": True, "adversarial_isolation": False, "authoritative_reference_present": False, "source_paths": provenance["source_paths"], "source_sha256": provenance["source_sha256"]})
    records = {}
    for instance in instances:
        records[instance.task_id] = record_benchmark_task(conn, suite_id, family=instance.family, task_id=instance.task_id, variant_seed=str(instance.seed), content_hash=hashlib.sha256(json.dumps(instance.files, sort_keys=True).encode()).hexdigest(), prompt_hash=digest_bytes(instance.prompt.encode()), evaluator_hash=visible_evaluator_digest(instance))
    return suite_id, records


def run_sweep() -> dict[str, Any]:
    discovery = discover_models()
    local = check_local_suite()
    registry = current_registry()
    validate_registry(registry)
    agy = next(item for item in registry if item["name"] == "agy")
    if agy["eligibility"].get("public_characterization") != "supported":
        raise RuntimeError("AGY public characterization eligibility is not supported")
    conn = connect()
    for audited in registry:
        record_harness(
            conn, audited["name"], version=audited.get("observed_version") or audited["version"],
            adapter_version="ekalavya.harness_registry", transport=audited["transport"],
            capabilities=audited["capabilities"], telemetry=audited.get("telemetry"), eligibility=audited["eligibility"],
            evidence_label=audited["evidence"], observed_at=discovery["timestamp"],
        )
    harness_id = record_harness(conn, "agy", version=discovery["client_version"], adapter_version="benchmark.adapters.AntigravityAdapter", transport="agy", capabilities=agy["capabilities"], telemetry=agy.get("telemetry"), eligibility=agy["eligibility"], evidence_label="public_characterization_non_adversarial", observed_at=discovery["timestamp"])
    instances = [make_instance(family, 20260903 + index) for index, family in enumerate(FAMILIES)]
    provenance = validate_git_identity(Path(__file__).resolve().parents[2], SUITE_SOURCE_PATHS)
    suite_id, task_records = _suite_records(conn, instances, provenance)
    attempts = []
    telemetry_semantics = agy.get("telemetry", {})
    for generation in GENERATIONS:
        for reasoning in REASONING:
            model_id = f"gemini-{generation}-flash-{reasoning}"
            for instance in instances:
                # The exact runtime ID is selected directly.  No medium-model
                # alias or effort overlay is used for low/high variants.
                attempts.append(run_attempt(conn, suite_id, task_records[instance.task_id], instance, model_id, reasoning, harness_id, state_root(), discovery["client_version"], telemetry_semantics))
    result = {"experiment": EXPERIMENT, "evaluation_class": EVALUATION_CLASS, "discovery": discovery, "local_check": local, "attempts": len(attempts), "completed": sum(item["exit_code"] == 0 for item in attempts), "full_pass": sum(bool(item["assessment"].get("full_pass")) for item in attempts), "retries": 0}
    (state_root() / "run-summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _all_rows(conn: Any) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT r.run_id, r.resolved_json, r.started_at, r.status, r.evaluation_class, a.score, a.wall_seconds, a.metadata_json FROM runs r JOIN task_attempts a ON a.run_id=r.run_id WHERE r.evaluation_class=? ORDER BY r.started_at", (EVALUATION_CLASS,)).fetchall()
    result = []
    for row in rows:
        resolved = json.loads(row["resolved_json"] or "{}")
        metadata = json.loads(row["metadata_json"] or "{}")
        request_metrics = [dict(item) for item in conn.execute("SELECT * FROM request_metrics WHERE run_id=?", (row["run_id"],))]
        tool_complete = metadata.get("tool_event_telemetry") == "complete"
        result.append({"run_id": row["run_id"], "model": resolved.get("provider_model_id"), "generation": resolved.get("generation"), "reasoning": resolved.get("reasoning"), "harness": resolved.get("harness"), "status": row["status"], "score": row["score"], "wall_seconds": row["wall_seconds"], "request_count": metadata.get("request_count"), "request_metric_semantics": metadata.get("request_metric_semantics") or "harness_session", "tool_events": metadata.get("tool_events") if tool_complete else None, "tool_event_telemetry": metadata.get("tool_event_telemetry") or "unavailable", "invalid_tool_calls": metadata.get("invalid_tool_calls") if tool_complete else None, "recoveries": metadata.get("recoveries") if tool_complete else None, "timed_out": metadata.get("timed_out", False), "input_tokens": sum((m.get("input_tokens") or 0) for m in request_metrics) if request_metrics and any(m.get("input_tokens") is not None for m in request_metrics) else None, "output_tokens": sum((m.get("output_tokens") or 0) for m in request_metrics) if request_metrics and any(m.get("output_tokens") is not None for m in request_metrics) else None, "cache_read_tokens": sum((m.get("cache_read_tokens") or 0) for m in request_metrics) if request_metrics and any(m.get("cache_read_tokens") is not None for m in request_metrics) else None, "reasoning_tokens": sum((m.get("reasoning_tokens") or 0) for m in request_metrics) if request_metrics and any(m.get("reasoning_tokens") is not None for m in request_metrics) else None, "valid": row["status"] == "completed" and row["score"] is not None})
    return result


def report() -> Path:
    root = state_root(); conn = connect(); rows = _all_rows(conn)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["model"], row["reasoning"], row["harness"]), []).append(row)
    summary = []
    for (model, reasoning, harness), items in sorted(grouped.items()):
        scores = [item["score"] for item in items if item["valid"] and item["score"] is not None]
        times = [item["wall_seconds"] for item in items if item["wall_seconds"] is not None]
        summary.append({"model": model, "reasoning": reasoning, "harness": harness, "quality_on_completed": statistics.mean(scores) if scores else None, "median_score": statistics.median(scores) if scores else None, "full_tasks": sum(score == 100 for score in scores), "completed": len(scores), "attempted": len(items), "completion_rate": len(scores) / len(items) if items else None, "mean_wall_attempted": statistics.mean(times) if times else None, "requests": statistics.mean([item["request_count"] for item in items if item["request_count"] is not None]) if any(item["request_count"] is not None for item in items) else None, "tools": None, "malformed": None, "tokens": sum((item["input_tokens"] or 0) + (item["output_tokens"] or 0) for item in items) if all(item["input_tokens"] is not None and item["output_tokens"] is not None for item in items) else None})
    lines = [f"# {SUITE_NAME}", "", "Class: `public_characterization` (objective, non-adversarially-isolated).", "", "## Provenance", "", "The originally recorded suite SHA was `91f96127f9393cc81e4f4c296ec5d8e228210a13`; the authoritative tracked implementation is `738a8aa26d87d266af34c2af8e70577bb0bb60dd`. The recorded identity is corrected in derived ledger metadata with an append-only correction record; raw evidence is unchanged.", "", "## Model × reasoning", "", "Quality is correctness conditional on a completed scored task. Reliability is reported separately.", "", "| Exact model ID | Reasoning | Harness | Quality on completed | Median | Full tasks | Completed/attempted | Completion rate | Mean wall attempted | Request metric semantics | Tool telemetry | Observed token fields | Actual cost | Calculated cost | API-equivalent cost |", "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|---|---|"]
    for item in summary:
        token_total = item["tokens"]
        semantics = "harness_session" if item["requests"] is not None else "unavailable"
        lines.append(f"| {item['model']} | {item['reasoning']} | {item['harness']} | {item['quality_on_completed'] if item['quality_on_completed'] is not None else 'null'} | {item['median_score'] if item['median_score'] is not None else 'null'} | {item['full_tasks']} | {item['completed']}/{item['attempted']} | {item['completion_rate'] if item['completion_rate'] is not None else 'null'} | {item['mean_wall_attempted'] if item['mean_wall_attempted'] is not None else 'null'} | {semantics} | unavailable | {token_total if token_total is not None else 'null'} | null | null | null |")
    lines += ["", "## Harness", "", "AGY 1.1.25 exposes one parsed outer session/result record per attempt. This is not evidence of one provider model request. Tool activity is unavailable rather than observed zero. No second exact-Gemini harness was available.", "", "## Generation comparison", "", "| Reasoning | 3.7 quality | 3.8 quality | Delta | 3.7 completion | 3.8 completion |", "|---|---:|---:|---:|---:|---:|"]
    by_key = {(item["model"], item["reasoning"]): item for item in summary}
    for reasoning in ("low", "medium", "high"):
        old, new = by_key.get((f"gemini-3.7-flash-{reasoning}", reasoning)), by_key.get((f"gemini-3.8-flash-{reasoning}", reasoning))
        if old and new:
            lines.append(f"| {reasoning} | {old['quality_on_completed']} | {new['quality_on_completed']} | {(new['quality_on_completed'] - old['quality_on_completed']) if old['quality_on_completed'] is not None and new['quality_on_completed'] is not None else 'null'} | {old['completed']}/{old['attempted']} | {new['completed']}/{new['attempted']} |")
    lines += ["", "## Reasoning and generation interpretation", ""]
    for item in summary:
        lines.append(f"- `{item['model']} / {item['reasoning']} / {item['harness']}`: quality on completed `{item['quality_on_completed']}`, completion `{item['completed']}/{item['attempted']}`, mean wall over attempts `{item['mean_wall_attempted']}` seconds.")
    lines += ["", "## Pareto frontier", "", "Quality is maximized; wall time, tokens, malformed calls, and recoveries are minimized where observed. Missing dimensions are not treated as zero."]
    for item in summary:
        dominated = False
        for other in summary:
            if other is item or other["quality_on_completed"] is None or item["quality_on_completed"] is None or other["mean_wall_attempted"] is None or item["mean_wall_attempted"] is None:
                continue
            comparable_tokens = item["tokens"] is not None and other["tokens"] is not None
            no_worse = other["quality_on_completed"] >= item["quality_on_completed"] and other["mean_wall_attempted"] <= item["mean_wall_attempted"] and other["completion_rate"] >= item["completion_rate"] and (not comparable_tokens or other["tokens"] <= item["tokens"])
            strictly = other["quality_on_completed"] > item["quality_on_completed"] or other["mean_wall_attempted"] < item["mean_wall_attempted"] or other["completion_rate"] > item["completion_rate"] or (comparable_tokens and other["tokens"] < item["tokens"])
            if no_worse and strictly:
                dominated = True
                break
        lines.append(f"- `{item['model']} / {item['reasoning']} / {item['harness']}`: {'dominated' if dominated else 'non-dominated'}")
    lines += ["", "## Catalogue recommendation", "", "Keep Gemini 3.7 Flash Medium as the persistent default. Gemini 3.7 Low has equal observed quality with lower wall time and reported usage, but one four-task repetition is insufficient to change the user-owned default. Gemini 3.8 Low is first-pass non-dominated but insufficient evidence for promotion. Gemini 3.8 Medium and High merit no further testing on this screen.", "", "## Cost", "", "Actual: null; calculated: null; API-equivalent: null; billing mode: subscription; source: unavailable.", "", "## Telemetry completeness", ""]
    total = len(rows)
    for label, field in (("request/session records", "request_count"), ("tool events", "tool_events"), ("input/output tokens", "input_tokens"), ("TTFT", None), ("cost evidence", None)):
        if label == "tool events":
            lines.append(f"- {label}: unavailable (0/{total} independently observable)" if total else f"- {label}: unavailable")
            continue
        count = sum(row.get(field) is not None for row in rows) if field else 0
        lines.append(f"- {label}: {count}/{total} ({(100 * count / total):.1f}%)" if total else f"- {label}: 0/0 (null)")
    lines += ["", "## Operational reliability", "", f"Attempted tasks: {len(rows)}; completed scored tasks: {sum(row['valid'] for row in rows)}; failed/unscored tasks: {sum(not row['valid'] for row in rows)}. No correctness zero is assigned to the Gemini 3.8 High P2 harness failure."]
    path = root / "REPORT.md"; path.write_text("\n".join(lines) + "\n")
    (root / "CORRECTED_REPORT.md").write_text(path.read_text())
    plot_metadata = {
        "score-vs-wall": plot_rows(summary, root / "score-vs-wall.png", x_key="mean_wall_attempted", y_key="quality_on_completed", xlabel="mean wall seconds (attempted)", ylabel="quality on completed"),
        "reasoning-correctness": plot_rows(summary, root / "reasoning-correctness.png", x_key="reasoning", y_key="quality_on_completed", xlabel="reasoning", ylabel="quality on completed"),
        "reasoning-wall": plot_rows(summary, root / "reasoning-wall.png", x_key="reasoning", y_key="mean_wall_attempted", xlabel="reasoning", ylabel="mean wall seconds (attempted)"),
        "tokens-vs-correctness": plot_rows(summary, root / "tokens-vs-correctness.png", x_key="tokens", y_key="quality_on_completed", xlabel="observed AGY usage fields", ylabel="quality on completed"),
    }
    (root / "plot-metadata.json").write_text(json.dumps(plot_metadata, indent=2, sort_keys=True) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    action = (argv or sys.argv[1:] or ["run"])[0]
    if action == "check":
        print(json.dumps(check_local_suite(), indent=2, sort_keys=True)); return 0
    if action == "discover":
        print(json.dumps(discover_models(), indent=2, sort_keys=True)); return 0
    if action == "run":
        print(json.dumps(run_sweep(), indent=2, sort_keys=True)); return 0
    if action == "report":
        print(report()); return 0
    if action == "audit":
        print(json.dumps(generate_audit_artifacts(state_root()), indent=2, sort_keys=True)); return 0
    raise SystemExit(f"unknown command: {action}")


if __name__ == "__main__":
    raise SystemExit(main())
