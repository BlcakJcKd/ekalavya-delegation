"""Private SQLite ledger and conservative legacy evidence importer."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import SCHEMA_VERSION
from .schema import CandidateIdentity

EVALUATION_CLASSES = {"ordinary", "public_characterization", "hidden_benchmark", "unknown"}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_versions(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL, checksum TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS model_families(id INTEGER PRIMARY KEY, provider TEXT NOT NULL, family_key TEXT NOT NULL, lineage TEXT, display_name TEXT, UNIQUE(provider,family_key));
CREATE TABLE IF NOT EXISTS models(id INTEGER PRIMARY KEY, identity_key TEXT NOT NULL UNIQUE, provider TEXT NOT NULL, family_id INTEGER REFERENCES model_families(id), family TEXT, provider_model_id TEXT, display_name TEXT, generation TEXT, variant TEXT, capabilities_json TEXT, architecture TEXT, parameter_count TEXT, active_parameter_count TEXT, quantization TEXT, serving_engine TEXT, serving_engine_version TEXT, hardware_profile TEXT, lifecycle TEXT NOT NULL DEFAULT 'candidate', discovered_at TEXT, retired_at TEXT);
CREATE TABLE IF NOT EXISTS model_availability(id INTEGER PRIMARY KEY, model_id INTEGER NOT NULL REFERENCES models(id), state TEXT NOT NULL, observed_at TEXT NOT NULL, source TEXT, details_json TEXT);
CREATE TABLE IF NOT EXISTS harnesses(id INTEGER PRIMARY KEY, name TEXT NOT NULL, version TEXT, adapter_version TEXT, transport TEXT, capabilities_json TEXT, telemetry_json TEXT, eligibility_json TEXT, evidence_label TEXT, observed_at TEXT, UNIQUE(name,version,adapter_version));
CREATE TABLE IF NOT EXISTS serving_engines(id INTEGER PRIMARY KEY, name TEXT NOT NULL, version TEXT, UNIQUE(name,version));
CREATE TABLE IF NOT EXISTS hardware_profiles(id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, details_json TEXT);
CREATE TABLE IF NOT EXISTS profiles(id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, description TEXT, default_identity_key TEXT, permitted_candidates_json TEXT, required_capabilities_json TEXT, writable INTEGER, reasoning_policy TEXT, default_reasoning TEXT, enabled INTEGER NOT NULL DEFAULT 1);
CREATE TABLE IF NOT EXISTS benchmark_suites(id INTEGER PRIMARY KEY, name TEXT NOT NULL, layer TEXT NOT NULL, version TEXT NOT NULL, evaluation_class TEXT NOT NULL DEFAULT 'unknown', git_sha TEXT, metadata_json TEXT, UNIQUE(name,version,git_sha));
CREATE TABLE IF NOT EXISTS benchmark_suite_corrections(id INTEGER PRIMARY KEY, suite_id INTEGER NOT NULL REFERENCES benchmark_suites(id), originally_recorded_git_sha TEXT, corrected_git_sha TEXT NOT NULL, corrected_at TEXT NOT NULL, reason TEXT NOT NULL, evidence_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS benchmark_tasks(id INTEGER PRIMARY KEY, suite_id INTEGER REFERENCES benchmark_suites(id), family TEXT NOT NULL, task_id TEXT NOT NULL, variant_seed TEXT, content_hash TEXT, prompt_hash TEXT, evaluator_hash TEXT, baseline_score REAL, baseline_check_vector_json TEXT, task_spec_hash TEXT, allowed_edit_manifest_hash TEXT, reference_validation_passed INTEGER, reference_validation_at TEXT, UNIQUE(suite_id,task_id,variant_seed));
CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT, profile TEXT, requested_json TEXT NOT NULL, resolved_json TEXT, resolution_reason TEXT, provider TEXT, identity_key TEXT, harness_id INTEGER, engine_id INTEGER, hardware_id INTEGER, billing_mode TEXT, evaluation_class TEXT NOT NULL DEFAULT 'unknown', raw_evidence_path TEXT, raw_evidence_sha256 TEXT, status TEXT);
CREATE TABLE IF NOT EXISTS task_attempts(id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id), task_id INTEGER REFERENCES benchmark_tasks(id), score REAL, public_score REAL, hidden_score REAL, invariant_score REAL, api_score REAL, scope_compliant INTEGER, wall_seconds REAL, baseline_score REAL, baseline_check_vector_json TEXT, final_check_vector_json TEXT, delta_score REAL, normalized_improvement REAL, evaluator_tampering INTEGER, prohibited_changed_files_json TEXT, metadata_json TEXT);
CREATE TABLE IF NOT EXISTS request_metrics(id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id), ordinal INTEGER, started_at TEXT, ended_at TEXT, model TEXT, provider TEXT, input_tokens INTEGER, output_tokens INTEGER, cache_read_tokens INTEGER, cache_write_tokens INTEGER, reasoning_tokens INTEGER, ttft_seconds REAL, wall_seconds REAL, stop_reason TEXT, metadata_json TEXT);
CREATE TABLE IF NOT EXISTS tool_events(id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id), request_id INTEGER REFERENCES request_metrics(id), ordinal INTEGER, tool_name TEXT, validity TEXT, error TEXT, recovered INTEGER, alternate_tool INTEGER, metadata_json TEXT);
CREATE TABLE IF NOT EXISTS pricing_snapshots(id INTEGER PRIMARY KEY, provider TEXT NOT NULL, effective_at TEXT NOT NULL, currency TEXT, prices_json TEXT NOT NULL, source TEXT, UNIQUE(provider,effective_at,prices_json));
CREATE TABLE IF NOT EXISTS cost_observations(id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id), billing_mode TEXT, provider_reported_cost REAL, calculated_cost REAL, api_equivalent_cost REAL, currency TEXT, cost_source TEXT, price_snapshot_id INTEGER REFERENCES pricing_snapshots(id), input_tokens INTEGER, output_tokens INTEGER, cached_input_tokens INTEGER, cache_write_tokens INTEGER, reasoning_tokens INTEGER);
CREATE TABLE IF NOT EXISTS promotion_events(id INTEGER PRIMARY KEY, identity_key TEXT, from_state TEXT, to_state TEXT, occurred_at TEXT, reason TEXT, promotion_basis TEXT);
CREATE TABLE IF NOT EXISTS retirement_events(id INTEGER PRIMARY KEY, identity_key TEXT, occurred_at TEXT, reason TEXT);
CREATE TABLE IF NOT EXISTS default_changes(id INTEGER PRIMARY KEY, profile TEXT, old_identity_key TEXT, new_identity_key TEXT, occurred_at TEXT, reason TEXT);
CREATE TABLE IF NOT EXISTS resolution_decisions(id INTEGER PRIMARY KEY, run_id TEXT REFERENCES runs(run_id), requested_json TEXT NOT NULL, resolved_json TEXT, state TEXT NOT NULL, reason TEXT, alternatives_json TEXT);
CREATE TABLE IF NOT EXISTS imported_evidence(source_path TEXT NOT NULL, source_sha256 TEXT NOT NULL, imported_at TEXT NOT NULL, record_count INTEGER NOT NULL, PRIMARY KEY(source_path,source_sha256));
"""


def default_state_dir() -> Path:
    root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")).expanduser()
    return root / "ekalavya"


def default_db_path() -> Path:
    return default_state_dir() / "ledger.sqlite3"


def _secure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Additive migrations for ledgers created before evaluation classes."""
    _ensure_column(conn, "harnesses", "capabilities_json", "TEXT")
    _ensure_column(conn, "harnesses", "telemetry_json", "TEXT")
    _ensure_column(conn, "harnesses", "eligibility_json", "TEXT")
    _ensure_column(conn, "harnesses", "evidence_label", "TEXT")
    _ensure_column(conn, "harnesses", "observed_at", "TEXT")
    _ensure_column(conn, "benchmark_suites", "evaluation_class", "TEXT NOT NULL DEFAULT 'unknown'")
    _ensure_column(conn, "runs", "evaluation_class", "TEXT NOT NULL DEFAULT 'unknown'")
    _ensure_column(conn, "promotion_events", "promotion_basis", "TEXT")
    for column, definition in (
        ("baseline_score", "REAL"),
        ("baseline_check_vector_json", "TEXT"),
        ("task_spec_hash", "TEXT"),
        ("allowed_edit_manifest_hash", "TEXT"),
        ("reference_validation_passed", "INTEGER"),
        ("reference_validation_at", "TEXT"),
    ):
        _ensure_column(conn, "benchmark_tasks", column, definition)
    for column, definition in (
        ("baseline_score", "REAL"),
        ("baseline_check_vector_json", "TEXT"),
        ("final_check_vector_json", "TEXT"),
        ("delta_score", "REAL"),
        ("normalized_improvement", "REAL"),
        ("evaluator_tampering", "INTEGER"),
        ("prohibited_changed_files_json", "TEXT"),
    ):
        _ensure_column(conn, "task_attempts", column, definition)
    conn.execute("UPDATE benchmark_suites SET evaluation_class='unknown' WHERE evaluation_class IS NULL OR evaluation_class='' ")
    conn.execute("UPDATE runs SET evaluation_class='unknown' WHERE evaluation_class IS NULL OR evaluation_class='' ")


def connect(path: Path | None = None) -> sqlite3.Connection:
    target = path or default_db_path()
    _secure_dir(target.parent)
    conn = sqlite3.connect(target)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA_SQL)
    _migrate_schema(conn)
    checksum = hashlib.sha256(SCHEMA_SQL.encode()).hexdigest()
    conn.execute("INSERT OR IGNORE INTO schema_versions VALUES(?,?,?)", (SCHEMA_VERSION, datetime.now(timezone.utc).isoformat(), checksum))
    conn.commit()
    if target.exists():
        os.chmod(target, 0o600)
    return conn


def record_run(conn: sqlite3.Connection, run_id: str, requested: dict[str, Any], *, resolved: dict[str, Any] | None = None, status: str = "resolved", evaluation_class: str = "unknown", **fields: Any) -> None:
    if evaluation_class not in EVALUATION_CLASSES:
        raise ValueError(f"invalid evaluation class: {evaluation_class!r}")
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("INSERT OR REPLACE INTO runs(run_id,started_at,ended_at,profile,requested_json,resolved_json,resolution_reason,provider,identity_key,harness_id,engine_id,hardware_id,billing_mode,evaluation_class,raw_evidence_path,raw_evidence_sha256,status) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (run_id, fields.pop("started_at", now), fields.pop("ended_at", None), fields.pop("profile", requested.get("profile")), json.dumps(requested, sort_keys=True), json.dumps(resolved, sort_keys=True) if resolved is not None else None, fields.pop("resolution_reason", None), fields.pop("provider", None), fields.pop("identity_key", None), fields.pop("harness_id", None), fields.pop("engine_id", None), fields.pop("hardware_id", None), fields.pop("billing_mode", None), evaluation_class, fields.pop("raw_evidence_path", None), fields.pop("raw_evidence_sha256", None), status))
    conn.commit()


def record_resolution(conn: sqlite3.Connection, run_id: str, requested: dict[str, Any], resolution: dict[str, Any]) -> None:
    conn.execute("INSERT INTO resolution_decisions(run_id,requested_json,resolved_json,state,reason,alternatives_json) VALUES(?,?,?,?,?,?)", (run_id, json.dumps(requested, sort_keys=True), json.dumps(resolution.get("resolved"), sort_keys=True), resolution.get("state", "resolved"), resolution.get("reason"), json.dumps(resolution.get("alternatives", []), sort_keys=True)))
    conn.commit()


def record_price_snapshot(conn: sqlite3.Connection, provider: str, effective_at: str, prices: dict[str, Any], *, currency: str | None = None, source: str = "provider") -> int:
    encoded = json.dumps(prices, sort_keys=True)
    existing = conn.execute("SELECT id,prices_json FROM pricing_snapshots WHERE provider=? AND effective_at=? ORDER BY id LIMIT 1", (provider, effective_at)).fetchone()
    if existing and existing[1] != encoded:
        raise ValueError("price snapshot is immutable for a provider/effective timestamp")
    row = conn.execute("INSERT OR IGNORE INTO pricing_snapshots(provider,effective_at,currency,prices_json,source) VALUES(?,?,?,?,?)", (provider, effective_at, currency, encoded, source)).lastrowid
    if not row:
        row = conn.execute("SELECT id FROM pricing_snapshots WHERE provider=? AND effective_at=? AND prices_json=?", (provider, effective_at, encoded)).fetchone()[0]
    conn.commit()
    return int(row)


def upsert_model(
    conn: sqlite3.Connection,
    identity: CandidateIdentity,
    *,
    lifecycle: str = "candidate",
    discovered_at: str | None = None,
    identity_key: str | None = None,
) -> int:
    """Insert a model identity without fabricating provider metadata."""
    if lifecycle not in {"candidate", "current", "previous", "retired", "rejected", "removed"}:
        raise ValueError(f"invalid lifecycle: {lifecycle}")
    family_id = None
    if identity.family:
        conn.execute("INSERT OR IGNORE INTO model_families(provider,family_key,display_name) VALUES(?,?,?)", (identity.provider, identity.family, identity.display_name))
        family_id = conn.execute("SELECT id FROM model_families WHERE provider=? AND family_key=?", (identity.provider, identity.family)).fetchone()[0]
    values = identity.as_dict()
    stored_identity_key = identity_key or identity.identity_key
    conn.execute("""INSERT INTO models(identity_key,provider,family_id,family,provider_model_id,display_name,generation,variant,capabilities_json,architecture,parameter_count,active_parameter_count,quantization,serving_engine,serving_engine_version,hardware_profile,lifecycle,discovered_at)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(identity_key) DO UPDATE SET lifecycle=excluded.lifecycle, discovered_at=COALESCE(excluded.discovered_at,models.discovered_at)""", (stored_identity_key, values["provider"], family_id, values["family"], values["provider_model_id"], values["display_name"], values["generation"], values["variant"], json.dumps(values["capabilities"], sort_keys=True), values["architecture"], values["parameter_count"], values["active_parameter_count"], values["quantization"], values["serving_engine"], values["serving_engine_version"], values["hardware_profile"], lifecycle, discovered_at))
    conn.commit()
    return int(conn.execute("SELECT id FROM models WHERE identity_key=?", (stored_identity_key,)).fetchone()[0])


def record_availability(conn: sqlite3.Connection, model_id: int, *, state: str, observed_at: str | None = None, source: str | None = None, details: dict[str, Any] | None = None) -> None:
    conn.execute("INSERT INTO model_availability(model_id,state,observed_at,source,details_json) VALUES(?,?,?,?,?)", (model_id, state, observed_at or datetime.now(timezone.utc).isoformat(), source, json.dumps(details or {}, sort_keys=True)))
    conn.commit()


def record_promotion_event(
    conn: sqlite3.Connection,
    identity_key: str,
    *,
    from_state: str | None,
    to_state: str,
    reason: str,
    promotion_basis: str,
    occurred_at: str | None = None,
) -> int:
    """Record an explicit lifecycle promotion with its evidentiary basis."""
    if promotion_basis not in {"quality_superiority", "operational_efficiency", "manual", "unspecified"}:
        raise ValueError(f"invalid promotion basis: {promotion_basis}")
    cursor = conn.execute(
        "INSERT INTO promotion_events(identity_key,from_state,to_state,occurred_at,reason,promotion_basis) VALUES(?,?,?,?,?,?)",
        (identity_key, from_state, to_state, occurred_at or datetime.now(timezone.utc).isoformat(), reason, promotion_basis),
    )
    conn.commit()
    return int(cursor.lastrowid)


def record_default_change(
    conn: sqlite3.Connection,
    profile: str,
    *,
    old_identity_key: str | None,
    new_identity_key: str,
    reason: str,
    occurred_at: str | None = None,
) -> int:
    """Record an explicit user-owned profile default change."""
    cursor = conn.execute(
        "INSERT INTO default_changes(profile,old_identity_key,new_identity_key,occurred_at,reason) VALUES(?,?,?,?,?)",
        (profile, old_identity_key, new_identity_key, occurred_at or datetime.now(timezone.utc).isoformat(), reason),
    )
    conn.commit()
    return int(cursor.lastrowid)


def record_harness(conn: sqlite3.Connection, name: str, *, version: str | None = None, adapter_version: str | None = None, transport: str | None = None, capabilities: dict[str, Any] | None = None, telemetry: dict[str, Any] | None = None, eligibility: dict[str, str] | None = None, evidence_label: str | None = None, observed_at: str | None = None) -> int:
    encoded_capabilities = json.dumps(capabilities, sort_keys=True) if capabilities is not None else None
    encoded_telemetry = json.dumps(telemetry, sort_keys=True) if telemetry is not None else None
    encoded_eligibility = json.dumps(eligibility, sort_keys=True) if eligibility is not None else None
    conn.execute("INSERT OR IGNORE INTO harnesses(name,version,adapter_version,transport,capabilities_json,telemetry_json,eligibility_json,evidence_label,observed_at) VALUES(?,?,?,?,?,?,?,?,?)", (name, version, adapter_version, transport, encoded_capabilities, encoded_telemetry, encoded_eligibility, evidence_label, observed_at))
    conn.execute("UPDATE harnesses SET capabilities_json=COALESCE(?,capabilities_json), telemetry_json=COALESCE(?,telemetry_json), eligibility_json=COALESCE(?,eligibility_json), evidence_label=COALESCE(?,evidence_label), observed_at=COALESCE(?,observed_at) WHERE name IS ? AND version IS ? AND adapter_version IS ?", (encoded_capabilities, encoded_telemetry, encoded_eligibility, evidence_label, observed_at, name, version, adapter_version))
    conn.commit()
    return int(conn.execute("SELECT id FROM harnesses WHERE name IS ? AND version IS ? AND adapter_version IS ?", (name, version, adapter_version)).fetchone()[0])


def record_benchmark_suite(conn: sqlite3.Connection, name: str, layer: str, version: str, *, git_sha: str | None = None, metadata: dict[str, Any] | None = None, evaluation_class: str = "unknown") -> int:
    if evaluation_class not in EVALUATION_CLASSES:
        raise ValueError(f"invalid evaluation class: {evaluation_class!r}")
    conn.execute("INSERT OR IGNORE INTO benchmark_suites(name,layer,version,evaluation_class,git_sha,metadata_json) VALUES(?,?,?,?,?,?)", (name, layer, version, evaluation_class, git_sha, json.dumps(metadata or {}, sort_keys=True)))
    conn.commit()
    return int(conn.execute("SELECT id FROM benchmark_suites WHERE name=? AND version=? AND git_sha IS ?", (name, version, git_sha)).fetchone()[0])


def record_benchmark_suite_correction(
    conn: sqlite3.Connection,
    suite_id: int,
    corrected_git_sha: str,
    *,
    reason: str,
    evidence: dict[str, Any] | None = None,
    corrected_at: str | None = None,
) -> int:
    """Correct derived suite identity while preserving the prior value."""
    row = conn.execute("SELECT git_sha FROM benchmark_suites WHERE id=?", (suite_id,)).fetchone()
    if row is None:
        raise ValueError(f"unknown benchmark suite: {suite_id}")
    old_sha = row[0]
    if old_sha == corrected_git_sha:
        return int(conn.execute("SELECT id FROM benchmark_suite_corrections WHERE suite_id=? ORDER BY id DESC LIMIT 1", (suite_id,)).fetchone()[0]) if conn.execute("SELECT 1 FROM benchmark_suite_corrections WHERE suite_id=?", (suite_id,)).fetchone() else 0
    when = corrected_at or datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO benchmark_suite_corrections(suite_id,originally_recorded_git_sha,corrected_git_sha,corrected_at,reason,evidence_json) VALUES(?,?,?,?,?,?)",
        (suite_id, old_sha, corrected_git_sha, when, reason, json.dumps(evidence or {}, sort_keys=True)),
    )
    conn.execute("UPDATE benchmark_suites SET git_sha=? WHERE id=?", (corrected_git_sha, suite_id))
    conn.commit()
    return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def record_benchmark_task(conn: sqlite3.Connection, suite_id: int, *, family: str, task_id: str, variant_seed: str, content_hash: str, prompt_hash: str, evaluator_hash: str, baseline_score: float | None = None, baseline_check_vector: list[bool] | None = None, task_spec_hash: str | None = None, allowed_edit_manifest_hash: str | None = None, reference_validation_passed: bool | None = None, reference_validation_at: str | None = None) -> int:
    conn.execute("INSERT OR IGNORE INTO benchmark_tasks(suite_id,family,task_id,variant_seed,content_hash,prompt_hash,evaluator_hash,baseline_score,baseline_check_vector_json,task_spec_hash,allowed_edit_manifest_hash,reference_validation_passed,reference_validation_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (suite_id, family, task_id, variant_seed, content_hash, prompt_hash, evaluator_hash, baseline_score, json.dumps(baseline_check_vector) if baseline_check_vector is not None else None, task_spec_hash, allowed_edit_manifest_hash, None if reference_validation_passed is None else int(reference_validation_passed), reference_validation_at))
    conn.commit()
    return int(conn.execute("SELECT id FROM benchmark_tasks WHERE suite_id=? AND task_id=? AND variant_seed=?", (suite_id, task_id, variant_seed)).fetchone()[0])


def record_task_attempt(conn: sqlite3.Connection, run_id: str, task_id: int, *, score: float | None = None, public_score: float | None = None, hidden_score: float | None = None, invariant_score: float | None = None, api_score: float | None = None, scope_compliant: bool | None = None, wall_seconds: float | None = None, baseline_score: float | None = None, baseline_check_vector: list[bool] | None = None, final_check_vector: list[bool] | None = None, delta_score: float | None = None, normalized_improvement: float | None = None, evaluator_tampering: bool | None = None, prohibited_changed_files: list[str] | None = None, metadata: dict[str, Any] | None = None) -> int:
    conn.execute("INSERT INTO task_attempts(run_id,task_id,score,public_score,hidden_score,invariant_score,api_score,scope_compliant,wall_seconds,baseline_score,baseline_check_vector_json,final_check_vector_json,delta_score,normalized_improvement,evaluator_tampering,prohibited_changed_files_json,metadata_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (run_id, task_id, score, public_score, hidden_score, invariant_score, api_score, None if scope_compliant is None else int(scope_compliant), wall_seconds, baseline_score, json.dumps(baseline_check_vector) if baseline_check_vector is not None else None, json.dumps(final_check_vector) if final_check_vector is not None else None, delta_score, normalized_improvement, None if evaluator_tampering is None else int(evaluator_tampering), json.dumps(prohibited_changed_files or [], sort_keys=True) if prohibited_changed_files is not None else None, json.dumps(metadata or {}, sort_keys=True)))
    conn.commit()
    return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def record_request_metric(conn: sqlite3.Connection, run_id: str, metric: dict[str, Any]) -> int:
    conn.execute("INSERT INTO request_metrics(run_id,ordinal,started_at,ended_at,model,provider,input_tokens,output_tokens,cache_read_tokens,cache_write_tokens,reasoning_tokens,ttft_seconds,wall_seconds,stop_reason,metadata_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (run_id, metric.get("ordinal"), metric.get("request_start"), metric.get("request_end"), metric.get("model"), metric.get("provider"), metric.get("input_tokens"), metric.get("output_tokens"), metric.get("cache_read_tokens"), metric.get("cache_write_tokens"), metric.get("reasoning_tokens"), metric.get("ttft_seconds"), metric.get("wall_seconds"), metric.get("stop_reason"), json.dumps(metric, sort_keys=True)))
    conn.commit()
    return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def record_tool_event(conn: sqlite3.Connection, run_id: str, event: dict[str, Any], *, request_id: int | None = None) -> int:
    conn.execute("INSERT INTO tool_events(run_id,request_id,ordinal,tool_name,validity,error,recovered,alternate_tool,metadata_json) VALUES(?,?,?,?,?,?,?,?,?)", (run_id, request_id, event.get("ordinal"), event.get("tool_name"), event.get("validity"), event.get("error"), event.get("recovered"), event.get("alternate_tool"), json.dumps(event, sort_keys=True)))
    conn.commit()
    return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def finalize_run(conn: sqlite3.Connection, run_id: str, *, ended_at: str | None = None, status: str = "completed", raw_evidence_path: str | None = None, raw_evidence_sha256: str | None = None) -> None:
    conn.execute("UPDATE runs SET ended_at=?,status=?,raw_evidence_path=COALESCE(?,raw_evidence_path),raw_evidence_sha256=COALESCE(?,raw_evidence_sha256) WHERE run_id=?", (ended_at or datetime.now(timezone.utc).isoformat(), status, raw_evidence_path, raw_evidence_sha256, run_id))
    conn.commit()


def record_cost(conn: sqlite3.Connection, run_id: str, *, billing_mode: str, provider_reported_cost: float | None = None, calculated_cost: float | None = None, api_equivalent_cost: float | None = None, currency: str | None = None, cost_source: str = "unavailable", price_snapshot_id: int | None = None, input_tokens: int | None = None, output_tokens: int | None = None, cached_input_tokens: int | None = None, cache_write_tokens: int | None = None, reasoning_tokens: int | None = None) -> None:
    if billing_mode not in {"metered_api", "subscription", "local", "unknown"}:
        raise ValueError("invalid billing mode")
    conn.execute("INSERT INTO cost_observations(run_id,billing_mode,provider_reported_cost,calculated_cost,api_equivalent_cost,currency,cost_source,price_snapshot_id,input_tokens,output_tokens,cached_input_tokens,cache_write_tokens,reasoning_tokens) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (run_id, billing_mode, provider_reported_cost, calculated_cost, api_equivalent_cost, currency, cost_source, price_snapshot_id, input_tokens, output_tokens, cached_input_tokens, cache_write_tokens, reasoning_tokens))
    conn.commit()


def import_legacy_state(conn: sqlite3.Connection, root: Path) -> dict[str, int]:
    """Import hashes and JSON records without moving, interpreting, or exposing them."""
    counts = {"files": 0, "records": 0, "skipped": 0}
    if not root.is_dir():
        return counts
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.suffix.lower() not in {".json", ".jsonl", ".md", ".csv", ".txt"}:
            continue
        try:
            data = path.read_bytes()
        except OSError:
            counts["skipped"] += 1
            continue
        digest = hashlib.sha256(data).hexdigest()
        key = str(path.resolve())
        if conn.execute("SELECT 1 FROM imported_evidence WHERE source_path=? AND source_sha256=?", (key, digest)).fetchone():
            counts["skipped"] += 1
            continue
        records = 1
        if path.suffix.lower() == ".jsonl":
            records = sum(1 for line in data.decode("utf-8", "replace").splitlines() if line.strip())
        conn.execute("INSERT INTO imported_evidence VALUES(?,?,?,?)", (key, digest, datetime.now(timezone.utc).isoformat(), records))
        if path.name == "execution.json":
            try:
                raw = json.loads(data.decode("utf-8", "replace"))
            except (ValueError, UnicodeDecodeError):
                raw = None
            if isinstance(raw, dict):
                safe = {field: raw.get(field) for field in ("delegate", "provider", "requested_model", "requested_effort", "started_at", "ended_at", "exit_code", "response_status", "response_recorded", "response_sha256")}
                legacy_run_id = "legacy:" + hashlib.sha256((key + "\0" + digest).encode()).hexdigest()
                conn.execute("INSERT OR IGNORE INTO runs(run_id,started_at,ended_at,profile,requested_json,resolved_json,provider,raw_evidence_path,raw_evidence_sha256,status) VALUES(?,?,?,?,?,?,?,?,?,?)", (legacy_run_id, raw.get("started_at") or datetime.now(timezone.utc).isoformat(), raw.get("ended_at"), raw.get("delegate"), json.dumps(safe, sort_keys=True), json.dumps({"provider_model_id": raw.get("requested_model"), "provider": raw.get("provider")}, sort_keys=True), raw.get("provider"), key, digest, "imported"))
        counts["files"] += 1; counts["records"] += records
    conn.commit()
    return counts


def import_execution_runs(conn: sqlite3.Connection, root: Path) -> int:
    """Backfill structured runs from already-indexed, whitelisted metadata."""
    imported = 0
    if not root.is_dir():
        return imported
    for path in sorted(root.rglob("execution.json")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            raw = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(raw, dict):
            continue
        data = path.read_bytes(); digest = hashlib.sha256(data).hexdigest(); key = str(path.resolve())
        run_id = "legacy:" + hashlib.sha256((key + "\0" + digest).encode()).hexdigest()
        safe = {field: raw.get(field) for field in ("delegate", "provider", "requested_model", "requested_effort", "started_at", "ended_at", "exit_code", "response_status", "response_recorded", "response_sha256")}
        cursor = conn.execute("INSERT OR IGNORE INTO runs(run_id,started_at,ended_at,profile,requested_json,resolved_json,provider,raw_evidence_path,raw_evidence_sha256,status) VALUES(?,?,?,?,?,?,?,?,?,?)", (run_id, raw.get("started_at") or datetime.now(timezone.utc).isoformat(), raw.get("ended_at"), raw.get("delegate"), json.dumps(safe, sort_keys=True), json.dumps({"provider_model_id": raw.get("requested_model"), "provider": raw.get("provider")}, sort_keys=True), raw.get("provider"), key, digest, "imported"))
        imported += int(cursor.rowcount == 1)
    conn.commit()
    return imported
