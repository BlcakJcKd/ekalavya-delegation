"""Deterministic local usage summaries and controls."""

from __future__ import annotations

import csv
import io
import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any

from . import SCHEMA_VERSION
from .ledger import clear_observability, delete_feedback, record_quota_snapshot, set_feedback
from .quota import collect_snapshots


def period_bounds(value: str, *, now: datetime | None = None) -> tuple[datetime | None, datetime, str]:
    now = now or datetime.now(timezone.utc)
    if value == "all":
        return None, now, "all"
    if value.endswith("d") and value[:-1].isdigit() and int(value[:-1]) > 0:
        days = int(value[:-1]); return now - timedelta(days=days), now, value
    raise ValueError("period must be all or a positive number of days such as 7d")


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values); pos = (len(ordered) - 1) * q; lo = math.floor(pos); hi = math.ceil(pos)
    if lo == hi: return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _metric(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return {"value": sum(values) if values else None, "available": len(values), "total": len(rows), "coverage": len(values) / len(rows) if rows else 0.0}


def _load(conn: Any, start: datetime | None, end: datetime, filters: dict[str, str | None]) -> list[dict[str, Any]]:
    clauses = ["o.event_kind='delegation'", "r.evaluation_class='unknown'"]
    params: list[Any] = []
    if start: clauses.append("o.created_at >= ?"); params.append(start.isoformat())
    clauses.append("o.created_at < ?"); params.append(end.isoformat())
    for column, key in (("o.task", "task"), ("o.requested_profile", "profile"), ("r.provider", "provider"), ("COALESCE(o.provider_reported_model_id,o.requested_provider_model_id)", "model")):
        if filters.get(key): clauses.append(f"{column}=?"); params.append(filters[key])
    rows = conn.execute("SELECT o.*,r.provider AS provider,r.started_at,r.evaluation_class,f.outcome AS feedback_outcome FROM run_observability o JOIN runs r ON r.run_id=o.run_id LEFT JOIN user_feedback f ON f.run_id=o.run_id WHERE " + " AND ".join(clauses) + " ORDER BY o.created_at,o.run_id", params).fetchall()
    return [dict(row) for row in rows]


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    success = sum(r.get("execution_status") == "success" for r in rows); failed = sum(r.get("execution_status") == "failure" for r in rows); aborted = sum(r.get("execution_status") == "aborted" for r in rows)
    walls = [float(r["wall_seconds"]) for r in rows if isinstance(r.get("wall_seconds"), (int, float))]
    feedback = [r.get("feedback_outcome") for r in rows if r.get("feedback_outcome") in {"useful", "mixed", "not-useful"}]
    feedback_counts = {key: feedback.count(key) for key in ("useful", "mixed", "not-useful")}
    return {"runs": len(rows), "successful_runs": success, "failed_runs": failed, "aborted_runs": aborted, "success_rate": success / len(rows) if rows else None, "reported_tokens": _metric(rows, "total_tokens"), "input_tokens": _metric(rows, "input_tokens"), "output_tokens": _metric(rows, "output_tokens"), "reasoning_tokens": _metric(rows, "reasoning_tokens"), "cache_read_tokens": _metric(rows, "cache_read_tokens"), "latency_seconds": {"median": _quantile(walls, .5) if len(walls) >= 3 else None, "p90": _quantile(walls, .9) if len(walls) >= 10 else None, "available": len(walls), "p90_minimum_sample": 10}, "feedback": {"available": len(feedback), "unrated": len(rows) - len(feedback), "counts": feedback_counts, "useful_rate": feedback_counts["useful"] / len(feedback) if len(feedback) >= 3 else None}}


def _groups(rows: list[dict[str, Any]], by: str | None) -> list[dict[str, Any]]:
    if not by: return []
    keymap = {"task": "task", "profile": "requested_profile", "provider": "provider", "model": "provider_reported_model_id"}
    field = keymap[by]; grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if by == "model":
            key = row.get(field) or (f"requested:{row.get('requested_provider_model_id')}" if row.get("requested_provider_model_id") else "effective-model-unavailable")
        else:
            key = row.get(field) or "unspecified"
        grouped.setdefault(key, []).append(row)
    return [{"key": key, "summary": _summary(group)} for key, group in sorted(grouped.items())]


def build_usage(conn: Any, *, period: str = "7d", by: str | None = None, filters: dict[str, str | None] | None = None, quotas: list[dict[str, Any]] | None = None, _now: datetime | None = None) -> dict[str, Any]:
    start, end, label = period_bounds(period, now=_now); rows = _load(conn, start, end, filters or {})
    return {"schema_version": SCHEMA_VERSION, "scope": {"local_history": "only Ekalavya-observed delegations on this machine", "account_quota": "provider-reported only when a safe adapter exists", "native_same_provider_activity": "outside local history", "external_machine_activity": "outside local history"}, "period": {"label": label, "start_utc": start.isoformat() if start else None, "end_utc": end.isoformat()}, "coverage": {"observability_events": {"available": len(rows), "eligible": len(rows)}, "token_telemetry": {"available": sum(r.get("total_tokens") is not None for r in rows), "eligible": len(rows)}, "reasoning_tokens": {"available": sum(r.get("reasoning_tokens") is not None for r in rows), "eligible": len(rows)}}, "summary": _summary(rows), "groups": _groups(rows, by), "quota": quotas or [], "warnings": ["Totals describe only unambiguously observed Ekalavya delegations; they are not provider-account totals.", "Cross-provider token counts are reported units, not directly comparable efficiency measures."]}


def build_insights(conn: Any, **kwargs: Any) -> dict[str, Any]:
    now = datetime.now(timezone.utc); usage = build_usage(conn, _now=now, **kwargs); summary = usage["summary"]; insights = []
    if summary["runs"]: insights.append({"kind": "descriptive", "message": "Most-used delegate/task/profile are available via eka usage --by; no quality claim is inferred."})
    if usage["coverage"]["token_telemetry"]["available"] < usage["coverage"]["token_telemetry"]["eligible"]: insights.append({"kind": "coverage_gap", "message": "Token telemetry is partial; totals cover only rows with reported tokens."})
    if summary["feedback"]["unrated"] and summary["runs"]: insights.append({"kind": "feedback_gap", "message": f"{summary['feedback']['unrated']} observed run(s) have no user outcome rating."})
    if any(isinstance(row.get("percentage"), (int, float)) and row["percentage"] >= 80 for row in usage.get("quota", [])):
        insights.append({"kind": "quota_pressure", "message": "A safely observed capacity/quota snapshot is at or above 80%; scope and source are shown in the quota section."})
    if kwargs.get("period", "7d") != "all":
        start, end, label = period_bounds(kwargs.get("period", "7d"), now=now)
        assert start is not None
        duration = end - start
        previous_rows = _load(conn, start - duration, start, kwargs.get("filters") or {})
        previous = _summary(previous_rows)
        usage["trend"] = {"comparison": "previous_equal_utc_period", "current": {"start_utc": start.isoformat(), "end_utc": end.isoformat()}, "previous": {"start_utc": (start - duration).isoformat(), "end_utc": start.isoformat()}, "supported": summary["runs"] >= 5 and len(previous_rows) >= 5, "current_runs": summary["runs"], "previous_runs": previous["runs"], "current_reported_tokens": summary["reported_tokens"]["value"] if summary["runs"] >= 5 else None, "previous_reported_tokens": previous["reported_tokens"]["value"] if len(previous_rows) >= 5 else None}
        if not usage["trend"]["supported"]: insights.append({"kind": "low_sample", "message": "Previous-period comparisons require at least five observed runs in both periods."})
    if not usage["summary"]["runs"]: insights.append({"kind": "no_data", "message": "No unambiguously observed delegations in this period."})
    usage["insights"] = insights; return usage


def export_usage(conn: Any, *, fmt: str = "json", **kwargs: Any) -> str:
    payload = build_usage(conn, **kwargs)
    if fmt == "json": return json.dumps(payload, indent=2, sort_keys=True)
    groups = payload["groups"] or build_usage(conn, **{**kwargs, "by": "task"})["groups"]
    output = io.StringIO(); writer = csv.DictWriter(output, fieldnames=["key", "runs", "reported_tokens"]); writer.writeheader()
    for group in groups: writer.writerow({"key": group["key"], "runs": group["summary"]["runs"], "reported_tokens": group["summary"]["reported_tokens"]["value"]})
    return output.getvalue()


def refresh_quotas(conn: Any, *, live_vllm: bool = False) -> list[dict[str, Any]]:
    snapshots = collect_snapshots(live_vllm=live_vllm)
    for snapshot in snapshots: record_quota_snapshot(conn, snapshot)
    return snapshots


__all__ = ["build_usage", "build_insights", "export_usage", "period_bounds", "refresh_quotas", "set_feedback", "delete_feedback", "clear_observability"]
