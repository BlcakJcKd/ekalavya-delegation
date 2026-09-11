"""Privacy-safe, non-inferencing usage enrichment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .ledger import record_run_observability, record_safe_request_metric

PROHIBITED_KEYS = {
    "prompt", "prompt_text", "response", "final_answer", "completion", "completion_text",
    "reasoning", "reasoning_content", "chain_of_thought", "tool_arguments", "tool_args",
    "command", "argv", "workspace", "path", "absolute_path", "repository_path",
    "username", "email", "identity", "credential", "credentials", "api_key", "apikey",
    "token", "access_token", "cookie", "cookies", "raw_payload", "payload", "metadata",
}

SAFE_ANALYTICS_KEYS = {
    "ordinal", "request_start", "request_end", "model", "provider", "input_tokens",
    "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens",
    "ttft_seconds", "wall_seconds", "stop_reason", "total_tokens",
    "input_tokens_provenance", "output_tokens_provenance", "reasoning_tokens_provenance",
    "cache_read_tokens_provenance", "cache_write_tokens_provenance", "total_tokens_provenance",
}


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def safe_usage_from_execution(raw: dict[str, Any]) -> dict[str, Any]:
    """Project only provider/harness usage fields from an execution record."""
    usage = raw.get("provider_reported_usage")
    if not isinstance(usage, dict):
        usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    input_tokens = _int(usage.get("input_tokens", usage.get("prompt_tokens")))
    output_tokens = _int(usage.get("output_tokens", usage.get("completion_tokens")))
    reasoning_tokens = _int(usage.get("reasoning_tokens"))
    cache_read = _int(usage.get("cache_read_tokens", usage.get("cached_input_tokens")))
    cache_write = _int(usage.get("cache_write_tokens"))
    total = _int(usage.get("total_tokens"))
    if total is None and input_tokens is not None and output_tokens is not None:
        total = input_tokens + output_tokens
    provenance = raw.get("usage_provenance") if raw.get("usage_provenance") in {"provider_reported", "harness_reported"} else "provider_reported"
    return {
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens, "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "uncached_input_tokens": input_tokens - cache_read if input_tokens is not None and cache_read is not None and cache_read <= input_tokens else None,
        "total_tokens": total,
        "input_tokens_provenance": provenance if input_tokens is not None else "unavailable",
        "output_tokens_provenance": provenance if output_tokens is not None else "unavailable",
        "reasoning_tokens_provenance": provenance if reasoning_tokens is not None else "unavailable",
        "cache_read_tokens_provenance": provenance if cache_read is not None else "unavailable",
        "cache_write_tokens_provenance": provenance if cache_write is not None else "unavailable",
        "uncached_input_tokens_provenance": "derived" if input_tokens is not None and cache_read is not None and cache_read <= input_tokens else "unavailable",
        "total_tokens_provenance": provenance if usage.get("total_tokens") is not None else ("derived" if total is not None else "unavailable"),
        "token_telemetry_status": "complete" if total is not None else ("partial" if any(x is not None for x in (input_tokens, output_tokens, reasoning_tokens, cache_read, cache_write)) else "unavailable"),
        "provider_reported_model_id": raw.get("provider_reported_model_id") if isinstance(raw.get("provider_reported_model_id"), str) and len(raw["provider_reported_model_id"]) <= 256 else None,
    }


def persist_execution_observability(conn: Any, run_id: str, *, run_data: dict[str, Any], execution: dict[str, Any] | None = None) -> None:
    """Persist enrichment after execution; callers must catch failures."""
    execution = execution or {}
    raw = {key: value for key, value in execution.items() if key not in PROHIBITED_KEYS}
    usage = safe_usage_from_execution(raw)
    status = str(execution.get("state", "unknown"))
    if status == "completed":
        execution_status = "success"
    elif status in {"failed", "aborted", "incomplete-infrastructure-run"}:
        execution_status = "aborted" if "incomplete" in status or execution.get("timed_out") else "failure"
    else:
        execution_status = status
    data = {
        **run_data, **usage,
        "execution_status": execution_status,
        "failure_category": execution.get("error_category") if isinstance(execution.get("error_category"), str) else None,
        "wall_seconds": _float(execution.get("wall_seconds")),
        "provider_request_count": execution.get("request_count") if isinstance(execution.get("request_count"), int) else (1 if run_data.get("harness_name") == "vllm" else None),
        "telemetry_status": "unavailable" if execution.get("telemetry_parse_warning") else ("complete" if execution else "unavailable"),
    }
    record_run_observability(conn, run_id, data)
    if execution:
        metric = {key: usage[key] for key in ("input_tokens", "output_tokens", "reasoning_tokens", "cache_read_tokens", "cache_write_tokens", "total_tokens", "input_tokens_provenance", "output_tokens_provenance", "reasoning_tokens_provenance", "cache_read_tokens_provenance", "cache_write_tokens_provenance", "total_tokens_provenance")}
        metric.update({"ordinal": 1, "model": usage.get("provider_reported_model_id"), "provider": run_data.get("primary_provider")})
        if any(metric.get(key) is not None for key in ("input_tokens", "output_tokens", "reasoning_tokens", "cache_read_tokens", "cache_write_tokens", "total_tokens")):
            record_safe_request_metric(conn, run_id, metric)


def assert_safe_analytics_payload(payload: dict[str, Any]) -> None:
    """Test helper/guard for callers handling external wrapper dictionaries."""
    if not isinstance(payload, dict):
        raise ValueError("analytics payload must be an object")
    if any(not isinstance(key, str) for key in payload):
        raise ValueError("analytics payload keys must be strings")
    leaked = {key for key in payload if key.lower() in PROHIBITED_KEYS}
    unknown = set(payload) - SAFE_ANALYTICS_KEYS
    nested = {key for key, value in payload.items() if isinstance(value, (dict, list, tuple))}
    if leaked or unknown or nested:
        raise ValueError(f"unsafe analytics payload fields: {sorted(leaked | unknown | nested)!r}")
