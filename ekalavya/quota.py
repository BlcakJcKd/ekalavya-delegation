"""Quota/capacity adapter boundary. Hosted sources are intentionally honest unknowns."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from delegation.vllm import inspect_vllm_live_routes, inspect_vllm_routes

CAPABILITIES = ("exact", "partial", "interactive_only", "unavailable")
PUBLIC_FIELDS = ("provider", "scope_kind", "scope_key", "resource_kind", "window_kind", "window_label", "used_value", "remaining_value", "limit_value", "percentage", "units", "reset_at", "observed_at", "source", "capability", "accuracy", "error_category")


def public_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {key: snapshot.get(key) for key in PUBLIC_FIELDS}


class QuotaAdapter(Protocol):
    provider: str
    capability: str

    def snapshot(self) -> list[dict[str, Any]]: ...


def _hosted(provider: str, capability: str, reason: str) -> dict[str, Any]:
    return {"provider": provider, "scope_kind": "account", "scope_key": None, "resource_kind": "provider_quota", "window_kind": None, "window_label": None, "used_value": None, "remaining_value": None, "limit_value": None, "percentage": None, "units": None, "reset_at": None, "observed_at": datetime.now(timezone.utc).isoformat(), "source": "not_collected", "capability": capability, "accuracy": "unknown", "error_category": reason}


def hosted_quota_statuses() -> list[dict[str, Any]]:
    return [_hosted("codex", "interactive_only", "provider usage settings are interactive-only in v0.4"), _hosted("claude", "interactive_only", "provider usage command is interactive-only in v0.4"), _hosted("gemini", "interactive_only", "provider /stats is interactive-only in v0.4"), _hosted("deepseek", "partial", "documented limits exist but no safe local account snapshot source"), _hosted("minimax", "partial", "documented limits exist but no safe local account snapshot source")]


def local_vllm_snapshots(*, live: bool = False) -> list[dict[str, Any]]:
    routes = inspect_vllm_routes()
    if not routes:
        return []
    if not live:
        return [{"provider": "vllm", "scope_kind": "local_capacity", "scope_key": route, "resource_kind": "local_capacity", "window_kind": "instant", "window_label": "configured route", "used_value": None, "remaining_value": None, "limit_value": None, "percentage": None, "units": None, "reset_at": None, "observed_at": datetime.now(timezone.utc).isoformat(), "source": "local_configuration", "capability": "partial" if info.provider else "unavailable", "accuracy": "configuration", "error_category": info.error} for route, info in routes.items()]
    statuses = inspect_vllm_live_routes(routes)
    result = []
    for route, status in statuses.items():
        result.append({"provider": "vllm", "scope_kind": "local_capacity", "scope_key": route, "resource_kind": "kv_cache", "window_kind": "instant", "window_label": "scheduler", "used_value": status.kv_cache_usage_perc, "remaining_value": 100.0 - status.kv_cache_usage_perc if status.kv_cache_usage_perc is not None else None, "limit_value": 100.0 if status.kv_cache_usage_perc is not None else None, "percentage": status.kv_cache_usage_perc, "units": "percent", "reset_at": None, "observed_at": datetime.now(timezone.utc).isoformat(), "source": "vllm_get_metrics", "capability": "partial" if status.state != "UNKNOWN" else "unavailable", "accuracy": "reported", "error_category": status.reason})
    return result


def collect_snapshots(*, live_vllm: bool = False) -> list[dict[str, Any]]:
    return hosted_quota_statuses() + local_vllm_snapshots(live=live_vllm)
