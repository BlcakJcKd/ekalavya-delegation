"""Status-report helpers for the Ekalavya control plane.

Never invokes a delegate CLI and never queries quota; quota availability is
user-managed (see docs/DELEGATE_CONFIGURATION.md).
"""

from __future__ import annotations

import argparse
import json as jsonlib
import shutil
from importlib import metadata
from pathlib import Path
from typing import Any

from . import routing
from .config import load_config
from .paths import config_path, log_root
from .status import compute_status
from .vllm import inspect_vllm_live_routes, inspect_vllm_routes, vllm_config_path

DIST_NAME = "ekalavya-delegation"


def _runtime_version() -> str:
    try:
        return metadata.version(DIST_NAME)
    except metadata.PackageNotFoundError:
        return "dev (not installed; running from a source checkout)"


def skill_status() -> dict[str, Any]:
    source = Path.home() / ".agents" / "skills" / "delegation" / "SKILL.md"
    claude_link = Path.home() / ".claude" / "skills" / "delegation"
    return {
        "source_installed": source.is_file(),
        "source_path": str(source),
        "claude_code_discovers": claude_link.exists(),
        "claude_code_path": str(claude_link),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Zero-model-call view of the effective delegation landscape")
    parser.add_argument(
        "--primary",
        help="declared primary provider identity, e.g. claude-code, codex, gemini, manual",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--live", action="store_true",
        help="GET safe vLLM load/metrics snapshots (never performs inference)",
    )
    return parser


def build_report(
    primary: str | None,
    which=shutil.which,
    config: dict | None = None,
    vllm_routes: dict | None = None,
    live: bool = False,
    live_status: dict | None = None,
    route_readiness: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    resolved_config = config if config is not None else load_config()
    resolved_vllm = dict(vllm_routes) if vllm_routes is not None else (
        inspect_vllm_routes() if config is None else {}
    )
    # Preserve a user-owned availability preference even if the local route
    # definition was removed or became unreadable; status reports that state
    # instead of silently dropping the route.
    for route in resolved_config.get("vllm", {}):
        if route not in resolved_vllm:
            from .vllm import VLLMRouteInfo
            resolved_vllm[route] = VLLMRouteInfo(
                route, None, "local vLLM route definition is missing", "invalid-configuration"
            )
    normalized_primary = routing.normalize_primary(primary)
    routes = compute_status(
        resolved_config, primary=primary, which=which, vllm_routes=resolved_vllm
    )
    route_payloads = [r.as_dict() for r in routes]
    for route in route_payloads:
        readiness = (route_readiness or {}).get(route["route"])
        if readiness is None:
            continue
        route.update(readiness)
        # A route can only be effectively executable when its current local
        # binding and readiness proof agree.  Persisted evidence is exposed as
        # diagnostic context, never as an override for a broken binding.
        if route["effective"] == "available" and readiness.get("harness_ready") != "ready":
            route["effective"] = "unavailable"
            route["effective_enabled"] = False
            route["effective_reason"] = readiness.get("readiness_reason")
    report = {
        "config_path": str(config_path()),
        "vllm_config_path": str(vllm_config_path()),
        "state_log_path": str(log_root()),
        "runtime_version": _runtime_version(),
        "skill": skill_status(),
        "declared_primary": normalized_primary or "not-declared",
        "quota": "user-managed / unknown",
        "routes": route_payloads,
    }
    if live:
        observed = live_status if live_status is not None else inspect_vllm_live_routes(resolved_vllm)
        report["live_vllm"] = {name: value.as_dict() for name, value in observed.items()}
    return report


def _print_human(report: dict[str, Any]) -> None:
    print(f"Config:  {report['config_path']}")
    print(f"vLLM:    {report['vllm_config_path']}")
    print(f"State:   {report['state_log_path']}")
    print(f"Runtime: {report['runtime_version']}")
    skill = report["skill"]
    skill_line = "installed" if skill["source_installed"] else "not installed"
    discover_line = (
        "discovered by Claude Code" if skill["claude_code_discovers"] else "not linked for Claude Code"
    )
    print(f"Skill:   {skill_line} ({skill['source_path']}); {discover_line}")
    print(f"Primary: {report['declared_primary']}")
    print(f"Quota:   {report['quota']}")
    print()
    header = (
        f"{'Route':<15} {'Provider':<9} {'Transport':<10} {'Billing':<8} {'Maturity':<13} "
        f"{'Model cfg':<10} {'Provider cfg':<12} {'Route type':<14} {'Effective':<25} {'Model':<24} "
        f"{'Shared':<8} {'Conc.':<6} {'Think':<6} {'Default':<8} {'Cap':<6} Reason"
    )
    print(header)
    print("-" * len(header))
    for route in report["routes"]:
        config_state = "enabled" if route["configured_enabled"] else "disabled"
        provider_state = "enabled" if route.get("provider_enabled", True) else "disabled"
        reason = route["effective_reason"] or route["configured_reason"] or ""
        print(
            f"{route['route']:<15} {route['provider']:<9} {route['transport']:<10} "
            f"{route['billing']:<8} {route['maturity']:<13} {config_state:<10} "
            f"{provider_state:<12} {route['route_type']:<14} {route['effective']:<25} "
            f"{(route.get('model') or ''):<24} "
            f"{('yes' if route.get('shared_compute') else 'no' if route.get('shared_compute') is not None else ''):<8} "
            f"{(route.get('max_concurrency') or '')!s:<6} "
            f"{('on' if route.get('thinking_default') else 'off' if route.get('thinking_default') is not None else ''):<6} "
            f"{(route.get('default_max_tokens') or '')!s:<8} "
            f"{(route.get('max_tokens_cap') or '')!s:<6} {reason}"
        )
    if "live_vllm" in report:
        print()
        print("Live vLLM observability (GET /load and /metrics only):")
        for name, live in sorted(report["live_vllm"].items()):
            values = [f"live: {live['state']}"]
            for key, label in (
                ("server_load", "load"), ("running", "running"), ("waiting", "waiting"),
                ("kv_cache_usage_perc", "KV cache"), ("recent_requests", "recent requests"),
                ("recent_prompt_tokens", "recent prompt tokens"),
                ("recent_generation_tokens", "recent generation tokens"),
                ("recent_preemptions", "recent preemptions"),
            ):
                if live.get(key) is not None:
                    suffix = "%" if key == "kv_cache_usage_perc" else ""
                    values.append(f"{label}: {live[key]}{suffix}")
            if live.get("prefix_caching") is not None:
                values.append(f"prefix caching: {'enabled' if live['prefix_caching'] else 'disabled'}")
            if live.get("engine_sleep_state"):
                values.append(f"sleep: {live['engine_sleep_state']}")
            if live.get("reason"):
                values.append(f"reason: {live['reason']}")
            print(f"  {name}: " + ", ".join(values))
