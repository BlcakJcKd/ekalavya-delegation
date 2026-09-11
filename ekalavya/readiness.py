"""Shared local execution-binding and status-readiness checks.

These checks deliberately use only local configuration, the audited harness
registry, PATH, and already-persisted discovery facts.  They never invoke a
provider CLI or alter user-owned control files.
"""

from __future__ import annotations

import shutil
from typing import Any, Callable

from delegation import routing

from .harness_registry import audited_registry


def _registry_capability(harness: str) -> tuple[bool, str | None]:
    """Return ordinary-execution support for audited harnesses.

    Routes whose wrapper is not in the benchmark registry retain their
    established wrapper contract; the registry is an extra capability gate for
    harnesses it explicitly audits.
    """
    record = next((item for item in audited_registry() if item.name == harness), None)
    if record is None:
        return True, None
    if record.eligibility.get("ordinary") != "supported":
        return False, f"harness {harness} is not supported for ordinary execution"
    if record.capabilities.get("provider_transport_available") != "supported":
        return False, f"harness {harness} provider transport is not ready"
    return True, None


def binding_preflight(
    candidate: dict[str, Any],
    route: str | None,
    harness: str | None,
    *,
    which: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    """Validate the deterministic local execution contract before a launch."""
    which = which or shutil.which
    if not route:
        return {
            "ok": False,
            "reason": "resolved candidate has no configured execution adapter",
            "harness_detected": False,
            "harness_capability": "unavailable",
        }
    if route.startswith("vllm:"):
        if harness not in {None, "vllm"}:
            return {
                "ok": False,
                "reason": f"configured vLLM route requires harness 'vllm', got {harness!r}",
                "harness_detected": False,
                "harness_capability": "unavailable",
            }
        return {"ok": True, "reason": None, "harness_detected": True, "harness_capability": "supported", "harness": "vllm"}

    from delegation.core import DELEGATES

    spec = DELEGATES.get(route)
    if spec is None:
        return {
            "ok": False,
            "reason": f"configured execution adapter {route!r} is unavailable",
            "harness_detected": False,
            "harness_capability": "unavailable",
        }
    if candidate.get("provider") != routing.ROUTE_PROVIDER.get(route):
        return {
            "ok": False,
            "reason": f"execution adapter {route!r} does not match provider {candidate.get('provider')!r}",
            "harness_detected": False,
            "harness_capability": "unavailable",
        }
    if harness != spec.executable:
        return {
            "ok": False,
            "reason": f"execution adapter {route!r} requires harness {spec.executable!r}, got {harness!r}",
            "harness_detected": False,
            "harness_capability": "unavailable",
        }
    capable, capability_reason = _registry_capability(harness)
    detected = bool(which(harness))
    if not capable:
        return {"ok": False, "reason": capability_reason, "harness_detected": detected, "harness_capability": "unsupported", "harness": harness}
    if not detected:
        return {"ok": False, "reason": f"{harness} not found on PATH", "harness_detected": False, "harness_capability": "supported", "harness": harness}
    return {"ok": True, "reason": None, "harness_detected": True, "harness_capability": "supported", "harness": harness}


def profile_readiness(
    route_name: str,
    profiles: list[dict[str, Any]],
    catalogue: list[dict[str, Any]],
    availability: dict[str, dict[str, Any]] | None = None,
    *,
    which: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    """Inspect a profile's default binding without resolving or probing a provider."""
    which = which or shutil.which
    profile = next((item for item in profiles if item.get("name") == route_name), None)
    if profile is None or not profile.get("default_identity_key"):
        return {
            "profile_configured": False,
            "harness_ready": "unavailable",
            "readiness_reason": "profile is not configured",
            "harness_detected": False,
            "harness_capability": "unavailable",
            "availability_observed_at": None,
        }
    candidate = next((item for item in catalogue if item.get("identity_key") == profile.get("default_identity_key")), None)
    if candidate is None:
        return {
            "profile_configured": False,
            "harness_ready": "unavailable",
            "readiness_reason": "configured profile default is missing from the catalogue",
            "harness_detected": False,
            "harness_capability": "unavailable",
            "availability_observed_at": None,
        }
    selected_harness = profile.get("harness") or candidate.get("harness") or candidate.get("serving_engine") or candidate.get("transport")
    selected_route = candidate.get("execution_route") or candidate.get("route") or candidate.get("legacy_route")
    checked = binding_preflight(candidate, selected_route, selected_harness, which=which)
    observed = (availability or {}).get(str(profile["default_identity_key"]))
    if not checked["ok"]:
        return {
            "profile_configured": True,
            "execution_route": selected_route,
            "resolved_harness": selected_harness,
            "harness_ready": "unavailable",
            "readiness_reason": f"harness-unavailable: {checked['reason']}",
            "harness_detected": checked["harness_detected"],
            "harness_capability": checked["harness_capability"],
            "availability_observed_at": observed.get("observed_at") if observed else None,
            "availability_source": observed.get("source") if observed else None,
        }
    if observed is None or observed.get("state") != "available":
        return {
            "profile_configured": True,
            "execution_route": selected_route,
            "resolved_harness": selected_harness,
            "harness_ready": "unconfirmed",
            "readiness_reason": "harness readiness is not locally confirmed for the configured profile",
            "harness_detected": checked["harness_detected"],
            "harness_capability": checked["harness_capability"],
            "availability_observed_at": observed.get("observed_at") if observed else None,
            "availability_source": observed.get("source") if observed else None,
        }
    return {
        "profile_configured": True,
        "execution_route": selected_route,
        "resolved_harness": selected_harness,
        "harness_ready": "ready",
        "readiness_reason": None,
        "harness_detected": checked["harness_detected"],
        "harness_capability": checked["harness_capability"],
        "availability_observed_at": observed.get("observed_at"),
        "availability_source": observed.get("source"),
    }
