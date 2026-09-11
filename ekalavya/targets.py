"""Configured recommendation/run target construction shared by CLI surfaces."""

from __future__ import annotations

from typing import Any

from delegation.vllm import inspect_vllm_routes

from .schema import CandidateIdentity


def named_route_profile(name: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Expose a configured named vLLM route through the generic run contract."""
    if not name.startswith("vllm:") or not name[5:]:
        return None
    route = name[5:]
    info = inspect_vllm_routes().get(route)
    if info is None or info.provider is None:
        return None
    provider = info.provider
    identity = CandidateIdentity("vllm", "openai-compatible", provider.model, route, capabilities={"harness_values": ["vllm"]})
    entry = identity.as_dict()
    entry.update({"identity_key": identity.identity_key, "lifecycle": "current", "execution_route": f"vllm:{route}", "transport": "openai-compatible", "harness": "vllm", "harness_version": None})
    return ({"name": name, "description": f"Configured named vLLM route {route}", "default_identity_key": identity.identity_key, "permitted_candidates": [identity.identity_key], "reasoning_policy": "overrideable"}, entry)
