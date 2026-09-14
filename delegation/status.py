"""Zero-model-call computation of the effective delegation landscape.

Combines user config (``delegation.config``) with the fixed route/provider
tables (``delegation.routing``), local named vLLM schema inspection, and, for
fixed routes with an external wrapper, whether that wrapper's executable is
actually on PATH. Never invokes a delegate CLI, resolves credentials, or
queries quota -- quota availability is user-managed in this version (see
docs/DELEGATE_CONFIGURATION.md).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Any, Callable

from . import routing
from .vllm import VLLMRouteInfo


@dataclass(frozen=True)
class RouteStatus:
    route: str
    provider: str
    transport: str  # the CLI frontend that executes the call; may differ from `provider`
    billing: str  # "quota" | "payg" -- display only, never a routing input
    maturity: str  # "stable" | "experimental" -- display only, never a routing input
    configured_enabled: bool
    configured_reason: str | None
    provider_enabled: bool | None
    provider_reason: str | None
    effective_enabled: bool
    route_type: str  # "external" | "same-provider" | "native-only"
    effective: str  # also "invalid configuration" | "missing credential reference"
    effective_reason: str | None
    executable: str | None  # the wrapper executable name, or None if not applicable
    executable_available: bool | None  # None when the route has no external wrapper
    source: str = "fixed"
    model: str | None = None
    shared_compute: bool | None = None
    max_concurrency: int | None = None
    thinking_default: bool | None = None
    default_max_tokens: int | None = None
    max_tokens_cap: int | None = None
    credential_configured: bool | None = None

    @property
    def max_tokens(self) -> int | None:
        """Backward-compatible alias for the normal output default."""
        return self.default_max_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "provider": self.provider,
            "transport": self.transport,
            "billing": self.billing,
            "maturity": self.maturity,
            "configured_enabled": self.configured_enabled,
            "configured_reason": self.configured_reason,
            "provider_enabled": self.provider_enabled,
            "provider_reason": self.provider_reason,
            "effective_enabled": self.effective_enabled,
            "route_type": self.route_type,
            "effective": self.effective,
            "effective_reason": self.effective_reason,
            "executable": self.executable,
            "executable_available": self.executable_available,
            "source": self.source,
            "model": self.model,
            "shared_compute": self.shared_compute,
            "max_concurrency": self.max_concurrency,
            "thinking_default": self.thinking_default,
            # Keep max_tokens for consumers of the original status schema.
            "max_tokens": self.default_max_tokens,
            "default_max_tokens": self.default_max_tokens,
            "max_tokens_cap": self.max_tokens_cap,
            "credential_configured": self.credential_configured,
        }


def _configured(config: dict, route: str) -> tuple[bool, str | None, bool, str | None, bool, str | None]:
    provider = routing.ROUTE_PROVIDER[route]
    provider_entry = config["providers"].get(provider, {"enabled": True})
    model_entry = config["models"].get(route, {"enabled": True})
    model_enabled = bool(model_entry.get("enabled", True))
    model_reason = model_entry.get("reason")
    provider_enabled = bool(provider_entry.get("enabled", True))
    provider_reason = provider_entry.get("reason")
    if not provider_enabled:
        return model_enabled, model_reason, provider_enabled, provider_reason, False, provider_reason or "provider-disabled"
    if not model_enabled:
        return model_enabled, model_reason, provider_enabled, provider_reason, False, model_reason
    return model_enabled, model_reason, provider_enabled, provider_reason, True, None


def compute_status(
    config: dict,
    primary: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
    vllm_routes: dict[str, VLLMRouteInfo] | None = None,
) -> list[RouteStatus]:
    """Compute the effective status of every route for a declared primary.

    ``which`` is injectable so tests can simulate executable presence/absence
    without touching real PATH lookups or invoking anything.
    """
    normalized_primary = routing.normalize_primary(primary)
    results: list[RouteStatus] = []
    for route in sorted(routing.MODELS, key=lambda r: (routing.ROUTE_PROVIDER[r], r)):
        provider = routing.ROUTE_PROVIDER[route]
        enabled, reason, provider_enabled, provider_reason, effective_enabled, unavailable_reason = _configured(config, route)
        rtype = routing.route_type(route, normalized_primary)
        executable = routing.ROUTE_EXECUTABLE[route]
        executable_available = bool(which(executable)) if executable else None

        if not effective_enabled:
            effective, effective_reason = "disabled", unavailable_reason
        elif rtype == "native-only":
            effective, effective_reason = "native-only", "no external wrapper for this route"
        elif rtype == "same-provider":
            effective, effective_reason = (
                "native-only",
                f"primary is already {provider}; use native agent capability instead",
            )
        elif executable_available is False:
            effective, effective_reason = "executable missing", f"{executable} not found on PATH"
        else:
            effective, effective_reason = "available", None

        results.append(RouteStatus(
            route=route, provider=provider, transport=routing.ROUTE_TRANSPORT[route],
            billing=routing.ROUTE_BILLING[route], maturity=routing.ROUTE_MATURITY[route],
            configured_enabled=enabled, configured_reason=reason,
            provider_enabled=provider_enabled, provider_reason=provider_reason,
            effective_enabled=effective_enabled, route_type=rtype,
            effective=effective, effective_reason=effective_reason, executable=executable,
            executable_available=executable_available,
        ))
    for route in sorted(vllm_routes or {}):
        info = (vllm_routes or {})[route]
        entry = config.get("vllm", {}).get(route, {"enabled": True})
        configured_enabled = bool(entry.get("enabled", True))
        configured_reason = entry.get("reason")
        if info.provider is None:
            effective = "missing credential reference" if info.error_kind == "missing-credential-reference" else "invalid configuration"
            effective_reason = info.error
            model = shared_compute = max_concurrency = thinking_default = None
            default_max_tokens = max_tokens_cap = credential_configured = None
        else:
            provider = info.provider
            effective = "available" if configured_enabled else "disabled"
            effective_reason = configured_reason if not configured_enabled else None
            model = provider.model
            shared_compute = provider.shared_compute
            max_concurrency = provider.max_concurrency
            thinking_default = provider.thinking_default
            default_max_tokens = provider.default_max_tokens
            max_tokens_cap = provider.max_tokens_cap
            credential_configured = True
        results.append(RouteStatus(
            route=route, provider="vllm", transport="openai-compatible",
            billing="shared", maturity="configured", configured_enabled=configured_enabled,
            configured_reason=configured_reason, route_type="external", effective=effective,
            provider_enabled=True, provider_reason=None, effective_enabled=(effective == "available"),
            effective_reason=effective_reason, executable=None, executable_available=None,
            source="vllm", model=model, shared_compute=shared_compute,
            max_concurrency=max_concurrency, thinking_default=thinking_default,
            default_max_tokens=default_max_tokens,
            max_tokens_cap=max_tokens_cap,
            credential_configured=credential_configured,
        ))
    return results
