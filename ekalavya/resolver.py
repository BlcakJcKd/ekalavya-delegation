"""Explicit profile resolution; never auto-fails over or shops providers."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .catalogue import expand_runtime_variants
from .deepseek import DEEPSEEK_PRO_PROVIDER_MODEL_ID, DeepSeekExactIdentityError, assert_deepseek_pro_exact
from .readiness import binding_preflight
from .schema import CandidateIdentity, ReasoningPolicy, Resolution, RunIntent

SAME_PROVIDER_NATIVE = {
    "codex": "same-provider-native-required",
    "claude": "same-provider-native-required",
    "gemini": "same-provider-native-required",
}


def _availability_block(
    chosen: dict[str, Any], availability: dict[str, Any] | None,
) -> tuple[str, str] | None:
    """Return the user-owned availability reason, if this route is disabled."""
    if availability is None:
        return None
    execution_route = chosen.get("execution_route") or chosen.get("route") or chosen.get("legacy_route")
    if isinstance(execution_route, str) and execution_route.startswith("vllm:"):
        entry = availability.get("vllm", {}).get(execution_route[5:])
        if entry is not None and not entry.get("enabled", True):
            return entry.get("reason") or "disabled by user configuration", "model-disabled"
        return None
    provider = chosen.get("provider")
    provider_entry = availability.get("providers", {}).get(provider, {})
    if not provider_entry.get("enabled", True):
        return provider_entry.get("reason") or "disabled by user configuration", "provider-disabled"
    model = chosen.get("provider_model_id")
    # Legacy route names are retained on catalogue entries.  Runtime variants
    # inherit the same parent route, so this remains stable across generations.
    model_key = chosen.get("legacy_route") or chosen.get("route") or chosen.get("family")
    if model_key not in availability.get("models", {}):
        model_key = chosen.get("execution_route")
    model_entry = availability.get("models", {}).get(model_key, {})
    if not model_entry and model in availability.get("models", {}):
        model_entry = availability["models"][model]
    if not model_entry.get("enabled", True):
        return model_entry.get("reason") or "disabled by user configuration", "model-disabled"
    return None


def resolve(
    intent: RunIntent,
    profile: dict[str, Any],
    candidates: list[dict[str, Any]],
    availability: dict[str, Any] | None = None,
    *,
    now: datetime | None = None,
    provider_reported_model_id: str | None = None,
) -> Resolution:
    permitted = set(profile.get("permitted_candidates") or [])
    default_key = profile.get("default_identity_key")
    expanded = expand_runtime_variants(candidates)
    eligible = [c for c in expanded if c.get("lifecycle") in {"current", "previous", "candidate"} and (not permitted or c.get("identity_key") in permitted or c.get("catalogue_parent_identity_key") in permitted)]
    filters = {"provider": intent.provider, "family": intent.family, "provider_model_id": intent.model}
    filtered = [c for c in eligible if all(v is None or c.get(k) == v for k, v in filters.items())]
    alternatives = tuple({"identity_key": c.get("identity_key"), "provider": c.get("provider"), "family": c.get("family"), "provider_model_id": c.get("provider_model_id"), "lifecycle": c.get("lifecycle")} for c in eligible)
    explicit_candidate = bool(intent.model)
    if default_key and not any(c.get("identity_key") == default_key for c in filtered):
        if not filtered:
            return Resolution(intent, None, reason="configured profile default does not satisfy requested constraints", state="unavailable", alternatives=alternatives, reason_code="missing-route")
    requested_variant = intent.reasoning if intent.reasoning is not None else profile.get("default_reasoning")
    chosen = None
    if default_key:
        # A generation-level default resolves through its exact runtime
        # variant.  This preserves the parent lifecycle while ensuring the
        # provider receives gemini-3.x-flash-low/medium/high literally.
        chosen = next((c for c in filtered if c.get("catalogue_parent_identity_key") == default_key and (requested_variant is None or c.get("variant") == requested_variant)), None)
        chosen = chosen or next((c for c in filtered if c.get("identity_key") == default_key), None)
    elif len(filtered) == 1:
        chosen = filtered[0]
    if chosen is None and explicit_candidate and len(filtered) == 1:
        chosen = filtered[0]
    if chosen is None:
        return Resolution(intent, None, reason="profile has no unambiguous configured candidate; choose one explicitly", state="unavailable", alternatives=alternatives, reason_code="missing-route")
    if chosen.get("provider") == "deepseek" and chosen.get("provider_model_id") == DEEPSEEK_PRO_PROVIDER_MODEL_ID:
        try:
            assert_deepseek_pro_exact(now=now, provider_reported_model_id=provider_reported_model_id)
        except DeepSeekExactIdentityError as exc:
            return Resolution(intent, None, reason=str(exc), state="unavailable", alternatives=alternatives, reason_code="lifecycle-not-executable")
    availability_reason = _availability_block(chosen, availability)
    if availability_reason is not None:
        reason, reason_code = availability_reason
        return Resolution(intent, None, reason=reason, state="unavailable", alternatives=alternatives, reason_code=reason_code)
    primary = (intent.primary or "").strip().lower()
    aliases = {"codex-cli": "codex", "claude-code": "claude", "antigravity": "gemini", "agy": "gemini"}
    primary = aliases.get(primary, primary)
    if primary and primary == chosen.get("provider") and primary in SAME_PROVIDER_NATIVE:
        return Resolution(intent, None, reason=f"primary provider {primary} must use its native agent capability", state="same-provider-native-required", alternatives=alternatives, reason_code="same-provider-native-required")
    caps = chosen.get("capabilities") or {}
    policy = ReasoningPolicy(mode=profile.get("reasoning_policy", "overrideable"), default=intent.reasoning if intent.reasoning is not None else profile.get("default_reasoning"), supported=tuple(caps.get("reasoning_values") or ()))
    try:
        reasoning = policy.validate(intent.reasoning)
    except ValueError as exc:
        return Resolution(intent, None, reason=str(exc), state="invalid-reasoning", alternatives=alternatives, reason_code="unsupported-reasoning")
    supported_harnesses = tuple(caps.get("harness_values") or ())
    if not supported_harnesses:
        supported_harnesses = tuple(value for value in (chosen.get("harness"), chosen.get("serving_engine"), chosen.get("transport")) if value)
    if intent.harness and intent.harness not in supported_harnesses:
        return Resolution(intent, None, reason=f"unsupported harness {intent.harness!r}; supported: {list(supported_harnesses)!r}", state="invalid-harness", alternatives=alternatives, reason_code="unsupported-harness")
    route = chosen.get("execution_route") or chosen.get("route") or chosen.get("legacy_route")
    harness = intent.harness or profile.get("harness") or chosen.get("harness") or chosen.get("serving_engine") or chosen.get("transport")
    preflight = binding_preflight(
        chosen, route, harness,
        model=chosen.get("provider_model_id"),
        reasoning=reasoning,
    )
    if not preflight["ok"]:
        return Resolution(intent, None, reason=str(preflight["reason"]), state="harness-unavailable", alternatives=alternatives, reason_code=preflight.get("reason_code", "harness-unavailable"))
    candidate = CandidateIdentity(**{k: chosen.get(k) for k in CandidateIdentity.__dataclass_fields__})
    return Resolution(intent, candidate, reasoning, harness, chosen.get("harness_version") or chosen.get("serving_engine_version"), chosen.get("transport"), route, "configured profile default" if default_key else "explicit sole candidate", "resolved", alternatives)
