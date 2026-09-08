"""Small live model catalogue with explicit lifecycle transitions."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .schema import CandidateIdentity

LIVE_STATES = {"candidate", "current", "previous"}
ALL_STATES = LIVE_STATES | {"retired", "rejected", "removed"}
PROMOTION_BASES = {"quality_superiority", "operational_efficiency", "manual", "unspecified"}
_GEMINI_FLASH_ID = re.compile(r"^gemini-(?P<generation>\d+\.\d+)-flash-(?P<reasoning>low|medium|high)$")
_BOOTSTRAP_LIFECYCLE = {"3.6": "previous", "3.7": "current"}


def _private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)


def load_catalogue(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("catalogue must be a JSON list")
    return data


def save_catalogue(path: Path, entries: list[dict[str, Any]]) -> None:
    _private_dir(path.parent)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")
    os.chmod(tmp, 0o600)
    tmp.replace(path)
    os.chmod(path, 0o600)


def add_candidate(entries: list[dict[str, Any]], identity: CandidateIdentity, *, state: str = "candidate") -> list[dict[str, Any]]:
    if state not in LIVE_STATES:
        raise ValueError(f"invalid live catalogue state: {state}")
    key = identity.identity_key
    if any(e.get("identity_key") == key for e in entries):
        return entries
    result = list(entries)
    item = identity.as_dict()
    item.update({"identity_key": key, "lifecycle": state, "discovered_at": None})
    result.append(item)
    return result


def transition(entries: list[dict[str, Any]], identity_key: str, target: str) -> list[dict[str, Any]]:
    if target not in ALL_STATES:
        raise ValueError(f"invalid catalogue lifecycle: {target}")
    result = [dict(e) for e in entries]
    match = next((e for e in result if e.get("identity_key") == identity_key), None)
    if match is None:
        raise KeyError(identity_key)
    if target == "current":
        family = (match.get("provider"), match.get("family"))
        for entry in result:
            if (entry.get("provider"), entry.get("family")) == family and entry.get("lifecycle") == "current":
                entry["lifecycle"] = "previous"
    match["lifecycle"] = target
    return result


def selectable(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in entries if e.get("lifecycle") in {"current", "previous", "candidate"}]


def expand_runtime_variants(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose runtime variants while retaining lifecycle on the parent identity.

    A generation/family catalogue entry may carry exact provider runtime IDs
    (for example, reasoning variants).  Resolution sees those IDs as virtual
    candidates, but their lifecycle remains inherited from the parent entry.
    """
    expanded: list[dict[str, Any]] = []
    identity_fields = set(CandidateIdentity.__dataclass_fields__)
    for entry in entries:
        expanded.append(dict(entry))
        variants = entry.get("runtime_variants") or []
        for variant in variants:
            if not isinstance(variant, dict) or not variant.get("provider_model_id"):
                continue
            child = dict(entry)
            child.update({k: v for k, v in variant.items() if k not in {"lifecycle", "catalogue_key"}})
            if variant.get("reasoning"):
                child["variant"] = variant["reasoning"]
            child["lifecycle"] = entry.get("lifecycle", "candidate")
            child["catalogue_parent_identity_key"] = entry.get("identity_key")
            identity = CandidateIdentity(**{k: child.get(k) for k in identity_fields})
            child["identity_key"] = identity.identity_key
            expanded.append(child)
    return expanded


def _gemini_flash_generation(value: dict[str, Any] | str) -> str | None:
    if isinstance(value, dict):
        if value.get("provider") != "gemini" or value.get("family") != "flash":
            return None
        generation = value.get("generation")
        if generation:
            return str(generation)
        value = str(value.get("provider_model_id", ""))
    match = _GEMINI_FLASH_ID.fullmatch(value)
    return match.group("generation") if match else None


def _stable_gemini_flash_key(identity: CandidateIdentity) -> str:
    """Hash lifecycle identity fields while excluding harness version provenance."""
    version_neutral = CandidateIdentity(
        **{name: (None if name == "serving_engine_version" else getattr(identity, name)) for name in CandidateIdentity.__dataclass_fields__}
    )
    return version_neutral.identity_key


def _gemini_flash_record(
    generation: str,
    variants: list[dict[str, str]],
    *,
    observed_at: str,
    serving_engine_version: str | None,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    variants = sorted(variants, key=lambda item: item["provider_model_id"])
    medium = next((item for item in variants if item["provider_model_id"].endswith("-medium")), variants[0])
    identity = CandidateIdentity(
        provider="gemini", family="flash", provider_model_id=medium["provider_model_id"],
        display_name=f"Gemini {generation} Flash", generation=generation, variant="medium",
        capabilities={"reasoning_values": [item["provider_model_id"].rsplit("-", 1)[-1] for item in variants]},
        serving_engine="agy", serving_engine_version=serving_engine_version,
    )
    incoming = identity.as_dict()
    incoming.update({
        "identity_key": _stable_gemini_flash_key(identity),
        "catalogue_key": f"gemini:flash:{generation}",
        "lifecycle_scope": "generation_family",
        "lifecycle": _BOOTSTRAP_LIFECYCLE.get(generation, "candidate"),
        "default_runtime_variant": "medium",
        "runtime_variants": [
            {"provider_model_id": item["provider_model_id"], "display_name": item.get("display_name"), "reasoning": item["provider_model_id"].rsplit("-", 1)[-1]}
            for item in variants
        ],
        "discovery_source": "agy models",
        "discovery_timestamp": observed_at,
        "availability_observed_at": observed_at,
        "discovery_client_version": serving_engine_version,
        "transport": "agy",
    })
    if existing is None:
        return incoming

    merged = dict(existing)
    # serving_engine_version is historical identity provenance.  The latest
    # observation is recorded separately in discovery_client_version.
    for name, value in incoming.items():
        if name not in {"identity_key", "lifecycle", "serving_engine_version"}:
            merged[name] = value
    return merged


def canonicalize_gemini_flash_generations(
    entries: list[dict[str, Any]],
    discovered: list[dict[str, str]],
    *,
    observed_at: str,
    serving_engine_version: str | None = None,
) -> list[dict[str, Any]]:
    """Store Gemini Flash lifecycle by generation, with exact runtime variants."""
    discovered_by_generation: dict[str, list[dict[str, str]]] = {}
    for item in discovered:
        model_id = item.get("provider_model_id", "")
        match = _GEMINI_FLASH_ID.fullmatch(model_id)
        if match:
            discovered_by_generation.setdefault(match.group("generation"), []).append(item)
    result = [dict(entry) for entry in entries]
    existing_by_generation = {
        generation: index
        for index, entry in enumerate(result)
        if (generation := _gemini_flash_generation(entry)) is not None
    }
    for generation in sorted(discovered_by_generation):
        if not discovered_by_generation[generation]:
            continue
        index = existing_by_generation.get(generation)
        existing = result[index] if index is not None else None
        item = _gemini_flash_record(
            generation, discovered_by_generation[generation], observed_at=observed_at,
            serving_engine_version=serving_engine_version, existing=existing,
        )
        if existing is None:
            result.append(item)
            existing_by_generation[generation] = len(result) - 1
        else:
            result[index] = item
    return result


def merge_gemini_flash_discovery(
    entries: list[dict[str, Any]],
    discovered: list[dict[str, str]],
    *,
    observed_at: str,
    serving_engine_version: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Add or refresh Gemini Flash generation facts without changing defaults.

    The bootstrap route identity remains the 3.7 current anchor until an
    explicit promotion changes it.  A discovered 3.8 generation is therefore
    appended as a candidate, not treated as an implicit lifecycle migration.
    Existing exact identities retain lifecycle and promotion history.
    """
    canonical = canonicalize_gemini_flash_generations(
        [], discovered, observed_at=observed_at, serving_engine_version=serving_engine_version,
    )
    result = [dict(entry) for entry in entries]
    existing_by_generation = {
        generation: index
        for index, entry in enumerate(result)
        if (generation := _gemini_flash_generation(entry)) is not None
    }
    added: list[str] = []
    updated: list[str] = []
    registered: list[str] = []
    for incoming in canonical:
        generation = incoming.get("generation")
        index = existing_by_generation.get(generation)
        if index is not None:
            old = dict(result[index])
            if generation == "3.7" and not old.get("generation"):
                refreshed = dict(old)
                refreshed.update({
                    "discovery_source": "agy models",
                    "discovery_timestamp": observed_at,
                    "availability_observed_at": observed_at,
                    "discovered_runtime_variants": incoming["runtime_variants"],
                    "discovery_client_version": serving_engine_version,
                })
            else:
                refreshed = _gemini_flash_record(
                    generation, incoming["runtime_variants"], observed_at=observed_at,
                    serving_engine_version=serving_engine_version, existing=old,
                )
            key = str(old.get("identity_key"))
            if refreshed != old:
                result[index] = refreshed
                updated.append(key)
            registered.append(key)
            continue
        item = dict(incoming)
        result.append(item)
        key = str(item["identity_key"])
        existing_by_generation[generation] = len(result) - 1
        added.append(key)
        registered.append(key)
    return result, {"added": added, "updated": updated, "registered": registered}


def promote(
    entries: list[dict[str, Any]],
    identity_key: str,
    reason: str = "explicit promotion",
    *,
    promotion_basis: str = "unspecified",
) -> list[dict[str, Any]]:
    if promotion_basis not in PROMOTION_BASES:
        raise ValueError(f"invalid promotion basis: {promotion_basis}")
    result = transition(entries, identity_key, "current")
    for entry in result:
        if entry.get("identity_key") == identity_key:
            entry["promotion_basis"] = promotion_basis
            entry["promotion_reason"] = reason
            break
    return result


def reject(entries: list[dict[str, Any]], identity_key: str, reason: str = "explicit rejection") -> list[dict[str, Any]]:
    return transition(entries, identity_key, "rejected")


def retire(entries: list[dict[str, Any]], identity_key: str, reason: str = "removed or superseded") -> list[dict[str, Any]]:
    return transition(entries, identity_key, "retired")
