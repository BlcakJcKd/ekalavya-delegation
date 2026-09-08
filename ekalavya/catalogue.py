"""Small live model catalogue with explicit lifecycle transitions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .schema import CandidateIdentity

LIVE_STATES = {"candidate", "current", "previous"}
ALL_STATES = LIVE_STATES | {"retired", "rejected", "removed"}
PROMOTION_BASES = {"quality_superiority", "operational_efficiency", "manual", "unspecified"}


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


def canonicalize_gemini_flash_generations(
    entries: list[dict[str, Any]],
    discovered: list[dict[str, str]],
    *,
    observed_at: str,
    serving_engine_version: str | None = None,
) -> list[dict[str, Any]]:
    """Store Gemini Flash lifecycle by generation, with exact runtime variants."""
    generations = {"3.6": "previous", "3.7": "current", "3.8": "candidate"}
    discovered_by_generation: dict[str, list[dict[str, str]]] = {generation: [] for generation in generations}
    for item in discovered:
        model_id = item.get("provider_model_id", "")
        parts = model_id.split("-")
        if len(parts) == 4 and parts[0] == "gemini" and parts[2] == "flash" and parts[1] in generations:
            discovered_by_generation[parts[1]].append(item)
    def is_flash_generation_entry(entry: dict[str, Any]) -> bool:
        if entry.get("provider") != "gemini" or entry.get("family") != "flash":
            return False
        model_id = entry.get("provider_model_id", "")
        return entry.get("generation") in generations or any(model_id.startswith(f"gemini-{generation}-flash-") for generation in generations)

    existing = [e for e in entries if not is_flash_generation_entry(e)]
    result = list(existing)
    for generation, lifecycle in generations.items():
        variants = sorted(discovered_by_generation[generation], key=lambda item: item["provider_model_id"])
        if not variants:
            continue
        medium = next((item for item in variants if item["provider_model_id"].endswith("-medium")), variants[0])
        old = next((e for e in entries if e.get("provider") == "gemini" and e.get("family") == "flash" and (e.get("generation") == generation or e.get("provider_model_id", "").startswith(f"gemini-{generation}-flash-"))), {})
        identity = CandidateIdentity(
            provider="gemini", family="flash", provider_model_id=medium["provider_model_id"],
            display_name=f"Gemini {generation} Flash", generation=generation, variant="medium",
            capabilities={"reasoning_values": [item["provider_model_id"].rsplit("-", 1)[-1] for item in variants]},
            serving_engine="agy", serving_engine_version=serving_engine_version,
        )
        item = dict(old)
        item.update(identity.as_dict())
        item.update({
            "identity_key": identity.identity_key,
            "catalogue_key": f"gemini:flash:{generation}",
            "lifecycle_scope": "generation_family",
            "lifecycle": lifecycle,
            "default_runtime_variant": "medium",
            "runtime_variants": [
                {"provider_model_id": variant["provider_model_id"], "display_name": variant.get("display_name"), "reasoning": variant["provider_model_id"].rsplit("-", 1)[-1]}
                for variant in variants
            ],
            "discovery_source": "agy models",
            "discovery_timestamp": observed_at,
            "availability_observed_at": observed_at,
            "transport": "agy",
        })
        result.append(item)
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
    exact = {entry.get("identity_key"): index for index, entry in enumerate(result)}
    has_seed_37 = any(
        entry.get("provider") == "gemini"
        and entry.get("family") == "flash"
        and str(entry.get("provider_model_id", "")).startswith("gemini-3.7-flash-")
        and not entry.get("generation")
        for entry in result
    )
    added: list[str] = []
    updated: list[str] = []
    registered: list[str] = []
    for incoming in canonical:
        generation = incoming.get("generation")
        # Keep the deterministic bootstrap current identity as the sole 3.7
        # current anchor.  Record availability facts on it without replacing
        # its identity or changing profile/default policy.
        if generation == "3.7" and has_seed_37:
            seed_index = next(i for i, entry in enumerate(result) if entry.get("provider") == "gemini" and entry.get("family") == "flash" and str(entry.get("provider_model_id", "")).startswith("gemini-3.7-flash-") and not entry.get("generation"))
            seed = dict(result[seed_index])
            seed.update({
                "discovery_source": "agy models",
                "discovery_timestamp": observed_at,
                "availability_observed_at": observed_at,
                "discovered_runtime_variants": incoming["runtime_variants"],
                "discovery_client_version": serving_engine_version,
            })
            if seed != result[seed_index]:
                result[seed_index] = seed
                updated.append(str(seed.get("identity_key")))
            registered.append(str(seed.get("identity_key")))
            continue
        key = incoming["identity_key"]
        if key in exact:
            old = dict(result[exact[key]])
            lifecycle = old.get("lifecycle", "candidate")
            promotion = {name: old[name] for name in ("promotion_basis", "promotion_reason") if name in old}
            old.update(incoming)
            old["lifecycle"] = lifecycle
            old.update(promotion)
            if old != result[exact[key]]:
                result[exact[key]] = old
                updated.append(key)
            registered.append(key)
            continue
        # The repository lifecycle seed is explicit only for historical 3.6;
        # every newly observed generation (including 3.8 and later) is a
        # candidate until the user promotes it.
        item = dict(incoming)
        item["lifecycle"] = "previous" if generation == "3.6" else "candidate"
        result.append(item)
        exact[key] = len(result) - 1
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
