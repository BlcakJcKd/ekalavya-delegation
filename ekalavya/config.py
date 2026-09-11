"""Reversible legacy configuration migration and Ekalavya state paths."""

from __future__ import annotations

import os
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .schema import CandidateIdentity
from .deepseek import (
    DEEPSEEK_FLASH_PROVIDER_MODEL_ID,
    DEEPSEEK_PRO_PROVIDER_MODEL_ID,
    DEEPSEEK_REASONING_MAPPING,
    DEEPSEEK_PROVIDER_REASONING_LEVELS,
    DEEPSEEK_V4_FLASH_DISPLAY_NAME,
    DEEPSEEK_V4_FLASH_PROVIDER_MODEL_ID,
    DEEPSEEK_V4_PRO_DISPLAY_NAME,
    DEEPSEEK_V41_FLASH_DISPLAY_NAME,
)


def config_root() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")).expanduser() / "ekalavya"


def load_profiles(path: Path | None = None) -> list[dict[str, object]]:
    """Load the private profile control file, treating an absent file as empty."""
    target = path or config_root() / "profiles.json"
    if not target.is_file():
        return []
    value = json.loads(target.read_text())
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("profiles must be a JSON list of objects")
    return value


def save_profiles(profiles: list[dict[str, object]], path: Path | None = None) -> None:
    """Atomically write profiles with the same private permissions as catalogue."""
    target = path or config_root() / "profiles.json"
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(target.parent, 0o700)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(profiles, indent=2, sort_keys=True) + "\n")
    os.chmod(tmp, 0o600)
    tmp.replace(target)
    os.chmod(target, 0o600)


def permit_profile_candidates(
    profiles: list[dict[str, object]], profile_name: str, candidates: list[str],
) -> list[dict[str, object]]:
    """Append known candidate identities without changing a profile default."""
    updated = [dict(profile) for profile in profiles]
    profile = next((item for item in updated if item.get("name") == profile_name), None)
    if profile is None:
        raise ValueError(f"unknown profile: {profile_name}")
    allowed = list(profile.get("permitted_candidates") or [])
    for candidate in candidates:
        if candidate not in allowed:
            allowed.append(candidate)
    profile["permitted_candidates"] = allowed
    return updated


def legacy_root() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")).expanduser() / "agent-delegation"


def migrate_legacy_config(source: Path | None = None, target: Path | None = None) -> dict[str, object]:
    source = source or legacy_root(); target = target or config_root()
    report: dict[str, object] = {"source": str(source), "target": str(target), "copied": [], "skipped": [], "conflicts": []}
    if not source.is_dir():
        report["decision"] = "legacy config absent; created no replacement"
        return report
    if source.is_symlink() or target.is_symlink():
        report["decision"] = "refused symlinked config root"
        report["conflicts"] = ["symlinked-root"]
        return report
    target.mkdir(parents=True, exist_ok=True, mode=0o700); os.chmod(target, 0o700)
    for item in sorted(source.rglob("*")):
        rel = item.relative_to(source); dest = target / rel
        if item.is_symlink():
            report["skipped"].append(str(rel)); continue
        if item.is_dir():
            dest.mkdir(parents=True, exist_ok=True, mode=0o700); os.chmod(dest, 0o700); continue
        if dest.exists():
            if dest.read_bytes() == item.read_bytes(): report["skipped"].append(str(rel))
            else: report["conflicts"].append(str(rel))
            continue
        dest.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copy2(item, dest)
        os.chmod(dest, item.stat().st_mode & 0o777)
        report["copied"].append(str(rel))
    report["decision"] = "copied without deleting legacy tree; rerunnable and conflict-preserving"
    report["timestamp"] = datetime.now(timezone.utc).isoformat()
    return report


def ensure_control_files(target: Path | None = None) -> dict[str, object]:
    """Create additive Ekalavya catalogue/profile files from fixed route metadata."""
    target = target or config_root(); target.mkdir(parents=True, exist_ok=True, mode=0o700); os.chmod(target, 0o700)
    catalogue_path, profiles_path = target / "catalogue.json", target / "profiles.json"
    catalogue_exists, profiles_exist = catalogue_path.exists(), profiles_path.exists()
    if catalogue_exists != profiles_exist:
        present = [name for name, exists in (("catalogue.json", catalogue_exists), ("profiles.json", profiles_exist)) if exists]
        missing = [name for name, exists in (("catalogue.json", catalogue_exists), ("profiles.json", profiles_exist)) if not exists]
        return {"created": [], "skipped": present, "status": "incomplete", "missing": missing}
    if catalogue_exists and profiles_exist:
        return {"created": [], "skipped": ["catalogue.json", "profiles.json"], "status": "complete"}
    from delegation import routing
    from delegation.core import DELEGATES
    catalogue=[]; profiles=[]
    for route, spec in sorted(DELEGATES.items()):
        if route == "flash":
            # The initial control files need a historical, deterministic
            # identity so discovery can add/promote newer generations.  This
            # is catalogue bootstrap data only; execution always receives an
            # exact model from the resolved catalogue variant.
            identity = CandidateIdentity(
                "gemini", "flash", "gemini-3.7-flash-medium", "flash",
                capabilities={"reasoning_values": ["medium"]},
            )
            item = identity.as_dict()
            item.update({
                "identity_key": identity.identity_key,
                "lifecycle": "current",
                "legacy_route": route,
                "execution_route": route,
                "harness": spec.executable,
                "transport": routing.ROUTE_TRANSPORT.get(route),
            })
            catalogue.append(item)
            profiles.append({
                "name": route,
                "description": "Gemini Flash",
                "default_identity_key": identity.identity_key,
                "permitted_candidates": [identity.identity_key],
                "reasoning_policy": "overrideable",
                "default_reasoning": "medium",
            })
            continue
        if route == "deepseek-flash":
            capabilities = {
                "reasoning_values": [spec.effort] if spec.effort else [],
                "provider_reasoning_values": list(DEEPSEEK_PROVIDER_REASONING_LEVELS),
                "reasoning_compatibility": DEEPSEEK_REASONING_MAPPING,
                "context_tokens": 1_000_000,
                "max_output_tokens": 384_000,
                "thinking": True,
                "non_thinking": True,
                "tool_calls": True,
                "responses_api": True,
                "anthropic_api": True,
                "input_modalities": ["text"],
                "provider_native_vision": True,
                "harness_multimodal": False,
            }
            historical = CandidateIdentity(
                "deepseek", "flash", DEEPSEEK_V4_FLASH_PROVIDER_MODEL_ID,
                DEEPSEEK_V4_FLASH_DISPLAY_NAME, generation="v4",
                capabilities=capabilities,
            )
            old_item = historical.as_dict()
            old_item.update({
                "identity_key": historical.identity_key,
                "lifecycle": "retired",
                "provider_aliases": [DEEPSEEK_V4_FLASH_PROVIDER_MODEL_ID, "deepseek-v4-flash-vision-exp"],
                "historical_only": True,
                "transport": routing.ROUTE_TRANSPORT.get(route),
            })
            current = CandidateIdentity(
                "deepseek", "flash", DEEPSEEK_FLASH_PROVIDER_MODEL_ID,
                DEEPSEEK_V41_FLASH_DISPLAY_NAME, generation="v4.1",
                capabilities=capabilities,
            )
            item = current.as_dict()
            item.update({
                "identity_key": current.identity_key,
                "lifecycle": "candidate",
                "legacy_route": route,
                "execution_route": route,
                "harness": spec.executable,
                "provider_aliases": [DEEPSEEK_V4_FLASH_PROVIDER_MODEL_ID, "deepseek-v4-flash-vision-exp"],
                "transport": routing.ROUTE_TRANSPORT.get(route),
            })
            catalogue.extend([old_item, item])
            profiles.append({
                "name": route,
                "description": DEEPSEEK_V41_FLASH_DISPLAY_NAME,
                "default_identity_key": current.identity_key,
                "permitted_candidates": [current.identity_key],
                "reasoning_policy": "fixed",
                "default_reasoning": spec.effort,
            })
            continue
        if route == "deepseek-pro":
            identity = CandidateIdentity(
                "deepseek", "pro", DEEPSEEK_PRO_PROVIDER_MODEL_ID,
                DEEPSEEK_V4_PRO_DISPLAY_NAME, generation="v4",
                capabilities={
                    "reasoning_values": [spec.effort] if spec.effort else [],
                    "provider_reasoning_values": list(DEEPSEEK_PROVIDER_REASONING_LEVELS),
                    "reasoning_compatibility": DEEPSEEK_REASONING_MAPPING,
                    "input_modalities": ["text"],
                    "harness_multimodal": False,
                },
            )
            item = identity.as_dict()
            item.update({
                "identity_key": identity.identity_key,
                "lifecycle": "current",
                "legacy_route": route,
                "execution_route": route,
                "harness": spec.executable,
                "transport": routing.ROUTE_TRANSPORT.get(route),
            })
            catalogue.append(item)
            profiles.append({
                "name": route,
                "description": DEEPSEEK_V4_PRO_DISPLAY_NAME,
                "default_identity_key": identity.identity_key,
                "permitted_candidates": [identity.identity_key],
                "reasoning_policy": "fixed",
                "default_reasoning": spec.effort,
            })
            continue
        identity = CandidateIdentity(routing.ROUTE_PROVIDER[route], route, spec.model, route, capabilities={"reasoning_values": [spec.effort] if spec.effort else []})
        item = identity.as_dict(); item.update({"identity_key": identity.identity_key, "lifecycle": "current", "legacy_route": route, "execution_route": route, "harness": spec.executable, "transport": routing.ROUTE_TRANSPORT.get(route)})
        catalogue.append(item)
        profiles.append({"name": route, "description": f"Stable explicit route {route}", "default_identity_key": identity.identity_key, "permitted_candidates": [identity.identity_key], "reasoning_policy": "fixed", "default_reasoning": spec.effort})
    created: list[str] = []
    skipped: list[str] = []
    from .catalogue import save_catalogue
    save_catalogue(catalogue_path, catalogue)
    created.append("catalogue.json")
    save_profiles(profiles, profiles_path)
    created.append("profiles.json")
    return {"created": created, "skipped": skipped, "status": "complete"}
