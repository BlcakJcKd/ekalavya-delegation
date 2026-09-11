"""User-owned Ekalavya availability and route-policy configuration.

Persisted as TOML at ``delegation.paths.config_path()`` (XDG config dir). This
file describes availability policy and optional user routing preferences --
which providers/routes are eligible in principle, including named local vLLM
routes. It must never contain credentials, tokens, or
provider secrets, and it cannot redefine what a pinned route (for example, ``flash``)
resolves to at runtime; that pin lives in ``delegation.core.DELEGATES`` and
is not configurable here. See docs/DELEGATE_CONFIGURATION.md.

This is user-owned persistent state. An AI primary agent may read, inspect,
and respect it, but must not mutate it merely on its own judgement (for example,
"quota looks low") -- persistent mutation requires either explicit user
instruction or an explicit human-directed Ekalavya configuration action. A task-scoped
instruction such as "don't use Codex for this task" is a session constraint,
not a reason to call ``set_enabled`` here.
"""

from __future__ import annotations

import os
import tomllib
import math
import json
from pathlib import Path
from typing import Any

from .paths import config_path
from .routing import DEFAULT_DISABLED_REASON, EXPERIMENTAL_PAYG_NAMES, MODELS, PROVIDERS
from ekalavya.schema import normalize_task

_SECTIONS = {"providers": PROVIDERS, "models": MODELS}
_CONFIG_SECTIONS = {"providers", "models", "vllm", "routing"}
_ALLOWED_ENTRY_KEYS = {"enabled", "reason"}
_ROUTING_KEYS = {"preferences", "reserves"}
_PREFERENCE_KEYS = {"preferred_targets", "allowed_targets", "excluded_targets"}
_RESERVE_KEYS = {"provider", "scope_kind", "resource_kind", "window_kind", "minimum_remaining_fraction"}


def _default_entry(name: str) -> dict[str, Any]:
    """The default entry for a name absent from a config file.

    Stable providers/routes default to enabled. Experimental PAYG
    providers/routes (``routing.EXPERIMENTAL_PAYG_NAMES``) default to
    disabled with a standard reason -- both for a fresh install and when
    merging into an existing config that predates them, so a PAYG route
    never becomes eligible merely because a user's config file is older
    than it. Enabling one is always an explicit Ekalavya configuration action.
    """
    if name in EXPERIMENTAL_PAYG_NAMES:
        return {"enabled": False, "reason": DEFAULT_DISABLED_REASON}
    return {"enabled": True}


def default_config(vllm_routes: set[str] | tuple[str, ...] = ()) -> dict[str, dict[str, dict[str, Any]]]:
    """The config used when no config file exists yet.

    Everything is enabled except experimental PAYG providers/routes, which
    default to disabled -- see ``_default_entry``.
    """
    result = {
        section: {name: _default_entry(name) for name in names}
        for section, names in _SECTIONS.items()
    }
    if vllm_routes:
        result["vllm"] = {name: _default_entry(name) for name in sorted(vllm_routes)}
    return result


def _validate_entry(section: str, name: str, entry: Any) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError(f"{section}.{name} must be a table, got {type(entry).__name__}")
    if "enabled" not in entry:
        raise ValueError(f"{section}.{name}.enabled is required")
    if not isinstance(entry["enabled"], bool):
        raise ValueError(f"{section}.{name}.enabled must be a boolean")
    if "reason" in entry and not isinstance(entry["reason"], str):
        raise ValueError(f"{section}.{name}.reason must be a string")
    extra = set(entry) - _ALLOWED_ENTRY_KEYS
    if extra:
        raise ValueError(f"{section}.{name} has unsupported field(s): {sorted(extra)}")
    result: dict[str, Any] = {"enabled": entry["enabled"]}
    if entry.get("reason"):
        result["reason"] = entry["reason"]
    return result


def normalize_route_target(value: str) -> str:
    """Return the stable internal spelling for a routing target."""
    if not isinstance(value, str) or not value:
        raise ValueError("routing target must be a non-empty string")
    if value == "primary-native":
        return value
    if value in MODELS:
        return f"profile:{value}"
    if value.startswith("profile:"):
        profile = value[8:]
        if profile not in MODELS:
            raise ValueError(f"unknown profile target: {value!r}")
        return value
    if value.startswith("vllm:"):
        from .vllm import is_vllm_route_name
        name = value[5:]
        if not is_vllm_route_name(name):
            raise ValueError(f"invalid vLLM routing target: {value!r}")
        return value
    raise ValueError(f"unknown routing target: {value!r}")


def _target_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be an array of routing targets")
    targets = [normalize_route_target(item) for item in value]
    if len(set(targets)) != len(targets):
        raise ValueError(f"{field} must not contain duplicate targets")
    return targets


def _validate_preference_conflicts(task: str, preference: dict[str, list[str]]) -> None:
    preferred = set(preference.get("preferred_targets", []))
    allowed = set(preference.get("allowed_targets", []))
    excluded = set(preference.get("excluded_targets", []))
    if preferred & excluded:
        raise ValueError(f"routing preference {task!r} has targets in both preferred_targets and excluded_targets")
    if allowed & excluded:
        raise ValueError(f"routing preference {task!r} has targets in both allowed_targets and excluded_targets")


def _validate_routing(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("[routing] must be a table")
    extra = set(raw) - _ROUTING_KEYS
    if extra:
        raise ValueError(f"[routing] has unsupported field(s): {sorted(extra)}")
    result: dict[str, Any] = {"preferences": {}, "reserves": {}}
    preferences = raw.get("preferences", {})
    if not isinstance(preferences, dict):
        raise ValueError("[routing.preferences] must be a table")
    for task, item in preferences.items():
        try:
            normalized_task = normalize_task(task)
        except ValueError as exc:
            raise ValueError(f"invalid routing task {task!r}: {exc}") from None
        if normalized_task == "unspecified":
            raise ValueError("routing preferences cannot be configured for unspecified")
        if not isinstance(item, dict):
            raise ValueError(f"routing preference {task!r} must be a table")
        unknown = set(item) - _PREFERENCE_KEYS
        if unknown:
            raise ValueError(f"routing preference {task!r} has unsupported field(s): {sorted(unknown)}")
        preference: dict[str, list[str]] = {}
        for field in _PREFERENCE_KEYS:
            if field in item:
                preference[field] = _target_list(item[field], f"routing preference {task!r}.{field}")
        _validate_preference_conflicts(task, preference)
        result["preferences"][normalized_task] = preference
    reserves = raw.get("reserves", {})
    if not isinstance(reserves, dict):
        raise ValueError("[routing.reserves] must be a table")
    for name, item in reserves.items():
        if not isinstance(name, str) or not name or not isinstance(item, dict):
            raise ValueError("routing reserve entries must be named tables")
        unknown = set(item) - _RESERVE_KEYS
        missing = _RESERVE_KEYS - set(item)
        if unknown or missing:
            raise ValueError(f"routing reserve {name!r} fields invalid; missing={sorted(missing)}, unsupported={sorted(unknown)}")
        provider = item["provider"]
        if provider not in PROVIDERS and provider != "vllm":
            raise ValueError(f"routing reserve {name!r} has unknown provider {provider!r}")
        fraction = item["minimum_remaining_fraction"]
        if not isinstance(fraction, (int, float)) or isinstance(fraction, bool) or not math.isfinite(float(fraction)) or not 0 <= float(fraction) <= 1:
            raise ValueError(f"routing reserve {name!r}.minimum_remaining_fraction must be a finite value from 0 to 1")
        for key in ("scope_kind", "resource_kind", "window_kind"):
            if not isinstance(item[key], str) or not item[key]:
                raise ValueError(f"routing reserve {name!r}.{key} must be a non-empty string")
        result["reserves"][name] = {
            "provider": provider,
            "scope_kind": item["scope_kind"],
            "resource_kind": item["resource_kind"],
            "window_kind": item["window_kind"],
            "minimum_remaining_fraction": float(fraction),
        }
    return result


def parse_config(
    raw: dict[str, Any],
    vllm_routes: set[str] | tuple[str, ...] = (),
) -> dict[str, dict[str, dict[str, Any]]]:
    """Validate a raw parsed-TOML dict against the known schema.

    Unknown provider/model names are rejected (no silently-ignored typos);
    a name absent from an otherwise-valid file defaults to enabled with no
    reason, matching a fresh install.
    """
    unknown_sections = set(raw) - _CONFIG_SECTIONS
    if unknown_sections:
        raise ValueError(f"unknown config section(s): {sorted(unknown_sections)}")
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for section, names in _SECTIONS.items():
        section_raw = raw.get(section, {})
        if not isinstance(section_raw, dict):
            raise ValueError(f"[{section}] must be a table")
        unknown = set(section_raw) - set(names)
        if unknown:
            raise ValueError(f"unknown {section[:-1]}(s) in config: {sorted(unknown)}")
        result[section] = {
            name: _validate_entry(section, name, section_raw[name]) if name in section_raw
            else _default_entry(name)
            for name in names
        }
    raw_vllm = raw.get("vllm", {})
    if not isinstance(raw_vllm, dict):
        raise ValueError("[vllm] must be a table")
    route_names = set(vllm_routes) | set(raw_vllm)
    from .vllm import is_vllm_route_name
    invalid_routes = [name for name in route_names if not is_vllm_route_name(name)]
    if invalid_routes:
        raise ValueError(f"invalid vLLM route name(s): {sorted(invalid_routes, key=str)}")
    if route_names or "vllm" in raw:
        result["vllm"] = {
            name: _validate_entry("vllm", name, raw_vllm[name]) if name in raw_vllm
            else _default_entry(name)
            for name in sorted(route_names)
        }
    if "routing" in raw:
        result["routing"] = _validate_routing(raw["routing"])
    return result


def load_config(path: Path | None = None) -> dict[str, dict[str, dict[str, Any]]]:
    """Load and validate the config, or return the default if none exists yet."""
    target = path or config_path()
    # The availability layer discovers named local vLLM routes, but never
    # imports credentials or contacts their endpoints.  Existing vLLM entries
    # in config.toml are also retained so removing a local route cannot erase
    # its user-owned preference.
    from .vllm import vllm_route_names

    # An explicit config path is used by tests/tools for an isolated config
    # layer; do not accidentally couple it to the caller's machine-local
    # vllm.toml.  The normal no-argument runtime path discovers local routes.
    local_routes = vllm_route_names() if path is None else set()
    if not target.is_file():
        return default_config(local_routes)
    raw = tomllib.loads(target.read_text())
    configured_routes = raw.get("vllm", {})
    if not isinstance(configured_routes, dict):
        raise ValueError("[vllm] must be a table")
    return parse_config(raw, local_routes | set(configured_routes))


def _render_toml(config: dict[str, dict[str, dict[str, Any]]]) -> str:
    lines = [
        "# Ekalavya user configuration",
        "#",
        "# Describes Ekalavya availability and optional route policy (which providers/routes",
        "# are eligible in principle, plus user-owned target preferences). It must never contain credentials, tokens,",
        "# or provider secrets. See docs/DELEGATE_CONFIGURATION.md.",
        "",
    ]
    for section in ("providers", "models", "vllm"):
        if section not in config:
            continue
        for name in sorted(config[section]):
            entry = config[section][name]
            lines.append(f"[{section}.{name}]")
            lines.append(f"enabled = {'true' if entry['enabled'] else 'false'}")
            reason = entry.get("reason")
            if reason:
                escaped = reason.replace("\\", "\\\\").replace('"', '\\"')
                lines.append(f'reason = "{escaped}"')
            lines.append("")
    routing_policy = config.get("routing")
    if routing_policy is not None:
        for task in sorted(routing_policy.get("preferences", {})):
            entry = routing_policy["preferences"][task]
            escaped_task = task.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'[routing.preferences."{escaped_task}"]')
            for field in ("preferred_targets", "allowed_targets", "excluded_targets"):
                if field in entry:
                    rendered = ", ".join(json.dumps(value) for value in entry[field])
                    lines.append(f"{field} = [{rendered}]")
            lines.append("")
        for name in sorted(routing_policy.get("reserves", {})):
            entry = routing_policy["reserves"][name]
            escaped_name = name.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'[routing.reserves."{escaped_name}"]')
            for field in ("provider", "scope_kind", "resource_kind", "window_kind"):
                lines.append(f"{field} = {json.dumps(entry[field])}")
            lines.append(f"minimum_remaining_fraction = {entry['minimum_remaining_fraction']}")
            lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def save_config(config: dict[str, dict[str, dict[str, Any]]], path: Path | None = None) -> Path:
    """Atomically write the config: render to a temp file, then rename in place."""
    target = path or config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(target.parent, 0o700)
    tmp = target.with_name(f"{target.name}.tmp-{os.getpid()}")
    tmp.write_text(_render_toml(config))
    os.chmod(tmp, 0o600)
    tmp.replace(target)
    os.chmod(target, 0o600)
    return target


def set_enabled(
    config: dict[str, dict[str, dict[str, Any]]],
    section: str,
    name: str,
    enabled: bool,
    reason: str | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Return a new config with one entry's enabled/reason updated.

    Enabling always clears any existing reason (a reason describes why an
    entry is disabled; carrying a stale one forward is confusing). Disabling
    sets ``reason`` when given; when not given, any existing reason on that
    entry is left untouched, so re-disabling an already-disabled route
    without a fresh reason does not erase the last recorded one.
    """
    if section not in {"providers", "models", "vllm"}:
        raise ValueError(f"unknown config section: {section!r}")
    known = config.get(section, {}) if section == "vllm" else dict.fromkeys(_SECTIONS[section])
    if name not in known:
        noun = "vllm route" if section == "vllm" else section[:-1]
        raise ValueError(f"unknown {noun}: {name!r}; known: {sorted(known)}")
    updated = {sec: {n: dict(entry) for n, entry in entries.items()} for sec, entries in config.items()}
    entry = dict(updated[section][name])
    entry["enabled"] = enabled
    if enabled:
        entry.pop("reason", None)
    elif reason is not None:
        if reason:
            entry["reason"] = reason
        else:
            entry.pop("reason", None)
    updated[section][name] = entry
    return updated


def set_routing_preference(config: dict[str, Any], task: str, field: str, targets: list[str]) -> dict[str, Any]:
    """Return config with one explicit, validated routing target list replaced."""
    if field not in _PREFERENCE_KEYS:
        raise ValueError(f"unknown routing preference field: {field!r}")
    normalized_task = normalize_task(task)
    if normalized_task == "unspecified":
        raise ValueError("routing preferences cannot be configured for unspecified")
    normalized = _target_list(targets, field)
    updated = {section: ({name: dict(value) for name, value in entries.items()} if isinstance(entries, dict) else entries) for section, entries in config.items()}
    policy = updated.setdefault("routing", {"preferences": {}, "reserves": {}})
    policy["preferences"] = {name: dict(value) for name, value in policy.get("preferences", {}).items()}
    preference = dict(policy["preferences"].get(normalized_task, {}))
    if normalized:
        preference[field] = normalized
    else:
        preference.pop(field, None)
    _validate_preference_conflicts(normalized_task, preference)
    if preference:
        policy["preferences"][normalized_task] = preference
    else:
        policy["preferences"].pop(normalized_task, None)
    return updated


def set_routing_reserve(config: dict[str, Any], name: str, reserve: dict[str, Any] | None) -> dict[str, Any]:
    """Set or remove one advanced quota-reserve selector."""
    updated = {section: ({key: dict(value) for key, value in entries.items()} if isinstance(entries, dict) else entries) for section, entries in config.items()}
    policy = updated.setdefault("routing", {"preferences": {}, "reserves": {}})
    policy["reserves"] = {key: dict(value) for key, value in policy.get("reserves", {}).items()}
    if reserve is None:
        policy["reserves"].pop(name, None)
    else:
        policy["reserves"][name] = dict(reserve)
    # Reuse the same validator used for manually edited TOML before saving.
    policy.update(_validate_routing(policy))
    return updated
