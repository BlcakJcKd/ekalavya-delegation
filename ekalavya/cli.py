"""Canonical Ekalavya CLI with read-only status/history/spend foundations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from delegation import routing
from delegation.config import load_config, save_config, set_enabled
from delegation.status_cli import _print_human, build_report
from delegation.vllm import inspect_vllm_routes
from delegation.paths import state_dir

from . import __version__
from .catalogue import PROMOTION_BASES, load_catalogue, merge_gemini_flash_discovery, promote, save_catalogue
from .config import config_root, load_profiles, migrate_legacy_config, permit_profile_candidates, save_profiles
from .discovery import DiscoveryError, discover_gemini
from .executor import execute
from .harness_registry import current_registry, validate_registry
from .ledger import connect, default_db_path, finalize_run, normalize_cutoff, record_availability, record_default_change, record_promotion_event, record_resolution, record_run, upsert_model
from .migrate import migrate_all
from .resolver import resolve
from benchmark.review_bundle import create_review_bundle
from .schema import CandidateIdentity, RunIntent
from .telemetry import persist_execution_observability
from .quota import collect_snapshots, public_snapshot
from .usage import build_insights, build_usage, clear_observability, delete_feedback, export_usage, refresh_quotas, set_feedback


def _paths() -> tuple[Path, Path, Path]:
    root = config_root(); return root, root / "catalogue.json", root / "profiles.json"


def _profiles(path: Path) -> list[dict[str, Any]]:
    return load_profiles(path)


def _refresh_gemini_catalogue() -> dict[str, Any]:
    """Discover and register Gemini candidates without altering profile defaults."""
    discovery = discover_gemini()
    root, catalogue_path, profiles_path = _paths()
    current = load_catalogue(catalogue_path)
    updated, merge = merge_gemini_flash_discovery(
        current,
        discovery["models"],
        observed_at=str(discovery["observed_at"]),
        serving_engine_version=str(discovery["client_version"]),
    )
    profiles = _profiles(profiles_path)
    permitted = [key for key in merge["registered"] if any(entry.get("identity_key") == key and entry.get("provider") == "gemini" and entry.get("family") == "flash" for entry in updated)]
    changed_profiles = permit_profile_candidates(profiles, "flash", permitted)
    # Validate all derived state before writing either control file.  Each
    # individual file write is atomic and private; a failed discovery writes
    # neither catalogue nor profile state.
    if updated != current:
        save_catalogue(catalogue_path, updated)
    if changed_profiles != profiles:
        save_profiles(changed_profiles, profiles_path)
    conn = connect()
    by_key = {entry.get("identity_key"): entry for entry in updated}
    for key in merge["registered"]:
        entry = by_key.get(key)
        if not entry:
            continue
        identity = CandidateIdentity(**{name: entry.get(name) for name in CandidateIdentity.__dataclass_fields__})
        model_id = upsert_model(conn, identity, identity_key=str(entry["identity_key"]), lifecycle=str(entry.get("lifecycle", "candidate")), discovered_at=str(discovery["observed_at"]))
        record_availability(conn, model_id, state="available", observed_at=str(discovery["observed_at"]), source="agy models", details={"provider_model_id": identity.provider_model_id, "client_version": discovery["client_version"]})
    return {
        **discovery,
        "added_candidates": len(merge["added"]),
        "updated_candidates": len(merge["updated"]),
        "registered_identity_keys": merge["registered"],
        "auto_promoted": 0,
        "profile_default_changed": False,
        "default_reasoning_changed": False,
    }


def _validated_source_candidates(incoming: object) -> list[dict[str, Any]]:
    if not isinstance(incoming, list):
        raise ValueError("models refresh source must contain a JSON list")
    result: list[dict[str, Any]] = []
    for raw in incoming:
        if not isinstance(raw, dict):
            raise ValueError("models refresh source entries must be objects")
        if not isinstance(raw.get("provider"), str) or not raw["provider"]:
            raise ValueError("models refresh source entry requires provider")
        if not isinstance(raw.get("provider_model_id"), str) or not raw["provider_model_id"]:
            raise ValueError("models refresh source entry requires provider_model_id")
        item = dict(raw)
        identity = CandidateIdentity(**{name: item.get(name) for name in CandidateIdentity.__dataclass_fields__})
        supplied = item.get("identity_key")
        if supplied is not None and supplied != identity.identity_key:
            raise ValueError("models refresh source identity_key does not match identity fields")
        item["identity_key"] = identity.identity_key
        result.append(item)
    return result


def _named_route_profile(name: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Expose a configured named vLLM route through the generic run contract."""
    if not name.startswith("vllm:") or not name[5:]:
        return None
    route = name[5:]
    info = inspect_vllm_routes().get(route)
    config = load_config()
    if info is None or info.provider is None:
        return None
    provider = info.provider
    identity = CandidateIdentity(
        "vllm", "openai-compatible", provider.model, route,
        capabilities={"harness_values": ["vllm"]},
    )
    entry = identity.as_dict()
    entry.update({
        "identity_key": identity.identity_key,
        "lifecycle": "current",
        "execution_route": f"vllm:{route}",
        "transport": "openai-compatible",
        "harness": "vllm",
        "harness_version": None,
    })
    profile = {
        "name": name,
        "description": f"Configured named vLLM route {route}",
        "default_identity_key": identity.identity_key,
        "permitted_candidates": [identity.identity_key],
        "reasoning_policy": "overrideable",
    }
    return profile, entry


def _json_or_text(value: Any, as_json: bool) -> None:
    if as_json: print(json.dumps(value, indent=2, sort_keys=True))
    else: print(value if isinstance(value, str) else json.dumps(value, indent=2, sort_keys=True))


def _vllm_summary() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, info in inspect_vllm_routes().items():
        provider = info.provider
        result[name] = {
            "state": "available" if provider else info.error_kind or "invalid-configuration",
            "model": provider.model if provider else None,
            "shared_compute": provider.shared_compute if provider else None,
            "max_concurrency": provider.max_concurrency if provider else None,
            "thinking_default": provider.thinking_default if provider else None,
            "default_max_tokens": provider.default_max_tokens if provider else None,
            "max_tokens_cap": provider.max_tokens_cap if provider else None,
            "credential_configured": bool(provider) if provider else None,
        }
    return result


def _availability_payload(config: dict[str, Any]) -> dict[str, Any]:
    effective_models = {}
    for model in routing.MODELS:
        provider = routing.ROUTE_PROVIDER[model]
        provider_entry = config["providers"][provider]
        model_entry = config["models"][model]
        provider_enabled = bool(provider_entry.get("enabled", True))
        configured_enabled = bool(model_entry.get("enabled", True))
        reason = None
        if not provider_enabled:
            reason = provider_entry.get("reason") or "disabled by provider"
        elif not configured_enabled:
            reason = model_entry.get("reason") or "disabled by model"
        effective_models[model] = {
            "configured_enabled": configured_enabled,
            "configured_reason": model_entry.get("reason"),
            "provider": provider,
            "provider_enabled": provider_enabled,
            "provider_reason": provider_entry.get("reason"),
            "effective_enabled": provider_enabled and configured_enabled,
            "unavailable_reason": reason,
        }
    return {
        "availability": config,
        "effective_models": effective_models,
        "vllm_routes": _vllm_summary(),
    }


def _persisted_model_availability() -> dict[str, dict[str, str]]:
    """Read prior local discovery facts without creating or migrating a ledger."""
    path = default_db_path()
    if not path.is_file():
        return {}
    try:
        conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
        rows = conn.execute(
            "SELECT m.identity_key,a.state,a.observed_at,a.source "
            "FROM model_availability a JOIN models m ON m.id=a.model_id "
            "WHERE a.id IN (SELECT MAX(id) FROM model_availability GROUP BY model_id)"
        ).fetchall()
    except sqlite3.DatabaseError:
        return {}
    finally:
        if "conn" in locals():
            conn.close()
    return {
        str(identity_key): {"state": str(state), "observed_at": str(observed_at), "source": str(source) if source is not None else "unknown"}
        for identity_key, state, observed_at, source in rows
    }


def cmd_status(args: argparse.Namespace) -> int:
    root, cat, prof = _paths(); entries = load_catalogue(cat); profiles = _profiles(prof)
    try:
        from .readiness import profile_readiness

        observed = _persisted_model_availability()
        readiness = {
            route: profile_readiness(route, profiles, entries, observed)
            for route in routing.MODELS
        }
        routing_report = build_report(
            getattr(args, "primary", None), live=getattr(args, "live", False),
            route_readiness=readiness,
        )
    except ValueError as exc:
        print(f"status error: {exc}", file=sys.stderr); return 2
    result = {"product": "Ekalavya", "version": __version__, "primary": getattr(args, "primary", None), "config_root": str(root), "ledger": str(default_db_path()), "profiles": [{"name": p.get("name"), "default": p.get("default_identity_key"), "reasoning_policy": p.get("reasoning_policy", "overrideable"), "availability": "configured" if p.get("default_identity_key") else "not-configured"} for p in profiles], "catalogue": [{k: e.get(k) for k in ("provider", "family", "provider_model_id", "lifecycle", "identity_key")} for e in entries], "routing": routing_report}
    if args.json:
        _json_or_text(result, True)
    else:
        _print_human(routing_report)
    return 0


def cmd_profiles(args: argparse.Namespace) -> int:
    _, _, path = _paths(); _json_or_text(_profiles(path), args.json); return 0


def cmd_models(args: argparse.Namespace) -> int:
    _, path, _ = _paths()
    if getattr(args, "action", None) == "promote":
        target = getattr(args, "target", None)
        entries = load_catalogue(path)
        match = next((entry for entry in entries if entry.get("identity_key") == target), None)
        if match is None:
            print(f"unknown catalogue identity: {target}; use eka models --json to inspect exact identity keys", file=sys.stderr)
            return 2
        reason = getattr(args, "promotion_reason", None) or "explicit promotion"
        updated = promote(entries, target, reason, promotion_basis=args.basis)
        if getattr(args, "set_default", False):
            profile_name = getattr(args, "profile", None)
            if not profile_name:
                print("--set-default requires --profile PROFILE", file=sys.stderr)
                return 2
            profiles_path = _paths()[2]
            profiles = _profiles(profiles_path)
            profile = next((item for item in profiles if item.get("name") == profile_name), None)
            if profile is None:
                print(f"unknown profile: {profile_name}", file=sys.stderr)
                return 2
            old_default_identity_key = profile.get("default_identity_key")
            profiles = permit_profile_candidates(profiles, profile_name, [target])
            profile = next(item for item in profiles if item.get("name") == profile_name)
            profile["default_identity_key"] = target
            if getattr(args, "default_reasoning", None):
                profile["default_reasoning"] = args.default_reasoning
            profile["promotion_basis"] = args.basis
            profile["promotion_reason"] = reason
            save_profiles(profiles, profiles_path)
        save_catalogue(path, updated)
        conn = connect()
        record_promotion_event(conn, target, from_state=match.get("lifecycle"), to_state="current", reason=reason, promotion_basis=args.basis)
        if getattr(args, "set_default", False) and old_default_identity_key != target:
            record_default_change(conn, profile_name, old_identity_key=old_default_identity_key, new_identity_key=target, reason=reason)
        _json_or_text({"action": "promote", "identity_key": target, "promotion_basis": args.basis, "promotion_reason": reason, "set_default": bool(getattr(args, "set_default", False))}, args.json)
        return 0
    if getattr(args, "action", None) == "refresh":
        if bool(args.source) == bool(getattr(args, "provider", None)):
            print("models refresh requires exactly one of --source FILE or --provider gemini", file=sys.stderr); return 2
        if getattr(args, "provider", None) == "gemini":
            try:
                _json_or_text(_refresh_gemini_catalogue(), args.json)
            except (DiscoveryError, ValueError, OSError) as exc:
                print(f"Gemini discovery failed; catalogue unchanged: {exc}", file=sys.stderr)
                return 2
            return 0
        try:
            incoming = _validated_source_candidates(json.loads(Path(args.source).read_text()))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"models refresh source rejected: {exc}", file=sys.stderr); return 2
        current = load_catalogue(path); known = {e.get("identity_key") for e in current}; added = 0
        conn = connect(); observed_at = datetime.now(timezone.utc).isoformat()
        for raw in incoming:
            item = dict(raw)
            existing = next((e for e in current if e.get("identity_key") == item["identity_key"]), None)
            item["lifecycle"] = existing.get("lifecycle", "candidate") if existing else "candidate"
            if item.get("identity_key") not in known:
                current.append(item); known.add(item.get("identity_key")); added += 1
            identity = CandidateIdentity(**{k: item.get(k) for k in CandidateIdentity.__dataclass_fields__})
            model_id = upsert_model(conn, identity, lifecycle=item["lifecycle"], discovered_at=observed_at)
            record_availability(conn, model_id, state="available", observed_at=observed_at, source=item.get("discovery_source", "provider discovery"), details={k: v for k, v in item.items() if k not in CandidateIdentity.__dataclass_fields__})
        save_catalogue(path, current); _json_or_text({"added_candidates": added, "auto_promoted": 0}, args.json); return 0
    entries = load_catalogue(path)
    routes = _vllm_summary()
    _json_or_text({"catalogue": entries, "named_routes": routes} if routes else entries, args.json); return 0


def cmd_config(args: argparse.Namespace) -> int:
    if getattr(args, "action", None) == "migrate":
        result = migrate_all()
        control = result.get("config", {}).get("control_files", {})
        _json_or_text(result, args.json)
        if control.get("status") == "incomplete":
            print("config migration stopped: incomplete catalogue/profiles pair; restore the missing file explicitly", file=sys.stderr)
            return 2
        return 0
    try:
        config = load_config()
    except ValueError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    action = getattr(args, "action", None)
    if action in (None, "list"):
        if action is None and not getattr(args, "json", False) and sys.stdin.isatty() and sys.stdout.isatty():
            from delegation.config_tui import run_interactive_config
            return run_interactive_config()
        root, _, _ = _paths(); _json_or_text({"config_root": str(root), "migration": "explicit via eka config migrate", **_availability_payload(config)}, args.json); return 0
    target = getattr(args, "target", None)
    if not target:
        print(f"config {action} requires a target", file=sys.stderr); return 2
    if action in {"enable-model", "disable-model"}:
        if target not in routing.MODELS:
            print(f"unknown model: {target}; known: {', '.join(routing.MODELS)}", file=sys.stderr); return 2
        section = "models"
        enabled = action == "enable-model"
        updated = set_enabled(config, section, target, enabled, reason=getattr(args, "reason", None))
    elif action in {"enable", "disable"}:
        if target in routing.MODELS: section = "models"
        elif target in config.get("vllm", {}): section = "vllm"
        else:
            known = ", ".join(sorted(set(routing.MODELS) | set(config.get("vllm", {}))))
            print(f"unknown model or vLLM route: {target}; known: {known}", file=sys.stderr); return 2
        updated = set_enabled(config, section, target, action == "enable", reason=getattr(args, "reason", None))
    elif action in {"enable-provider", "disable-provider"}:
        if target not in routing.PROVIDERS:
            print(f"unknown provider: {target}; known: {', '.join(routing.PROVIDERS)}", file=sys.stderr); return 2
        updated = set_enabled(config, "providers", target, action == "enable-provider", reason=getattr(args, "reason", None))
        section = "providers"
    else:
        print(f"unknown config action: {action}", file=sys.stderr); return 2
    save_config(updated)
    _json_or_text({"action": action, "target": target, **_availability_payload(updated)}, args.json)
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    """Human onboarding: selection is explicit, harness readiness is advisory."""
    from delegation.config_tui import run_interactive_setup, setup_detection, setup_readiness

    try:
        config = load_config()
    except ValueError as exc:
        print(f"setup config error: {exc}", file=sys.stderr)
        return 2
    detection = setup_detection()
    if args.json:
        _json_or_text({"interactive": False, "readiness": setup_readiness(config, detection)}, True)
        return 0
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print("eka setup requires an interactive TTY; use `eka setup --json` for read-only readiness, then explicit `eka config` commands for automation.", file=sys.stderr)
        return 2
    result = run_interactive_setup()
    if result["cancelled"]:
        print("eka setup: cancelled, no changes made")
        return 0
    print("eka setup: saved" if result["changed"] else "eka setup: no changes made")
    for line in result.get("changes", []):
        print(f"  {line}")
    readiness = result["readiness"]
    for warning in readiness["warnings"]:
        print(f"  warning: {warning}")
    selected = {item["provider"]: item for item in readiness["providers"] if item["selected"]}
    gemini = selected.get("gemini")
    if gemini and "flash" in gemini["selected_models"]:
        if gemini["detected"]:
            try:
                refresh = _refresh_gemini_catalogue()
                print(f"  Gemini discovery: registered {refresh['added_candidates']} candidate(s), updated {refresh['updated_candidates']} identity record(s)")
            except (DiscoveryError, ValueError, OSError) as exc:
                print(f"  Gemini remains selected; discovery prerequisite unresolved: {exc}")
        else:
            print("  Gemini remains selected; next action: install/authenticate the AGY harness, then run `eka models refresh --provider gemini`.")
    for provider, item in selected.items():
        if provider != "gemini" and not item["detected"]:
            print(f"  {item['label']} remains selected; harness not detected. Install/authenticate it using its provider-owned setup, then run `eka status`.")
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    conn = connect(); clauses=[]; values=[]
    if args.profile: clauses.append("profile=?"); values.append(args.profile)
    if args.provider: clauses.append("provider=?"); values.append(args.provider)
    if args.model: clauses.append("identity_key=?"); values.append(args.model)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = [dict(r) for r in conn.execute("SELECT run_id,started_at,ended_at,profile,provider,identity_key,status FROM runs" + where + " ORDER BY started_at DESC LIMIT ?", (*values, args.limit))]
    _json_or_text(rows, args.json); return 0


def cmd_spend(args: argparse.Namespace) -> int:
    conn = connect(); rows = [dict(r) for r in conn.execute("SELECT billing_mode, cost_source, currency, SUM(provider_reported_cost) AS provider_reported, SUM(calculated_cost) AS calculated, SUM(api_equivalent_cost) AS api_equivalent, COUNT(*) AS observations FROM cost_observations GROUP BY billing_mode,cost_source,currency")]
    _json_or_text({"semantics": {"actual": "provider_reported_cost", "calculated": "calculated_cost", "api_equivalent": "api_equivalent_cost", "unknown": "null; never inferred"}, "groups": rows}, args.json); return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    root, cat, prof = _paths(); integrity = True
    if default_db_path().exists():
        try: integrity = connect().execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        except sqlite3.DatabaseError: integrity = False
    legacy = state_dir() / "delegate_runs"
    legacy_private = True
    if legacy.exists():
        paths = [legacy, *legacy.rglob("*")]
        legacy_private = all((path.stat().st_mode & 0o777) == (0o700 if path.is_dir() else 0o600) for path in paths if not path.is_symlink())
    checks = {"config_dir_private": root.exists() and (root.stat().st_mode & 0o777) == 0o700 if root.exists() else True, "catalogue_readable": not cat.exists() or cat.is_file(), "profiles_readable": not prof.exists() or prof.is_file(), "control_file_pair_complete": cat.exists() == prof.exists(), "availability_config_readable": not (root / "config.toml").exists() or (root / "config.toml").is_file(), "ledger_parent": default_db_path().parent.exists(), "ledger_integrity": integrity, "legacy_evidence_private": legacy_private}
    _json_or_text(checks, args.json); return 0 if all(checks.values()) else 1


def cmd_run(args: argparse.Namespace) -> int:
    root, cat, prof = _paths(); profiles = {p.get("name"): p for p in _profiles(prof)}
    catalogue = load_catalogue(cat)
    dynamic = _named_route_profile(args.profile)
    if dynamic:
        profiles[args.profile], named_entry = dynamic
        catalogue = catalogue + [named_entry]
    if args.profile not in profiles:
        print(f"profile unavailable: {args.profile}; no automatic provider failover", file=sys.stderr); return 2
    try:
        intent = RunIntent(args.profile, args.provider, args.family, args.model, args.reasoning, args.harness, str(args.workspace) if args.workspace else None, str(args.prompt_file) if args.prompt_file else None, args.primary, args.timeout, args.task)
    except ValueError as exc:
        print(f"invalid task: {exc}", file=sys.stderr); return 2
    resolution = resolve(intent, profiles[args.profile], catalogue, availability=load_config()); record = resolution.as_dict(); run_id = uuid.uuid4().hex
    conn = connect(); record_run(conn, run_id, intent.__dict__, resolved=record.get("resolved"), status=resolution.state, resolution_reason=resolution.reason, provider=(resolution.candidate.provider if resolution.candidate else None), identity_key=(resolution.candidate.identity_key if resolution.candidate else None)); record_resolution(conn, run_id, intent.__dict__, record)
    if args.prompt_file and resolution.state != "resolved": _json_or_text({"run_id": run_id, **record}, args.json); return 3
    if args.prompt_file:
        if not args.workspace:
            _json_or_text({"run_id": run_id, **record, "execution": {"state": "workspace-required", "reason": "a writable/read-only workspace must be explicit"}}, args.json); return 3
        execution = execute(record, args.prompt_file, args.workspace, primary=args.primary, timeout_seconds=args.timeout)
        evidence = execution.get("evidence")
        if evidence:
            evidence_path = Path(str(evidence))
            metadata_path = evidence_path / "execution.json"
            digest = hashlib.sha256(metadata_path.read_bytes()).hexdigest() if metadata_path.is_file() else None
            finalize_run(conn, run_id, status="completed" if execution.get("state") == "completed" else "failed", raw_evidence_path=str(evidence_path), raw_evidence_sha256=digest)
        run_data = {"task": intent.task, "primary_provider": args.primary, "requested_profile": intent.profile, "requested_provider_model_id": args.model or (resolution.candidate.provider_model_id if resolution.candidate else None), "resolved_catalogue_identity_key": (resolution.candidate.identity_key if resolution.candidate else None), "resolved_family": (resolution.candidate.family if resolution.candidate else None), "resolved_generation": (resolution.candidate.generation if resolution.candidate else None), "reasoning_level": resolution.resolved_reasoning, "harness_name": resolution.resolved_harness, "harness_version": resolution.resolved_harness_version, "transport": resolution.transport, "invocation_basis": "explicit_profile", "effective_identity_status": "provider_reported" if execution.get("provider_reported_model_id") else "unavailable"}
        telemetry_warning = None
        try:
            persist_execution_observability(conn, run_id, run_data=run_data, execution=execution)
        except Exception as exc:  # optional enrichment must not invalidate provider result
            telemetry_warning = "usage telemetry could not be persisted"
            print(f"warning: {telemetry_warning}: {type(exc).__name__}", file=sys.stderr)
        payload = {"run_id": run_id, **record, "execution": execution}
        if telemetry_warning: payload["telemetry_warning"] = telemetry_warning
        _json_or_text(payload, args.json); return 0 if execution.get("state") == "completed" else 4
    _json_or_text({"run_id": run_id, **record}, args.json); return 0 if resolution.state == "resolved" else 3


def cmd_bench(args: argparse.Namespace) -> int:
    if getattr(args, "bench_action", None) == "bundle":
        try:
            result = create_review_bundle(args.experiment, output=args.output)
        except (OSError, ValueError) as exc:
            print(f"review bundle error: {exc}", file=sys.stderr); return 2
        _json_or_text(result, args.json); return 0
    if getattr(args, "bench_action", None) in {"status", "harnesses"}:
        records = current_registry(); validate_registry(records)
        if args.bench_action == "status":
            payload = {"execution_classes": ["ordinary", "public_characterization", "hidden_benchmark"], "harnesses": [{"name": item["name"], "version": item["version"], "installed": item["installed"], "eligibility": item["eligibility"], "reason": item["reason"]} for item in records]}
        else:
            payload = {"harnesses": records}
        _json_or_text(payload, args.json); return 0
    print("Ekalavya benchmark subsystem delegates to the existing benchmark.runner; no benchmark mutation is performed by this command.")
    return 0


def _usage_filters(args: argparse.Namespace) -> dict[str, str | None]:
    return {"task": getattr(args, "task", None), "profile": getattr(args, "profile", None), "provider": getattr(args, "provider", None), "model": getattr(args, "model", None)}


def cmd_usage(args: argparse.Namespace) -> int:
    if args.action == "prune" and not args.before:
        print("usage prune requires --before <OFFSET-AWARE-ISO-TIMESTAMP>; use usage reset --yes to clear all observability data", file=sys.stderr)
        return 2
    if args.action in {"delete", "reset"} and not args.yes:
        print(f"usage {args.action} requires --yes; use usage prune --before <OFFSET-AWARE-ISO-TIMESTAMP> for bounded deletion", file=sys.stderr)
        return 2
    if args.action != "prune" and args.before:
        print("usage --before is valid only with usage prune", file=sys.stderr)
        return 2
    if args.action == "prune":
        try:
            # Validate before opening the ledger: connecting can create or
            # migrate it, so a malformed cutoff must be a true no-op.
            args.before = normalize_cutoff(args.before)
        except ValueError as exc:
            print(f"usage prune: {exc}; provide an offset-aware ISO timestamp", file=sys.stderr)
            return 2
    conn = connect()
    if args.action in {"delete", "reset", "prune"}:
        before = args.before if args.action == "prune" else None
        try:
            result = clear_observability(conn, before=before)
        except ValueError as exc:
            print(f"usage {args.action}: {exc}", file=sys.stderr)
            return 2
        result["preserved"] = ["runs", "resolution_decisions", "promotion_events", "default_changes", "benchmark evidence", "retained responses", "catalogue/model provenance", "cost_observations"]
        _json_or_text(result, args.json); return 0
    if args.action == "export":
        output = export_usage(conn, fmt=args.format, period=args.period, by=args.by, filters=_usage_filters(args))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700); args.output.write_text(output, encoding="utf-8"); os.chmod(args.output, 0o600)
        else: print(output, end="" if output.endswith("\n") else "\n")
        return 0
    if args.refresh_quota:
        try: refresh_quotas(conn, live_vllm=True)
        except Exception as exc: print(f"warning: quota refresh unavailable: {type(exc).__name__}", file=sys.stderr)
    persisted = [dict(row) for row in conn.execute("SELECT * FROM quota_snapshots q WHERE q.id IN (SELECT MAX(id) FROM quota_snapshots GROUP BY provider,scope_kind,scope_key,resource_kind,window_kind,window_label)")]
    quotas = [public_snapshot(item) for item in collect_snapshots(live_vllm=False) + persisted]
    payload = build_usage(conn, period=args.period, by=args.by, filters=_usage_filters(args), quotas=quotas)
    if args.json: _json_or_text(payload, True)
    else:
        print("Ekalavya Usage")
        print("Provider headroom/capacity")
        for quota in quotas:
            value = quota["remaining_value"] if quota.get("remaining_value") is not None else quota.get("percentage")
            shown = f"{value}{quota.get('units') or ''}" if value is not None else "unknown"
            print(f"  {quota['provider']:<10} {quota['scope_kind']:<15} {shown:<12} {quota['capability']} ({quota['source']})")
        print("  unknown means telemetry unavailable; local_capacity is not provider account quota.")
        summary = payload["summary"]
        feedback = summary["feedback"]
        print(f"Local Ekalavya history — {args.period}")
        print(f"Runs: {summary['runs']}")
        print(f"Successful: {summary['successful_runs']}")
        print(f"Failed: {summary['failed_runs']}")
        if summary["aborted_runs"]:
            print(f"Aborted: {summary['aborted_runs']}")
        if summary["success_rate"] is not None:
            print(f"Success rate: {summary['success_rate']:.0%}")
        elif summary["non_terminal_runs"]:
            print("Success rate: unavailable (non-terminal runs present)")
        print(f"Reported tokens: {payload['summary']['reported_tokens']['value'] if payload['summary']['reported_tokens']['value'] is not None else 'telemetry unavailable'}")
        print(f"Telemetry coverage: {payload['coverage']['token_telemetry']['available']}/{payload['coverage']['token_telemetry']['eligible']} runs")
        print(f"Reasoning-token coverage: {payload['coverage']['reasoning_tokens']['available']}/{payload['coverage']['reasoning_tokens']['eligible']} runs")
        print(f"Feedback: {feedback['available']}/{summary['runs']} rated")
        for outcome in ("useful", "mixed", "not-useful"):
            print(f"  {outcome}: {feedback['counts'][outcome]}")
        if feedback["useful_rate"] is not None:
            print(f"  useful rate: {feedback['useful_rate']:.0%}")
        print("Scope: Local Ekalavya history is not total provider-account usage.")
        print("May exclude native same-provider agents, direct provider/web/desktop activity, other machines, and activity outside Ekalavya.")
        if args.by:
            for group in payload["groups"]: print(f"  {group['key']}: {group['summary']['runs']} run(s)")
    return 0


def cmd_insights(args: argparse.Namespace) -> int:
    conn = connect()
    persisted = [dict(row) for row in conn.execute("SELECT * FROM quota_snapshots q WHERE q.id IN (SELECT MAX(id) FROM quota_snapshots GROUP BY provider,scope_kind,scope_key,resource_kind,window_kind,window_label)")]
    payload = build_insights(conn, period=args.period, by=args.by, filters=_usage_filters(args), quotas=[public_snapshot(item) for item in collect_snapshots(live_vllm=False) + persisted])
    _json_or_text(payload, args.json)
    return 0


def cmd_feedback(args: argparse.Namespace) -> int:
    conn = connect()
    try:
        if not args.delete and not args.outcome:
            raise ValueError("feedback requires --outcome or --delete")
        if args.delete: delete_feedback(conn, args.run_id)
        else: set_feedback(conn, args.run_id, args.outcome)
    except ValueError as exc:
        print(str(exc), file=sys.stderr); return 2
    _json_or_text({"run_id": args.run_id, "deleted": bool(args.delete), "outcome": None if args.delete else args.outcome}, args.json); return 0


def _parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(prog="ekalavya", description="Ekalavya delegation control plane")
    p.add_argument("--version", action="version", version=__version__); sub=p.add_subparsers(dest="command", required=True)
    def common(q): q.add_argument("--json", action="store_true")
    q=sub.add_parser("status", help="network-free catalogue/profile overview"); q.add_argument("--primary"); q.add_argument("--live", action="store_true", help="perform explicit GET-only shared-route observability checks"); common(q); q.set_defaults(func=cmd_status)
    q=sub.add_parser("profiles", help="list stable capability profiles, not raw model IDs"); common(q); q.set_defaults(func=cmd_profiles)
    q=sub.add_parser("models", help="list catalogue identities; promotion is explicit"); q.add_argument("action", nargs="?", choices=["refresh", "promote"], default=None); q.add_argument("target", nargs="?"); q.add_argument("--source", type=Path); q.add_argument("--provider", choices=["gemini"]); q.add_argument("--basis", choices=sorted(PROMOTION_BASES - {"unspecified"}), default="unspecified"); q.add_argument("--promotion-reason"); q.add_argument("--set-default", action="store_true"); q.add_argument("--profile"); q.add_argument("--default-reasoning", choices=["low", "medium", "high"]); common(q); q.set_defaults(func=cmd_models)
    q=sub.add_parser("config", help="inspect or explicitly mutate user-owned availability configuration"); q.add_argument("action", nargs="?", choices=["list", "migrate", "enable", "disable", "enable-provider", "disable-provider", "enable-model", "disable-model"]); q.add_argument("target", nargs="?"); q.add_argument("--reason"); common(q); q.set_defaults(func=cmd_config)
    q=sub.add_parser("setup", help="interactive first-run integration and profile selection"); common(q); q.set_defaults(func=cmd_setup)
    q=sub.add_parser("history"); q.add_argument("--profile"); q.add_argument("--provider"); q.add_argument("--model"); q.add_argument("--limit", type=int, default=20); common(q); q.set_defaults(func=cmd_history)
    q=sub.add_parser("spend"); common(q); q.set_defaults(func=cmd_spend)
    q=sub.add_parser("doctor"); common(q); q.set_defaults(func=cmd_doctor)
    q=sub.add_parser("bench", help="read-only benchmark and harness inspection"); bench_sub=q.add_subparsers(dest="bench_action")
    b=bench_sub.add_parser("status", help="show harness eligibility by execution class"); common(b); b.set_defaults(func=cmd_bench)
    b=bench_sub.add_parser("harnesses", help="show detailed harness capabilities"); common(b); b.set_defaults(func=cmd_bench)
    b=bench_sub.add_parser("bundle", help="create an allowlisted private experiment review bundle"); b.add_argument("experiment"); b.add_argument("--output", type=Path); common(b); b.set_defaults(func=cmd_bench)
    q.set_defaults(func=cmd_bench, bench_action=None, json=False)
    q=sub.add_parser("run"); q.add_argument("profile"); q.add_argument("--provider"); q.add_argument("--family"); q.add_argument("--model"); q.add_argument("--reasoning"); q.add_argument("--harness"); q.add_argument("--workspace", type=Path); q.add_argument("--prompt-file", type=Path); q.add_argument("--primary"); q.add_argument("--timeout", type=int, default=None); q.add_argument("--task", default="unspecified"); q.add_argument("--json", action="store_true"); q.set_defaults(func=cmd_run)
    q=sub.add_parser("usage", help="inspect local observed usage and honest quota status"); q.add_argument("action", nargs="?", choices=["inspect", "export", "prune", "delete", "reset"]); q.add_argument("--period", default="7d"); q.add_argument("--by", choices=["task", "profile", "provider", "model"]); q.add_argument("--task"); q.add_argument("--profile"); q.add_argument("--provider"); q.add_argument("--model"); q.add_argument("--refresh-quota", action="store_true"); q.add_argument("--before"); q.add_argument("--yes", action="store_true", help="confirm complete observability reset"); q.add_argument("--format", choices=["json", "csv"], default="json"); q.add_argument("--output", type=Path); common(q); q.set_defaults(func=cmd_usage)
    q=sub.add_parser("insights", help="deterministic local usage insights"); q.add_argument("--period", default="7d"); q.add_argument("--by", choices=["task", "profile", "provider", "model"]); q.add_argument("--task"); q.add_argument("--profile"); q.add_argument("--provider"); q.add_argument("--model"); common(q); q.set_defaults(func=cmd_insights)
    q=sub.add_parser("feedback", help="record or delete current categorical feedback"); q.add_argument("run_id"); q.add_argument("--outcome", choices=["useful", "mixed", "not-useful"]); q.add_argument("--delete", action="store_true"); common(q); q.set_defaults(func=cmd_feedback)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "models": args.refresh = args.action == "refresh"
    return args.func(args)
