"""Read-only, lexicographic Ekalavya route recommendations."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import sqlite3
from pathlib import Path
from typing import Any

from delegation import routing

from .config import load_profiles
from .readiness import profile_readiness
from .resolver import SAME_PROVIDER_NATIVE, resolve
from .route_evidence import applicable_records, load_registry
from .schema import RunIntent, normalize_task
from .targets import named_route_profile


ROUTE_SCHEMA_VERSION = 1
TERMINAL = {"success", "failure", "aborted"}


def _reason_code(result: Any, config: dict[str, Any], profile: str) -> str:
    # The resolver is authoritative for lifecycle, route, harness, and
    # availability classification.  Keep the prose reason display-only.
    if getattr(result, "reason_code", None):
        return result.reason_code
    if result.state == "harness-unavailable":
        return "harness-unavailable"
    if result.state == "invalid-reasoning":
        return "unsupported-reasoning"
    if result.state == "unavailable":
        return "missing-route"
    return result.state.replace("_", "-")


def _history(db_path: Path | None, task: str) -> dict[tuple[str, str | None], dict[str, Any]]:
    """Load only aggregate-safe ordinary delegation observations, read-only."""
    if db_path is None or not db_path.is_file():
        return {}
    try:
        conn = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT o.requested_profile,o.resolved_catalogue_identity_key,o.execution_status,"
            "o.wall_seconds,f.outcome FROM run_observability o JOIN runs r ON r.run_id=o.run_id "
            "LEFT JOIN user_feedback f ON f.run_id=o.run_id "
            "WHERE o.event_kind='delegation' AND r.evaluation_class='unknown' AND o.task=?",
            (task,),
        ).fetchall()
    except sqlite3.DatabaseError:
        return {}
    finally:
        if "conn" in locals():
            conn.close()
    grouped: dict[tuple[str, str | None], list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault((str(row["requested_profile"]), row["resolved_catalogue_identity_key"]), []).append(row)
    result: dict[tuple[str, str | None], dict[str, Any]] = {}
    for key, values in grouped.items():
        outcomes = Counter(str(row["outcome"]) for row in values if row["outcome"] in {"useful", "mixed", "not-useful"})
        terminals = [row for row in values if row["execution_status"] in TERMINAL]
        successful = [row for row in values if row["execution_status"] == "success"]
        walls = [float(row["wall_seconds"]) for row in successful if isinstance(row["wall_seconds"], (int, float))]
        ordered_walls = sorted(walls)
        median = None
        if len(ordered_walls) == len(successful) and len(ordered_walls) >= 5:
            middle = len(ordered_walls) // 2
            median = ordered_walls[middle] if len(ordered_walls) % 2 else (ordered_walls[middle - 1] + ordered_walls[middle]) / 2
        result[key] = {
            "runs": len(values), "rated": sum(outcomes.values()), "feedback": dict(outcomes),
            "terminal": len(terminals), "successful": len(successful),
            "success_rate": (sum(row["execution_status"] == "success" for row in terminals) / len(terminals)) if len(terminals) == len(values) and terminals else None,
            "wall_coverage": len(walls) / len(successful) if successful else 0.0,
            "median_wall_seconds": median,
        }
    return result


def _quota_exclusion(candidate: dict[str, Any], reserves: dict[str, Any], snapshots: list[dict[str, Any]], now: datetime) -> dict[str, Any] | None:
    for name, reserve in reserves.items():
        if reserve["provider"] != candidate["provider"]:
            continue
        matching = [item for item in snapshots if all(item.get(key) == reserve[key] for key in ("provider", "scope_kind", "resource_kind", "window_kind"))]
        for item in matching:
            remaining, limit = item.get("remaining_value"), item.get("limit_value")
            reset_at = item.get("reset_at")
            details = item.get("details_json", item.get("details", {}))
            if isinstance(details, str):
                try:
                    details = json.loads(details)
                except ValueError:
                    details = {}
            fresh_until = details.get("fresh_until") if isinstance(details, dict) else None
            try:
                reset = datetime.fromisoformat(str(reset_at).replace("Z", "+00:00")) if reset_at else None
                freshness = datetime.fromisoformat(str(fresh_until).replace("Z", "+00:00")) if fresh_until else reset
            except ValueError:
                reset = freshness = None
            # Adapters must supply either a still-open reset window or an
            # explicit fresh_until contract; absence is honest unknown.
            if not isinstance(remaining, (int, float)) or not isinstance(limit, (int, float)) or limit <= 0 or freshness is None or freshness <= now:
                continue
            fraction = float(remaining) / float(limit)
            if fraction < reserve["minimum_remaining_fraction"]:
                return {"code": "quota-reserve", "reserve": name, "remaining_fraction": fraction, "minimum_remaining_fraction": reserve["minimum_remaining_fraction"], "observed_at": item.get("observed_at"), "reset_at": reset_at, "fresh_until": fresh_until}
    return None


def _winner_by_feedback(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates or not all(item["history"]["rated"] >= 5 for item in candidates):
        return None
    def value(item: dict[str, Any]) -> tuple[float, int]:
        feedback = item["history"]["feedback"]; rated = item["history"]["rated"]
        return ((feedback.get("useful", 0) - feedback.get("not-useful", 0)) / rated, feedback.get("useful", 0))
    values = {item["target"]: value(item) for item in candidates}
    best = max(values.values())
    winners = [item for item in candidates if values[item["target"]] == best]
    return winners[0] if len(winners) == 1 else None


def _winner_by_operational(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str | None]:
    if candidates and all(item["history"]["terminal"] >= 5 and item["history"]["success_rate"] is not None for item in candidates):
        highest = max(item["history"]["success_rate"] for item in candidates)
        winners = [item for item in candidates if item["history"]["success_rate"] == highest]
        if len(winners) == 1:
            return winners[0], "operational-success"
        candidates = winners
    if candidates and all(item["history"]["successful"] >= 5 and item["history"]["wall_coverage"] == 1.0 and item["history"]["median_wall_seconds"] is not None for item in candidates):
        lowest = min(item["history"]["median_wall_seconds"] for item in candidates)
        winners = [item for item in candidates if item["history"]["median_wall_seconds"] == lowest]
        if len(winners) == 1:
            return winners[0], "operational-latency"
    return None, None


def _winner_by_benchmark(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Rank only a complete explicitly comparable portfolio; absence is neutral."""
    if not candidates:
        return None
    groups = None
    for candidate in candidates:
        available = {
            item["comparison_group_id"] for item in candidate["benchmark"]
            if item["comparability"]["correctness_comparable"] and item["comparability"]["scope_comparable"]
        }
        groups = available if groups is None else groups & available
    if not groups:
        return None
    group = sorted(groups)[0]
    values: dict[str, tuple[float, float]] = {}
    for candidate in candidates:
        record = next(item for item in candidate["benchmark"] if item["comparison_group_id"] == group)
        outcome = record["outcome"]
        values[candidate["target"]] = (float(outcome["correctness_rate"]), float(outcome["scope_compliance_rate"]))
    best = max(values.values())
    winners = [item for item in candidates if values[item["target"]] == best]
    return winners[0] if len(winners) == 1 else None


def recommend(*, task: str, primary: str | None, config: dict[str, Any], profiles: list[dict[str, Any]], catalogue: list[dict[str, Any]], observed_availability: dict[str, dict[str, Any]], db_path: Path | None = None, quota_snapshots: list[dict[str, Any]] | None = None, registry: dict[str, Any] | None = None, now: datetime | None = None) -> dict[str, Any]:
    """Return a deterministic recommendation without provider/model activity."""
    task = normalize_task(task)
    # A direct library caller gets the same once-per-calculation behavior as
    # the CLI, while the CLI explicitly loads and passes its validated object.
    registry = registry if registry is not None else load_registry()
    normalized_primary = routing.normalize_primary(primary)
    now = now or datetime.now(timezone.utc)
    policy = config.get("routing", {}).get("preferences", {}).get(task, {})
    preferred = list(policy.get("preferred_targets", []))
    allowed = set(policy.get("allowed_targets", []))
    excluded = set(policy.get("excluded_targets", []))
    history = _history(db_path, task)
    candidates: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    warnings: list[str] = []

    native_eligible = normalized_primary in SAME_PROVIDER_NATIVE
    if primary is None or normalized_primary is None:
        warnings.append("primary-unknown: primary-native was not evaluated; pass --primary when known")
    elif not native_eligible:
        warnings.append(f"primary-native is not defined for declared primary {normalized_primary}")
    if native_eligible:
        native = {"target": "primary-native", "kind": "primary-native", "provider": normalized_primary, "profile": None, "identity": None, "history": {"runs": 0, "rated": 0, "feedback": {}, "terminal": 0, "successful": 0, "success_rate": None, "wall_coverage": 0.0, "median_wall_seconds": None}, "benchmark": []}
        if "primary-native" in excluded:
            exclusions.append({"target": "primary-native", "code": "user-excluded"})
        elif allowed and "primary-native" not in allowed:
            exclusions.append({"target": "primary-native", "code": "not-in-allowlist"})
        else:
            candidates.append(native)

    def inspect_profile(name: str, profile: dict[str, Any], entries: list[dict[str, Any]], *, is_vllm: bool = False) -> None:
        target = f"vllm:{name[5:]}" if is_vllm else f"profile:{name}"
        result = resolve(RunIntent(name, task=task), profile, entries, availability=config, now=now)
        if result.state != "resolved" or result.candidate is None:
            exclusions.append({"target": target, "code": _reason_code(result, config, name), "detail": result.reason})
            return
        if normalized_primary and result.candidate.provider == normalized_primary and normalized_primary in SAME_PROVIDER_NATIVE:
            exclusions.append({"target": target, "code": "native-only", "detail": f"same provider as declared primary {normalized_primary}"})
            return
        if not is_vllm:
            readiness = profile_readiness(name, profiles, catalogue, observed_availability)
            if readiness.get("harness_ready") != "ready":
                exclusions.append({"target": target, "code": "availability-unconfirmed" if readiness.get("harness_ready") == "unconfirmed" else "harness-unavailable", "detail": readiness.get("readiness_reason")})
                return
        if target in excluded:
            exclusions.append({"target": target, "code": "user-excluded"})
            return
        if allowed and target not in allowed:
            exclusions.append({"target": target, "code": "not-in-allowlist"})
            return
        identity_key = result.candidate.identity_key
        item = {"target": target, "kind": "vllm-route" if is_vllm else "ekalavya-profile", "provider": result.candidate.provider, "profile": name, "identity": result.candidate.as_dict() | {"identity_key": identity_key, "reasoning": result.resolved_reasoning, "harness": result.resolved_harness, "execution_route": result.execution_route}, "history": history.get((name, identity_key), {"runs": 0, "rated": 0, "feedback": {}, "terminal": 0, "successful": 0, "success_rate": None, "wall_coverage": 0.0, "median_wall_seconds": None}), "benchmark": []}
        item["benchmark"] = [record for record in applicable_records(task, target, registry=registry) if record["identity"].get("provider_model_id") == result.candidate.provider_model_id]
        reserve = _quota_exclusion(item, config.get("routing", {}).get("reserves", {}), quota_snapshots or [], now)
        if reserve:
            exclusions.append({"target": target, **reserve})
            return
        candidates.append(item)

    for profile in profiles:
        if isinstance(profile.get("name"), str):
            inspect_profile(str(profile["name"]), profile, catalogue)
    for name in sorted(config.get("vllm", {})):
        dynamic = named_route_profile(f"vllm:{name}")
        if dynamic is None:
            exclusions.append({"target": f"vllm:{name}", "code": "missing-route"})
        else:
            inspect_profile(dynamic[0]["name"], dynamic[0], [dynamic[1]], is_vllm=True)

    candidates.sort(key=lambda item: (item["provider"], item["target"], (item["identity"] or {}).get("identity_key", "")))
    recommendation: dict[str, Any] | None = None
    basis = "none-eligible"
    if preferred:
        for rank, target in enumerate(preferred, start=1):
            match = next((item for item in candidates if item["target"] == target), None)
            if match is not None:
                recommendation = match | {"preference_rank": rank}
                basis = "explicit-preference"
                break
        if recommendation is None:
            basis = "no-preferred-target"
            warnings.append("no preferred target is currently eligible")
    elif native_eligible and any(item["target"] == "primary-native" for item in candidates):
        recommendation = next(item for item in candidates if item["target"] == "primary-native")
        basis = "default-primary-native"
    else:
        comparable = [item for item in candidates if item["kind"] != "primary-native"]
        winner = _winner_by_feedback(comparable)
        if winner is not None:
            recommendation, basis = winner, "local-feedback"
        else:
            winner = _winner_by_benchmark(comparable)
            if winner is not None:
                recommendation, basis = winner, "benchmark-comparison"
            else:
                winner, tier = _winner_by_operational(comparable)
                if winner is not None:
                    recommendation, basis = winner, str(tier)
                elif len(comparable) == 1:
                    recommendation, basis = comparable[0], "sole-eligible-target"
                elif comparable:
                    basis = "tie"

    alternatives = [item for item in candidates if recommendation is None or item["target"] != recommendation["target"]]
    kind = recommendation["kind"] if recommendation else ("none-eligible" if not candidates else "no-defensible-distinction")
    trace = [{"target": item["target"], "eligible": True, "history": item["history"], "benchmark_records": [record["evidence_id"] for record in item["benchmark"]]} for item in candidates]
    trace.extend({"target": item["target"], "eligible": False, "exclusion": item["code"], "detail": item.get("detail")} for item in exclusions)
    return {"schema_version": ROUTE_SCHEMA_VERSION, "task": task, "primary": {"requested": primary, "normalized": normalized_primary}, "recommendation": recommendation, "recommendation_kind": kind, "decision_basis": basis, "decisive_tier": basis, "alternatives": alternatives, "excluded": sorted(exclusions, key=lambda item: (item["target"], item["code"])), "evidence": {"policy": policy, "quota_snapshot_count": len(quota_snapshots or []), "history_scope": "local Ekalavya ordinary delegations only", "benchmark_registry": "route_evidence.v1"}, "warnings": warnings, "executed": False, "trace": trace}
