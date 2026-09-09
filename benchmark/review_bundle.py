"""Allowlisted, privacy-scanned review bundles for private experiments."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ekalavya.ledger import default_state_dir


ROOT_ARTIFACTS = (
    "REPORT.md", "run-summary.json", "discovery.json", "plot-metadata.json",
    "AUDIT_REPORT.md", "CORRECTED_REPORT.md", "telemetry-semantics.md",
    "token-semantics.md", "task-check-matrix.md", "task-check-matrix.csv",
    "configuration-summary.json", "VALIDATION_REPORT.md", "calibration-summary.json",
)
OPTIONAL_PLOTS = (
    "reasoning-correctness.png", "reasoning-wall.png", "score-vs-wall.png",
    "tokens-vs-correctness.png", "baseline-vs-final.png", "delta-by-configuration.png",
    "final-vs-wall.png", "final-vs-tokens.png",
)
OPTIONAL_DIRS = ("provenance", "validation", "task-specifications", "verifier-contracts", "edit-scopes", "calibration-evidence")
PUBLIC_SNAPSHOT_DIR = "baseline-task-snapshots"
REQUESTED_SEMANTICS = {
    "request_metric_semantics", "tool_event_telemetry", "token_metric_semantics",
}
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|password|secret|credential)\s*[:=]\s*[\"']?([A-Za-z0-9_./+=:-]{8,})"
)
_PRIVATE_URL = re.compile(r"(?i)https?://[^\s/]+(?:/[^\s]*)?")
# Ekalavya's public shell tooling targets POSIX environments.  These are the
# home-root forms used by Linux and macOS; keep the replacement deliberately
# limited to an absolute home prefix so the exported relative suffix remains.
_USER_HOME_PREFIX = re.compile(r"(?<![A-Za-z0-9_])(?:/home|/Users)/[A-Za-z0-9._-]+(?=/|$)")
_ABSOLUTE_HOME = re.compile(r"(?<![A-Za-z0-9_])(?:/home|/Users)/[A-Za-z0-9._-]+(?:/[^\s\"']*)?")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sanitize(data: bytes, state_root: Path) -> tuple[bytes, bool]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data, False
    replacements = {
        str(state_root): "<EKALAVYA_STATE>",
        str(Path.home()): "<USER_HOME>",
    }
    changed = False
    for old, new in replacements.items():
        if old in text:
            text = text.replace(old, new)
            changed = True
    sanitized, count = _USER_HOME_PREFIX.subn("<USER_HOME>", text)
    if count:
        text = sanitized
        changed = True
    return text.encode("utf-8"), changed


def privacy_scan(root: Path) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for match in _SECRET_ASSIGNMENT.finditer(text):
            value = match.group(1).lower()
            if value not in {"null", "none", "false", "unavailable"}:
                findings.append({"file": path.relative_to(root).as_posix(), "kind": "secret-like assignment"})
        for match in _PRIVATE_URL.finditer(text):
            if not match.group(0).startswith(("https://example.", "http://example.")):
                findings.append({"file": path.relative_to(root).as_posix(), "kind": "endpoint-like URL"})
        if _ABSOLUTE_HOME.search(text):
            findings.append({"file": path.relative_to(root).as_posix(), "kind": "unsanitized absolute home path"})
    return {"status": "pass" if not findings else "fail", "findings": findings}


def _git_identity() -> str | None:
    try:
        import subprocess
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _attempt_counts(state: Path) -> dict[str, int]:
    evidence = sorted({path for directory in ("evidence", "calibration-evidence") for path in (state / directory).glob("*.json")})
    attempts = completed = timeouts = 0
    for path in evidence:
        try:
            item = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        attempts += 1
        if item.get("timed_out") is True or item.get("status") == "explicit_timeout":
            timeouts += 1
        elif item.get("status") == "completed" or (item.get("exit_code") == 0 and item.get("status") not in {"evaluator_tampering", "harness_failure", "malformed_evaluator"}):
            completed += 1
    stop_record = state / "validation" / "calibration-stop.json"
    if stop_record.is_file():
        try:
            item = json.loads(stop_record.read_text())
        except (OSError, ValueError):
            item = {}
        if item.get("status") == "harness_failure":
            attempts += 1
    return {"attempts": attempts, "completed": completed, "failed": attempts - completed - timeouts, "timeouts": timeouts}


def _evaluation_class(state: Path, experiment: str) -> str:
    summary = state / "run-summary.json"
    if summary.is_file():
        try:
            value = json.loads(summary.read_text()).get("evaluation_class")
            if value:
                return str(value)
        except (OSError, ValueError):
            pass
    return "public_characterization" if experiment.startswith("public-characterization-") else "unknown"


def _experiment_git_identity(state: Path) -> str | None:
    summary = state / "run-summary.json"
    if summary.is_file():
        try:
            value = json.loads(summary.read_text()).get("suite_git_sha")
            if value:
                return str(value)
        except (OSError, ValueError):
            pass
    return None


def create_review_bundle(
    experiment: str,
    *,
    output: Path | None = None,
    state_dir: Path | None = None,
) -> dict[str, Any]:
    """Create a portable review copy from an explicit artifact allowlist."""
    state = (state_dir or default_state_dir() / "experiments" / experiment).resolve()
    if not state.is_dir():
        raise FileNotFoundError(f"experiment state not found: {state}")
    bundle = (output or state / "review-bundle").resolve()
    archive = bundle.with_suffix(".zip")
    if bundle == state or bundle == state.parent:
        raise ValueError("review bundle output must be a child or separate directory")
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    if archive.exists():
        archive.unlink()

    candidates: list[Path] = []
    for name in ROOT_ARTIFACTS + OPTIONAL_PLOTS:
        path = state / name
        if path.is_file():
            candidates.append(path)
    for directory in OPTIONAL_DIRS:
        source = state / directory
        if source.is_dir():
            candidates.extend(p for p in sorted(source.rglob("*")) if p.is_file() and "__pycache__" not in p.parts)
    if _evaluation_class(state, experiment) == "public_characterization":
        source = state / PUBLIC_SNAPSHOT_DIR
        if source.is_dir():
            candidates.extend(p for p in sorted(source.rglob("*")) if p.is_file() and "__pycache__" not in p.parts)
    evidence = state / "evidence"
    if evidence.is_dir():
        candidates.extend(p for p in sorted(evidence.glob("*.json")) if p.is_file())
    if not candidates:
        raise ValueError(f"no approved review artifacts found in {state}")

    sanitized = False
    included: list[str] = []
    for source in sorted(set(candidates)):
        relative = source.relative_to(state)
        destination = bundle / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        data, changed = _sanitize(source.read_bytes(), state)
        destination.write_bytes(data)
        sanitized = sanitized or changed
        included.append(relative.as_posix())

    counts = _attempt_counts(state)
    payload_files = sorted(included)
    correction = None
    provenance = bundle / "provenance" / "correction-summary.json"
    if provenance.is_file():
        try:
            correction = json.loads(provenance.read_text())
        except ValueError:
            correction = {"status": "malformed correction summary"}
    manifest = {
        "experiment": experiment,
        "evaluation_class": _evaluation_class(state, experiment),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_code_git_identity": (correction or {}).get("corrected_suite_sha") or _experiment_git_identity(state) or _git_identity(),
        "report_generation_code_identity": _git_identity(),
        **counts,
        "included_files": payload_files + ["MANIFEST.md", "manifest.json"],
        "sha256": {name: _sha256(bundle / name) for name in payload_files},
        "sha256_exclusions": {
            "manifest.json": "self-referential metadata; its digest cannot be recorded inside itself",
        },
        "privacy_scan": None,
        "sanitization_applied": sanitized,
        "workspaces_included": False,
        "raw_provider_traces_included": False,
        "credentials_included": False,
        "configuration_included": False,
        "ledger_included": False,
        "allowlist": {"root_artifacts": list(ROOT_ARTIFACTS), "optional_plots": list(OPTIONAL_PLOTS), "optional_directories": list(OPTIONAL_DIRS), "public_snapshot_directory": PUBLIC_SNAPSHOT_DIR, "evidence_glob": "evidence/*.json"},
    }
    if correction is not None:
        manifest["provenance_correction"] = {key: correction.get(key) for key in ("originally_recorded_suite_sha", "corrected_suite_sha", "correction_reason", "correction_timestamp", "ledger_correction_id")}
    scan = privacy_scan(bundle)
    manifest["privacy_scan"] = scan
    (bundle / "MANIFEST.md").write_text(_manifest_markdown(manifest) + "\n")
    manifest["sha256"]["MANIFEST.md"] = _sha256(bundle / "MANIFEST.md")
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(p for p in bundle.rglob("*") if p.is_file()):
            handle.write(path, path.relative_to(bundle).as_posix())
    return {"bundle": str(bundle), "archive": str(archive), "manifest": manifest}


def _manifest_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Private experiment review bundle", "", f"- Experiment: `{manifest['experiment']}`",
        f"- Evaluation class: `{manifest['evaluation_class']}`",
        f"- Generated: `{manifest['generated_at']}`",
        f"- Experiment code Git identity: `{manifest.get('experiment_code_git_identity') or 'null'}`",
        f"- Attempts/completed/failed/timeouts: `{manifest['attempts']}/{manifest['completed']}/{manifest['failed']}/{manifest['timeouts']}`",
        f"- Sanitization applied: `{str(manifest['sanitization_applied']).lower()}`",
        f"- Workspaces included: `{str(manifest['workspaces_included']).lower()}`",
        f"- Raw provider traces included: `{str(manifest['raw_provider_traces_included']).lower()}`",
        f"- Credentials/configuration/ledger included: `false/false/false`", "",
        "## Included files", "",
    ]
    for name in manifest.get("included_files", []):
        lines.append(f"- `{name}` — `{manifest['sha256'].get(name, 'generated after manifest pass')}`")
    lines += ["", "## Privacy scan", "", f"Status: `{(manifest.get('privacy_scan') or {}).get('status', 'pending')}`"]
    return "\n".join(lines)
