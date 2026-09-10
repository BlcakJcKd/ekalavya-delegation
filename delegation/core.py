"""Narrow, auditable delegated-consultation command construction and execution.

The default mode is read-only consultation.  It intentionally has no command
for implementation delegation yet: writes need an explicit future scope and
approval design rather than becoming an accidental default.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import routing
from ekalavya.deepseek import assert_deepseek_pro_exact
from .paths import log_root as _xdg_log_root
from .retention import persist_response, persist_text

READ_ONLY_CLAUDE_TOOLS: tuple[str, ...] = ("Read", "Glob", "Grep")
DEFAULT_TIMEOUT_SECONDS = 300

# Inherited-environment marker that identifies a process already running
# inside a delegated (child) context.  The wrapper rejects any invocation
# where this is already present and >= 1, and always sets it to "1" in a
# spawned delegate's environment.  This only protects the approved wrappers
# in this module; it cannot stop a delegate from invoking some other
# installed CLI directly.
DELEGATION_DEPTH_ENV = "AGENT_DELEGATION_DEPTH"

# Provenance-only identifier for whichever agent/process is acting as the
# primary owner (e.g. "codex", "claude-code", "manual"). Recorded for audit
# purposes; it is never used to make a safety decision.
DELEGATION_CALLER_ENV = "AGENT_DELEGATION_CALLER"
DEFAULT_CALLER = "unknown"


@dataclass(frozen=True)
class DelegateSpec:
    """Pinned configuration for one operational consultation delegate."""

    name: str
    executable: str
    model: str
    effort: str | None
    mode: str = "consult"


DELEGATES: dict[str, DelegateSpec] = {
    "flash": DelegateSpec("flash", "agy", "gemini-3.7-flash-medium", "medium"),
    "haiku": DelegateSpec("haiku", "claude", "claude-haiku-4-5-20251001", "medium"),
    "sonnet": DelegateSpec("sonnet", "claude", "claude-sonnet-5", "medium"),
    # Stable cross-provider routes for non-Codex primaries. Their inference
    # provider is OpenAI/Codex and their transport is the normal authenticated
    # `codex` CLI. The same-provider guard makes these native-only for a Codex
    # primary, which must use native Codex agents instead.
    "terra": DelegateSpec("terra", "codex", "gpt-5.6-terra", "medium"),
    "luna": DelegateSpec("luna", "codex", "gpt-5.6-luna", "medium"),
    # Experimental PAYG routes (see docs/PAYG_DELEGATES.md). Transport is the
    # `codex-deepseek`/`codex-minimax` launchers -- independently verified,
    # pre-existing Codex provider-profile wrappers that pin `--profile
    # deepseek`/`--profile minimax` and resolve their API key from the login
    # keyring; this package never handles that key. Effort is pinned to
    # "high" for both: DeepSeek's own catalog exposes low/high/max (no
    # "medium"), and MiniMax's profile is already pinned to "high" locally
    # -- neither is an invented flag.
    "deepseek-pro": DelegateSpec("deepseek-pro", "codex-deepseek", "deepseek-v4-pro", "high"),
    "deepseek-flash": DelegateSpec("deepseek-flash", "codex-deepseek", "deepseek-flash", "high"),
    "minimax-m3": DelegateSpec("minimax-m3", "codex-minimax", "MiniMax-M3", "high"),
}


def _read_only_instruction(task: str, workspace: Path) -> str:
    return f"""You are a read-only consultation delegate.

Assigned workspace: {workspace}
Allowed scope: inspect only files inside that workspace. Do not read parent
directories, home directories, credentials, benchmark private material, or
other workspaces. Do not edit, create, delete, rename, or execute files. Do
not use network tools. Do not invoke, request, or delegate to another agent or
CLI. Return a concise evidence-backed consultation in your final response,
including relevant file paths and uncertainties.

Task:
{task}
"""


def build_argv(spec: DelegateSpec, workspace: Path, task: str) -> list[str]:
    """Build a documented, non-bypass read-only consultation argv list.

    The final prompt remains one argv item.  The caller uses ``workspace`` as
    process CWD; Codex routes additionally receive a read-only sandbox and
    use the normal authenticated Codex CLI. The self-provider guard prevents
    a Codex primary from using that transport as a recursive same-provider hop.
    """
    prompt = _read_only_instruction(task, workspace)
    if spec.name in {"haiku", "sonnet"}:
        return [
            "claude", "--output-format", "json", "--no-session-persistence",
            "--safe-mode", "--permission-mode", "plan",
            "--tools", ",".join(READ_ONLY_CLAUDE_TOOLS),
            "--allowedTools", ",".join(READ_ONLY_CLAUDE_TOOLS),
            "--model", spec.model, "--effort", spec.effort or "medium",
            "-p", prompt,
        ]
    if spec.name == "flash":
        return [
            "agy", "--output-format", "json", "--mode", "plan", "--sandbox",
            "--model", spec.model, "--effort", spec.effort or "medium",
            "-p", prompt,
        ]
    if spec.executable in {"codex-deepseek", "codex-minimax"}:
        # These launchers already bake in `--profile deepseek`/`--profile
        # minimax` and keyring credential retrieval (see
        # docs/PAYG_DELEGATES.md); this argv only adds the same read-only,
        # non-interactive, structured-output shape as the reference `codex`
        # case below, plus an explicit model/effort pin so the route is
        # never left to the profile's own default.
        return [
            spec.executable, "exec", "--ephemeral", "--skip-git-repo-check",
            "--sandbox", "read-only", "--cd", str(workspace), "--json",
            "--model", spec.model, "--config", f'model_reasoning_effort="{spec.effort or "high"}"',
            prompt,
        ]
    if spec.name in {"codex", "terra", "luna"}:
        return [
            "codex", "exec", "--ephemeral", "--skip-git-repo-check",
            "--sandbox", "read-only", "--cd", str(workspace),
            "--model", spec.model, "--config", f'model_reasoning_effort="{spec.effort or "medium"}"',
            prompt,
        ]
    raise ValueError(f"unsupported delegate: {spec.name}")


def default_log_root() -> Path:
    """The XDG state directory's ``delegate_runs``, independent of ``__file__``.

    This resolves the same way whether ``delegation`` is imported from a
    source checkout or an installed site-packages copy, and never writes
    into the source repository or an installed package directory. Pass
    ``log_root=`` to override, e.g. for tests or a dev-time preference.
    """
    return _xdg_log_root()


def _check_recursion_guard() -> None:
    """Reject an invocation already running inside a delegated context.

    Only the inherited ``AGENT_DELEGATION_DEPTH`` environment marker is
    consulted; a caller-supplied argument could not be trusted for this.
    Any value >= 1, including a malformed one, is rejected fail-closed.
    """
    raw = os.environ.get(DELEGATION_DEPTH_ENV)
    if raw is None:
        return
    try:
        depth = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"recursive delegation rejected: malformed {DELEGATION_DEPTH_ENV}={raw!r}"
        ) from exc
    if depth >= 1:
        raise ValueError(
            f"recursive delegation rejected: {DELEGATION_DEPTH_ENV}={depth} indicates this "
            "process is already running inside a delegated context; a delegate must never "
            "invoke another delegate"
        )


def _resolve_caller(caller: str | None) -> str:
    return caller or os.environ.get(DELEGATION_CALLER_ENV) or DEFAULT_CALLER


def _check_self_provider_guard(delegate_name: str, primary: str | None) -> str | None:
    """Reject a delegate whose provider matches the declared primary's own.

    Distinct from :func:`_check_recursion_guard`: the recursion guard stops
    ``delegate -> wrapper -> another delegate`` using an inherited, hard-to-
    forge environment marker. This guard stops ``primary -> external call to
    its own provider`` using a caller-declared ``primary`` value -- it is a
    routing/policy nudge, not a security boundary, and it is only enforced
    when a primary is actually declared (fail-open on an absent value, since
    it cannot otherwise be verified). Returns the normalized primary (or
    ``None`` if undeclared) for the caller to record.
    """
    normalized = routing.normalize_primary(primary)
    if normalized is None or normalized == "manual":
        return normalized
    if routing.ROUTE_PROVIDER.get(delegate_name) == normalized:
        raise ValueError(
            f"same-provider external delegation disabled: primary={normalized!r} cannot "
            f"externally invoke its own provider via Ekalavya; use a native "
            "agent/subagent capability instead"
        )
    return normalized


def _validate_scope(workspace: Path) -> Path:
    resolved = workspace.expanduser().resolve()
    marker = resolved / ".delegation-scope.json"
    if not resolved.is_dir():
        raise ValueError(f"workspace does not exist or is not a directory: {resolved}")
    if any(path.is_symlink() for path in resolved.rglob("*")):
        raise ValueError("read-only consultation workspace must not contain symlinks")
    if not marker.is_file():
        raise ValueError(
            f"read-only consultation requires scope marker: {marker}; "
            "create it only in a dedicated, least-privilege workspace"
        )
    try:
        scope = json.loads(marker.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid scope marker JSON: {marker}") from exc
    if not isinstance(scope, dict) or scope.get("mode") != "read-only":
        raise ValueError("scope marker must contain {\"mode\": \"read-only\"}")
    return resolved


def _record_path(log_root: Path, delegate: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return log_root / f"{timestamp}-{delegate}-{uuid.uuid4().hex[:10]}"


def _capture_text(value: object) -> str:
    """Normalize subprocess captures without putting raw values in metadata."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value if isinstance(value, str) else str(value)


def _validated_log_root(log_root: Path, workspace: Path) -> Path:
    """Resolve a private run root outside both the scope and Git trees."""
    resolved = log_root.expanduser().resolve()
    if resolved == workspace or resolved.is_relative_to(workspace):
        raise ValueError("log root must be outside the consulted workspace")
    if any((ancestor / ".git").exists() for ancestor in (resolved, *resolved.parents)):
        raise ValueError("log root must be outside a repository")
    return resolved


def run_consultation(
    delegate_name: str,
    workspace: Path,
    task: str,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    log_root: Path | None = None,
    caller: str | None = None,
    primary: str | None = None,
    now=None,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[int, Path]:
    """Run exactly one read-only delegate and retain an auditable local record.

    The environment is passed through only to support the user's authenticated
    CLI sessions; it is never serialized.  The spawned delegate's environment
    always has ``AGENT_DELEGATION_DEPTH`` set to ``"1"`` so an approved
    wrapper cannot be recursively invoked from inside it.  Logs are outside
    ``workspace`` so the administrator's evidence collection does not modify
    the consulted project.  A timeout is represented as exit code 124.

    ``primary``, if given, is checked by the self-provider guard (see
    :func:`_check_self_provider_guard`) and recorded for audit; it is a
    distinct mechanism from the recursion-depth guard above.
    """
    _check_recursion_guard()
    if delegate_name not in DELEGATES:
        raise ValueError(f"unknown delegate: {delegate_name}")
    if delegate_name == "deepseek-pro":
        assert_deepseek_pro_exact(now=now)
    if timeout_seconds <= 0:
        raise ValueError("timeout must be positive")
    normalized_primary = _check_self_provider_guard(delegate_name, primary)
    resolved_caller = _resolve_caller(caller)
    workspace = _validate_scope(workspace)
    spec = DELEGATES[delegate_name]
    executable = shutil.which(spec.executable)
    if not executable:
        raise RuntimeError(f"delegate executable is unavailable: {spec.executable}")
    argv = build_argv(spec, workspace, task)
    resolved_log_root = _validated_log_root(log_root or default_log_root(), workspace)
    record_dir = _record_path(resolved_log_root, delegate_name)
    record_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    # The run directory is private response state, including when the caller's
    # umask is permissive.  ``persist_text`` repeats this check before each
    # capture so a replaced/symlinked directory fails closed.
    os.chmod(record_dir, 0o700)
    prompt = argv[-1]
    persist_text(record_dir, "prompt.md", prompt)
    started_at = datetime.now(timezone.utc).isoformat()
    begun = time.monotonic()
    stdout = ""
    stderr = ""
    timed_out = False
    child_env = dict(os.environ)
    child_env[DELEGATION_DEPTH_ENV] = "1"
    try:
        completed = run(
            argv, cwd=workspace, text=True, capture_output=True, timeout=timeout_seconds,
            env=child_env,
        )
        exit_code = completed.returncode
        stdout = _capture_text(completed.stdout)
        stderr = _capture_text(completed.stderr)
    except subprocess.TimeoutExpired as exc:
        exit_code = 124
        timed_out = True
        stdout = _capture_text(exc.stdout)
        stderr = _capture_text(exc.stderr)
    wall_seconds = time.monotonic() - begun

    provider_success = exit_code == 0
    inference_occurred = provider_success and bool(stdout.strip())
    response_status = "text-returned" if stdout.strip() else "empty-response"
    error_category: str | None = None
    response_metadata: dict[str, object] = {
        "response_recorded": False,
        "response_file": None,
        "response_length_bytes": 0,
        "response_sha256": None,
    }
    try:
        response_metadata = persist_response(record_dir, stdout)
        if stdout.strip() and response_metadata.get("response_recorded") is not True:
            raise OSError("response persistence did not confirm a recorded response")
    except OSError:
        # The child has already completed successfully, but returning its text
        # without a durable copy would recreate the incident this contract is
        # designed to prevent.  Keep the provider/inference facts separate and
        # make the wrapper failure deterministic; do not issue a retry.
        if stdout.strip():
            response_status = "response-retention-failure"
            error_category = "response-retention"
            response_metadata = {
                "response_recorded": False,
                "response_file": None,
                "response_length_bytes": 0,
                "response_sha256": None,
            }
            if provider_success:
                exit_code = 1
            stderr = (
                f"{stderr.rstrip(chr(10))}{chr(10) if stderr else ''}"
                "delegation response-retention failure: textual response was not "
                "durably saved; no retry performed\n"
            )
        else:
            # There is no textual result to retain.  Preserve the established
            # empty-response semantics even if writing the empty capture fails.
            response_metadata = {
                "response_recorded": False,
                "response_file": None,
                "response_length_bytes": 0,
                "response_sha256": None,
            }
    persist_text(record_dir, "stderr.txt", stderr)
    record = {
        "delegate": spec.name,
        "mode": spec.mode,
        "provider": routing.ROUTE_PROVIDER.get(
            spec.name, "codex" if spec.executable == "codex" else spec.executable
        ),
        "transport": routing.ROUTE_TRANSPORT.get(spec.name, spec.executable),
        "caller": resolved_caller,
        "declared_primary": normalized_primary or "not-declared",
        "requested_model": spec.model,
        "requested_effort": spec.effort,
        "workspace": str(workspace),
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": wall_seconds,
        "timeout_seconds": timeout_seconds,
        "timed_out": timed_out,
        "exit_code": exit_code,
        "provider_success": provider_success,
        "inference_occurred": inference_occurred,
        "response_status": response_status,
        "error_category": error_category,
        **response_metadata,
        "argv": argv[:-1] + ["<PROMPT>"],
        "prompt_file": "prompt.md",
        "stdout_file": "stdout.txt",
        "stderr_file": "stderr.txt",
        "permission_strategy": (
            "Claude plan mode with Read/Glob/Grep only"
            if spec.name in {"haiku", "sonnet"}
            else "Antigravity plan mode with sandbox"
            if spec.name == "flash"
            else "Codex exec read-only sandbox"
            if spec.executable == "codex"
            else "provider launcher read-only sandbox"
        ),
        "environment_captured": False,
        "recursive_delegation_enabled": False,
        "child_delegation_depth": 1,
    }
    persist_text(record_dir, "execution.json", json.dumps(record, indent=2, sort_keys=True) + "\n")
    return exit_code, record_dir
