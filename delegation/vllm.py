"""Bounded direct HTTP consultation for named OpenAI-compatible vLLM routes.

This module deliberately does not use a coding-agent CLI as transport.  A
named provider is loaded from the user's XDG configuration, a single minimal
Chat Completions request is made, and the result is returned through the same
stdout/stderr/evidence contract as the existing consultation runner.

Provider configuration and reliability records are machine-local.  Secrets
are resolved only at request time from an environment variable or the Ubuntu
login keyring and are never written to evidence, exceptions, or issue logs.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from .core import (
    DEFAULT_TIMEOUT_SECONDS, _check_recursion_guard, _record_path, _validated_log_root,
    _validate_scope, default_log_root,
)
from .paths import config_dir, state_dir
from .retention import persist_response, persist_text

VLLM_CONFIG_FILENAME = "vllm.toml"
# Kept for compatibility with the existing user-owned path.  New successful
# runs are not written here; the normal delegate_runs execution record is the
# audit log for all attempts.
VLLM_ISSUE_FILENAME = "vllm_issues.jsonl"
VLLM_LOCK_FILENAME = "vllm.request.lock"
DEFAULT_MAX_TOKENS = 512
DEFAULT_MAX_TOKENS_CAP = 512
HARD_MAX_TOKENS = 8192
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_OBSERVABILITY_BYTES = 4 * 1024 * 1024
MAX_PROVIDER_REPORTED_MODEL_ID = 256
KEYRING_LOOKUP_TIMEOUT_SECONDS = 10
OBSERVABILITY_TIMEOUT_SECONDS = 5
LIVE_STATUS_INTERVAL_SECONDS = 2.0
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_ENV_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class VLLMConfigurationError(ValueError):
    """A safe, user-actionable local provider configuration error."""


class VLLMFailure(RuntimeError):
    """A bounded request failure with a sanitized category and no raw body."""

    def __init__(self, category: str, message: str, *, http_status: int | None = None,
                 timed_out: bool = False) -> None:
        super().__init__(message)
        self.category = category
        self.http_status = http_status
        self.timed_out = timed_out


@dataclass(frozen=True)
class VLLMProvider:
    name: str
    model: str
    base_url: str
    credential_source: str
    keyring_service: str | None = None
    keyring_provider: str | None = None
    shared_compute: bool = True
    max_concurrency: int = 1
    thinking_default: bool = False
    default_max_tokens: int = DEFAULT_MAX_TOKENS
    max_tokens_cap: int = DEFAULT_MAX_TOKENS_CAP

    @property
    def max_tokens(self) -> int:
        """Backward-compatible name for the normal request default."""
        return self.default_max_tokens


@dataclass(frozen=True)
class VLLMLiveStatus:
    """A conservative, GET-only scheduler view for one local vLLM route."""

    state: str
    server_load: float | int | None = None
    running: float | int | None = None
    waiting: float | int | None = None
    kv_cache_usage_perc: float | None = None
    recent_requests: float | None = None
    recent_prompt_tokens: float | None = None
    recent_generation_tokens: float | None = None
    recent_preemptions: float | None = None
    prefix_caching: bool | None = None
    engine_sleep_state: str | None = None
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "server_load": self.server_load,
            "running": self.running,
            "waiting": self.waiting,
            "kv_cache_usage_perc": self.kv_cache_usage_perc,
            "recent_requests": self.recent_requests,
            "recent_prompt_tokens": self.recent_prompt_tokens,
            "recent_generation_tokens": self.recent_generation_tokens,
            "recent_preemptions": self.recent_preemptions,
            "prefix_caching": self.prefix_caching,
            "engine_sleep_state": self.engine_sleep_state,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class VLLMRouteInfo:
    """Safe local inspection result for control-plane consumers.

    ``provider`` is present only when the complete route definition validates.
    ``error`` is deliberately limited to local schema information; it never
    contains an endpoint, credential value, or server response.
    """

    name: str
    provider: VLLMProvider | None
    error: str | None = None
    error_kind: str | None = None


@dataclass(frozen=True)
class VLLMRunResult:
    """A result whose text has already been retained in private run state."""

    exit_code: int
    record_dir: Path
    text: str
    diagnostics: str
    response_recorded: bool = False

    # Preserve the familiar ``code, record_dir = run(...)`` shape for callers
    # of the adapter while exposing whether the durable response invariant held.
    def __iter__(self) -> Iterator[object]:
        yield self.exit_code
        yield self.record_dir

    def __getitem__(self, index: int) -> object:
        if index == 0:
            return self.exit_code
        if index == 1:
            return self.record_dir
        raise IndexError(index)


def vllm_config_path() -> Path:
    """Return the machine-local named-provider configuration path."""
    return config_dir() / VLLM_CONFIG_FILENAME


def vllm_issue_path() -> Path:
    """Return the machine-local reliability JSONL path."""
    return state_dir() / VLLM_ISSUE_FILENAME


def vllm_lock_path() -> Path:
    """Return the machine-local inter-process request lock path."""
    return state_dir() / VLLM_LOCK_FILENAME


def vllm_route_names(path: Path | None = None) -> set[str]:
    """Return syntactically valid named routes found in local TOML.

    This is a discovery helper for the offline control plane.  It parses only
    local configuration and never contacts the configured endpoint.
    """
    target = path or vllm_config_path()
    if not target.is_file():
        return set()
    import tomllib

    try:
        raw = tomllib.loads(target.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return set()
    providers = raw.get("providers", {})
    if not isinstance(providers, dict):
        return set()
    return {route for route in providers if isinstance(route, str) and _NAME_RE.fullmatch(route)}


def is_vllm_route_name(value: Any) -> bool:
    """Return whether a value is a safe named-route identifier."""
    return isinstance(value, str) and bool(_NAME_RE.fullmatch(value))


def _require_string(raw: dict[str, Any], key: str, route: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise VLLMConfigurationError(f"vLLM provider {route!r} requires a non-empty {key}")
    return value.strip()


def _validate_base_url(value: str, route: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise VLLMConfigurationError(f"vLLM provider {route!r} has an invalid base_url")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise VLLMConfigurationError(f"vLLM provider {route!r} base_url must not contain credentials or query data")
    return value.rstrip("/")


def _validate_credential_fields(raw: dict[str, Any], route: str) -> tuple[str, str | None, str | None]:
    source = _require_string(raw, "credential_source", route)
    if source.startswith("env:"):
        variable = source[4:].strip()
        if not _ENV_RE.fullmatch(variable):
            raise VLLMConfigurationError(f"vLLM provider {route!r} has an invalid environment credential reference")
        return source, None, None
    if source == "keyring":
        service = _require_string(raw, "keyring_service", route)
        provider = _require_string(raw, "keyring_provider", route)
        if any("\n" in value or "\r" in value for value in (service, provider)):
            raise VLLMConfigurationError(f"vLLM provider {route!r} has invalid keyring attributes")
        return source, service, provider
    raise VLLMConfigurationError(
        f"vLLM provider {route!r} credential_source must be env:NAME or keyring"
    )


def _validate_int(raw: dict[str, Any], key: str, default: int, route: str, *, maximum: int | None = None) -> int:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise VLLMConfigurationError(f"vLLM provider {route!r} {key} must be a positive integer")
    if maximum is not None and value > maximum:
        raise VLLMConfigurationError(f"vLLM provider {route!r} {key} exceeds the bounded maximum")
    return value


def _load_vllm_raw(path: Path) -> dict[str, Any]:
    import tomllib

    try:
        raw = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise VLLMConfigurationError("vLLM provider configuration could not be parsed") from exc
    providers = raw.get("providers", {})
    if not isinstance(providers, dict):
        raise VLLMConfigurationError("vLLM config [providers] must be a table")
    return providers


def _parse_provider(route: str, entry: Any) -> VLLMProvider:
    if not isinstance(route, str) or not _NAME_RE.fullmatch(route):
        raise VLLMConfigurationError("vLLM provider names must be short alphanumeric names with . _ or -")
    if not isinstance(entry, dict):
        raise VLLMConfigurationError(f"vLLM provider {route!r} must be a table")
    allowed = {
        "model", "base_url", "credential_source", "keyring_service", "keyring_provider",
        "shared_compute", "max_concurrency", "thinking_default", "max_tokens",
        "default_max_tokens", "max_tokens_cap",
    }
    extra = set(entry) - allowed
    if extra:
        raise VLLMConfigurationError(f"vLLM provider {route!r} has unsupported field(s): {sorted(extra)}")
    if "max_tokens" in entry and ({"default_max_tokens", "max_tokens_cap"} & set(entry)):
        raise VLLMConfigurationError(
            f"vLLM provider {route!r} must use either legacy max_tokens or default_max_tokens/max_tokens_cap"
        )
    model = _require_string(entry, "model", route)
    if any(ord(char) < 32 for char in model):
        raise VLLMConfigurationError(f"vLLM provider {route!r} model contains control characters")
    base_url = _validate_base_url(_require_string(entry, "base_url", route), route)
    credential_source, service, provider = _validate_credential_fields(entry, route)
    shared_compute = entry.get("shared_compute", True)
    if not isinstance(shared_compute, bool):
        raise VLLMConfigurationError(f"vLLM provider {route!r} shared_compute must be a boolean")
    max_concurrency = _validate_int(entry, "max_concurrency", 1, route)
    if shared_compute and max_concurrency != 1:
        raise VLLMConfigurationError("shared vLLM providers must set max_concurrency = 1")
    thinking_default = entry.get("thinking_default", False)
    if not isinstance(thinking_default, bool):
        raise VLLMConfigurationError(f"vLLM provider {route!r} thinking_default must be a boolean")
    if "max_tokens" in entry:
        # Legacy configuration is deliberately not widened: its old value is
        # both the default and the local cap.
        default_max_tokens = _validate_int(entry, "max_tokens", DEFAULT_MAX_TOKENS, route, maximum=HARD_MAX_TOKENS)
        max_tokens_cap = default_max_tokens
    else:
        default_max_tokens = _validate_int(
            entry, "default_max_tokens", DEFAULT_MAX_TOKENS, route, maximum=HARD_MAX_TOKENS
        )
        max_tokens_cap = _validate_int(
            entry, "max_tokens_cap", DEFAULT_MAX_TOKENS_CAP, route, maximum=HARD_MAX_TOKENS
        )
        if default_max_tokens > max_tokens_cap:
            raise VLLMConfigurationError(
                f"vLLM provider {route!r} default_max_tokens must not exceed max_tokens_cap"
            )
    return VLLMProvider(
        name=route, model=model, base_url=base_url, credential_source=credential_source,
        keyring_service=service, keyring_provider=provider, shared_compute=shared_compute,
        max_concurrency=max_concurrency, thinking_default=thinking_default,
        default_max_tokens=default_max_tokens, max_tokens_cap=max_tokens_cap,
    )


def inspect_vllm_routes(path: Path | None = None) -> dict[str, VLLMRouteInfo]:
    """Inspect named local routes without credential lookup or network I/O."""
    target = path or vllm_config_path()
    if not target.is_file():
        return {}
    try:
        raw = _load_vllm_raw(target)
    except VLLMConfigurationError:
        return {}
    result: dict[str, VLLMRouteInfo] = {}
    for route, entry in raw.items():
        if not isinstance(route, str) or not _NAME_RE.fullmatch(route):
            continue
        try:
            provider = _parse_provider(route, entry)
        except VLLMConfigurationError as exc:
            missing_reference = (
                isinstance(entry, dict)
                and "credential_source" not in entry
            ) or (
                isinstance(entry, dict)
                and entry.get("credential_source") == "keyring"
                and ("keyring_service" not in entry or "keyring_provider" not in entry)
            )
            result[route] = VLLMRouteInfo(
                route, None, "credential reference is missing" if missing_reference else "invalid vLLM route configuration",
                "missing-credential-reference" if missing_reference else "invalid-configuration",
            )
        else:
            result[route] = VLLMRouteInfo(route, provider)
    return result


def load_vllm_config(path: Path | None = None) -> dict[str, VLLMProvider]:
    """Load named providers from local TOML; missing config means no routes."""
    target = path or vllm_config_path()
    if not target.is_file():
        return {}
    return {route: _parse_provider(route, entry) for route, entry in _load_vllm_raw(target).items()}


def _observability_base_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url.rstrip("/"))
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3].rstrip("/")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _observability_get(
    base_url: str,
    path: str,
    *,
    opener: Callable[..., Any],
) -> tuple[int | None, bytes, str | None]:
    """GET one safe observability endpoint without credentials."""
    request = urllib.request.Request(
        _observability_base_url(base_url) + path,
        headers={"Accept": "application/json, text/plain; version=0.0.4"},
        method="GET",
    )
    try:
        with opener(request, timeout=OBSERVABILITY_TIMEOUT_SECONDS) as response:
            body = response.read(MAX_OBSERVABILITY_BYTES + 1)
            if len(body) > MAX_OBSERVABILITY_BYTES:
                return int(response.status), b"", "response-too-large"
            return int(response.status), body, None
    except urllib.error.HTTPError as exc:
        exc.close()
        return int(exc.code), b"", "http-failure"
    except (OSError, TimeoutError, socket.timeout, urllib.error.URLError):
        return None, b"", "unreachable"


def _metric_labels(value: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for key, raw in re.findall(r'([A-Za-z_][A-Za-z0-9_]*)="((?:\\.|[^"\\])*)"', value):
        labels[key] = raw.replace(r'\"', '"').replace(r'\\', '\\').replace(r'\n', "\n")
    return labels


def _parse_prometheus_metrics(body: bytes) -> dict[str, list[tuple[dict[str, str], float]]]:
    """Parse enough Prometheus text format for stable scheduler metrics."""
    result: dict[str, list[tuple[dict[str, str], float]]] = {}
    for line in body.decode("utf-8", "replace").splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([^\s{]+)(?:\{(.*)\})?\s+([-+0-9.eEInfNaN]+)(?:\s+\d+)?$", line)
        if not match:
            continue
        try:
            value = float(match.group(3))
        except ValueError:
            continue
        result.setdefault(match.group(1), []).append((_metric_labels(match.group(2) or ""), value))
    return result


def _metric_total(
    metrics: dict[str, list[tuple[dict[str, str], float]]],
    name: str,
    model: str,
) -> float | None:
    values = metrics.get(name)
    if not values:
        return None
    model_values = [(labels, value) for labels, value in values if labels.get("model_name") in (None, model)]
    if not model_values:
        return None
    return sum(value for _labels, value in model_values)


def _metric_label_value(
    metrics: dict[str, list[tuple[dict[str, str], float]]],
    name: str,
    label: str,
    model: str,
) -> str | None:
    for labels, value in metrics.get(name, []):
        if labels.get("model_name") not in (None, model) or value <= 0:
            continue
        return labels.get(label)
    return None


def _live_snapshot(provider: VLLMProvider, *, opener: Callable[..., Any]) -> tuple[dict[str, Any] | None, str | None]:
    load_status, load_body, load_error = _observability_get(provider.base_url, "/load", opener=opener)
    metrics_status, metrics_body, metrics_error = _observability_get(provider.base_url, "/metrics", opener=opener)
    if load_status in {401, 403} or metrics_status in {401, 403}:
        return None, "observability requires authentication"
    if load_status != 200 or metrics_status != 200:
        return None, "observability endpoint unavailable"
    try:
        load = json.loads(load_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, "load response is incompatible"
    if not isinstance(load, dict):
        return None, "load response is incompatible"
    metrics = _parse_prometheus_metrics(metrics_body)
    running = _metric_total(metrics, "vllm:num_requests_running", provider.model)
    waiting = _metric_total(metrics, "vllm:num_requests_waiting", provider.model)
    kv = _metric_total(metrics, "vllm:kv_cache_usage_perc", provider.model)
    if kv is not None:
        kv = float(kv)
    return {
        "server_load": load.get("server_load") if isinstance(load.get("server_load"), (int, float)) else None,
        "running": running,
        "waiting": waiting,
        "kv_cache_usage_perc": kv,
        "prompt_tokens": _metric_total(metrics, "vllm:prompt_tokens_total", provider.model),
        "generation_tokens": _metric_total(metrics, "vllm:generation_tokens_total", provider.model),
        "requests": _metric_total(metrics, "vllm:request_success_total", provider.model),
        "preemptions": _metric_total(metrics, "vllm:num_preemptions_total", provider.model),
        "prefix_caching": _metric_label_value(metrics, "vllm:cache_config_info", "enable_prefix_caching", provider.model),
        "engine_sleep_state": _metric_label_value(metrics, "vllm:engine_sleep_state", "sleep_state", provider.model),
    }, load_error or metrics_error


def _counter_delta(first: dict[str, Any], second: dict[str, Any], key: str) -> float | None:
    before, after = first.get(key), second.get(key)
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
        return None
    return max(0.0, float(after) - float(before))


def _classify_live(first: dict[str, Any], second: dict[str, Any]) -> VLLMLiveStatus:
    running, waiting = second.get("running"), second.get("waiting")
    if not isinstance(running, (int, float)) or not isinstance(waiting, (int, float)):
        return VLLMLiveStatus("UNKNOWN", reason="required scheduler metrics unavailable")
    recent_requests = _counter_delta(first, second, "requests")
    recent_prompt = _counter_delta(first, second, "prompt_tokens")
    recent_generation = _counter_delta(first, second, "generation_tokens")
    recent_preemptions = _counter_delta(first, second, "preemptions")
    counters_available = any(value is not None for value in (recent_requests, recent_prompt, recent_generation))
    kv = second.get("kv_cache_usage_perc")
    if waiting > 0 or (recent_preemptions or 0) > 0 or (isinstance(kv, (int, float)) and kv >= 95.0):
        state = "PRESSURED"
    elif running > 0 or any((value or 0) > 0 for value in (recent_requests, recent_prompt, recent_generation)):
        state = "ACTIVE"
    elif not counters_available:
        state = "UNKNOWN"
    else:
        state = "IDLE"
    return VLLMLiveStatus(
        state,
        server_load=second.get("server_load"),
        running=running,
        waiting=waiting,
        kv_cache_usage_perc=kv,
        recent_requests=recent_requests,
        recent_prompt_tokens=recent_prompt,
        recent_generation_tokens=recent_generation,
        recent_preemptions=recent_preemptions,
        prefix_caching=(second.get("prefix_caching") or "").lower() == "true" if second.get("prefix_caching") is not None else None,
        engine_sleep_state=second.get("engine_sleep_state"),
    )


def inspect_vllm_live_routes(
    routes: dict[str, VLLMRouteInfo],
    *,
    opener: Callable[..., Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    interval_seconds: float = LIVE_STATUS_INTERVAL_SECONDS,
) -> dict[str, VLLMLiveStatus]:
    """Collect two unauthenticated GET snapshots for valid local routes."""
    get = opener or urllib.request.urlopen
    result: dict[str, VLLMLiveStatus] = {}
    for name in sorted(routes):
        info = routes[name]
        if info.provider is None:
            result[name] = VLLMLiveStatus("UNKNOWN", reason=info.error or "invalid local route")
            continue
        first, first_error = _live_snapshot(info.provider, opener=get)
        if first is None:
            result[name] = VLLMLiveStatus("UNKNOWN", reason=first_error or "observability unavailable")
            continue
        sleep(interval_seconds)
        second, second_error = _live_snapshot(info.provider, opener=get)
        if second is None:
            result[name] = VLLMLiveStatus("UNKNOWN", reason=second_error or "observability unavailable")
            continue
        result[name] = _classify_live(first, second)
    return result


def _resolve_credential(provider: VLLMProvider) -> str:
    if provider.credential_source.startswith("env:"):
        variable = provider.credential_source[4:]
        value = os.environ.get(variable)
        if not value:
            raise VLLMFailure("credential-unavailable", "vLLM credential is not available")
        return value
    assert provider.keyring_service is not None and provider.keyring_provider is not None
    try:
        completed = subprocess.run(
            ["secret-tool", "lookup", "service", provider.keyring_service, "provider", provider.keyring_provider],
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
            timeout=KEYRING_LOOKUP_TIMEOUT_SECONDS, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise VLLMFailure("credential-unavailable", "vLLM keyring credential could not be read") from exc
    # stdout contains the credential on success. It is deliberately consumed
    # only in memory and is never included in any diagnostic or audit field.
    value = completed.stdout.strip()
    if completed.returncode != 0 or not value:
        raise VLLMFailure("credential-unavailable", "vLLM keyring credential is not available")
    return value


def _http_post(url: str, headers: dict[str, str], body: bytes, timeout: int) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    deadline = time.monotonic() + timeout

    def remaining() -> float:
        value = deadline - time.monotonic()
        if value <= 0:
            raise VLLMFailure("request-timeout", "vLLM request timed out", timed_out=True)
        return value

    def read_bounded(response: Any) -> bytes:
        chunks: list[bytes] = []
        total = 0
        while total <= MAX_RESPONSE_BYTES:
            seconds = remaining()
            # urllib's response ultimately wraps a socket. Resetting its
            # per-read timeout to the remaining wall-clock budget prevents a
            # streaming server from extending a request forever by sending a
            # byte just before each socket timeout.
            raw = getattr(response, "fp", None)
            raw = getattr(raw, "raw", raw)
            sock = getattr(raw, "_sock", None)
            if sock is not None and hasattr(sock, "settimeout"):
                sock.settimeout(seconds)
            try:
                chunk = response.read(min(65536, MAX_RESPONSE_BYTES + 1 - total))
            except (TimeoutError, socket.timeout) as exc:
                raise VLLMFailure("request-timeout", "vLLM request timed out", timed_out=True) from exc
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
            total += len(chunk)
        return b"".join(chunks)

    try:
        with urllib.request.urlopen(request, timeout=remaining()) as response:
            return int(response.status), read_bounded(response)
    except urllib.error.HTTPError as exc:
        # Read and discard only enough to close the response; the body may
        # contain private provider diagnostics and is never logged.
        exc.close()
        raise VLLMFailure(_http_category(exc.code), "vLLM server returned an HTTP failure", http_status=exc.code) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise VLLMFailure("request-timeout", "vLLM request timed out", timed_out=True) from exc
    except urllib.error.URLError as exc:
        reason = exc.reason
        timed_out = isinstance(reason, (TimeoutError, socket.timeout))
        raise VLLMFailure(
            "request-timeout" if timed_out else "connection-error",
            "vLLM request timed out" if timed_out else "vLLM server could not be reached",
            timed_out=timed_out,
        ) from exc
    except OSError as exc:
        timed_out = isinstance(exc, (TimeoutError, socket.timeout))
        raise VLLMFailure(
            "request-timeout" if timed_out else "connection-error",
            "vLLM request timed out" if timed_out else "vLLM server could not be reached",
            timed_out=timed_out,
        ) from exc


def _http_category(status: int) -> str:
    if status in {401, 403}:
        return "authentication-failure"
    if status in {404, 405}:
        return "api-compatibility-failure"
    if status == 429:
        return "rate-limited"
    if 500 <= status <= 599:
        return "server-failure"
    return "http-failure"


def _response_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise VLLMFailure("malformed-response", "vLLM response JSON has an invalid shape")
    if payload.get("error") is not None:
        raise VLLMFailure("model-response-failure", "vLLM returned an error response")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise VLLMFailure("empty-response", "vLLM returned no textual choice")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise VLLMFailure("malformed-response", "vLLM response has no message object")
    if message.get("refusal"):
        raise VLLMFailure("model-refusal", "vLLM model refused the request")
    content = message.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = [part.get("text", "") for part in content if isinstance(part, dict) and isinstance(part.get("text", ""), str)]
        text = "".join(parts)
    else:
        text = ""
    if not text.strip():
        raise VLLMFailure("empty-response", "vLLM returned an empty textual response")
    return text


@contextmanager
def _request_lock(path: Path | None = None) -> Iterator[None]:
    """Acquire a non-blocking machine-local lock; kernel releases it on exit."""
    target = path or vllm_lock_path()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_RDWR | os.O_CREAT
    fd = os.open(target, flags, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise VLLMFailure("concurrency-busy", "another shared vLLM request is already active") from exc
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _safe_machine_label() -> str:
    value = os.environ.get("AGENT_DELEGATION_MACHINE_LABEL") or socket.gethostname() or "local-machine"
    value = re.sub(r"[^A-Za-z0-9_.-]", "-", value)
    return value[:64] or "local-machine"


class _VLLMTermination(BaseException):
    def __init__(self, signal_number: int) -> None:
        super().__init__(signal_number)
        self.signal_number = signal_number


@contextmanager
def _termination_guard() -> Iterator[None]:
    """Turn operator interrupt/TERM into a finalized incomplete run record."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def handler(signal_number: int, _frame: Any) -> None:
        raise _VLLMTermination(signal_number)

    old_int = signal.signal(signal.SIGINT, handler)
    old_term = signal.signal(signal.SIGTERM, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


def _append_issue(record: dict[str, Any], path: Path | None = None) -> Path:
    target = path or vllm_issue_path()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), "machine_label": _safe_machine_label(), **record}
    fd = os.open(target, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        with os.fdopen(fd, "a", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.write("\n")
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    return target


def _record_execution(record_dir: Path, data: dict[str, Any]) -> None:
    persist_text(record_dir, "execution.json", json.dumps(data, indent=2, sort_keys=True) + "\n")


def run_vllm_consultation(
    route: str,
    workspace: Path,
    task: str,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_tokens: int | None = None,
    thinking: bool | None = None,
    config_path: Path | None = None,
    log_root: Path | None = None,
    issue_path: Path | None = None,
    lock_path: Path | None = None,
    transport: Callable[[str, dict[str, str], bytes, int], tuple[int, bytes]] = _http_post,
    credential_loader: Callable[[VLLMProvider], str] = _resolve_credential,
) -> VLLMRunResult:
    """Run exactly one bounded direct request and return ``(exit, evidence)``."""
    if timeout_seconds <= 0:
        raise VLLMConfigurationError("timeout must be positive")
    _check_recursion_guard()
    workspace = _validate_scope(workspace)
    providers = load_vllm_config(config_path)
    if route not in providers:
        raise VLLMConfigurationError(f"unknown vLLM provider route: {route!r}")
    provider = providers[route]
    if not provider.shared_compute or provider.max_concurrency != 1:
        raise VLLMConfigurationError("shared vLLM providers must enable shared_compute with max_concurrency = 1")
    requested_tokens = provider.default_max_tokens if max_tokens is None else max_tokens
    if isinstance(requested_tokens, bool) or not isinstance(requested_tokens, int) or requested_tokens <= 0:
        raise VLLMConfigurationError(
            f"requested max_tokens={requested_tokens!r} is invalid; "
            f"local route default={provider.default_max_tokens}, cap={provider.max_tokens_cap}; "
            "request rejected before model inference"
        )
    if requested_tokens > provider.max_tokens_cap:
        raise VLLMConfigurationError(
            f"requested max_tokens={requested_tokens} exceeds local route cap={provider.max_tokens_cap}; "
            "request rejected before model inference"
        )
    effective_thinking = provider.thinking_default if thinking is None else thinking
    try:
        resolved_log_root = _validated_log_root(log_root or default_log_root(), workspace)
    except ValueError as exc:
        raise VLLMConfigurationError(str(exc)) from exc
    record_dir = _record_path(resolved_log_root, f"vllm-{route}")
    record_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(record_dir, 0o700)
    started_at = datetime.now(timezone.utc).isoformat()
    begun = time.monotonic()
    exit_code = 1
    timed_out = False
    http_status: int | None = None
    error_category: str | None = None
    result_state = "infrastructure-failure"
    response = ""
    stderr = ""
    provider_success = False
    inference_occurred = False
    provider_reported_usage: dict[str, int] = {}
    provider_reported_model_id: str | None = None
    response_metadata: dict[str, Any] = {
        "response_recorded": False,
        "response_file": None,
        "response_length_bytes": 0,
        "response_sha256": None,
    }
    try:
        with _termination_guard():
            with _request_lock(lock_path):
                credential = credential_loader(provider)
                payload = {
                    "model": provider.model,
                    "messages": [{"role": "user", "content": task}],
                    "max_tokens": requested_tokens,
                    "chat_template_kwargs": {"enable_thinking": effective_thinking},
                }
                body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                status, raw = transport(
                    provider.base_url + "/chat/completions",
                    {"Authorization": f"Bearer {credential}", "Content-Type": "application/json"},
                    body,
                    timeout_seconds,
                )
                http_status = status
                if status < 200 or status >= 300:
                    raise VLLMFailure(_http_category(status), "vLLM server returned an HTTP failure", http_status=status)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise VLLMFailure("malformed-response", "vLLM response exceeded the bounded response size")
                try:
                    parsed = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise VLLMFailure("malformed-response", "vLLM response was not valid JSON") from exc
                response = _response_text(parsed)
                # Keep only documented numeric usage and explicit model identity;
                # never retain the provider response object in execution metadata.
                raw_usage = parsed.get("usage") if isinstance(parsed, dict) else None
                if isinstance(raw_usage, dict):
                    provider_reported_usage = {
                        key: raw_usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens", "reasoning_tokens", "cached_input_tokens", "cache_read_tokens", "cache_write_tokens")
                        if isinstance(raw_usage.get(key), int) and not isinstance(raw_usage.get(key), bool) and raw_usage[key] >= 0
                    }
                else:
                    provider_reported_usage = {}
                provider_reported_model_id = parsed.get("model") if isinstance(parsed, dict) and isinstance(parsed.get("model"), str) and len(parsed["model"]) <= MAX_PROVIDER_REPORTED_MODEL_ID else None
                provider_success = True
                inference_occurred = True
                try:
                    response_metadata = persist_response(record_dir, response)
                    if response_metadata.get("response_recorded") is not True:
                        raise OSError("response persistence did not confirm a recorded response")
                except OSError:
                    # The provider completed, but returning a response that
                    # exists only in memory would make a successful HTTP call
                    # unrecoverable if the caller loses terminal output.
                    response = ""
                    exit_code = 1
                    result_state = "response-retention-failure"
                    error_category = "response-retention"
                    response_metadata = {
                        "response_recorded": False,
                        "response_file": None,
                        "response_length_bytes": 0,
                        "response_sha256": None,
                    }
                    stderr = (
                        "vLLM response-retention failure: textual response was not "
                        "durably saved; no retry performed\n"
                    )
                else:
                    exit_code = 0
                    result_state = "text-returned"
    except VLLMFailure as exc:
        error_category = exc.category
        timed_out = exc.timed_out
        http_status = exc.http_status if exc.http_status is not None else http_status
        exit_code = 124 if timed_out else 1
        if exc.category in {"malformed-response", "empty-response", "model-response-failure", "model-refusal"}:
            result_state = "model/response-failure"
        # Keep the diagnostic category-only. Production transports already
        # sanitize their own exceptions, but this also prevents a future
        # transport implementation from accidentally reflecting a URL, body,
        # credential, or provider-specific error string.
        stderr = f"vLLM {result_state}: {exc.category}\n"
    except _VLLMTermination as exc:
        error_category = "manual-termination"
        result_state = "incomplete-infrastructure-run"
        exit_code = 128 + exc.signal_number
        stderr = "vLLM incomplete-infrastructure-run: manual-termination\n"
    except KeyboardInterrupt:
        error_category = "manual-termination"
        result_state = "incomplete-infrastructure-run"
        exit_code = 130
        stderr = "vLLM incomplete-infrastructure-run: manual-termination\n"
    elapsed = time.monotonic() - begun
    persist_text(record_dir, "stderr.txt", stderr)
    execution = {
        "adapter": "openai-compatible-vllm",
        "route": route,
        "provider_type": "openai_compatible_vllm",
        "model": provider.model,
        "operation_class": "bounded-chat-completion",
        "thinking": effective_thinking,
        "max_tokens": requested_tokens,
        "default_max_tokens": provider.default_max_tokens,
        "max_tokens_cap": provider.max_tokens_cap,
        "timeout_seconds": timeout_seconds,
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": elapsed,
        "http_status": http_status,
        "timed_out": timed_out,
        "exit_code": exit_code,
        "provider_success": provider_success,
        "inference_occurred": inference_occurred,
        "response_status": result_state,
        "error_category": error_category,
        "retry": False,
        "fallback": None,
        "shared_compute": True,
        "max_concurrency": 1,
        "prompt_recorded": False,
        "credential_recorded": False,
        "stderr_file": "stderr.txt",
        "provider_reported_usage": provider_reported_usage if provider_success else {},
        "provider_reported_model_id": provider_reported_model_id if provider_success else None,
        "request_count": 1 if provider_success else 0,
        **response_metadata,
    }
    _record_execution(record_dir, execution)
    if result_state != "text-returned" and error_category != "response-retention":
        _append_issue({
            "adapter": "openai-compatible-vllm",
            "route": route,
            "provider_type": "openai_compatible_vllm",
            "client_version": "ekalavya-delegation",
            "model": provider.model,
            "operation_class": "bounded-chat-completion",
            "result_state": result_state,
            "http_status": http_status,
            "duration_seconds": elapsed,
            "timeout": timed_out,
            "thinking": effective_thinking,
            "error_category": error_category,
            "retry": False,
            "fallback": None,
        }, issue_path)
    return VLLMRunResult(
        exit_code, record_dir, response, stderr,
        bool(response_metadata.get("response_recorded")),
    )
