"""Zero-inference provider discovery used by the Ekalavya control plane.

Discovery is deliberately separate from lifecycle promotion.  The only live
adapter currently supported here is AGY's documented tabular ``models`` list.
AGY 1.1.27 documents no machine-readable listing flag, so parsing is strict:
unexpected non-empty lines and incomplete Flash generations are rejected.
"""

from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone
from typing import Any, Callable


class DiscoveryError(ValueError):
    """A provider listing could not be safely turned into catalogue facts."""


_AGY_PREAMBLE = {"Fetching available models..."}
_FLASH_ID = re.compile(r"^gemini-(?P<generation>\d+\.\d+)-flash-(?P<reasoning>low|medium|high)$")
_REQUIRED_REASONING = {"low", "medium", "high"}


def _capture(argv: list[str], *, run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> tuple[int, str, str]:
    """Run a documented read-only CLI probe without inheriting caller stdin."""
    try:
        completed = run(argv, text=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return -1, "", str(exc)
    return completed.returncode, completed.stdout, completed.stderr


def parse_agy_models(stdout: str) -> list[dict[str, str]]:
    """Parse AGY's documented tabular listing without accepting partial rows."""
    rows: list[dict[str, str]] = []
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line or line in _AGY_PREAMBLE:
            continue
        if "\t" not in raw:
            raise DiscoveryError(f"unrecognized AGY models output line: {line!r}")
        model_id, display_name = raw.split("\t", 1)
        model_id, display_name = model_id.strip(), display_name.strip()
        if not model_id or not display_name:
            raise DiscoveryError("malformed AGY models row")
        rows.append({"provider_model_id": model_id, "display_name": display_name})
    if not rows:
        raise DiscoveryError("AGY models returned no parseable rows")
    return rows


def gemini_flash_generations(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return only complete exact Gemini Flash generations from an AGY listing."""
    selected: list[dict[str, str]] = []
    found: dict[str, set[str]] = {}
    for row in rows:
        match = _FLASH_ID.match(row["provider_model_id"])
        if not match:
            continue
        selected.append(row)
        found.setdefault(match.group("generation"), set()).add(match.group("reasoning"))
    if not selected:
        raise DiscoveryError("AGY models did not expose Gemini Flash runtime IDs")
    incomplete = sorted(generation for generation, values in found.items() if values != _REQUIRED_REASONING)
    if incomplete:
        raise DiscoveryError("incomplete Gemini Flash reasoning variants: " + ", ".join(incomplete))
    return selected


def discover_gemini(*, run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> dict[str, Any]:
    """Discover exact Gemini Flash runtime IDs; this never invokes a model."""
    version_code, version_out, version_err = _capture(["agy", "--version"], run=run)
    if version_code != 0 or not version_out.strip():
        raise DiscoveryError("agy --version failed: " + (version_err.strip() or "unknown error"))
    models_code, models_out, models_err = _capture(["agy", "models"], run=run)
    if models_code != 0:
        raise DiscoveryError("agy models failed: " + (models_err.strip() or "unknown error"))
    rows = gemini_flash_generations(parse_agy_models(models_out))
    return {
        "provider": "gemini",
        "client": "agy",
        "client_version": version_out.strip(),
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "models": rows,
    }
