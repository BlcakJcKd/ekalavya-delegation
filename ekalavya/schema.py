"""Stable, JSON-friendly control-plane identities and policies."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any


TASK_UNSPECIFIED = "unspecified"
TASK_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def normalize_task(value: str | None) -> str:
    """Validate the explicit, bounded task classification contract."""
    if value is None or value == "":
        return TASK_UNSPECIFIED
    if not isinstance(value, str) or not TASK_RE.fullmatch(value):
        raise ValueError("task must be a lowercase categorical slug (1-64 chars; [a-z0-9][a-z0-9._-]*)")
    return value


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


@dataclass(frozen=True)
class CandidateIdentity:
    provider: str
    family: str | None = None
    provider_model_id: str | None = None
    display_name: str | None = None
    generation: str | None = None
    variant: str | None = None
    capabilities: dict[str, Any] = field(default_factory=dict)
    architecture: str | None = None
    parameter_count: str | None = None
    active_parameter_count: str | None = None
    quantization: str | None = None
    serving_engine: str | None = None
    serving_engine_version: str | None = None
    hardware_profile: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def identity_key(self) -> str:
        return canonical_hash(self.as_dict())


@dataclass(frozen=True)
class ReasoningPolicy:
    mode: str = "overrideable"  # fixed | overrideable | required
    default: Any = None
    supported: tuple[Any, ...] = ()

    def validate(self, requested: Any) -> Any:
        value = self.default if requested is None else requested
        if self.mode == "required" and value is None:
            raise ValueError("reasoning is required for this candidate")
        # None means "provider default" for an overrideable policy; it is an
        # explicit absence of an override, not an unsupported setting.
        if value is None:
            return None
        if self.supported and value not in self.supported:
            raise ValueError(f"unsupported reasoning setting {value!r}; supported: {list(self.supported)!r}")
        if self.mode == "fixed" and requested is not None and requested != self.default:
            raise ValueError(f"reasoning is fixed at {self.default!r}")
        return value


@dataclass(frozen=True)
class RunIntent:
    profile: str
    provider: str | None = None
    family: str | None = None
    model: str | None = None
    reasoning: Any = None
    harness: str | None = None
    workspace: str | None = None
    prompt_file: str | None = None
    primary: str | None = None
    timeout_seconds: int | None = None
    task: str = TASK_UNSPECIFIED

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", normalize_task(self.task))


@dataclass(frozen=True)
class Resolution:
    intent: RunIntent
    candidate: CandidateIdentity | None
    resolved_reasoning: Any = None
    resolved_harness: str | None = None
    resolved_harness_version: str | None = None
    transport: str | None = None
    execution_route: str | None = None
    reason: str = ""
    state: str = "resolved"
    alternatives: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        resolved = self.candidate.as_dict() if self.candidate else None
        if resolved is not None:
            resolved.update({
                "reasoning": self.resolved_reasoning,
                "harness": self.resolved_harness,
                "harness_version": self.resolved_harness_version,
                "transport": self.transport,
                "execution_route": self.execution_route,
            })
        return {
            "requested": asdict(self.intent),
            "resolved": resolved,
            "resolved_reasoning": self.resolved_reasoning,
            "resolved_harness": self.resolved_harness,
            "resolved_harness_version": self.resolved_harness_version,
            "transport": self.transport,
            "execution_route": self.execution_route,
            "reason": self.reason,
            "state": self.state,
            "alternatives": list(self.alternatives),
        }
