"""Provider-contract facts and exact-identity guards for DeepSeek."""

from __future__ import annotations

from datetime import datetime, timezone


DEEPSEEK_FLASH_PROVIDER_MODEL_ID = "deepseek-flash"
DEEPSEEK_V4_FLASH_PROVIDER_MODEL_ID = "deepseek-v4-flash"
DEEPSEEK_V4_FLASH_VISION_EXP_PROVIDER_MODEL_ID = "deepseek-v4-flash-vision-exp"
DEEPSEEK_PRO_PROVIDER_MODEL_ID = "deepseek-v4-pro"
DEEPSEEK_V41_FLASH_DISPLAY_NAME = "DeepSeek V4.1 Flash"
DEEPSEEK_V4_FLASH_DISPLAY_NAME = "DeepSeek V4 Flash"
DEEPSEEK_V4_PRO_DISPLAY_NAME = "DeepSeek V4 Pro"

# 12:00 Beijing time on 2026-09-14 is 04:00 UTC. Keep this as an injected,
# deterministic contract boundary rather than consulting wall-clock time in
# tests or at catalogue construction time.
DEEPSEEK_PRO_CUTOFF_UTC = datetime(2026, 9, 14, 4, 0, tzinfo=timezone.utc)

DEEPSEEK_REASONING_MAPPING: dict[str, str] = {
    "minimal": "low",
    "low": "low",
    "medium": "high",
    "high": "high",
    "xhigh": "high",
    "max": "max",
    "ultra": "max",
}
DEEPSEEK_PROVIDER_REASONING_LEVELS = ("low", "high", "max")
DEEPSEEK_EKALAVYA_REASONING_LEVELS = tuple(DEEPSEEK_REASONING_MAPPING)

# The endpoint slug is not evidence of the effective model after the cutoff.
# This allow-list is intentionally limited to an exact provider-reported
# version; the current provider contract does not report one before execution.
DEEPSEEK_EXACT_PRO_EFFECTIVE_IDS = frozenset({"DeepSeek-V4-Pro-0813"})


class DeepSeekExactIdentityError(RuntimeError):
    """Raised when an endpoint cannot prove the exact requested model."""


def _utc(value: datetime | None) -> datetime:
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def map_reasoning_level(level: str | None) -> str | None:
    """Map Ekalavya's explicit reasoning vocabulary to DeepSeek's values."""
    if level is None:
        return None
    try:
        return DEEPSEEK_REASONING_MAPPING[level]
    except KeyError:
        raise ValueError(
            f"unsupported DeepSeek reasoning setting {level!r}; "
            f"supported: {list(DEEPSEEK_EKALAVYA_REASONING_LEVELS)!r}"
        ) from None


def assert_deepseek_pro_exact(
    *, now: datetime | None = None, provider_reported_model_id: str | None = None,
) -> None:
    """Fail closed once DeepSeek no longer guarantees V4 Pro at its slug.

    ``provider_reported_model_id`` is accepted only as an independently
    reported exact effective identity. The requested endpoint slug is not
    accepted as proof and is never copied into that field.
    """
    if _utc(now) < DEEPSEEK_PRO_CUTOFF_UTC:
        return
    if provider_reported_model_id in DEEPSEEK_EXACT_PRO_EFFECTIVE_IDS:
        return
    raise DeepSeekExactIdentityError(
        "deepseek-pro is unavailable after 2026-09-14T04:00:00Z: "
        "DeepSeek documents deepseek-v4-pro as serving V4.1 Flash until a "
        "future exact Pro identity is provider-reported; no fallback to "
        "deepseek-flash is performed"
    )
