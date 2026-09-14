"""Provider-contract facts and exact-identity guards for DeepSeek."""

from __future__ import annotations

DEEPSEEK_FLASH_PROVIDER_MODEL_ID = "deepseek-flash"
DEEPSEEK_V4_FLASH_PROVIDER_MODEL_ID = "deepseek-v4-flash"
DEEPSEEK_V4_FLASH_VISION_EXP_PROVIDER_MODEL_ID = "deepseek-v4-flash-vision-exp"
DEEPSEEK_PRO_PROVIDER_MODEL_ID = "deepseek-v4-pro"
DEEPSEEK_V41_FLASH_DISPLAY_NAME = "DeepSeek V4.1 Flash"
DEEPSEEK_V4_FLASH_DISPLAY_NAME = "DeepSeek V4 Flash"
DEEPSEEK_V4_PRO_DISPLAY_NAME = "DeepSeek V4 Pro"

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

# These are the only effective identities accepted for the exact Pro route.
# A missing report is allowed before execution because the registered route
# already pins the requested provider model. A contradictory report is never
# silently treated as an alias or fallback.
DEEPSEEK_EXACT_PRO_EFFECTIVE_IDS = frozenset({
    DEEPSEEK_PRO_PROVIDER_MODEL_ID,
    "DeepSeek-V4-Pro-0813",
})


class DeepSeekExactIdentityError(RuntimeError):
    """Raised when a provider reports a different effective model."""


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
    *, provider_reported_model_id: str | None = None,
) -> None:
    """Validate an independently reported effective identity for Pro.

    The exact registered route establishes the pre-execution binding. If a
    provider response supplies an effective identity, it must still match the
    current V4 Pro contract; missing or absent telemetry remains unknown rather
    than being treated as a substitution.
    """
    if provider_reported_model_id is None:
        return
    if provider_reported_model_id in DEEPSEEK_EXACT_PRO_EFFECTIVE_IDS:
        return
    raise DeepSeekExactIdentityError(
        "deepseek-pro exact identity mismatch: expected DeepSeek V4 Pro "
        f"({DEEPSEEK_PRO_PROVIDER_MODEL_ID} / DeepSeek-V4-Pro-0813), "
        f"provider reported {provider_reported_model_id!r}; no fallback or "
        "model substitution is performed"
    )
