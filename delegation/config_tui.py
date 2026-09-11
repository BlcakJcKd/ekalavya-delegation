"""Interactive terminal helpers for Ekalavya availability configuration.

Split deliberately in two layers:

- Pure state-transition functions (``build_rows``, ``toggle``, ``set_reason``,
  ``rows_to_config``, ``diff_summary``) operate on plain data and contain all
  the actual logic. These are unit-tested directly, without a TTY.
- The curses event loop (``run_interactive_config``, ``_loop``, ``_render``)
  drives those functions.  Its save/cancel boundary is unit-tested with a
  fake wrapper; terminal rendering remains deliberately thin.

Toggling a provider row never cascades to its models' own enabled
preference: a disabled provider overrides *effective* availability (computed
in ``delegation.status``) without erasing which models were individually
enabled, so re-enabling the provider later restores them automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import shutil
from typing import Any

from . import routing
from .config import load_config, save_config, set_routing_preference
from .vllm import VLLMRouteInfo, inspect_vllm_routes

PROVIDER_ORDER: tuple[str, ...] = ("gemini", "claude", "codex", "deepseek", "minimax")
PROVIDER_LABELS: dict[str, str] = {
    "gemini": "Gemini", "claude": "Claude", "codex": "Codex",
    "deepseek": "DeepSeek", "minimax": "MiniMax",
}
MODEL_LABELS: dict[str, str] = {
    "flash": "Gemini Flash", "sonnet": "Claude Sonnet", "haiku": "Claude Haiku", "terra": "Codex Terra", "luna": "Codex Luna",
    "deepseek-pro": "DeepSeek V4 Pro", "deepseek-flash": "DeepSeek V4.1 Flash",
    "minimax-m3": "MiniMax M3",
}


@dataclass(frozen=True)
class Row:
    kind: str  # "provider" | "model" | "vllm"
    section: str  # "providers" | "models" | "vllm"
    name: str
    label: str
    enabled: bool
    reason: str | None
    provider: str | None  # owning provider name for a model row; None for a provider row
    details: tuple[str, ...] = ()
    effective_enabled: bool | None = None
    effective_reason: str | None = None

    @property
    def configured_enabled(self) -> bool:
        return self.enabled


def build_rows(
    config: dict[str, Any],
    vllm_routes: dict[str, VLLMRouteInfo] | None = None,
) -> list[Row]:
    rows: list[Row] = []
    for provider in PROVIDER_ORDER:
        entry = config["providers"][provider]
        rows.append(Row(
            "provider", "providers", provider, PROVIDER_LABELS[provider],
            entry["enabled"], entry.get("reason"), None,
            effective_enabled=bool(entry.get("enabled", True)),
            effective_reason=entry.get("reason") if not entry.get("enabled", True) else None,
        ))
    for model in sorted(routing.MODELS, key=lambda name: (routing.ROUTE_PROVIDER[name], name)):
        provider = routing.ROUTE_PROVIDER[model]
        provider_entry = config["providers"][provider]
        m_entry = config["models"][model]
        provider_enabled = bool(provider_entry.get("enabled", True))
        model_enabled = bool(m_entry.get("enabled", True))
        effective_reason = None
        if not provider_enabled:
            effective_reason = provider_entry.get("reason")
        elif not model_enabled:
            effective_reason = m_entry.get("reason")
        rows.append(Row(
            "model", "models", model, MODEL_LABELS[model],
            model_enabled, m_entry.get("reason"), provider, (),
            provider_enabled and model_enabled, effective_reason,
        ))
    # Keep this pure by default; the interactive entry point supplies the
    # machine-local inspection explicitly.  This also keeps library callers
    # and tests independent of whichever vllm.toml happens to exist.
    routes = vllm_routes or {}
    for name in sorted(set(routes) | set(config.get("vllm", {}))):
        entry = config.get("vllm", {}).get(name, {"enabled": True})
        info = routes.get(name)
        details: tuple[str, ...]
        if info and info.provider:
            provider = info.provider
            details = (
                f"model: {provider.model}",
                "type: shared vLLM / OpenAI-compatible",
                f"shared compute: {'yes' if provider.shared_compute else 'no'}",
                f"concurrency: {provider.max_concurrency}",
                f"thinking default: {'on' if provider.thinking_default else 'off'}",
                f"output default: {provider.default_max_tokens} tokens",
                f"output cap: {provider.max_tokens_cap} tokens",
                "credential: configured reference",
            )
            effective_enabled = bool(entry.get("enabled", True))
            effective_reason = entry.get("reason") if not effective_enabled else None
        elif info:
            details = (f"state: {info.error_kind or 'invalid configuration'}",)
            effective_enabled = False
            effective_reason = info.error
        else:
            details = ("state: local vLLM route definition missing",)
            effective_enabled = False
            effective_reason = "local vLLM route definition missing"
        rows.append(Row(
            "vllm", "vllm", name, name, entry.get("enabled", True),
            entry.get("reason"), None, details, effective_enabled, effective_reason,
        ))
    return rows


def toggle(rows: list[Row], index: int) -> list[Row]:
    """Flip one row's enabled state; enabling clears a stale reason."""
    row = rows[index]
    updated = replace(row, enabled=not row.enabled)
    if updated.enabled:
        updated = replace(updated, reason=None)
    return rows[:index] + [updated] + rows[index + 1:]


def set_reason(rows: list[Row], index: int, reason: str | None) -> list[Row]:
    updated = replace(rows[index], reason=(reason or None))
    return rows[:index] + [updated] + rows[index + 1:]


def rows_to_config(original: dict[str, Any], rows: list[Row]) -> dict[str, Any]:
    updated = {section: {n: dict(e) for n, e in entries.items()} for section, entries in original.items()}
    for row in rows:
        updated.setdefault(row.section, {})
        entry: dict[str, Any] = {"enabled": row.enabled}
        if row.reason:
            entry["reason"] = row.reason
        updated[row.section][row.name] = entry
    return updated


def diff_summary(original: dict[str, Any], rows: list[Row]) -> list[str]:
    lines = []
    for row in rows:
        before = original.get(row.section, {}).get(row.name, {"enabled": True})
        before_reason = before.get("reason") or None
        if before.get("enabled", True) != row.enabled or before_reason != row.reason:
            before_state = "enabled" if before.get("enabled", True) else "disabled"
            after_state = "enabled" if row.enabled else "disabled"
            reason = f' (reason: "{row.reason}")' if row.reason else ""
            lines.append(f"{row.name}: {before_state} -> {after_state}{reason}")
    return lines


def _row_height(row: Row) -> int:
    return 1 + len(row.details)


def _render(stdscr, rows: list[Row], cursor: int, *, title: str = "Ekalavya Availability") -> None:
    import curses

    stdscr.erase()
    height, width = stdscr.getmaxyx()
    stdscr.addstr(0, 0, title[: max(1, width - 1)], curses.A_BOLD)
    screen_lines: list[tuple[str, int, int | None]] = []
    section_titles = (("providers", "Providers"), ("models", "Models"), ("vllm", "Routes / vLLM"))
    for section, title in section_titles:
        section_rows = [(i, row) for i, row in enumerate(rows) if row.section == section]
        if not section_rows:
            continue
        screen_lines.append((title, curses.A_BOLD, None))
        for i, row in section_rows:
            indent = "    " if row.kind == "model" else ""
            box = "[x]" if row.enabled else "[ ]"
            badge = "   PAYG · experimental" if routing.PROVIDER_BILLING.get(row.provider or row.name) == "payg" else ""
            line = f"{indent}{box} {row.label}{badge}"
            attr = curses.A_REVERSE if i == cursor else curses.A_NORMAL
            screen_lines.append((line, attr, i))
            configured = "enabled" if row.enabled else "disabled"
            effective = "enabled" if row.effective_enabled is not False else "disabled"
            status = f"configured: {configured} · effective: {effective}"
            if row.effective_reason:
                status += f" ({row.effective_reason})"
            screen_lines.append((status, curses.A_DIM, None))
            for detail in row.details:
                screen_lines.append((detail, curses.A_DIM, None))
    cursor_line = next((line for line, (_, _, row_index) in enumerate(screen_lines) if row_index == cursor), 0)
    visible_height = max(1, height - 6)
    start = max(0, min(cursor_line, max(0, len(screen_lines) - visible_height)))
    for y, (text, attr, _) in enumerate(screen_lines[start:start + visible_height], start=2):
        try:
            stdscr.addstr(y, 0, text[: max(1, width - 1)], attr)
        except Exception:
            pass
    footer_y = max(0, height - 3)
    footer = "↑/↓ navigate  Space toggle  r reason  Enter/s save  q cancel"
    try:
        stdscr.addstr(footer_y, 0, footer[: max(1, width - 1)])
    except Exception:
        pass
    stdscr.refresh()


def _prompt_reason(stdscr) -> str | None:
    import curses

    height, width = stdscr.getmaxyx()
    y = max(0, height - 2)
    prompt = "Reason (blank to clear), Enter to confirm: "
    curses.echo()
    try:
        stdscr.addstr(y, 0, prompt[: max(1, width - 1)])
        stdscr.clrtoeol()
        stdscr.refresh()
        col = min(len(prompt), max(0, width - 2))
        raw = stdscr.getstr(y, col).decode("utf-8", errors="replace")
    except Exception:
        raw = ""
    finally:
        curses.noecho()
    return raw.strip() or None


def _prompt_text(stdscr, prompt: str) -> str | None:
    import curses
    height, width = stdscr.getmaxyx(); y = max(0, height - 2)
    curses.echo()
    try:
        stdscr.addstr(y, 0, prompt[: max(1, width - 1)]); stdscr.clrtoeol(); stdscr.refresh()
        raw = stdscr.getstr(y, min(len(prompt), max(0, width - 2))).decode("utf-8", errors="replace").strip()
    except Exception:
        raw = ""
    finally:
        curses.noecho()
    return raw or None


def _routing_setup_loop(stdscr, config: dict[str, Any]) -> dict[str, Any] | None:
    """Small staged setup editor for ordered task preferences only."""
    import curses
    updated = config
    while True:
        stdscr.erase(); height, width = stdscr.getmaxyx()
        stdscr.addstr(0, 0, "Ekalavya Setup — Route preferences", curses.A_BOLD)
        stdscr.addstr(2, 0, "a add/update task   d delete task   s save   q cancel")
        preferences = updated.get("routing", {}).get("preferences", {})
        y = 4
        if not preferences:
            stdscr.addstr(y, 0, "No task preferences configured (native/default policy remains unchanged).")
        for task in sorted(preferences):
            targets = preferences[task].get("preferred_targets", [])
            stdscr.addstr(y, 0, f"{task}: " + " > ".join(targets)[: max(1, width - len(task) - 3)])
            y += 1
            if y >= height - 3: break
        stdscr.refresh(); key = stdscr.getch()
        if key in (curses.KEY_ENTER, 10, 13, ord("s")):
            return updated
        if key == ord("q"):
            return None
        if key == ord("a"):
            task = _prompt_text(stdscr, "Task slug: ")
            targets = _prompt_text(stdscr, "Ordered targets (comma-separated; e.g. sonnet,primary-native): ")
            if task and targets:
                try:
                    updated = set_routing_preference(updated, task, "preferred_targets", [item.strip() for item in targets.split(",") if item.strip()])
                except ValueError as exc:
                    _prompt_text(stdscr, f"Invalid preference ({exc}); Enter: ")
        if key == ord("d"):
            task = _prompt_text(stdscr, "Task slug to delete: ")
            if task:
                try:
                    updated = set_routing_preference(updated, task, "preferred_targets", [])
                except ValueError as exc:
                    _prompt_text(stdscr, f"Invalid task ({exc}); Enter: ")


def _loop(stdscr, rows: list[Row], *, title: str = "Ekalavya Availability") -> list[Row] | None:
    import curses

    curses.curs_set(0)
    cursor = 0
    while True:
        _render(stdscr, rows, cursor, title=title)
        key = stdscr.getch()
        if key in (curses.KEY_UP, ord("k")):
            cursor = max(0, cursor - 1)
        elif key in (curses.KEY_DOWN, ord("j")):
            cursor = min(len(rows) - 1, cursor + 1)
        elif key == ord(" "):
            rows = toggle(rows, cursor)
        elif key == ord("r"):
            reason = _prompt_reason(stdscr)
            rows = set_reason(rows, cursor, reason)
        elif key in (curses.KEY_ENTER, 10, 13, ord("s")):
            return rows
        elif key == ord("q"):
            return None


def run_interactive_config() -> int:
    import curses

    config = load_config()
    rows = build_rows(config, inspect_vllm_routes())
    result = curses.wrapper(_loop, rows)
    if result is None:
        print("eka config: cancelled, no changes made")
        return 0
    changes = diff_summary(config, result)
    if not changes:
        print("eka config: no changes made")
        return 0
    save_config(rows_to_config(config, result))
    print("eka config: saved")
    for line in changes:
        print(f"  {line}")
    return 0


def setup_detection(which=shutil.which) -> dict[str, dict[str, object]]:
    """Report executable presence only; authentication is intentionally unknown."""
    result: dict[str, dict[str, object]] = {}
    for provider in PROVIDER_ORDER:
        executables = sorted({routing.ROUTE_EXECUTABLE[route] for route in routing.MODELS if routing.ROUTE_PROVIDER[route] == provider and routing.ROUTE_EXECUTABLE[route]})
        found = [name for name in executables if which(name)]
        result[provider] = {
            "label": PROVIDER_LABELS[provider],
            "executables": executables,
            "detected": bool(found),
            "detected_executables": found,
            "authentication": "not checked",
        }
    return result


def setup_readiness(config: dict[str, Any], detection: dict[str, dict[str, object]]) -> dict[str, object]:
    """Keep configured selection separate from executable/auth readiness."""
    providers: list[dict[str, object]] = []
    warnings: list[str] = []
    for provider in PROVIDER_ORDER:
        selected = bool(config["providers"][provider].get("enabled", True))
        models = [name for name in routing.MODELS if routing.ROUTE_PROVIDER[name] == provider and config["models"][name].get("enabled", True)]
        if selected and not models:
            warnings.append(f"{PROVIDER_LABELS[provider]} is selected with no enabled model profiles")
        providers.append({
            "provider": provider,
            "label": PROVIDER_LABELS[provider],
            "selected": selected,
            "selected_models": models,
            "detected": detection[provider]["detected"],
            "authentication": "not checked",
        })
    return {"providers": providers, "warnings": warnings}


def build_setup_rows(config: dict[str, Any], detection: dict[str, dict[str, object]], vllm_routes: dict[str, VLLMRouteInfo] | None = None) -> list[Row]:
    """Reuse availability rows while adding advisory setup information."""
    rows: list[Row] = []
    for row in build_rows(config, vllm_routes):
        if row.kind == "provider":
            item = detection[row.name]
            state = "detected" if item["detected"] else "not detected / optional"
            rows.append(replace(row, details=(f"harness: {state}", "authentication: not checked")))
        elif row.kind == "model":
            rows.append(replace(row, details=(f"profile id: {row.name}",)))
        else:
            rows.append(row)
    return rows


def run_interactive_setup() -> dict[str, object]:
    """Stage onboarding choices in the existing checkbox UI and save on demand."""
    import curses

    config = load_config()
    detection = setup_detection()
    rows = build_setup_rows(config, detection, inspect_vllm_routes())
    result = curses.wrapper(lambda stdscr: _loop(stdscr, rows, title="Ekalavya Setup"))
    if result is None:
        return {"cancelled": True, "changed": False, "readiness": setup_readiness(config, detection)}
    availability_updated = rows_to_config(config, result)
    routing_updated = curses.wrapper(lambda stdscr: _routing_setup_loop(stdscr, availability_updated))
    if routing_updated is None:
        return {"cancelled": True, "changed": False, "readiness": setup_readiness(config, detection)}
    # Compatibility for thin test/fake wrappers that return the availability
    # rows for every wrapper call rather than driving the second editor.
    if not isinstance(routing_updated, dict):
        routing_updated = availability_updated
    updated = routing_updated
    changes = diff_summary(config, result)
    if updated.get("routing") != config.get("routing"):
        changes.append("routing preferences updated")
    if changes:
        save_config(updated)
    return {
        "cancelled": False,
        "changed": bool(changes),
        "changes": changes,
        "config": updated,
        "readiness": setup_readiness(updated, detection),
    }
