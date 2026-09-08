"""Explicit, reversible migration of legacy user state into Ekalavya."""

from __future__ import annotations

import os
from pathlib import Path

from .config import ensure_control_files, migrate_legacy_config
from .ledger import connect, default_state_dir, import_execution_runs, import_legacy_state


def migrate_all(*, legacy_config: Path | None = None, new_config: Path | None = None, legacy_state: Path | None = None, db: Path | None = None) -> dict[str, object]:
    config_report = migrate_legacy_config(legacy_config, new_config)
    control_files = ensure_control_files(new_config)
    config_report["control_files"] = control_files
    state_base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")).expanduser()
    ledger_path = str(db or (default_state_dir() / "ledger.sqlite3"))
    if control_files.get("status") == "incomplete":
        return {"config": config_report, "state": {"skipped": "incomplete catalogue/profiles pair"}, "ledger": ledger_path, "reversible": True}
    old_state = legacy_state or (state_base / "agent-delegation")
    conn = connect(db)
    state_report = import_legacy_state(conn, old_state)
    state_report["structured_runs_imported"] = import_execution_runs(conn, old_state)
    return {"config": config_report, "state": state_report, "ledger": ledger_path, "reversible": True}
