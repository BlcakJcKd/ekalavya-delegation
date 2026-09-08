#!/usr/bin/env bash
# Install the delegation runtime as user-level, cross-project commands.
#
# What this does, in order:
#   1. `pipx install --force` the `ekalavya` package from this checkout,
#      providing the repo-independent `ekalavya` and `eka` control-plane
#      commands (pipx builds and copies the package into its own venv).
#   2. create initial XDG config.toml, catalogue.json, and profiles.json
#      only when each is absent
#   3. install the delegation skill to ~/.agents/skills/delegation/SKILL.md
#      (a copy, not a symlink into this repo -- see docs/USER_INSTALLATION.md
#      for why) and link it for Claude Code discovery if that pattern is
#      present on this machine
#
# No sudo. No provider authentication changes. No model calls. Safe to
# re-run for an update/reinstall -- it never touches an existing config, and
# only ever writes its own skill file.
#
# Usage:
#   scripts/install-user-delegation.sh            # install/update
#   scripts/install-user-delegation.sh --uninstall # remove the runtime
#
# Uninstall removes the pipx install and the skill files it created. It does
# NOT delete your config (~/.config/ekalavya/) or logs
# (~/.local/state/ekalavya/) -- remove those yourself if you want a full wipe.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE_NAME="ekalavya-delegation"

XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
XDG_STATE_HOME="${XDG_STATE_HOME:-$HOME/.local/state}"
CONFIG_DIR="$XDG_CONFIG_HOME/ekalavya"
CONFIG_FILE="$CONFIG_DIR/config.toml"
STATE_LOG_DIR="$XDG_STATE_HOME/ekalavya/delegate_runs"

AGENTS_SKILL_DIR="$HOME/.agents/skills/delegation"
AGENTS_SKILL_FILE="$AGENTS_SKILL_DIR/SKILL.md"
CLAUDE_SKILL_LINK="$HOME/.claude/skills/delegation"

uninstall() {
  echo "Uninstalling ${PACKAGE_NAME} (pipx)..."
  if command -v pipx >/dev/null 2>&1; then
    pipx uninstall "$PACKAGE_NAME" || echo "  (pipx reported nothing to uninstall)"
  else
    echo "  pipx not found; nothing to uninstall via pipx"
  fi
  if [ -L "$CLAUDE_SKILL_LINK" ]; then
    rm "$CLAUDE_SKILL_LINK"
    echo "Removed $CLAUDE_SKILL_LINK"
  fi
  if [ -f "$AGENTS_SKILL_FILE" ]; then
    rm "$AGENTS_SKILL_FILE"
    rmdir "$AGENTS_SKILL_DIR" 2>/dev/null || true
    echo "Removed $AGENTS_SKILL_FILE"
  fi
  echo "Config preserved at: $CONFIG_FILE (delete manually if you want a full wipe)"
  echo "Logs preserved at:   $STATE_LOG_DIR"
  exit 0
}

if [ "${1:-}" = "--uninstall" ]; then
  uninstall
fi

echo "== 1/3: installing user-level commands (pipx) =="
cd "$REPO_ROOT"
if ! command -v pipx >/dev/null 2>&1; then
  echo "pipx not found on PATH. Install it first (no sudo needed), e.g.:" >&2
  echo "  python3 -m pip install --user pipx && python3 -m pipx ensurepath" >&2
  exit 1
fi
pipx install --force "$REPO_ROOT"

# pipx can leave dangling app links when a package drops console scripts.
# Remove only links whose target is this package's venv, preserving unrelated
# commands and the two canonical links installed by the current package.
PIPX_HOME_DIR="$(pipx environment --value PIPX_HOME)"
PIPX_BIN_DIR="$(pipx environment --value PIPX_BIN_DIR)"
PIPX_VENV_DIR="$PIPX_HOME_DIR/venvs/$PACKAGE_NAME"
if [ -d "$PIPX_BIN_DIR" ]; then
  for app_link in "$PIPX_BIN_DIR"/*; do
    [ -L "$app_link" ] || continue
    app_name="$(basename "$app_link")"
    case "$app_name" in
      eka|ekalavya) continue ;;
    esac
    if [ "$(readlink "$app_link" 2>/dev/null || true)" = "$PIPX_VENV_DIR/bin/$app_name" ]; then
      rm "$app_link"
      echo "Removed stale Ekalavya app link: $app_link"
    fi
  done
fi

echo
echo "== shorthand collision check =="
CANONICAL_BIN="$(command -v ekalavya || true)"
if [ -n "${CANONICAL_BIN}" ]; then
  BIN_DIR="$(dirname "${CANONICAL_BIN}")"
  EXISTING_EKA="$(command -v eka || true)"
  if [ -z "${EXISTING_EKA}" ]; then
    ln -s "${CANONICAL_BIN}" "${BIN_DIR}/eka"
    echo "Installed shorthand: ${BIN_DIR}/eka -> ${CANONICAL_BIN}"
  elif [ "$(readlink "${EXISTING_EKA}" 2>/dev/null || true)" = "$PIPX_VENV_DIR/bin/eka" ] || [ "$(readlink -f "${EXISTING_EKA}" 2>/dev/null || true)" = "$(readlink -f "${CANONICAL_BIN}")" ]; then
    echo "Shorthand already points to this Ekalavya installation: ${EXISTING_EKA}"
  else
    echo "Shorthand collision: eka already resolves to ${EXISTING_EKA}; left it untouched" >&2
  fi
fi

echo
echo "== 2/3: initial config and control files =="
if [ -f "$CONFIG_FILE" ]; then
  echo "Existing config found, left untouched: $CONFIG_FILE"
else
  mkdir -p "$CONFIG_DIR"
  if ! python3 - "$CONFIG_FILE" <<'PY'
from delegation.config import default_config, save_config
from pathlib import Path
import sys
save_config(default_config(), Path(sys.argv[1]))
PY
  then
    echo "Core Ekalavya config bootstrap failed; installation aborted." >&2
    exit 1
  fi
  echo "Created default config: $CONFIG_FILE"
fi
if ! python3 - "$CONFIG_DIR" <<'PY'
from ekalavya.config import ensure_control_files
from pathlib import Path
import sys
config_dir = Path(sys.argv[1])
result = ensure_control_files(config_dir)
if result.get('status') == 'incomplete':
    missing = ', '.join(result.get('missing', []))
    raise SystemExit(f"Incomplete Ekalavya control-file pair; missing {missing}. Existing files were left untouched.")
for name in result['created']:
    print(f'Created default control file: {config_dir / name}')
for name in result['skipped']:
    print(f'Existing control file found, left untouched: {config_dir / name}')
PY
then
  echo "Core Ekalavya control-file bootstrap failed; installation aborted." >&2
  exit 1
fi

echo
echo "== 3/3: installing skill =="
mkdir -p "$AGENTS_SKILL_DIR"
cp "$REPO_ROOT/skills/delegation/SKILL.md" "$AGENTS_SKILL_FILE"
echo "Installed skill: $AGENTS_SKILL_FILE"
if [ -d "$HOME/.claude/skills" ]; then
  if [ -e "$CLAUDE_SKILL_LINK" ] || [ -L "$CLAUDE_SKILL_LINK" ]; then
    if [ -L "$CLAUDE_SKILL_LINK" ] && [ "$(readlink "$CLAUDE_SKILL_LINK")" = "../../.agents/skills/delegation" ]; then
      echo "Claude Code skill link already present: $CLAUDE_SKILL_LINK"
    else
      echo "Skipped: $CLAUDE_SKILL_LINK already exists and is not our link; not overwriting unrelated content"
    fi
  else
    ln -s "../../.agents/skills/delegation" "$CLAUDE_SKILL_LINK"
    echo "Linked for Claude Code: $CLAUDE_SKILL_LINK -> ../../.agents/skills/delegation"
  fi
else
  echo "No ~/.claude/skills directory found; skipping Claude Code link (skill source is still installed)"
fi

echo
echo "== Done =="
echo "Canonical commands:"
for cmd in ekalavya eka; do
  path="$(command -v "$cmd" 2>/dev/null || echo "NOT ON PATH")"
  echo "  $cmd: $path"
done
echo "Config: $CONFIG_FILE"
echo "Logs:   $STATE_LOG_DIR"
echo "Skill:  $AGENTS_SKILL_FILE"
echo
echo "If a command shows 'NOT ON PATH', run: pipx ensurepath (then open a new shell)"

# Provider/profile selection belongs to the explicit setup command.  Never
# attempt curses input in scripts, CI, or AI-agent installs without a TTY.
if [ -t 0 ] && [ -t 1 ]; then
  read -r -p "Run Ekalavya setup now? [y/N] " SETUP_REPLY || SETUP_REPLY=""
  case "$SETUP_REPLY" in
    [yY]|[yY][eE][sS]) ekalavya setup ;;
    *) echo "Next: Run eka setup to choose integrations and model profiles." ;;
  esac
else
  echo "Next: Run eka setup to choose integrations and model profiles."
fi
