from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "scripts" / "install-user-delegation.sh"


class InstallerTests(unittest.TestCase):
    def _write_executable(self, path: Path, contents: str) -> None:
        path.write_text(contents)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)

    def _run_installer(self, root: Path, *, fail_control_bootstrap: bool = False) -> subprocess.CompletedProcess[str]:
        fake_bin = root / "bin"
        fake_bin.mkdir(exist_ok=True)
        self._write_executable(
            fake_bin / "pipx",
            """#!/bin/sh
set -eu
case "$1" in
  install)
    venv="$PIPX_HOME/venvs/ekalavya-delegation"
    mkdir -p "$venv/bin" "$PIPX_BIN_DIR"
    for app in eka ekalavya; do
      printf '#!/bin/sh\\nexit 0\\n' > "$venv/bin/$app"
      chmod 755 "$venv/bin/$app"
      ln -sfn "$venv/bin/$app" "$PIPX_BIN_DIR/$app"
    done
    ;;
  environment)
    case "$3" in
      PIPX_HOME) printf '%s\\n' "$PIPX_HOME" ;;
      PIPX_BIN_DIR) printf '%s\\n' "$PIPX_BIN_DIR" ;;
      *) exit 2 ;;
    esac
    ;;
  *) exit 2 ;;
esac
""",
        )
        self._write_executable(
            fake_bin / "python3",
            """#!/bin/sh
if [ "${FAIL_CONTROL_BOOTSTRAP:-0}" = 1 ] && [ "${1:-}" = "-" ]; then
  case "${2:-}" in
    *.toml) ;;
    *) echo 'forced control bootstrap failure' >&2; exit 42 ;;
  esac
fi
exec /usr/bin/python3 "$@"
""",
        )
        config_home = root / "config"
        env = os.environ.copy()
        env.update({
            "HOME": str(root / "home"),
            "XDG_CONFIG_HOME": str(config_home),
            "XDG_STATE_HOME": str(root / "state"),
            "PIPX_HOME": str(root / "pipx"),
            "PIPX_BIN_DIR": str(root / "pipx-bin"),
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "PYTHONPATH": str(ROOT),
            "FAIL_CONTROL_BOOTSTRAP": "1" if fail_control_bootstrap else "0",
        })
        return subprocess.run(["bash", str(INSTALLER)], cwd=ROOT, env=env, text=True, input="", capture_output=True)

    def test_clean_install_creates_private_control_state_without_legacy_migration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            legacy = root / "config" / "agent-delegation"
            legacy.mkdir(parents=True)
            (legacy / "legacy.txt").write_text("must remain outside the new install")
            result = self._run_installer(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            config = root / "config" / "ekalavya"
            self.assertEqual({path.name for path in config.iterdir()}, {"config.toml", "catalogue.json", "profiles.json"})
            self.assertTrue((root / "home" / ".agents" / "skills" / "delegation" / "SKILL.md").is_file())
            self.assertEqual(stat.S_IMODE(config.stat().st_mode), 0o700)
            for name in ("config.toml", "catalogue.json", "profiles.json"):
                self.assertEqual(stat.S_IMODE((config / name).stat().st_mode), 0o600)
            self.assertEqual((legacy / "legacy.txt").read_text(), "must remain outside the new install")
            self.assertFalse((config / "legacy.txt").exists())
            self.assertNotIn("agy models", result.stdout + result.stderr)

    def test_reinstall_is_idempotent_and_preserves_user_owned_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertEqual(self._run_installer(root).returncode, 0)
            config = root / "config" / "ekalavya"
            owned = {name: (config / name).read_bytes() for name in ("config.toml", "catalogue.json", "profiles.json")}
            for name, contents in owned.items():
                (config / name).write_bytes(b"user-owned-" + contents)
            result = self._run_installer(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            for name, contents in owned.items():
                self.assertEqual((config / name).read_bytes(), b"user-owned-" + contents)

    def test_required_control_bootstrap_failure_stops_before_skill_installation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = self._run_installer(root, fail_control_bootstrap=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Core Ekalavya control-file bootstrap failed", result.stderr)
            config = root / "config" / "ekalavya"
            self.assertTrue((config / "config.toml").is_file())
            self.assertFalse((config / "catalogue.json").exists())
            self.assertFalse((config / "profiles.json").exists())
            self.assertFalse((root / "home" / ".agents" / "skills" / "delegation" / "SKILL.md").exists())

    def test_incomplete_control_file_pair_is_preserved_and_fails_clearly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "config" / "ekalavya"
            config.mkdir(parents=True)
            (config / "config.toml").write_text("user config\n")
            catalogue = config / "catalogue.json"
            catalogue.write_text("user catalogue\n")
            result = self._run_installer(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing profiles.json", result.stderr)
            self.assertIn("Core Ekalavya control-file bootstrap failed", result.stderr)
            self.assertEqual(catalogue.read_text(), "user catalogue\n")
            self.assertFalse((config / "profiles.json").exists())
            self.assertFalse((root / "home" / ".agents" / "skills" / "delegation" / "SKILL.md").exists())


if __name__ == "__main__":
    unittest.main()
