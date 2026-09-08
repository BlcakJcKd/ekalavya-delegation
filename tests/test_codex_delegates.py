"""Zero-model regression coverage for the stable OpenAI/Codex routes."""

from __future__ import annotations

import json
import subprocess
import tomllib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from delegation.config import load_config
from delegation.core import DELEGATES, build_argv, run_consultation


TASK = "Inspect only; return a concise consultation and do not modify files."


def _scope(root: Path) -> Path:
    workspace = root / "scope"
    workspace.mkdir()
    (workspace / ".delegation-scope.json").write_text('{"mode": "read-only"}\n')
    return workspace


class CodexRoutePinningTests(unittest.TestCase):
    def test_terra_and_luna_use_current_catalog_models_and_codex_transport(self):
        self.assertEqual(DELEGATES["terra"].executable, "codex")
        self.assertEqual(DELEGATES["terra"].model, "gpt-5.6-terra")
        self.assertEqual(DELEGATES["terra"].effort, "medium")
        self.assertEqual(DELEGATES["luna"].executable, "codex")
        self.assertEqual(DELEGATES["luna"].model, "gpt-5.6-luna")
        self.assertEqual(DELEGATES["luna"].effort, "medium")

    def test_codex_argv_is_read_only_and_has_no_json_or_output_file_requirement(self):
        for name in ("terra", "luna"):
            command = build_argv(DELEGATES[name], Path("/tmp/scoped workspace"), TASK)
            self.assertEqual(command[:2], ["codex", "exec"])
            self.assertEqual(command[command.index("--sandbox") + 1], "read-only")
            self.assertEqual(command[command.index("--model") + 1], DELEGATES[name].model)
            self.assertEqual(
                command[command.index("--config") + 1],
                'model_reasoning_effort="medium"',
            )
            self.assertNotIn("--json", command)
            self.assertNotIn("--output-last-message", command)
            self.assertIn(TASK, command[-1])
            self.assertNotIn("dangerously", " ".join(command))


class CodexRouteGuardAndContractTests(unittest.TestCase):
    def _fake_run(self, calls):
        def run(argv, **kwargs):
            calls.append((argv, kwargs))
            return subprocess.CompletedProcess(argv, 0, stdout="codex consultation\n", stderr="codex diagnostic\n")
        return run

    def test_claude_can_use_both_routes_and_audit_keeps_provider_transport_distinct(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = _scope(root)
            calls = []
            with patch("delegation.core.shutil.which", return_value="/usr/bin/codex"):
                for name in ("terra", "luna"):
                    code, record_dir = run_consultation(
                        name, workspace, TASK, log_root=root / name,
                        primary="claude-code", run=self._fake_run(calls),
                    )
                    self.assertEqual(code, 0)
                    metadata = json.loads((record_dir / "execution.json").read_text())
                    self.assertEqual(metadata["provider"], "codex")
                    self.assertEqual(metadata["transport"], "codex")
                    self.assertEqual(metadata["response_status"], "text-returned")
            self.assertEqual(len(calls), 2)

    def test_codex_primary_is_rejected_before_codex_process_launch(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = _scope(root)
            calls = []
            for name in ("terra", "luna"):
                with self.assertRaisesRegex(ValueError, "same-provider external delegation disabled"):
                    run_consultation(
                        name, workspace, TASK, log_root=root / name,
                        primary="codex", run=self._fake_run(calls),
                    )
            self.assertEqual(calls, [])

    def test_empty_response_and_timeout_semantics_are_unchanged(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            record = root / "empty-record"
            record.mkdir()
            (record / "stdout.txt").write_text(" \n")
            (record / "stderr.txt").write_text("")
            workspace = _scope(root)
            def timeout_run(argv, **kwargs):
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"], output="partial", stderr="quiet")

            with patch("delegation.core.shutil.which", return_value="/usr/bin/codex"):
                code, record_dir = run_consultation(
                    "luna", workspace, TASK, timeout_seconds=90,
                    log_root=root / "timeout-logs", primary="claude-code", run=timeout_run,
                )
            metadata = json.loads((record_dir / "execution.json").read_text())
            self.assertEqual(code, 124)
            self.assertTrue(metadata["timed_out"])
            self.assertEqual(metadata["timeout_seconds"], 90)


class CodexRouteInstallAndMigrationTests(unittest.TestCase):
    def test_existing_config_choices_are_preserved_when_new_route_entries_are_loaded(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text(
                "[providers.codex]\nenabled = true\n"
                "[providers.claude]\nenabled = true\n"
                "[models.terra]\nenabled = false\nreason = \"user choice\"\n"
                "[models.luna]\nenabled = true\n"
            )
            config = load_config(path)
            self.assertEqual(config["models"]["terra"], {"enabled": False, "reason": "user choice"})
            self.assertEqual(config["models"]["luna"], {"enabled": True})
            self.assertNotIn("api_key", path.read_text())

    def test_packaging_exposes_only_the_canonical_control_plane(self):
        root = Path(__file__).resolve().parents[1]
        metadata = tomllib.loads((root / "pyproject.toml").read_text())
        installer = (root / "scripts" / "install-user-delegation.sh").read_text()
        self.assertEqual(set(metadata["project"]["scripts"]), {"eka", "ekalavya"})
        self.assertIn("Canonical commands:", installer)
        self.assertIn("ensure_control_files", installer)
        self.assertIn("Existing control file found, left untouched", installer)
        self.assertNotIn("python -m unittest discover", installer)
        self.assertNotIn("python -m delegation.preflight", installer)
        self.assertNotIn("agent-delegation", installer)


if __name__ == "__main__":
    unittest.main()
