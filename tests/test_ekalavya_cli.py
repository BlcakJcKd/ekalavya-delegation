import io
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ekalavya.cli import main


class EkalavyaCliTests(unittest.TestCase):
    def _xdg(self, root: Path):
        return patch.dict(os.environ, {"XDG_CONFIG_HOME": str(root / "config"), "XDG_STATE_HOME": str(root / "state")}, clear=False)

    def _files(self, root: Path):
        config = root / "config" / "ekalavya"
        config.mkdir(parents=True)
        identity = {"provider": "claude", "family": "haiku", "provider_model_id": "claude-haiku", "display_name": "haiku", "capabilities": {"reasoning_values": ["medium"]}}
        identity["identity_key"] = "haiku-key"
        identity["lifecycle"] = "current"
        (config / "catalogue.json").write_text(json.dumps([identity]))
        (config / "profiles.json").write_text(json.dumps([{"name": "haiku", "default_identity_key": "haiku-key", "permitted_candidates": ["haiku-key"], "reasoning_policy": "fixed", "default_reasoning": "medium"}]))

    def test_status_accepts_primary_and_is_network_free(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self._files(root)
            with self._xdg(root), patch("ekalavya.cli.build_report", return_value={"routes": [], "live_vllm": {}}):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(main(["status", "--primary", "codex", "--json"]), 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["primary"], "codex")
            self.assertIn("routing", payload)

    def test_status_human_view_shows_provider_and_effective_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self._files(root)
            report = {"config_path": "config", "vllm_config_path": "vllm", "state_log_path": "state", "runtime_version": "test", "skill": {"source_installed": False, "source_path": "skill", "claude_code_discovers": False}, "declared_primary": "manual", "quota": "unknown", "routes": [{"route": "deepseek-flash", "provider": "deepseek", "transport": "codex", "billing": "payg", "maturity": "experimental", "configured_enabled": True, "configured_reason": None, "provider_enabled": False, "provider_reason": "peak hours", "effective_enabled": False, "route_type": "external", "effective": "disabled", "effective_reason": "peak hours", "model": None, "shared_compute": None, "max_concurrency": None, "thinking_default": None, "default_max_tokens": None, "max_tokens_cap": None}]}
            with self._xdg(root), patch("ekalavya.cli.build_report", return_value=report):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(main(["status"]), 0)
            text = output.getvalue()
            self.assertIn("Model cfg", text)
            self.assertIn("disabled", text)
            self.assertIn("peak hours", text)

    def test_config_mutation_is_explicit_and_json(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self._files(root)
            with self._xdg(root):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(main(["config", "disable-provider", "claude", "--reason", "maintenance", "--json"]), 0)
                payload = json.loads(output.getvalue())
                self.assertFalse(payload["availability"]["providers"]["claude"]["enabled"])

    def test_config_no_action_uses_tui_only_for_a_tty(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self._files(root)
            with self._xdg(root), patch("ekalavya.cli.sys.stdin.isatty", return_value=True), \
                 patch("ekalavya.cli.sys.stdout.isatty", return_value=True), \
                 patch("delegation.config_tui.run_interactive_config", return_value=0) as tui:
                self.assertEqual(main(["config"]), 0)
            tui.assert_called_once_with()

    def test_config_json_bypasses_tui_and_is_machine_readable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self._files(root)
            with self._xdg(root), patch("ekalavya.cli.sys.stdin.isatty", return_value=True), \
                 patch("ekalavya.cli.sys.stdout.isatty", return_value=True), \
                 patch("delegation.config_tui.run_interactive_config") as tui:
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(main(["config", "--json"]), 0)
                json.loads(output.getvalue())
            tui.assert_not_called()

    def test_model_promotion_requires_and_records_explicit_basis(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self._files(root)
            with self._xdg(root):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(main(["models", "promote", "haiku-key", "--basis", "operational_efficiency", "--promotion-reason", "measured efficiency", "--set-default", "--profile", "haiku", "--default-reasoning", "low", "--json"]), 0)
                payload = json.loads(output.getvalue())
                self.assertEqual(payload["promotion_basis"], "operational_efficiency")
                catalogue = json.loads((root / "config" / "ekalavya" / "catalogue.json").read_text())
                profile = json.loads((root / "config" / "ekalavya" / "profiles.json").read_text())[0]
                self.assertEqual(catalogue[0]["promotion_basis"], "operational_efficiency")
                self.assertEqual(profile["default_identity_key"], "haiku-key")
                self.assertEqual(profile["default_reasoning"], "low")

    def test_unknown_targets_are_rejected_without_persisting_new_keys(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self._files(root)
            with self._xdg(root):
                for argv, text in (
                    (["config", "disable-provider", "minimx"], "unknown provider"),
                    (["config", "disable-model", "deepseek-v4-flash"], "unknown model"),
                    (["config", "disable", "route-that-does-not-exist"], "unknown model or vLLM route"),
                ):
                    error = io.StringIO()
                    with redirect_stderr(error):
                        self.assertEqual(main(argv), 2)
                    self.assertIn(text, error.getvalue())
                self.assertFalse((root / "config" / "ekalavya" / "config.toml").exists())

    def test_doctor_reports_incomplete_control_file_pair(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._files(root)
            (root / "config" / "ekalavya" / "profiles.json").unlink()
            (root / "state" / "ekalavya").mkdir(parents=True)
            with self._xdg(root):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(main(["doctor", "--json"]), 1)
            self.assertFalse(json.loads(output.getvalue())["control_file_pair_complete"])

    def test_public_package_scripts_are_exactly_ekalavya_and_eka(self):
        import tomllib
        metadata = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
        self.assertEqual(set(metadata["project"]["scripts"]), {"eka", "ekalavya"})

    def test_required_run_persistence_remains_fail_closed_before_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self._files(root)
            prompt = root / "prompt.md"; prompt.write_text("review")
            with self._xdg(root), patch("ekalavya.cli.record_run", side_effect=OSError("ledger unavailable")), patch("ekalavya.cli.execute") as execute:
                with self.assertRaises(OSError): main(["run", "haiku", "--workspace", str(root), "--prompt-file", str(prompt), "--task", "review"])
            execute.assert_not_called()

    def test_optional_telemetry_failure_does_not_invalidate_completed_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self._files(root)
            prompt = root / "prompt.md"; prompt.write_text("review")
            evidence = root / "evidence"; evidence.mkdir(); (evidence / "execution.json").write_text("{}")
            with self._xdg(root), patch("ekalavya.cli.execute", return_value={"state": "completed", "evidence": str(evidence)}), patch("ekalavya.cli.persist_execution_observability", side_effect=OSError("analytics unavailable")):
                self.assertEqual(main(["run", "haiku", "--workspace", str(root), "--prompt-file", str(prompt), "--task", "review"]), 0)
            conn = sqlite3.connect(root / "state" / "ekalavya" / "ledger.sqlite3")
            self.assertEqual(conn.execute("SELECT status FROM runs").fetchone()[0], "completed")


if __name__ == "__main__":
    unittest.main()
