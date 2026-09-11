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
from ekalavya.ledger import connect, record_resolution, record_run, record_run_observability, set_feedback


class EkalavyaCliTests(unittest.TestCase):
    def _xdg(self, root: Path):
        return patch.dict(os.environ, {"XDG_CONFIG_HOME": str(root / "config"), "XDG_STATE_HOME": str(root / "state")}, clear=False)

    def _files(self, root: Path):
        config = root / "config" / "ekalavya"
        config.mkdir(parents=True)
        identity = {"provider": "claude", "family": "haiku", "provider_model_id": "claude-haiku", "display_name": "haiku", "execution_route": "haiku", "harness": "claude", "legacy_route": "haiku", "capabilities": {"reasoning_values": ["medium"], "harness_values": ["claude"]}}
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

    def test_run_persists_safe_provider_usage_with_canonical_cache_read_key(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self._files(root)
            prompt = root / "prompt.md"; prompt.write_text("review")
            evidence = root / "evidence"; evidence.mkdir(); (evidence / "execution.json").write_text("{}")
            execution = {"state": "completed", "evidence": str(evidence), "provider_reported_usage": {"input_tokens": 100, "output_tokens": 20, "cache_read_tokens": 40, "total_tokens": 120}, "provider_reported_model_id": "served-model", "wall_seconds": 1.0}
            with self._xdg(root), patch("ekalavya.cli.execute", return_value=execution):
                self.assertEqual(main(["run", "haiku", "--workspace", str(root), "--prompt-file", str(prompt), "--task", "review"]), 0)
            conn = sqlite3.connect(root / "state" / "ekalavya" / "ledger.sqlite3")
            row = conn.execute("SELECT cache_read_tokens,cache_read_tokens_provenance,metadata_json FROM request_metrics").fetchone()
            self.assertEqual(row, (40, "provider_reported", "{}"))
            self.assertIsNone(conn.execute("SELECT cache_write_tokens FROM request_metrics").fetchone()[0])

    def test_usage_human_summary_includes_outcomes_feedback_and_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self._xdg(root):
                conn = connect()
                record_run(conn, "success", {"profile": "haiku"})
                record_resolution(conn, "success", {}, {"state": "resolved"})
                record_run_observability(conn, "success", {"task": "review", "execution_status": "success", "total_tokens": 12, "total_tokens_provenance": "harness_reported"})
                set_feedback(conn, "success", "useful")
                conn.close()
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(main(["usage", "--period", "all"]), 0)
            text = output.getvalue()
            for expected in ("Successful: 1", "Failed: 0", "Success rate: 100%", "Feedback: 1/1 rated", "useful: 1", "Local Ekalavya history is not total provider-account usage", "direct provider/web/desktop activity", "other machines"):
                self.assertIn(expected, text)

    def test_usage_prune_requires_aware_cutoff_before_database_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self._xdg(root):
                error = io.StringIO()
                with redirect_stderr(error):
                    self.assertEqual(main(["usage", "prune"]), 2)
                self.assertIn("requires --before", error.getvalue())
                self.assertFalse((root / "state" / "ekalavya" / "ledger.sqlite3").exists())
                error = io.StringIO()
                with redirect_stderr(error):
                    self.assertEqual(main(["usage", "prune", "--before", "2026-01-02T00:00:00"]), 2)
                self.assertIn("offset-aware", error.getvalue())
                self.assertFalse((root / "state" / "ekalavya" / "ledger.sqlite3").exists())

    def test_usage_prune_is_bounded_and_reset_requires_yes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self._xdg(root):
                conn = connect()
                for run_id, created_at in (("old", "2026-01-01T00:00:00+00:00"), ("new", "2026-02-01T00:00:00+00:00")):
                    record_run(conn, run_id, {"profile": "haiku"}, started_at=created_at)
                    record_resolution(conn, run_id, {}, {"state": "resolved"})
                    record_run_observability(conn, run_id, {"task": "review", "execution_status": "success", "created_at": created_at})
                conn.close()
                self.assertEqual(main(["usage", "prune", "--before", "2026-01-15T00:00:00+00:00", "--json"]), 0)
                conn = connect()
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM run_observability").fetchone()[0], 1)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0], 2)
                conn.close()
                error = io.StringIO()
                with redirect_stderr(error):
                    self.assertEqual(main(["usage", "reset"]), 2)
                self.assertIn("requires --yes", error.getvalue())
                conn = connect(); self.assertEqual(conn.execute("SELECT COUNT(*) FROM run_observability").fetchone()[0], 1); conn.close()
                self.assertEqual(main(["usage", "reset", "--yes", "--json"]), 0)
                conn = connect(); self.assertEqual(conn.execute("SELECT COUNT(*) FROM run_observability").fetchone()[0], 0); self.assertEqual(conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0], 2); conn.close()


if __name__ == "__main__":
    unittest.main()
