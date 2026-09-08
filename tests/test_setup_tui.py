from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from delegation.config import default_config, set_enabled
from delegation.config_tui import build_setup_rows, rows_to_config, run_interactive_setup, setup_detection, setup_readiness, toggle
from ekalavya.cli import main


class SetupStateTests(unittest.TestCase):
    def test_detection_is_advisory_and_never_claims_authentication(self):
        detection = setup_detection(which=lambda name: "/bin/example" if name in {"agy", "codex"} else None)
        self.assertTrue(detection["gemini"]["detected"])
        self.assertFalse(detection["claude"]["detected"])
        self.assertEqual(detection["gemini"]["authentication"], "not checked")

    def test_provider_and_model_selection_remain_independent(self):
        config = set_enabled(default_config(), "providers", "deepseek", False, reason="not selected")
        config = set_enabled(config, "models", "deepseek-flash", True)
        readiness = setup_readiness(config, setup_detection(which=lambda _: None))
        deepseek = next(item for item in readiness["providers"] if item["provider"] == "deepseek")
        self.assertFalse(deepseek["selected"])
        self.assertIn("deepseek-flash", deepseek["selected_models"])

    def test_enabled_provider_without_model_is_reported_not_auto_filled(self):
        config = default_config()
        config = set_enabled(config, "models", "deepseek-flash", False, reason="not selected")
        config = set_enabled(config, "models", "deepseek-pro", False, reason="not selected")
        config = set_enabled(config, "providers", "deepseek", True)
        readiness = setup_readiness(config, setup_detection(which=lambda _: None))
        self.assertTrue(any("DeepSeek is selected with no enabled model profiles" in warning for warning in readiness["warnings"]))

    def test_model_profiles_can_be_selected_individually_under_one_provider(self):
        config = default_config()
        rows = build_setup_rows(config, setup_detection(which=lambda _: None))
        provider = next(i for i, row in enumerate(rows) if row.kind == "provider" and row.name == "deepseek")
        flash = next(i for i, row in enumerate(rows) if row.kind == "model" and row.name == "deepseek-flash")
        updated = rows_to_config(config, toggle(toggle(rows, provider), flash))
        readiness = setup_readiness(updated, setup_detection(which=lambda _: None))
        deepseek = next(item for item in readiness["providers"] if item["provider"] == "deepseek")
        self.assertTrue(deepseek["selected"])
        self.assertEqual(deepseek["selected_models"], ["deepseek-flash"])
        self.assertFalse(updated["models"]["deepseek-pro"]["enabled"])

    def test_rerun_can_reenable_provider_without_erasing_saved_model_choice(self):
        config = set_enabled(default_config(), "models", "deepseek-flash", True)
        disabled = set_enabled(config, "providers", "deepseek", False, reason="not selected")
        rerun = set_enabled(disabled, "providers", "deepseek", True)
        self.assertTrue(rerun["models"]["deepseek-flash"]["enabled"])
        readiness = setup_readiness(rerun, setup_detection(which=lambda _: None))
        deepseek = next(item for item in readiness["providers"] if item["provider"] == "deepseek")
        self.assertEqual(deepseek["selected_models"], ["deepseek-flash"])

    def test_setup_rows_use_human_labels_and_stable_ids(self):
        rows = build_setup_rows(default_config(), setup_detection(which=lambda name: "/bin/example" if name == "agy" else None))
        flash = next(row for row in rows if row.name == "flash")
        gemini = next(row for row in rows if row.name == "gemini")
        self.assertEqual(flash.label, "Gemini Flash")
        self.assertIn("profile id: flash", flash.details)
        self.assertIn("harness: detected", gemini.details[0])


class SetupInteractiveBoundaryTests(unittest.TestCase):
    def test_cancel_performs_zero_writes(self):
        config = default_config()
        with patch("delegation.config_tui.load_config", return_value=config), \
             patch("delegation.config_tui.setup_detection", return_value=setup_detection(which=lambda _: None)), \
             patch("delegation.config_tui.inspect_vllm_routes", return_value={}), \
             patch("curses.wrapper", return_value=None), \
             patch("delegation.config_tui.save_config") as save:
            result = run_interactive_setup()
        self.assertTrue(result["cancelled"])
        save.assert_not_called()

    def test_save_stages_only_selected_rows(self):
        config = default_config()
        detection = setup_detection(which=lambda name: "/bin/example" if name == "agy" else None)
        rows = build_setup_rows(config, detection, {})
        for name in ("claude", "codex", "deepseek", "minimax"):
            index = next(i for i, row in enumerate(rows) if row.kind == "provider" and row.name == name)
            rows = toggle(rows, index)
        with patch("delegation.config_tui.load_config", return_value=config), \
             patch("delegation.config_tui.setup_detection", return_value=detection), \
             patch("delegation.config_tui.inspect_vllm_routes", return_value={}), \
             patch("curses.wrapper", return_value=rows), \
             patch("delegation.config_tui.save_config") as save:
            result = run_interactive_setup()
        self.assertTrue(result["changed"])
        saved = save.call_args.args[0]
        self.assertTrue(saved["providers"]["gemini"]["enabled"])
        self.assertFalse(saved["providers"]["claude"]["enabled"])
        self.assertTrue(saved["models"]["flash"]["enabled"])


class SetupCliTests(unittest.TestCase):
    def test_json_is_read_only_readiness(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(root / "config"), "XDG_STATE_HOME": str(root / "state")}, clear=False):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(main(["setup", "--json"]), 0)
            self.assertIn('"interactive": false', output.getvalue())
            self.assertFalse((root / "config" / "ekalavya" / "config.toml").exists())

    def test_non_tty_never_launches_curses(self):
        with patch("ekalavya.cli.sys.stdin.isatty", return_value=False), \
             patch("ekalavya.cli.sys.stdout.isatty", return_value=False), \
             patch("delegation.config_tui.run_interactive_setup") as setup:
            error = io.StringIO()
            with redirect_stderr(error):
                self.assertEqual(main(["setup"]), 2)
        setup.assert_not_called()
        self.assertIn("interactive TTY", error.getvalue())


if __name__ == "__main__":
    unittest.main()
