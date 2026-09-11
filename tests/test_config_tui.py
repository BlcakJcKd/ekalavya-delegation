"""Tests for the pure state-transition functions behind Ekalavya config UI.

The curses event loop itself is not exercised here -- driving a real
terminal in a test is impractical and unnecessary; every requirement the
task lists (toggle, reason edit, save/cancel diff, provider/model
independence) lives in these pure functions and is covered without a TTY.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from delegation.config import default_config, set_enabled
from delegation.config_tui import build_rows, diff_summary, rows_to_config, run_interactive_config, run_interactive_setup, set_reason, toggle


class BuildRowsTests(unittest.TestCase):
    def test_rows_are_grouped_by_provider_with_provider_row_first(self):
        rows = build_rows(default_config())
        by_provider = {}
        for row in rows:
            key = row.name if row.kind == "provider" else row.provider
            by_provider.setdefault(key, []).append(row)
        for provider, group in by_provider.items():
            self.assertEqual(group[0].kind, "provider")
            self.assertEqual(group[0].name, provider)
            for model_row in group[1:]:
                self.assertEqual(model_row.kind, "model")
                self.assertEqual(model_row.provider, provider)

    def test_every_provider_and_model_appears_exactly_once(self):
        rows = build_rows(default_config())
        provider_names = [r.name for r in rows if r.kind == "provider"]
        model_names = [r.name for r in rows if r.kind == "model"]
        self.assertEqual(sorted(provider_names), ["claude", "codex", "deepseek", "gemini", "minimax"])
        self.assertEqual(
            sorted(model_names),
            ["deepseek-flash", "deepseek-pro", "flash", "haiku", "luna", "minimax-m3", "sonnet", "terra"],
        )

    def test_row_state_reflects_config(self):
        config = set_enabled(default_config(), "providers", "codex", False, reason="quota low")
        rows = build_rows(config)
        codex_row = next(r for r in rows if r.kind == "provider" and r.name == "codex")
        self.assertFalse(codex_row.enabled)
        self.assertEqual(codex_row.reason, "quota low")

    def test_display_labels_do_not_rename_stable_profile_ids(self):
        rows = build_rows(default_config())
        labels = {row.name: row.label for row in rows if row.kind == "model"}
        self.assertEqual(labels["flash"], "Gemini Flash")
        self.assertEqual(labels["haiku"], "Claude Haiku")
        self.assertEqual(labels["sonnet"], "Claude Sonnet")
        self.assertEqual(labels["luna"], "Codex Luna")
        self.assertEqual(labels["terra"], "Codex Terra")
        self.assertEqual(labels["flash"], "Gemini Flash")
        self.assertEqual({row.name for row in rows if row.kind == "model"} & {"flash", "haiku", "sonnet", "luna", "terra"}, {"flash", "haiku", "sonnet", "luna", "terra"})

    def test_sections_are_separate_and_provider_disable_does_not_disable_model_preference(self):
        config = set_enabled(default_config(), "providers", "deepseek", False, reason="peak hours")
        config = set_enabled(config, "models", "deepseek-flash", True)
        rows = build_rows(config)
        self.assertEqual([row.section for row in rows[:5]], ["providers"] * 5)
        self.assertTrue(all(row.section == "models" for row in rows[5:13]))
        model = next(row for row in rows if row.name == "deepseek-flash")
        self.assertTrue(model.configured_enabled)
        self.assertFalse(model.effective_enabled)
        self.assertEqual(model.effective_reason, "peak hours")

    def test_vllm_route_has_configured_and_effective_state(self):
        config = default_config({"qwen35-9b-craig"})
        rows = build_rows(config, {})
        route = next(row for row in rows if row.kind == "vllm")
        self.assertTrue(route.configured_enabled)
        self.assertFalse(route.effective_enabled)
        self.assertIn("missing", route.effective_reason)

    def test_payg_providers_and_routes_start_disabled_with_a_reason(self):
        rows = build_rows(default_config())
        for name in ("deepseek", "minimax"):
            row = next(r for r in rows if r.kind == "provider" and r.name == name)
            self.assertFalse(row.enabled)
            self.assertEqual(row.reason, "experimental PAYG; benchmark pending")
        for name in ("deepseek-pro", "deepseek-flash", "minimax-m3"):
            row = next(r for r in rows if r.kind == "model" and r.name == name)
            self.assertFalse(row.enabled)


class ToggleTests(unittest.TestCase):
    def test_toggle_flips_only_the_targeted_row(self):
        rows = build_rows(default_config())
        index = next(i for i, r in enumerate(rows) if r.name == "flash")
        updated = toggle(rows, index)
        self.assertNotEqual(updated[index].enabled, rows[index].enabled)
        for i, (a, b) in enumerate(zip(rows, updated)):
            if i != index:
                self.assertEqual(a, b)

    def test_toggle_does_not_cascade_to_child_model_rows(self):
        rows = build_rows(default_config())
        codex_index = next(i for i, r in enumerate(rows) if r.kind == "provider" and r.name == "codex")
        updated = toggle(rows, codex_index)
        self.assertFalse(updated[codex_index].enabled)
        model_rows = [r for r in updated if r.kind == "model" and r.provider == "codex"]
        self.assertTrue(all(r.enabled for r in model_rows), "child model rows must keep their own preference")

    def test_enabling_a_row_clears_its_reason(self):
        config = set_enabled(default_config(), "models", "flash", False, reason="x")
        rows = build_rows(config)
        index = next(i for i, r in enumerate(rows) if r.name == "flash")
        self.assertEqual(rows[index].reason, "x")
        updated = toggle(rows, index)
        self.assertTrue(updated[index].enabled)
        self.assertIsNone(updated[index].reason)

    def test_toggle_is_pure_original_list_untouched(self):
        rows = build_rows(default_config())
        index = 0
        before = rows[index]
        toggle(rows, index)
        self.assertEqual(rows[index], before)


class SetReasonTests(unittest.TestCase):
    def test_set_reason_updates_only_targeted_row(self):
        rows = build_rows(default_config())
        index = next(i for i, r in enumerate(rows) if r.name == "codex")
        updated = set_reason(rows, index, "weekly quota low")
        self.assertEqual(updated[index].reason, "weekly quota low")

    def test_blank_reason_clears_it(self):
        config = set_enabled(default_config(), "providers", "codex", False, reason="x")
        rows = build_rows(config)
        index = next(i for i, r in enumerate(rows) if r.name == "codex")
        updated = set_reason(rows, index, "")
        self.assertIsNone(updated[index].reason)


class RowsToConfigTests(unittest.TestCase):
    def test_round_trips_back_to_an_equivalent_config(self):
        original = default_config()
        rows = build_rows(original)
        reconstructed = rows_to_config(original, rows)
        self.assertEqual(reconstructed, original)

    def test_reflects_a_toggle(self):
        original = default_config()
        rows = build_rows(original)
        index = next(i for i, r in enumerate(rows) if r.name == "flash")
        rows = toggle(rows, index)
        updated = rows_to_config(original, rows)
        self.assertFalse(updated["models"]["flash"]["enabled"])
        self.assertTrue(original["models"]["flash"]["enabled"], "original must be untouched")


class DiffSummaryTests(unittest.TestCase):
    def test_no_changes_produces_empty_summary(self):
        original = default_config()
        rows = build_rows(original)
        self.assertEqual(diff_summary(original, rows), [])

    def test_toggle_produces_one_diff_line(self):
        original = default_config()
        rows = build_rows(original)
        index = next(i for i, r in enumerate(rows) if r.name == "flash")
        rows = toggle(rows, index)
        summary = diff_summary(original, rows)
        self.assertEqual(len(summary), 1)
        self.assertIn("flash", summary[0])
        self.assertIn("enabled -> disabled", summary[0])

    def test_reason_only_change_is_detected(self):
        original = default_config()
        rows = build_rows(original)
        index = next(i for i, r in enumerate(rows) if r.name == "codex")
        rows = toggle(rows, index)  # disable first so a reason is meaningful
        rows = set_reason(rows, index, "quota low")
        summary = diff_summary(original, rows)
        self.assertTrue(any("quota low" in line for line in summary))


class InteractiveBoundaryTests(unittest.TestCase):
    def test_cancel_does_not_save(self):
        config = default_config()
        with patch("delegation.config_tui.load_config", return_value=config), \
             patch("delegation.config_tui.inspect_vllm_routes", return_value={}), \
             patch("curses.wrapper", return_value=None), \
             patch("delegation.config_tui.save_config") as save:
            self.assertEqual(run_interactive_config(), 0)
        save.assert_not_called()

    def test_save_writes_only_after_wrapper_returns_staged_changes(self):
        config = default_config()
        rows = build_rows(config)
        index = next(i for i, row in enumerate(rows) if row.name == "flash")
        staged = toggle(rows, index)
        with patch("delegation.config_tui.load_config", return_value=config), \
             patch("delegation.config_tui.inspect_vllm_routes", return_value={}), \
             patch("curses.wrapper", return_value=staged), \
             patch("delegation.config_tui.save_config") as save:
            self.assertEqual(run_interactive_config(), 0)
        save.assert_called_once()
        self.assertFalse(save.call_args.args[0]["models"]["flash"]["enabled"])

    def test_invalid_staged_routing_config_is_rejected_before_save(self):
        config = default_config()
        rows = build_rows(config)
        invalid = {section: {name: dict(entry) for name, entry in entries.items()} for section, entries in config.items()}
        invalid["routing"] = {
            "preferences": {"review": {"preferred_targets": ["profile:sonnet"], "excluded_targets": ["profile:sonnet"]}},
            "reserves": {},
        }
        with patch("delegation.config_tui.load_config", return_value=config), \
             patch("delegation.config_tui.inspect_vllm_routes", return_value={}), \
             patch("curses.wrapper", side_effect=[rows, invalid]), \
             patch("delegation.config_tui.save_config") as save:
            with self.assertRaisesRegex(ValueError, "preferred_targets.*excluded_targets"):
                run_interactive_setup()
        save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
