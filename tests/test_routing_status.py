"""Tests for delegation.routing (primary alias / route-type classification)
and delegation.status (zero-model-call effective routing computation)."""

from __future__ import annotations

import unittest

from delegation.config import default_config, set_enabled
from delegation.routing import normalize_primary, route_type
from delegation.status import compute_status


class NormalizePrimaryTests(unittest.TestCase):
    def test_none_and_blank_are_undeclared(self):
        self.assertIsNone(normalize_primary(None))
        self.assertIsNone(normalize_primary(""))
        self.assertIsNone(normalize_primary("   "))

    def test_known_aliases_normalize_to_a_provider_or_manual(self):
        self.assertEqual(normalize_primary("claude-code"), "claude")
        self.assertEqual(normalize_primary("Claude"), "claude")
        self.assertEqual(normalize_primary("CODEX"), "codex")
        self.assertEqual(normalize_primary("antigravity"), "gemini")
        self.assertEqual(normalize_primary("agy"), "gemini")
        self.assertEqual(normalize_primary("manual"), "manual")
        self.assertEqual(normalize_primary("human"), "manual")

    def test_unknown_alias_raises(self):
        with self.assertRaisesRegex(ValueError, "unknown --primary value"):
            normalize_primary("bogus")

    def test_deepseek_and_minimax_transport_aliases_normalize_to_their_provider(self):
        # A primary's TRANSPORT (which CLI hosts it) is irrelevant to the
        # guard; only its inference PROVIDER matters, so every alias that
        # means "this primary's own inference is DeepSeek/MiniMax" must
        # normalize to that provider regardless of which CLI fronts it.
        for alias in ("deepseek", "claude-deepseek", "codex-deepseek"):
            self.assertEqual(normalize_primary(alias), "deepseek")
        for alias in ("minimax", "claude-minimax", "codex-minimax"):
            self.assertEqual(normalize_primary(alias), "minimax")


class RouteTypeTests(unittest.TestCase):
    def test_terra_and_luna_are_native_only_only_for_codex_primary(self):
        for primary in ("terra", "luna"):
            self.assertEqual(route_type(primary, "codex"), "same-provider")
            for other_primary in (None, "claude", "gemini", "manual"):
                self.assertEqual(route_type(primary, other_primary), "external")

    def test_wrapper_routes_are_external_for_a_different_or_undeclared_primary(self):
        self.assertEqual(route_type("flash", None), "external")
        self.assertEqual(route_type("flash", "claude"), "external")
        self.assertEqual(route_type("flash", "codex"), "external")
        self.assertEqual(route_type("haiku", "codex"), "external")
        self.assertEqual(route_type("sonnet", "gemini"), "external")
        self.assertEqual(route_type("terra", "claude"), "external")
        self.assertEqual(route_type("luna", "claude"), "external")

    def test_wrapper_routes_are_same_provider_for_a_matching_primary(self):
        self.assertEqual(route_type("haiku", "claude"), "same-provider")
        self.assertEqual(route_type("sonnet", "claude"), "same-provider")
        self.assertEqual(route_type("flash", "gemini"), "same-provider")
        self.assertEqual(route_type("terra", "codex"), "same-provider")
        self.assertEqual(route_type("luna", "codex"), "same-provider")

    def test_manual_primary_never_produces_same_provider(self):
        self.assertEqual(route_type("flash", "manual"), "external")
        self.assertEqual(route_type("haiku", "manual"), "external")

    def test_codex_and_claude_primaries_may_externally_reach_deepseek_and_minimax(self):
        # Same-provider protection is keyed on the INFERENCE PROVIDER, not
        # the transport executable: a Codex- or Claude-hosted primary is a
        # different provider than deepseek/minimax, so these are external.
        for primary in ("codex", "claude"):
            self.assertEqual(route_type("deepseek-pro", primary), "external")
            self.assertEqual(route_type("deepseek-flash", primary), "external")
            self.assertEqual(route_type("minimax-m3", primary), "external")

    def test_deepseek_primary_calling_a_deepseek_route_is_same_provider(self):
        self.assertEqual(route_type("deepseek-pro", "deepseek"), "same-provider")
        self.assertEqual(route_type("deepseek-flash", "deepseek"), "same-provider")
        # A DeepSeek primary is a different provider than minimax/claude/codex.
        self.assertEqual(route_type("minimax-m3", "deepseek"), "external")

    def test_minimax_primary_calling_the_minimax_route_is_same_provider(self):
        self.assertEqual(route_type("minimax-m3", "minimax"), "same-provider")
        self.assertEqual(route_type("deepseek-pro", "minimax"), "external")


class ComputeStatusTests(unittest.TestCase):
    def _which_all_present(self, name):
        return f"/usr/bin/{name}"

    def _which_none_present(self, name):
        return None

    def test_zero_model_calls_only_uses_injected_which(self):
        calls = []

        def spy_which(name):
            calls.append(name)
            return f"/usr/bin/{name}"

        compute_status(default_config(), primary="claude-code", which=spy_which)
        # only wrapper executables probed -- every route with an external
        # wrapper, enabled or not, since "enabled" is a separate axis from
        # "is the wrapper on this machine".
        self.assertEqual(set(calls), {"claude", "agy", "codex", "codex-deepseek", "codex-minimax"})

    def test_enabled_external_route_with_executable_present_is_available(self):
        results = {r.route: r for r in compute_status(default_config(), primary=None, which=self._which_all_present)}
        self.assertEqual(results["flash"].effective, "available")
        self.assertEqual(results["flash"].route_type, "external")

    def test_enabled_external_route_with_missing_executable_is_reported_clearly(self):
        results = {r.route: r for r in compute_status(default_config(), primary=None, which=self._which_none_present)}
        self.assertEqual(results["flash"].effective, "executable missing")
        self.assertIn("agy", results["flash"].effective_reason)

    def test_same_provider_route_is_native_only_regardless_of_executable(self):
        results = {r.route: r for r in compute_status(default_config(), primary="claude-code", which=self._which_all_present)}
        self.assertEqual(results["haiku"].effective, "native-only")
        self.assertEqual(results["sonnet"].effective, "native-only")
        self.assertEqual(results["flash"].effective, "available")  # gemini != claude primary

    def test_codex_routes_are_external_for_claude_primary(self):
        results = {r.route: r for r in compute_status(default_config(), primary="claude-code", which=self._which_all_present)}
        for route in ("terra", "luna"):
            self.assertEqual(results[route].effective, "available")
            self.assertEqual(results[route].route_type, "external")

    def test_codex_routes_are_native_only_for_codex_primary(self):
        results = {r.route: r for r in compute_status(default_config(), primary="codex", which=self._which_all_present)}
        for route in ("terra", "luna"):
            self.assertEqual(results[route].effective, "native-only")
            self.assertEqual(results[route].route_type, "same-provider")

    def test_provider_disable_overrides_individual_model_enabled_preference(self):
        config = set_enabled(default_config(), "providers", "codex", False, reason="weekly quota low")
        results = {r.route: r for r in compute_status(config, primary=None, which=self._which_all_present)}
        self.assertTrue(results["terra"].configured_enabled)
        self.assertTrue(results["terra"].provider_enabled is False)
        self.assertTrue(results["terra"].effective_enabled is False)
        self.assertEqual(results["terra"].provider_reason, "weekly quota low")
        self.assertEqual(results["terra"].effective, "disabled")
        self.assertEqual(results["terra"].effective_reason, "weekly quota low")
        self.assertTrue(results["luna"].configured_enabled)
        self.assertFalse(results["luna"].effective_enabled)

    def test_provider_disable_without_prose_reason_has_machine_readable_reason(self):
        config = set_enabled(default_config(), "providers", "deepseek", False, reason="")
        result = next(item for item in compute_status(config, primary=None, which=self._which_all_present) if item.route == "deepseek-flash")
        self.assertFalse(result.effective_enabled)
        self.assertEqual(result.effective_reason, "provider-disabled")

    def test_re_enabling_provider_restores_untouched_model_preferences(self):
        config = set_enabled(default_config(), "providers", "codex", False, reason="quota low")
        config = set_enabled(config, "providers", "codex", True)
        results = {r.route: r for r in compute_status(config, primary=None, which=self._which_all_present)}
        self.assertEqual(results["terra"].effective, "available")
        self.assertEqual(results["luna"].effective, "available")

    def test_model_disabled_reason_used_when_provider_enabled(self):
        config = set_enabled(default_config(), "models", "flash", False, reason="testing")
        results = {r.route: r for r in compute_status(config, primary=None, which=self._which_all_present)}
        self.assertEqual(results["flash"].effective, "disabled")
        self.assertEqual(results["flash"].effective_reason, "testing")

    def test_provider_reason_takes_precedence_over_model_reason_when_both_disabled(self):
        config = set_enabled(default_config(), "providers", "gemini", False, reason="provider-level")
        config = set_enabled(config, "models", "flash", False, reason="model-level")
        results = {r.route: r for r in compute_status(config, primary=None, which=self._which_all_present)}
        self.assertEqual(results["flash"].effective_reason, "provider-level")

    def test_route_status_is_json_serializable_via_as_dict(self):
        import json
        results = compute_status(default_config(), primary="codex", which=self._which_all_present)
        json.dumps([r.as_dict() for r in results])  # must not raise

    def test_unknown_primary_propagates_from_compute_status(self):
        with self.assertRaises(ValueError):
            compute_status(default_config(), primary="not-a-real-provider")

    def test_payg_routes_are_disabled_by_default_even_with_executable_present(self):
        results = {r.route: r for r in compute_status(default_config(), primary=None, which=self._which_all_present)}
        for route in ("deepseek-pro", "deepseek-flash", "minimax-m3"):
            self.assertEqual(results[route].effective, "disabled")
            self.assertEqual(results[route].effective_reason, "experimental PAYG; benchmark pending")

    def test_payg_route_metadata_is_reported(self):
        results = {r.route: r for r in compute_status(default_config(), primary=None, which=self._which_all_present)}
        for route in ("deepseek-pro", "deepseek-flash"):
            self.assertEqual(results[route].provider, "deepseek")
            self.assertEqual(results[route].transport, "codex")
            self.assertEqual(results[route].billing, "payg")
            self.assertEqual(results[route].maturity, "experimental")
        minimax = results["minimax-m3"]
        self.assertEqual(minimax.provider, "minimax")
        self.assertEqual(minimax.transport, "codex")
        self.assertEqual(minimax.billing, "payg")
        self.assertEqual(minimax.maturity, "experimental")

    def test_stable_route_metadata_is_reported_as_quota_and_stable(self):
        results = {r.route: r for r in compute_status(default_config(), primary=None, which=self._which_all_present)}
        for route in ("terra", "luna", "sonnet", "haiku", "flash"):
            self.assertEqual(results[route].billing, "quota")
            self.assertEqual(results[route].maturity, "stable")

    def test_deepseek_primary_makes_deepseek_routes_native_only_once_enabled(self):
        config = set_enabled(default_config(), "providers", "deepseek", True)
        config = set_enabled(config, "models", "deepseek-pro", True)
        results = {r.route: r for r in compute_status(config, primary="deepseek", which=self._which_all_present)}
        self.assertEqual(results["deepseek-pro"].effective, "native-only")
        self.assertEqual(results["deepseek-pro"].route_type, "same-provider")

    def test_minimax_primary_makes_minimax_route_native_only_once_enabled(self):
        config = set_enabled(default_config(), "providers", "minimax", True)
        config = set_enabled(config, "models", "minimax-m3", True)
        results = {r.route: r for r in compute_status(config, primary="minimax", which=self._which_all_present)}
        self.assertEqual(results["minimax-m3"].effective, "native-only")
        self.assertEqual(results["minimax-m3"].route_type, "same-provider")

    def test_enabling_deepseek_with_codex_primary_makes_it_available(self):
        config = set_enabled(default_config(), "providers", "deepseek", True)
        config = set_enabled(config, "models", "deepseek-pro", True)
        results = {r.route: r for r in compute_status(config, primary="codex", which=self._which_all_present)}
        self.assertEqual(results["deepseek-pro"].effective, "available")


if __name__ == "__main__":
    unittest.main()
