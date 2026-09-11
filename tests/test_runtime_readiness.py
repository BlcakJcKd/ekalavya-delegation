"""Status and execution share the same deterministic local binding contract."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from delegation.config import default_config, set_enabled
from delegation.status_cli import build_report
from ekalavya.readiness import profile_readiness
from ekalavya.resolver import resolve
from ekalavya.schema import RunIntent


def _candidate(**overrides):
    value = {
        "identity_key": "gemini-flash-current",
        "provider": "gemini",
        "family": "flash",
        "provider_model_id": "gemini-3.7-flash-medium",
        "display_name": "Gemini Flash",
        "lifecycle": "current",
        "execution_route": "flash",
        "legacy_route": "flash",
        "harness": "agy",
        "serving_engine": "agy",
        "transport": "agy",
        "capabilities": {"reasoning_values": ["medium"], "harness_values": ["agy"]},
    }
    value.update(overrides)
    return value


class RuntimeReadinessTests(unittest.TestCase):
    def setUp(self):
        self.profile = {"name": "flash", "default_identity_key": "gemini-flash-current", "permitted_candidates": ["gemini-flash-current"], "reasoning_policy": "fixed", "default_reasoning": "medium"}

    def test_stale_available_evidence_cannot_override_missing_binding(self):
        candidate = _candidate(execution_route=None, legacy_route=None, harness=None)
        readiness = profile_readiness("flash", [self.profile], [candidate], {"gemini-flash-current": {"state": "available", "observed_at": "2026-01-01T00:00:00+00:00", "source": "prior discovery"}}, which=lambda name: "/fake/agy")
        self.assertEqual(readiness["harness_ready"], "unavailable")
        self.assertIn("harness-unavailable", readiness["readiness_reason"])
        report = build_report(None, which=lambda name: "/fake/" + name, config=default_config(), route_readiness={"flash": readiness})
        flash = next(item for item in report["routes"] if item["route"] == "flash")
        self.assertEqual(flash["effective"], "unavailable")
        self.assertIn("harness-unavailable", flash["effective_reason"])
        with patch("ekalavya.readiness.shutil.which", return_value="/fake/agy"):
            result = resolve(RunIntent("flash"), self.profile, [candidate], default_config())
        self.assertEqual(result.state, "harness-unavailable")

    def test_correct_bound_and_observed_harness_is_effectively_available(self):
        candidate = _candidate()
        readiness = profile_readiness("flash", [self.profile], [candidate], {"gemini-flash-current": {"state": "available", "observed_at": "2026-01-01T00:00:00+00:00", "source": "local discovery"}}, which=lambda name: "/fake/agy")
        self.assertEqual(readiness["harness_ready"], "ready")
        report = build_report(None, which=lambda name: "/fake/" + name, config=default_config(), route_readiness={"flash": readiness})
        self.assertEqual(next(item for item in report["routes"] if item["route"] == "flash")["effective"], "available")

    def test_provider_disable_keeps_stored_model_but_disables_effective_route(self):
        candidate = _candidate()
        readiness = profile_readiness("flash", [self.profile], [candidate], {"gemini-flash-current": {"state": "available"}}, which=lambda name: "/fake/agy")
        config = set_enabled(default_config(), "providers", "gemini", False, reason="disabled for maintenance")
        report = build_report(None, which=lambda name: "/fake/" + name, config=config, route_readiness={"flash": readiness})
        flash = next(item for item in report["routes"] if item["route"] == "flash")
        self.assertEqual(flash["effective"], "disabled")
        self.assertFalse(flash["effective_enabled"])
        self.assertEqual(candidate["provider_model_id"], "gemini-3.7-flash-medium")

    def test_binary_presence_without_readiness_evidence_is_not_available(self):
        readiness = profile_readiness("flash", [self.profile], [_candidate()], {}, which=lambda name: "/fake/agy")
        self.assertTrue(readiness["harness_detected"])
        self.assertEqual(readiness["harness_ready"], "unconfirmed")
        report = build_report(None, which=lambda name: "/fake/" + name, config=default_config(), route_readiness={"flash": readiness})
        flash = next(item for item in report["routes"] if item["route"] == "flash")
        self.assertEqual(flash["effective"], "unavailable")
        self.assertIn("not locally confirmed", flash["effective_reason"])


if __name__ == "__main__":
    unittest.main()
