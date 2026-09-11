import tempfile
import unittest
import io
import os
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout

from delegation.config import default_config, parse_config, set_enabled, set_routing_preference
from ekalavya.config import ensure_control_files
from ekalavya.deepseek import DEEPSEEK_PRO_CUTOFF_UTC
from ekalavya.recommendation import _winner_by_feedback, recommend
from ekalavya.schema import CandidateIdentity, Resolution, RunIntent
from ekalavya.cli import _print_route_human, main


def resolved(profile: str, provider: str) -> Resolution:
    identity = CandidateIdentity(provider, profile, f"{provider}-{profile}", profile)
    return Resolution(RunIntent(profile), identity, resolved_harness="test", execution_route=profile)


class RouteRecommendationTests(unittest.TestCase):
    def setUp(self):
        self.config = default_config()
        self.profiles = [
            {"name": "sonnet", "default_identity_key": "sonnet", "permitted_candidates": ["sonnet"]},
            {"name": "flash", "default_identity_key": "flash", "permitted_candidates": ["flash"]},
        ]
        self.catalogue = []

    def _ready(self, *args, **kwargs):
        return {"harness_ready": "ready", "readiness_reason": None}

    def test_explicit_cross_provider_preference_beats_primary_native(self):
        self.config = set_routing_preference(self.config, "review", "preferred_targets", ["sonnet", "primary-native"])
        def fake_resolve(intent, *args, **kwargs):
            return resolved(intent.profile, "claude" if intent.profile == "sonnet" else "gemini")
        with patch("ekalavya.recommendation.resolve", side_effect=fake_resolve), patch("ekalavya.recommendation.profile_readiness", side_effect=self._ready):
            result = recommend(task="review", primary="codex", config=self.config, profiles=self.profiles, catalogue=self.catalogue, observed_availability={})
        self.assertEqual(result["recommendation"]["target"], "profile:sonnet")
        self.assertEqual(result["decision_basis"], "explicit-preference")
        self.assertFalse(result["executed"])

    def test_primary_native_is_default_without_task_preference(self):
        with patch("ekalavya.recommendation.resolve", side_effect=lambda intent, *args, **kwargs: resolved(intent.profile, "claude")), patch("ekalavya.recommendation.profile_readiness", side_effect=self._ready):
            result = recommend(task="review", primary="codex", config=self.config, profiles=self.profiles, catalogue=self.catalogue, observed_availability={})
        self.assertEqual(result["recommendation"]["target"], "primary-native")
        self.assertEqual(result["decision_basis"], "default-primary-native")

    def test_omitted_primary_never_creates_native_target(self):
        with patch("ekalavya.recommendation.resolve", side_effect=lambda intent, *args, **kwargs: resolved(intent.profile, "claude")), patch("ekalavya.recommendation.profile_readiness", side_effect=self._ready):
            result = recommend(task="review", primary=None, config=self.config, profiles=self.profiles, catalogue=self.catalogue, observed_availability={})
        self.assertIsNone(result["recommendation"])
        self.assertEqual(result["decision_basis"], "tie")
        self.assertTrue(any("primary-unknown" in warning for warning in result["warnings"]))

    def test_hard_exclusion_can_exclude_primary_native(self):
        self.config = set_routing_preference(self.config, "review", "excluded_targets", ["primary-native"])
        with patch("ekalavya.recommendation.resolve", side_effect=lambda intent, *args, **kwargs: resolved(intent.profile, "claude")), patch("ekalavya.recommendation.profile_readiness", side_effect=self._ready):
            result = recommend(task="review", primary="codex", config=self.config, profiles=self.profiles, catalogue=self.catalogue, observed_availability={})
        self.assertNotEqual((result["recommendation"] or {}).get("target"), "primary-native")
        self.assertIn({"target": "primary-native", "code": "user-excluded"}, result["excluded"])

    def test_each_profile_uses_the_canonical_resolver_once(self):
        with patch("ekalavya.recommendation.resolve", side_effect=lambda intent, *args, **kwargs: resolved(intent.profile, "claude")) as resolver, patch("ekalavya.recommendation.profile_readiness", side_effect=self._ready):
            recommend(task="review", primary=None, config=self.config, profiles=self.profiles, catalogue=self.catalogue, observed_availability={})
        self.assertEqual(resolver.call_count, len(self.profiles))

    def test_feedback_requires_five_ratings_for_every_compared_target(self):
        low = {"target": "profile:a", "history": {"rated": 4, "feedback": {"useful": 4}, "terminal": 0, "successful": 0, "success_rate": None, "wall_coverage": 0, "median_wall_seconds": None}}
        high = {"target": "profile:b", "history": {"rated": 5, "feedback": {"useful": 5}, "terminal": 0, "successful": 0, "success_rate": None, "wall_coverage": 0, "median_wall_seconds": None}}
        self.assertIsNone(_winner_by_feedback([low, high]))

    def test_routing_config_normalizes_bare_targets_and_rejects_conflicts(self):
        config = set_routing_preference(default_config(), "scientific-critique", "preferred_targets", ["sonnet", "primary-native"])
        self.assertEqual(config["routing"]["preferences"]["scientific-critique"]["preferred_targets"], ["profile:sonnet", "primary-native"])
        with self.assertRaises(ValueError):
            parse_config({"routing": {"preferences": {"review": {"allowed_targets": ["profile:sonnet"], "excluded_targets": ["sonnet"]}}}})

    def test_preferred_target_cannot_be_excluded_but_valid_preference_forms_remain_valid(self):
        config = set_routing_preference(default_config(), "review", "preferred_targets", ["sonnet"])
        self.assertEqual(config["routing"]["preferences"]["review"]["preferred_targets"], ["profile:sonnet"])
        config = set_routing_preference(config, "review", "allowed_targets", ["sonnet"])
        self.assertEqual(config["routing"]["preferences"]["review"]["allowed_targets"], ["profile:sonnet"])
        with self.assertRaisesRegex(ValueError, "preferred_targets.*excluded_targets"):
            set_routing_preference(config, "review", "excluded_targets", ["sonnet"])
        with self.assertRaisesRegex(ValueError, "preferred_targets.*excluded_targets"):
            parse_config({"routing": {"preferences": {"review": {"preferred_targets": ["primary-native"], "excluded_targets": ["primary-native"]}}}})

    def test_primary_native_allowed_and_excluded_is_rejected(self):
        config = set_routing_preference(default_config(), "review", "allowed_targets", ["primary-native"])
        with self.assertRaisesRegex(ValueError, "allowed_targets.*excluded_targets"):
            set_routing_preference(config, "review", "excluded_targets", ["primary-native"])

    def test_custom_task_tag_is_validated_and_preserved(self):
        config = parse_config({"routing": {"preferences": {"custom.audit": {"preferred_targets": ["sonnet"]}}}})
        self.assertEqual(config["routing"]["preferences"]["custom.audit"]["preferred_targets"], ["profile:sonnet"])

    def _recommend_one(self, config, profile, entry, *, now=None):
        return recommend(
            task="review", primary=None, config=config, profiles=[profile],
            catalogue=[entry], observed_availability={}, registry={"schema_version": 1, "records": []}, now=now,
        )

    def test_deepseek_pro_cutoff_is_lifecycle_exclusion(self):
        entry = {
            "provider": "deepseek", "family": "pro", "provider_model_id": "deepseek-v4-pro",
            "display_name": "DeepSeek V4 Pro", "capabilities": {"reasoning_values": ["high"]},
            "identity_key": "pro", "lifecycle": "current", "legacy_route": "deepseek-pro",
        }
        config = set_enabled(set_enabled(default_config(), "providers", "deepseek", True), "models", "deepseek-pro", True)
        payload = self._recommend_one(config, {"name": "deepseek-pro", "default_identity_key": "pro", "permitted_candidates": ["pro"]}, entry, now=DEEPSEEK_PRO_CUTOFF_UTC)
        self.assertEqual(payload["excluded"], [{"target": "profile:deepseek-pro", "code": "lifecycle-not-executable", "detail": payload["excluded"][0]["detail"]}])
        with redirect_stdout(io.StringIO()) as output:
            _print_route_human(payload, explain=True)
        self.assertIn("profile:deepseek-pro — lifecycle-not-executable", output.getvalue())

    def test_missing_execution_route_is_missing_route(self):
        entry = {"provider": "claude", "family": "haiku", "provider_model_id": "claude-haiku", "identity_key": "haiku", "lifecycle": "current"}
        payload = self._recommend_one(default_config(), {"name": "haiku", "default_identity_key": "haiku", "permitted_candidates": ["haiku"]}, entry)
        self.assertEqual(payload["excluded"][0]["code"], "missing-route")

    def test_missing_harness_is_harness_unavailable(self):
        entry = {"provider": "claude", "family": "haiku", "provider_model_id": "claude-haiku", "identity_key": "haiku", "lifecycle": "current", "legacy_route": "haiku", "execution_route": "haiku", "harness": "claude"}
        with patch("ekalavya.readiness.shutil.which", return_value=None):
            payload = self._recommend_one(default_config(), {"name": "haiku", "default_identity_key": "haiku", "permitted_candidates": ["haiku"]}, entry)
        self.assertEqual(payload["excluded"][0]["code"], "harness-unavailable")

    def test_provider_and_model_disabled_have_truthful_exclusions(self):
        entry = {"provider": "claude", "family": "haiku", "provider_model_id": "claude-haiku", "identity_key": "haiku", "lifecycle": "current", "legacy_route": "haiku", "execution_route": "haiku", "harness": "claude"}
        provider_disabled = set_enabled(default_config(), "providers", "claude", False)
        payload = self._recommend_one(provider_disabled, {"name": "haiku", "default_identity_key": "haiku", "permitted_candidates": ["haiku"]}, entry)
        self.assertEqual(payload["excluded"][0]["code"], "provider-disabled")
        model_disabled = set_enabled(default_config(), "models", "haiku", False)
        payload = self._recommend_one(model_disabled, {"name": "haiku", "default_identity_key": "haiku", "permitted_candidates": ["haiku"]}, entry)
        self.assertEqual(payload["excluded"][0]["code"], "model-disabled")

    def test_route_cli_never_calls_execution_or_creates_config(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(root / "config"), "XDG_STATE_HOME": str(root / "state")}, clear=False), patch("ekalavya.cli.execute", side_effect=AssertionError("must not execute")) as execute:
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main(["route", "--task", "review", "--json"]), 0)
            execute.assert_not_called()
            self.assertFalse((root / "config" / "ekalavya" / "config.toml").exists())

    def test_route_cli_loads_evidence_registry_once_for_multiple_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ensure_control_files(root / "config" / "ekalavya")
            with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(root / "config"), "XDG_STATE_HOME": str(root / "state")}, clear=False), patch("ekalavya.cli.load_registry", return_value={"schema_version": 1, "records": []}) as loader:
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main(["route", "--task", "review", "--json"]), 0)
            loader.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
