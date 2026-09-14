from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ekalavya.config import ensure_control_files
from ekalavya.deepseek import (
    DEEPSEEK_REASONING_MAPPING,
    DeepSeekExactIdentityError,
    assert_deepseek_pro_exact,
    map_reasoning_level,
)
from ekalavya.executor import execute
from ekalavya.readiness import binding_preflight, profile_readiness, resolved_harness_binding
from ekalavya.resolver import resolve
from ekalavya.schema import RunIntent


class DeepSeekLifecycleTests(unittest.TestCase):
    def _pro_fixture(self):
        entry = {
            "provider": "deepseek", "family": "pro", "provider_model_id": "deepseek-v4-pro",
            "display_name": "DeepSeek V4 Pro", "capabilities": {"reasoning_values": ["high"]},
            "identity_key": "pro", "lifecycle": "current", "legacy_route": "deepseek-pro",
            "execution_route": "deepseek-pro", "harness": "codex-deepseek",
        }
        profile = {
            "name": "deepseek-pro", "default_identity_key": "pro", "permitted_candidates": ["pro"],
            "reasoning_policy": "fixed", "default_reasoning": "high",
        }
        enabled = {"providers": {"deepseek": {"enabled": True}}, "models": {"deepseek-pro": {"enabled": True}}}
        return entry, profile, enabled

    def test_reasoning_mapping_preserves_ekalavya_vocabulary(self):
        self.assertEqual(DEEPSEEK_REASONING_MAPPING, {
            "minimal": "low", "low": "low", "medium": "high",
            "high": "high", "xhigh": "high", "max": "max", "ultra": "max",
        })
        for source, provider in DEEPSEEK_REASONING_MAPPING.items():
            self.assertEqual(map_reasoning_level(source), provider)

    def test_fresh_control_files_have_distinct_historical_and_v41_flash_identities(self):
        with TemporaryDirectory() as temp:
            report = ensure_control_files(Path(temp))
            self.assertEqual(report["status"], "complete")
            entries = json.loads((Path(temp) / "catalogue.json").read_text())
            flash = [entry for entry in entries if entry.get("provider") == "deepseek" and entry.get("family") == "flash"]
            self.assertEqual({entry["provider_model_id"] for entry in flash}, {"deepseek-v4-flash", "deepseek-flash"})
            old = next(entry for entry in flash if entry["provider_model_id"] == "deepseek-v4-flash")
            current = next(entry for entry in flash if entry["provider_model_id"] == "deepseek-flash")
            self.assertEqual(old["display_name"], "DeepSeek V4 Flash")
            self.assertEqual(old["lifecycle"], "retired")
            self.assertTrue(old["historical_only"])
            self.assertEqual(current["display_name"], "DeepSeek V4.1 Flash")
            self.assertEqual(current["generation"], "v4.1")
            self.assertEqual(current["lifecycle"], "candidate")
            self.assertEqual(current["execution_route"], "deepseek-flash")
            self.assertEqual(current["harness"], "codex-deepseek")
            self.assertNotIn("promotion_basis", current)
            self.assertEqual(current["capabilities"]["input_modalities"], ["text"])
            self.assertTrue(current["capabilities"]["provider_native_vision"])
            self.assertFalse(current["capabilities"]["harness_multimodal"])
            profile = next(item for item in json.loads((Path(temp) / "profiles.json").read_text()) if item["name"] == "deepseek-flash")
            self.assertEqual(profile["default_identity_key"], current["identity_key"])
            self.assertEqual(profile["permitted_candidates"], [current["identity_key"]])
            self.assertEqual(profile["default_reasoning"], "high")

    def test_flash_profile_resolves_canonical_endpoint_without_alias_fallback(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ensure_control_files(root)
            entries = json.loads((root / "catalogue.json").read_text())
            profile = next(item for item in json.loads((root / "profiles.json").read_text()) if item["name"] == "deepseek-flash")
            with patch("ekalavya.readiness.shutil.which", return_value="/fake/codex-deepseek"):
                result = resolve(RunIntent("deepseek-flash"), profile, entries, availability={"providers": {"deepseek": {"enabled": True}}, "models": {"deepseek-flash": {"enabled": True}}})
            self.assertEqual(result.state, "resolved")
            self.assertEqual(result.candidate.provider_model_id, "deepseek-flash")
            self.assertEqual(result.candidate.display_name, "DeepSeek V4.1 Flash")

    def test_legacy_v41_flash_without_harness_uses_registered_route_binding(self):
        """A missing old metadata field must not degrade a known exact route."""
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ensure_control_files(root)
            entries = json.loads((root / "catalogue.json").read_text())
            profiles = json.loads((root / "profiles.json").read_text())
            profile = next(item for item in profiles if item["name"] == "deepseek-flash")
            current = next(item for item in entries if item["identity_key"] == profile["default_identity_key"])
            current.pop("harness")
            current["lifecycle"] = "current"
            availability = {profile["default_identity_key"]: {"state": "available", "observed_at": "2026-09-13T00:00:00+00:00", "source": "prior discovery"}}
            config = {"providers": {"deepseek": {"enabled": True}}, "models": {"deepseek-flash": {"enabled": True}}}
            readiness = profile_readiness("deepseek-flash", profiles, entries, availability, which=lambda name: "/fake/codex-deepseek")
            self.assertEqual(readiness["harness_ready"], "ready")
            self.assertEqual(readiness["resolved_harness"], "codex-deepseek")
            with patch("ekalavya.readiness.shutil.which", return_value="/fake/codex-deepseek"):
                result = resolve(RunIntent("deepseek-flash"), profile, entries, availability=config)
            self.assertEqual(result.state, "resolved")
            self.assertEqual(result.candidate.provider_model_id, "deepseek-flash")
            self.assertEqual(result.resolved_reasoning, "high")
            self.assertEqual(result.resolved_harness, "codex-deepseek")
            record = result.as_dict()
            self.assertEqual(record["resolved"]["harness"], "codex-deepseek")
            checked = binding_preflight(record["resolved"], record["execution_route"], record["resolved_harness"], model="deepseek-flash", reasoning="high", which=lambda name: "/fake/codex-deepseek")
            self.assertTrue(checked["ok"])
            prompt = root / "prompt.md"
            prompt.write_text("binding check")
            captured = {}

            def fake_consultation(*args, **kwargs):
                captured.update(kwargs)
                evidence = root / "evidence"
                evidence.mkdir()
                (evidence / "execution.json").write_text("{}")
                return 0, evidence

            with patch("ekalavya.executor.binding_preflight", return_value={"ok": True}), patch("ekalavya.executor.run_consultation", side_effect=fake_consultation):
                outcome = execute(record, prompt, root)
            self.assertEqual(outcome["state"], "completed")
            self.assertEqual(captured["model"], "deepseek-flash")
            self.assertEqual(captured["effort"], "high")

    def test_missing_registered_deepseek_route_remains_unavailable(self):
        entry = {
            "provider": "deepseek", "family": "flash", "provider_model_id": "deepseek-flash",
            "display_name": "DeepSeek V4.1 Flash", "capabilities": {"reasoning_values": ["high"]},
            "identity_key": "flash", "lifecycle": "current", "execution_route": "not-a-route",
            "legacy_route": "not-a-route", "transport": "codex",
        }
        profile = {"name": "deepseek-flash", "default_identity_key": "flash", "permitted_candidates": ["flash"], "reasoning_policy": "fixed", "default_reasoning": "high"}
        readiness = profile_readiness("deepseek-flash", [profile], [entry], {"flash": {"state": "available"}}, which=lambda name: "/fake/codex-deepseek")
        self.assertEqual(readiness["harness_ready"], "unavailable")
        self.assertIn("execution adapter", readiness["readiness_reason"])

    def test_missing_route_does_not_infer_harness_from_transport(self):
        candidate = {
            "provider": "deepseek", "provider_model_id": "deepseek-flash",
            "transport": "codex",
        }
        self.assertIsNone(resolved_harness_binding(candidate, "not-a-route"))
        checked = binding_preflight(candidate, "not-a-route", None, which=lambda name: "/fake/codex-deepseek")
        self.assertFalse(checked["ok"])
        self.assertEqual(checked["reason_code"], "missing-route")

    def test_requested_harness_is_validated_against_independent_supported_contract(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ensure_control_files(root)
            entries = json.loads((root / "catalogue.json").read_text())
            profile = next(item for item in json.loads((root / "profiles.json").read_text()) if item["name"] == "deepseek-flash")
            current = next(item for item in entries if item["identity_key"] == profile["default_identity_key"])
            current.pop("harness")
            with patch("ekalavya.readiness.shutil.which", return_value="/fake/codex-deepseek"):
                valid = resolve(RunIntent("deepseek-flash", harness="codex-deepseek"), profile, entries, availability={"providers": {"deepseek": {"enabled": True}}, "models": {"deepseek-flash": {"enabled": True}}})
                invalid = resolve(RunIntent("deepseek-flash", harness="something-else"), profile, entries, availability={"providers": {"deepseek": {"enabled": True}}, "models": {"deepseek-flash": {"enabled": True}}})
            self.assertEqual(valid.state, "resolved")
            self.assertEqual(valid.resolved_harness, "codex-deepseek")
            self.assertEqual(invalid.state, "invalid-harness")
            self.assertEqual(invalid.reason_code, "unsupported-harness")

    def test_explicit_catalogue_harness_conflicting_with_route_fails_closed(self):
        entry = {
            "provider": "deepseek", "family": "flash", "provider_model_id": "deepseek-flash",
            "display_name": "DeepSeek V4.1 Flash", "capabilities": {"reasoning_values": ["high"]},
            "identity_key": "flash", "lifecycle": "current", "legacy_route": "deepseek-flash",
            "execution_route": "deepseek-flash", "harness": "codex",
        }
        profile = {"name": "deepseek-flash", "default_identity_key": "flash", "permitted_candidates": ["flash"], "reasoning_policy": "fixed", "default_reasoning": "high"}
        with patch("ekalavya.readiness.shutil.which", return_value="/fake/codex-deepseek"):
            result = resolve(RunIntent("deepseek-flash"), profile, [entry], availability={"providers": {"deepseek": {"enabled": True}}, "models": {"deepseek-flash": {"enabled": True}}})
        self.assertIsNone(result.candidate)
        self.assertEqual(result.reason_code, "unsupported-harness")

    def test_missing_harness_executable_after_valid_binding_is_unavailable(self):
        entry = {
            "provider": "deepseek", "family": "flash", "provider_model_id": "deepseek-flash",
            "display_name": "DeepSeek V4.1 Flash", "capabilities": {"reasoning_values": ["high"]},
            "identity_key": "flash", "lifecycle": "current", "legacy_route": "deepseek-flash",
            "execution_route": "deepseek-flash",
        }
        profile = {"name": "deepseek-flash", "default_identity_key": "flash", "permitted_candidates": ["flash"], "reasoning_policy": "fixed", "default_reasoning": "high"}
        with patch("ekalavya.readiness.shutil.which", return_value=None):
            result = resolve(RunIntent("deepseek-flash"), profile, [entry], availability={"providers": {"deepseek": {"enabled": True}}, "models": {"deepseek-flash": {"enabled": True}}})
        self.assertEqual(result.state, "harness-unavailable")
        self.assertEqual(result.reason_code, "harness-unavailable")

    def test_pro_resolves_exact_identity_before_and_after_former_cutoff(self):
        entry, profile, availability = self._pro_fixture()
        for observed_time in (
            datetime(2026, 9, 13, tzinfo=timezone.utc),
            datetime(2026, 9, 15, tzinfo=timezone.utc),
        ):
            with patch("ekalavya.readiness.shutil.which", return_value="/fake/codex-deepseek"):
                result = resolve(
                    RunIntent("deepseek-pro"), profile, [entry], availability=availability,
                    now=observed_time,
                )
            self.assertEqual(result.state, "resolved")
            self.assertEqual(result.candidate.provider_model_id, "deepseek-v4-pro")
            self.assertEqual(result.candidate.display_name, "DeepSeek V4 Pro")
            self.assertEqual(result.execution_route, "deepseek-pro")
            self.assertEqual(result.resolved_harness, "codex-deepseek")

    def test_pro_exact_identity_mismatch_fails_closed_without_substitution(self):
        entry, profile, availability = self._pro_fixture()
        with patch("ekalavya.readiness.shutil.which", return_value="/fake/codex-deepseek"):
            result = resolve(
                RunIntent("deepseek-pro"), profile, [entry],
                availability=availability,
                provider_reported_model_id="deepseek-flash",
            )
        self.assertEqual(result.state, "unavailable")
        self.assertEqual(result.reason_code, "exact-identity-mismatch")
        self.assertIsNone(result.candidate)
        self.assertIn("no fallback", result.reason)

    def test_provider_disabled_pro_stays_disabled_after_valid_binding(self):
        entry, profile, _ = self._pro_fixture()
        disabled = {"providers": {"deepseek": {"enabled": False}}, "models": {"deepseek-pro": {"enabled": True}}}
        with patch("ekalavya.readiness.shutil.which", return_value="/fake/codex-deepseek"):
            readiness = profile_readiness("deepseek-pro", [profile], [entry], {"pro": {"state": "available"}}, which=lambda name: "/fake/codex-deepseek")
            result = resolve(RunIntent("deepseek-pro"), profile, [entry], availability=disabled)
        self.assertEqual(readiness["harness_ready"], "ready")
        self.assertEqual(readiness["resolved_harness"], "codex-deepseek")
        self.assertEqual(result.state, "unavailable")
        self.assertEqual(result.reason_code, "provider-disabled")

    def test_exact_identity_guard_accepts_current_pro_reports_only(self):
        assert_deepseek_pro_exact()
        assert_deepseek_pro_exact(provider_reported_model_id="deepseek-v4-pro")
        assert_deepseek_pro_exact(provider_reported_model_id="DeepSeek-V4-Pro-0813")
        with self.assertRaises(DeepSeekExactIdentityError):
            assert_deepseek_pro_exact(provider_reported_model_id="DeepSeek V4.1 Flash")


if __name__ == "__main__":
    unittest.main()
