from __future__ import annotations

import json
import subprocess
import unittest
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from delegation.core import run_consultation
from ekalavya.config import ensure_control_files
from ekalavya.deepseek import (
    DEEPSEEK_PRO_CUTOFF_UTC,
    DEEPSEEK_REASONING_MAPPING,
    DeepSeekExactIdentityError,
    assert_deepseek_pro_exact,
    map_reasoning_level,
)
from ekalavya.resolver import resolve
from ekalavya.schema import RunIntent


class DeepSeekLifecycleTests(unittest.TestCase):
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

    def test_pro_cutoff_is_deterministic_on_both_sides(self):
        before = DEEPSEEK_PRO_CUTOFF_UTC - timedelta(seconds=1)
        after = DEEPSEEK_PRO_CUTOFF_UTC + timedelta(seconds=1)
        assert_deepseek_pro_exact(now=before)
        with self.assertRaises(DeepSeekExactIdentityError):
            assert_deepseek_pro_exact(now=after)
        assert_deepseek_pro_exact(now=after, provider_reported_model_id="DeepSeek-V4-Pro-0813")

    def test_resolver_fails_closed_for_pro_after_cutoff(self):
        entry = {"provider": "deepseek", "family": "pro", "provider_model_id": "deepseek-v4-pro", "display_name": "DeepSeek V4 Pro", "capabilities": {"reasoning_values": ["high"]}, "identity_key": "pro", "lifecycle": "current", "legacy_route": "deepseek-pro"}
        result = resolve(RunIntent("deepseek-pro"), {"default_identity_key": "pro", "permitted_candidates": ["pro"]}, [entry], now=DEEPSEEK_PRO_CUTOFF_UTC)
        self.assertEqual(result.state, "unavailable")
        self.assertIn("no fallback", result.reason)

    def test_pro_cutoff_blocks_launch_before_subprocess(self):
        with TemporaryDirectory() as temp:
            workspace = Path(temp)
            (workspace / ".delegation-scope.json").write_text('{"mode":"read-only"}')
            called = []
            def fake_run(*args, **kwargs):
                called.append(True)
                return subprocess.CompletedProcess(args[0], 0, "", "")
            with patch("delegation.core.shutil.which", return_value="/bin/codex-deepseek"):
                with self.assertRaises(DeepSeekExactIdentityError):
                    run_consultation("deepseek-pro", workspace, "check", now=DEEPSEEK_PRO_CUTOFF_UTC, run=fake_run)
            self.assertEqual(called, [])


if __name__ == "__main__":
    unittest.main()
