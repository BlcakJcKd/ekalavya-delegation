from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from delegation.config import default_config, save_config
from ekalavya.catalogue import load_catalogue
from ekalavya.cli import main
from ekalavya.config import ensure_control_files, load_profiles
from ekalavya.discovery import DiscoveryError, gemini_flash_generations, parse_agy_models


def discovery(timestamp: str = "2026-09-08T12:00:00+00:00") -> dict[str, object]:
    return {
        "provider": "gemini", "client": "agy", "client_version": "1.1.27", "observed_at": timestamp,
        "models": [
            {"provider_model_id": f"gemini-{generation}-flash-{reasoning}", "display_name": f"Gemini {generation} Flash ({reasoning.title()})"}
            for generation in ("3.6", "3.7", "3.8") for reasoning in ("low", "medium", "high")
        ],
    }


class GeminiDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {"XDG_CONFIG_HOME": str(self.root / "config"), "XDG_STATE_HOME": str(self.root / "state")}, clear=False)
        self.env.start()
        config_root = self.root / "config" / "ekalavya"
        config_root.mkdir(parents=True)
        save_config(default_config())
        ensure_control_files(config_root)
        self.catalogue = config_root / "catalogue.json"
        self.profiles = config_root / "profiles.json"

    def tearDown(self) -> None:
        self.env.stop()
        self.temp.cleanup()

    def _refresh(self) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with patch("ekalavya.cli.discover_gemini", return_value=discovery()), redirect_stdout(out), redirect_stderr(err):
            code = main(["models", "refresh", "--provider", "gemini", "--json"])
        return code, out.getvalue(), err.getvalue()

    def test_refresh_registers_one_3_8_candidate_without_duplicate_3_7_current(self):
        before_profiles = self.profiles.read_bytes()
        code, output, err = self._refresh()
        self.assertEqual(code, 0, err)
        payload = json.loads(output)
        self.assertEqual(payload["added_candidates"], 2)  # historical 3.6 plus new 3.8
        entries = load_catalogue(self.catalogue)
        current_37 = [e for e in entries if e.get("provider") == "gemini" and e.get("lifecycle") == "current" and str(e.get("provider_model_id")).startswith("gemini-3.7-flash-")]
        self.assertEqual(len(current_37), 1)
        candidate = next(e for e in entries if e.get("generation") == "3.8")
        self.assertEqual(candidate["lifecycle"], "candidate")
        self.assertEqual({item["provider_model_id"] for item in candidate["runtime_variants"]}, {"gemini-3.8-flash-low", "gemini-3.8-flash-medium", "gemini-3.8-flash-high"})
        profile = next(p for p in load_profiles(self.profiles) if p["name"] == "flash")
        self.assertEqual(profile["default_identity_key"], "d238f9b9c81bb54b9c4928e579b7d2b2ec16be0a861ed5da5a761d17abbad909")
        self.assertEqual(profile["default_reasoning"], "medium")
        self.assertNotEqual(self.profiles.read_bytes(), before_profiles)
        self.assertIn(candidate["identity_key"], profile["permitted_candidates"])
        # The seeded route remains the default until an explicit promotion.
        self.assertEqual(profile["default_identity_key"], "d238f9b9c81bb54b9c4928e579b7d2b2ec16be0a861ed5da5a761d17abbad909")

    def test_repeat_refresh_is_idempotent_and_preserves_lifecycle(self):
        self.assertEqual(self._refresh()[0], 0)
        first = json.loads(self.catalogue.read_text())
        self.assertEqual(self._refresh()[0], 0)
        second = json.loads(self.catalogue.read_text())
        self.assertEqual(len(first), len(second))
        candidate = next(e for e in second if e.get("generation") == "3.8")
        self.assertEqual(candidate["lifecycle"], "candidate")
        self.assertEqual(len([e for e in second if e.get("generation") == "3.8"]), 1)

    def test_failed_discovery_leaves_control_files_unchanged(self):
        before_catalogue, before_profiles = self.catalogue.read_bytes(), self.profiles.read_bytes()
        out, err = io.StringIO(), io.StringIO()
        with patch("ekalavya.cli.discover_gemini", side_effect=DiscoveryError("AGY unavailable")), redirect_stdout(out), redirect_stderr(err):
            self.assertEqual(main(["models", "refresh", "--provider", "gemini"]), 2)
        self.assertEqual(self.catalogue.read_bytes(), before_catalogue)
        self.assertEqual(self.profiles.read_bytes(), before_profiles)

    def test_registered_candidate_can_be_promoted_and_unknown_identity_stays_rejected(self):
        self.assertEqual(self._refresh()[0], 0)
        candidate = next(e for e in load_catalogue(self.catalogue) if e.get("generation") == "3.8")
        self.assertEqual(main(["models", "promote", "not-a-real-identity", "--basis", "manual"]), 2)
        self.assertEqual(main(["models", "promote", candidate["identity_key"], "--basis", "operational_efficiency", "--promotion-reason", "test", "--set-default", "--profile", "flash", "--default-reasoning", "low"]), 0)
        entries = load_catalogue(self.catalogue)
        promoted = next(e for e in entries if e.get("identity_key") == candidate["identity_key"])
        seed = next(e for e in entries if e.get("identity_key") == "d238f9b9c81bb54b9c4928e579b7d2b2ec16be0a861ed5da5a761d17abbad909")
        self.assertEqual(promoted["lifecycle"], "current")
        self.assertEqual(seed["lifecycle"], "previous")
        self.assertEqual(len([e for e in entries if e.get("provider") == "gemini" and e.get("family") == "flash" and e.get("lifecycle") == "current"]), 1)
        profile = next(p for p in load_profiles(self.profiles) if p["name"] == "flash")
        self.assertEqual(profile["default_identity_key"], candidate["identity_key"])
        self.assertEqual(profile["default_reasoning"], "low")

    def test_tabular_parser_rejects_partial_or_malformed_rows(self):
        with self.assertRaises(DiscoveryError):
            parse_agy_models("gemini-3.8-flash-low Gemini 3.8 Flash\n")
        with self.assertRaises(DiscoveryError):
            parse_agy_models("gemini-3.8-flash-low\t\n")
        rows = parse_agy_models("Fetching available models...\ngemini-3.8-flash-low\tGemini 3.8 Flash (Low)\nclaude-sonnet-4-6\tClaude Sonnet\n")
        self.assertEqual(rows[0]["provider_model_id"], "gemini-3.8-flash-low")

    def test_incomplete_gemini_generation_is_rejected_before_identity_derivation(self):
        rows = parse_agy_models(
            "gemini-3.8-flash-low\tGemini 3.8 Flash (Low)\n"
            "gemini-3.8-flash-medium\tGemini 3.8 Flash (Medium)\n"
        )
        with self.assertRaises(DiscoveryError):
            gemini_flash_generations(rows)


if __name__ == "__main__":
    unittest.main()
