import tempfile
import unittest
import json
import hashlib
import zipfile
import os
from pathlib import Path
from unittest.mock import patch

from benchmark.adapters import AntigravityAdapter
from benchmark.public_characterization.evaluate import evaluate
from benchmark.public_characterization.generate import make_instance, manifest, materialize, workspace_digest
from benchmark.public_characterization.runner import _update_profile_catalogue, check_local_suite
from benchmark.public_characterization.audit import _matrix
from benchmark.provenance import ProvenanceError, validate_git_identity
from benchmark.review_bundle import create_review_bundle
from ekalavya.catalogue import canonicalize_gemini_flash_generations, expand_runtime_variants
from ekalavya.config import ensure_control_files, load_profiles
from ekalavya.harness_registry import current_registry, validate_registry
from ekalavya.schema import CandidateIdentity, RunIntent
from ekalavya.resolver import resolve
from delegation.config import default_config, save_config


class PublicCharacterizationTests(unittest.TestCase):
    def test_instances_are_deterministic_and_have_no_authoritative_repair(self):
        first = make_instance("P1_multi_file_debug", 42)
        second = make_instance("P1_multi_file_debug", 42)
        self.assertEqual(first.files, second.files)
        self.assertEqual(first.prompt, second.prompt)
        self.assertNotIn("reference", "\n".join(first.files))
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "a", Path(temp) / "b"
            materialize(first, a); materialize(second, b)
            self.assertEqual(workspace_digest(a), workspace_digest(b))
            assessment = evaluate(first, a)
            self.assertTrue(assessment["objective"])
            self.assertFalse(assessment["adversarial_isolation"])
            self.assertFalse(assessment["authoritative_reference_present"])
            self.assertEqual(len(assessment["checks"]), 8)
            self.assertLess(assessment["score"], 100)

    def test_manifest_records_public_identity_hashes(self):
        item = make_instance("P4_compat_refactor", 44)
        data = manifest([item])
        self.assertEqual(data["evaluation_class"], "public_characterization")
        record = data["instances"][0]
        self.assertEqual(record["task_id"], item.task_id)
        self.assertEqual(len(record["workspace_hash"]), 64)
        self.assertEqual(len(record["visible_evaluator_hash"]), 64)

    def test_local_fixture_check_is_repeatable(self):
        self.assertEqual(check_local_suite(), check_local_suite())

    def test_exact_runtime_model_id_is_selected_without_reasoning_overlay(self):
        for model_id in ("gemini-3.7-flash-low", "gemini-3.7-flash-medium", "gemini-3.7-flash-high", "gemini-3.8-flash-low", "gemini-3.8-flash-medium", "gemini-3.8-flash-high"):
            argv = AntigravityAdapter(model=model_id, reasoning_effort=None).command(Path("/workspace"), "prompt", Path("/evidence"))
            self.assertEqual(argv[argv.index("--model") + 1], model_id)
            self.assertNotIn("--effort", argv)

    def test_catalogue_lifecycle_is_generation_level_and_variants_are_exact(self):
        observed = "2026-09-03T12:00:00+00:00"
        discovered = [{"provider_model_id": f"gemini-{generation}-flash-{reasoning}", "display_name": f"Gemini {generation} {reasoning}"} for generation in ("3.6", "3.7", "3.8") for reasoning in ("low", "medium", "high")]
        entries = canonicalize_gemini_flash_generations([], discovered, observed_at=observed, serving_engine_version="1.1.25")
        self.assertEqual({(item["generation"], item["lifecycle"], item["lifecycle_scope"]) for item in entries}, {("3.6", "previous", "generation_family"), ("3.7", "current", "generation_family"), ("3.8", "candidate", "generation_family")})
        self.assertTrue(all(item["discovery_timestamp"] == observed for item in entries))
        variants = expand_runtime_variants(entries)
        ids = {item["provider_model_id"] for item in variants if item.get("catalogue_parent_identity_key")}
        self.assertIn("gemini-3.8-flash-high", ids)
        current = next(item for item in entries if item["generation"] == "3.7")
        profile = {"default_identity_key": current["identity_key"], "permitted_candidates": [current["identity_key"]], "reasoning_policy": "overrideable", "default_reasoning": "medium"}
        result = resolve(RunIntent("flash", provider="gemini", reasoning="high"), profile, entries)
        self.assertEqual(result.state, "resolved")
        self.assertEqual(result.candidate.provider_model_id, "gemini-3.7-flash-high")
        newer = canonicalize_gemini_flash_generations([], discovered, observed_at=observed, serving_engine_version="1.1.28")
        self.assertEqual({item["generation"]: item["identity_key"] for item in entries}, {item["generation"]: item["identity_key"] for item in newer})

    def test_real_profile_catalogue_update_path_uses_merge_contract_and_saves_state(self):
        discovered = [{"provider_model_id": f"gemini-{generation}-flash-{reasoning}", "display_name": f"Gemini {generation} Flash ({reasoning})"} for generation in ("3.6", "3.7", "3.8") for reasoning in ("low", "medium", "high")]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(root / "config"), "XDG_STATE_HOME": str(root / "state")}, clear=False):
                config_root = root / "config" / "ekalavya"
                config_root.mkdir(parents=True)
                save_config(default_config())
                ensure_control_files(config_root)
                result = _update_profile_catalogue(discovered, "2026-09-08T12:00:00+00:00", "1.1.27")
                catalogue = json.loads((config_root / "catalogue.json").read_text())
                self.assertEqual(result["catalogue_entries"], len(catalogue))
                self.assertEqual(result["added_candidates"], 2)
                self.assertTrue(result["profile_updated"])
                profile = next(item for item in load_profiles(config_root / "profiles.json") if item["name"] == "flash")
                self.assertEqual(profile["default_reasoning"], "medium")
                candidate_key = next(entry["identity_key"] for entry in catalogue if entry.get("generation") == "3.8")
                self.assertIn(candidate_key, profile["permitted_candidates"])


class HarnessRegistryTests(unittest.TestCase):
    def test_registry_has_separate_eligibility_classes(self):
        registry = current_registry()
        validate_registry(registry)
        agy = next(item for item in registry if item["name"] == "agy")
        self.assertEqual(agy["eligibility"]["ordinary"], "supported")
        self.assertEqual(agy["eligibility"]["public_characterization"], "supported")
        self.assertEqual(agy["eligibility"]["hidden_benchmark"], "unsupported")
        self.assertIn("independent candidate-tool subprocess", agy["reason"])
        self.assertEqual(agy["version"], "1.1.26")
        self.assertEqual(agy["capabilities"]["filesystem_containment"], "unsupported")
        self.assertEqual(agy["capabilities"]["tool_network_containment"], "unsupported")
        self.assertEqual(agy["capabilities"]["tool_trace"], "unavailable")
        self.assertEqual(agy["telemetry"]["request_metric_semantics"], "harness_session")
        self.assertEqual(agy["telemetry"]["tool_event_telemetry"], "unavailable")


class ReviewBundleTests(unittest.TestCase):
    def test_bundle_counts_tampering_as_failed_not_completed(self):
        with tempfile.TemporaryDirectory() as temp:
            state = Path(temp) / "experiment"; (state / "evidence").mkdir(parents=True)
            (state / "REPORT.md").write_text("review\n")
            (state / "evidence/a.json").write_text(json.dumps({"exit_code": 0, "timed_out": False, "status": "evaluator_tampering"}))
            result = create_review_bundle("example", state_dir=state, output=Path(temp) / "bundle")
            self.assertEqual(result["manifest"]["completed"], 0)
            self.assertEqual(result["manifest"]["failed"], 1)
            self.assertEqual(result["manifest"]["timeouts"], 0)

    def test_bundle_is_allowlisted_and_archived_without_optional_plots(self):
        with tempfile.TemporaryDirectory() as temp:
            state = Path(temp) / "experiment"
            (state / "evidence").mkdir(parents=True)
            (state / "REPORT.md").write_text("state=/home/bivin/.local/state/ekalavya\n")
            (state / "run-summary.json").write_text(json.dumps({"attempts": 1}))
            (state / "evidence/a.json").write_text(json.dumps({"exit_code": 0, "timed_out": False}))
            (state / "ledger.sqlite3").write_text("must not copy")
            (state / "workspaces").mkdir()
            result = create_review_bundle("example", state_dir=state, output=Path(temp) / "review-bundle")
            bundle = Path(result["bundle"])
            self.assertTrue(Path(result["archive"]).is_file())
            self.assertFalse((bundle / "ledger.sqlite3").exists())
            self.assertFalse((bundle / "workspaces").exists())
            self.assertEqual((bundle / "REPORT.md").read_text(), "state=<USER_HOME>/.local/state/ekalavya\n")
            self.assertEqual(result["manifest"]["workspaces_included"], False)
            self.assertEqual(result["manifest"]["raw_provider_traces_included"], False)
            with zipfile.ZipFile(result["archive"]) as archive:
                names = set(archive.namelist())
            self.assertIn("manifest.json", names)
            self.assertNotIn("ledger.sqlite3", names)

            manifest_data = json.loads((bundle / "manifest.json").read_text())
            self.assertEqual(set(manifest_data["included_files"]), {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()})
            for name, digest in manifest_data["sha256"].items():
                self.assertEqual(hashlib.sha256((bundle / name).read_bytes()).hexdigest(), digest)
            self.assertIn("self-referential", manifest_data["sha256_exclusions"]["manifest.json"])

    def test_bundle_preserves_source_and_represents_provenance_correction(self):
        with tempfile.TemporaryDirectory() as temp:
            state = Path(temp) / "experiment"
            (state / "evidence").mkdir(parents=True)
            source = state / "evidence" / "attempt.json"
            source.write_text(json.dumps({"exit_code": 0, "timed_out": False, "path": "/home/bivin/private"}))
            before = source.read_bytes()
            (state / "REPORT.md").write_text("review\n")
            (state / "provenance").mkdir()
            (state / "provenance" / "correction-summary.json").write_text(json.dumps({
                "originally_recorded_suite_sha": "old",
                "corrected_suite_sha": "new",
                "correction_reason": "test",
                "correction_timestamp": "now",
                "ledger_correction_id": 7,
            }))
            result = create_review_bundle("public-characterization-v1", state_dir=state, output=Path(temp) / "bundle")
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(result["manifest"]["provenance_correction"]["originally_recorded_suite_sha"], "old")
            self.assertEqual(result["manifest"]["provenance_correction"]["corrected_suite_sha"], "new")
            sanitized = json.loads((Path(result["bundle"]) / "evidence/attempt.json").read_text())
            self.assertEqual(sanitized["path"], "<USER_HOME>/private")

    def test_matrix_extraction_is_deterministic_and_complete(self):
        item = make_instance("P1_multi_file_debug", 42)
        evidence = {"run_id": "run", "started_at": "2026-01-01T00:00:00Z", "resolved": {"provider_model_id": "model", "reasoning": "low"}, "task": {"family": item.family}, "exit_code": 0, "timed_out": False, "wall_seconds": 1.25, "assessment": {"full_pass": False, "checks": [{"name": f"c{i}", "passed": i % 2 == 0} for i in range(1, 9)]}}
        first = _matrix([evidence], {"run": {"status": "completed", "score": 50.0, "input_tokens": 1, "output_tokens": 2, "cache_read_tokens": 3, "reasoning_tokens": 4}})
        second = _matrix([evidence], {"run": {"status": "completed", "score": 50.0, "input_tokens": 1, "output_tokens": 2, "cache_read_tokens": 3, "reasoning_tokens": 4}})
        self.assertEqual(first, second)
        self.assertEqual(first[0]["passed_checks"], 4)
        self.assertEqual([first[0][f"check_{i}"] for i in range(1, 9)], ["fail", "pass", "fail", "pass", "fail", "pass", "fail", "pass"])

    def test_provenance_rejects_commit_without_suite_sources(self):
        with self.assertRaises(ProvenanceError):
            validate_git_identity(Path(__file__).resolve().parents[1], ("benchmark/public_characterization/runner.py",), git_sha="91f96127f9393cc81e4f4c296ec5d8e228210a13")


if __name__ == "__main__":
    unittest.main()
