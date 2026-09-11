import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ekalavya.catalogue import PROMOTION_BASES, add_candidate, promote, selectable
from ekalavya.config import ensure_control_files, migrate_legacy_config
from ekalavya.migrate import migrate_all
from ekalavya.ledger import SCHEMA_SQL, connect, import_legacy_state, record_benchmark_suite, record_benchmark_suite_correction, record_benchmark_task, record_cost, record_default_change, record_harness, record_price_snapshot, record_promotion_event, record_request_metric, record_run, record_task_attempt, upsert_model
from ekalavya.resolver import resolve
from ekalavya.schema import CandidateIdentity, RunIntent


class EkalavyaControlPlaneTests(unittest.TestCase):
    def test_catalogue_current_previous_and_family_identity(self):
        a = CandidateIdentity("google", "gemini", "gemini-3.7-flash", "Flash")
        b = CandidateIdentity("google", "gemini", "gemini-3.8-flash", "Flash")
        entries = add_candidate(add_candidate([], a), b)
        entries = promote(entries, a.identity_key)
        entries = promote(entries, b.identity_key)
        states = {e["provider_model_id"]: e["lifecycle"] for e in entries}
        self.assertEqual(states, {"gemini-3.7-flash": "previous", "gemini-3.8-flash": "current"})

    def test_promotion_records_basis_without_inventing_quality_claim(self):
        candidate = CandidateIdentity("gemini", "flash", "gemini-3.8-flash-medium")
        entries = add_candidate([], candidate)
        promoted = promote(entries, candidate.identity_key, "equal observed correctness and lower usage", promotion_basis="operational_efficiency")
        self.assertEqual(promoted[0]["lifecycle"], "current")
        self.assertEqual(promoted[0]["promotion_basis"], "operational_efficiency")
        self.assertEqual(promoted[0]["promotion_reason"], "equal observed correctness and lower usage")
        with self.assertRaises(ValueError):
            promote(entries, candidate.identity_key, promotion_basis="unverified_quality")
        self.assertIn("quality_superiority", PROMOTION_BASES)

    def test_promotion_ledger_field_is_additive_and_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            conn = connect(Path(d) / "ledger.sqlite3")
            event_id = record_promotion_event(conn, "gemini-3.8-key", from_state="candidate", to_state="current", reason="efficiency evidence", promotion_basis="operational_efficiency", occurred_at="2026-09-04T00:00:00+00:00")
            row = conn.execute("SELECT from_state,to_state,reason,promotion_basis FROM promotion_events WHERE id=?", (event_id,)).fetchone()
            self.assertEqual(tuple(row), ("candidate", "current", "efficiency evidence", "operational_efficiency"))
            with self.assertRaises(ValueError):
                record_promotion_event(conn, "x", from_state="candidate", to_state="current", reason="bad", promotion_basis="quality_claim_not_measured")

    def test_default_change_is_explicitly_auditable(self):
        with tempfile.TemporaryDirectory() as d:
            conn = connect(Path(d) / "ledger.sqlite3")
            event_id = record_default_change(conn, "flash", old_identity_key="old", new_identity_key="new", reason="operational efficiency", occurred_at="2026-09-04T00:00:00+00:00")
            row = conn.execute("SELECT profile,old_identity_key,new_identity_key,reason FROM default_changes WHERE id=?", (event_id,)).fetchone()
            self.assertEqual(tuple(row), ("flash", "old", "new", "operational efficiency"))

    def test_resolver_does_not_fail_over_and_enforces_native(self):
        c1 = CandidateIdentity("gemini", "gemini", "flash", capabilities={"reasoning_values": ["low", "high"]})
        c2 = CandidateIdentity("claude", "sonnet", "sonnet", capabilities={"reasoning_values": ["medium"]})
        profile = {"default_identity_key": c1.identity_key, "permitted_candidates": [c1.identity_key, c2.identity_key], "reasoning_policy": "overrideable"}
        native = resolve(RunIntent("coder", primary="gemini"), profile, [dict(c1.as_dict(), identity_key=c1.identity_key, lifecycle="current"), dict(c2.as_dict(), identity_key=c2.identity_key, lifecycle="current")])
        self.assertEqual(native.state, "same-provider-native-required")
        unavailable = resolve(RunIntent("coder", provider="claude"), profile, [dict(c1.as_dict(), identity_key=c1.identity_key, lifecycle="current"), dict(c2.as_dict(), identity_key=c2.identity_key, lifecycle="current")])
        self.assertEqual(unavailable.state, "unavailable")
        self.assertEqual(native.reason, "primary provider gemini must use its native agent capability")

    def test_resolver_blocks_a_model_when_its_provider_is_disabled(self):
        candidate = CandidateIdentity("deepseek", "deepseek-flash", "deepseek-v4-flash")
        entry = dict(candidate.as_dict(), identity_key=candidate.identity_key, lifecycle="current", legacy_route="deepseek-flash")
        profile = {"default_identity_key": candidate.identity_key, "permitted_candidates": [candidate.identity_key]}
        availability = {
            "providers": {"deepseek": {"enabled": False, "reason": "peak hours"}},
            "models": {"deepseek-flash": {"enabled": True}},
        }
        result = resolve(RunIntent("deepseek-flash"), profile, [entry], availability=availability)
        self.assertEqual(result.state, "unavailable")
        self.assertEqual(result.reason, "peak hours")
        self.assertIsNone(result.candidate)

    def test_reasoning_is_rejected_before_resolution(self):
        c = CandidateIdentity("local", "qwen", "qwen", capabilities={"reasoning_values": [False, True]})
        entry = dict(c.as_dict(), identity_key=c.identity_key, lifecycle="current")
        result = resolve(RunIntent("local-coder", reasoning="high"), {"default_identity_key": c.identity_key}, [entry])
        self.assertEqual(result.state, "invalid-reasoning")

    def test_harness_and_execution_route_are_resolved_without_coercion(self):
        candidate = CandidateIdentity("claude", "haiku", "claude-haiku", capabilities={"reasoning_values": ["medium"], "harness_values": ["claude"]})
        entry = dict(candidate.as_dict(), identity_key=candidate.identity_key, lifecycle="current", legacy_route="haiku", transport="claude", harness_version="1.2")
        profile = {"default_identity_key": candidate.identity_key, "reasoning_policy": "fixed", "default_reasoning": "medium"}
        with patch("ekalavya.readiness.shutil.which", return_value="/fake/claude"):
            resolved = resolve(RunIntent("haiku", harness="claude"), profile, [entry])
        self.assertEqual(resolved.state, "resolved")
        material = resolved.as_dict()["resolved"]
        self.assertEqual(material["execution_route"], "haiku")
        self.assertEqual(material["harness"], "claude")
        self.assertEqual(material["harness_version"], "1.2")
        invalid = resolve(RunIntent("haiku", harness="opencode"), profile, [entry])
        self.assertEqual(invalid.state, "invalid-harness")

    def test_explicit_previous_model_is_selectable_without_changing_default(self):
        current = CandidateIdentity("gemini", "flash", "gemini-3.7-flash-medium", capabilities={"reasoning_values": ["medium"]})
        previous = CandidateIdentity("gemini", "flash", "gemini-3.6-flash-medium", capabilities={"reasoning_values": ["medium"]})
        profile = {"default_identity_key": current.identity_key, "permitted_candidates": [current.identity_key, previous.identity_key], "reasoning_policy": "overrideable"}
        entries = [
            dict(current.as_dict(), identity_key=current.identity_key, lifecycle="current", execution_route="flash", legacy_route="flash", harness="agy", serving_engine="agy", transport="agy"),
            dict(previous.as_dict(), identity_key=previous.identity_key, lifecycle="previous", execution_route="flash", legacy_route="flash", harness="agy", serving_engine="agy", transport="agy"),
        ]
        with patch("ekalavya.readiness.shutil.which", return_value="/fake/agy"):
            result = resolve(RunIntent("flash", provider="gemini", model=previous.provider_model_id, reasoning="medium"), profile, entries)
        self.assertEqual(result.state, "resolved")
        self.assertEqual(result.candidate.provider_model_id, previous.provider_model_id)

    def test_ledger_schema_and_price_snapshot_immutability(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "ledger.sqlite3"; conn = connect(db)
            identity = CandidateIdentity("local", "qwen", "qwen3-coder:30b")
            upsert_model(conn, identity)
            sid = record_price_snapshot(conn, "x", "2026-09-03", {"input": 1}, currency="USD")
            sid2 = record_price_snapshot(conn, "x", "2026-09-03", {"input": 1}, currency="USD")
            self.assertEqual(sid, sid2); self.assertEqual(conn.execute("SELECT COUNT(*) FROM schema_versions").fetchone()[0], 1)
            with self.assertRaises(ValueError): record_price_snapshot(conn, "x", "2026-09-03", {"input": 2}, currency="USD")
            conn.execute("INSERT INTO runs(run_id,started_at,requested_json) VALUES('r','now','{}')")
            record_cost(conn, "r", billing_mode="subscription", cost_source="unavailable")
            row = conn.execute("SELECT provider_reported_cost,calculated_cost FROM cost_observations").fetchone()
            self.assertIsNone(row[0]); self.assertIsNone(row[1])

    def test_ledger_records_harness_request_reasoning_and_nullable_usage(self):
        with tempfile.TemporaryDirectory() as d:
            conn = connect(Path(d) / "ledger.sqlite3")
            harness_id = record_harness(conn, "agy", version="1.1.25", adapter_version="adapter", transport="agy")
            record_run(conn, "run", {"profile": "experiment"}, resolved={"provider_model_id": "gemini-3.8-flash-medium"}, harness_id=harness_id, provider="gemini", billing_mode="subscription")
            request_id = record_request_metric(conn, "run", {"ordinal": 1, "model": "gemini-3.8-flash-medium", "reasoning_tokens": 12})
            self.assertEqual(conn.execute("SELECT harness_id FROM runs WHERE run_id='run'").fetchone()[0], harness_id)
            self.assertEqual(conn.execute("SELECT reasoning_tokens FROM request_metrics WHERE id=?", (request_id,)).fetchone()[0], 12)
            record_cost(conn, "run", billing_mode="subscription", cost_source="unavailable")
            row = conn.execute("SELECT provider_reported_cost,api_equivalent_cost FROM cost_observations WHERE run_id='run'").fetchone()
            self.assertIsNone(row[0]); self.assertIsNone(row[1])

    def test_benchmark_metadata_columns_are_additive_and_nullable(self):
        with tempfile.TemporaryDirectory() as d:
            conn = connect(Path(d) / "ledger.sqlite3")
            suite_id = record_benchmark_suite(conn, "public", "public", "1", evaluation_class="public_characterization")
            task_id = record_benchmark_task(conn, suite_id, family="P1", task_id="p1", variant_seed="1", content_hash="c", prompt_hash="p", evaluator_hash="e", baseline_score=25.0, baseline_check_vector=[False, True], task_spec_hash="t", allowed_edit_manifest_hash="a", reference_validation_passed=True, reference_validation_at="now")
            record_run(conn, "run", {"profile": "flash"})
            attempt_id = record_task_attempt(conn, "run", task_id, score=75.0, baseline_score=25.0, baseline_check_vector=[False, True], final_check_vector=[True, True], delta_score=50.0, normalized_improvement=2/3, evaluator_tampering=False, prohibited_changed_files=[])
            task = conn.execute("SELECT baseline_score,baseline_check_vector_json,task_spec_hash,reference_validation_passed FROM benchmark_tasks WHERE id=?", (task_id,)).fetchone()
            attempt = conn.execute("SELECT baseline_score,final_check_vector_json,delta_score,evaluator_tampering FROM task_attempts WHERE id=?", (attempt_id,)).fetchone()
            self.assertEqual(tuple(task), (25.0, "[false, true]", "t", 1))
            self.assertEqual(tuple(attempt), (25.0, "[true, true]", 50.0, 0))

    def test_evaluation_classes_are_explicit_and_historical_defaults_are_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            conn = connect(Path(d) / "ledger.sqlite3")
            record_run(conn, "public", {"profile": "flash"}, evaluation_class="public_characterization")
            suite_id = record_benchmark_suite(conn, "public", "public", "1", evaluation_class="public_characterization")
            self.assertEqual(conn.execute("SELECT evaluation_class FROM runs WHERE run_id='public'").fetchone()[0], "public_characterization")
            self.assertEqual(conn.execute("SELECT evaluation_class FROM benchmark_suites WHERE id=?", (suite_id,)).fetchone()[0], "public_characterization")
            with self.assertRaises(ValueError): record_run(conn, "bad", {}, evaluation_class="not-a-class")

    def test_suite_provenance_correction_is_append_only(self):
        with tempfile.TemporaryDirectory() as d:
            conn = connect(Path(d) / "ledger.sqlite3")
            suite_id = record_benchmark_suite(conn, "public", "public", "1", git_sha="old", evaluation_class="public_characterization")
            correction_id = record_benchmark_suite_correction(conn, suite_id, "new", reason="implementation identity correction", evidence={"raw_preserved": True}, corrected_at="2026-01-01T00:00:00+00:00")
            self.assertEqual(conn.execute("SELECT git_sha FROM benchmark_suites WHERE id=?", (suite_id,)).fetchone()[0], "new")
            row = conn.execute("SELECT originally_recorded_git_sha,corrected_git_sha,corrected_at FROM benchmark_suite_corrections WHERE id=?", (correction_id,)).fetchone()
            self.assertEqual(tuple(row), ("old", "new", "2026-01-01T00:00:00+00:00"))

    def test_config_migration_preserves_source_and_permissions_idempotently(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root / "old"; target = root / "new"; source.mkdir(mode=0o700); secret = source / "config.toml"; secret.write_text("[providers]\n"); os.chmod(secret, 0o600)
            first = migrate_legacy_config(source, target); second = migrate_legacy_config(source, target)
            self.assertEqual(len(first["copied"]), 1); self.assertEqual(len(second["copied"]), 0); self.assertTrue(secret.exists()); self.assertEqual((target / "config.toml").stat().st_mode & 0o777, 0o600)

    def test_legacy_evidence_import_is_hash_idempotent_and_non_destructive(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "legacy"; root.mkdir(); source = root / "run.json"; source.write_text('{"provider":"local"}\n')
            conn = connect(Path(d) / "ledger.sqlite3")
            first = import_legacy_state(conn, root); second = import_legacy_state(conn, root)
            self.assertEqual(first["files"], 1); self.assertEqual(first["records"], 1); self.assertEqual(second["files"], 0); self.assertTrue(source.exists())
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM imported_evidence").fetchone()[0], 1)

    def test_execution_metadata_import_creates_a_safe_historical_run(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "legacy"; root.mkdir(); source = root / "execution.json"
            source.write_text(json.dumps({"delegate": "flash", "provider": "gemini", "requested_model": "gemini-x", "started_at": "2026-01-01T00:00:00Z", "response_recorded": True, "secret": "must-not-be-copied"}))
            conn = connect(Path(d) / "ledger.sqlite3"); import_legacy_state(conn, root)
            row = conn.execute("SELECT profile,provider,requested_json FROM runs").fetchone()
            self.assertEqual(row[0], "flash"); self.assertEqual(row[1], "gemini"); self.assertNotIn("must-not-be-copied", row[2])

    def test_control_file_migration_is_additive_and_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            report = ensure_control_files(root); self.assertEqual(set(report["created"]), {"catalogue.json", "profiles.json"})
            self.assertGreater(len(json.loads((root / "catalogue.json").read_text())), 1)
            self.assertNotIn("Migrated", (root / "profiles.json").read_text())
            self.assertEqual((root / "catalogue.json").stat().st_mode & 0o777, 0o600)
            self.assertEqual((root / "profiles.json").stat().st_mode & 0o777, 0o600)
            self.assertEqual(ensure_control_files(root)["created"], [])

    def test_bootstrap_leaves_both_existing_control_files_byte_identical(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ensure_control_files(root)
            original_catalogue = (root / "catalogue.json").read_bytes()
            original_profiles = (root / "profiles.json").read_bytes()
            report = ensure_control_files(root)
            self.assertEqual(report["status"], "complete")
            self.assertEqual((root / "catalogue.json").read_bytes(), original_catalogue)
            self.assertEqual((root / "profiles.json").read_bytes(), original_profiles)

    def test_bootstrap_reports_catalogue_only_incomplete_pair_without_rewriting(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ensure_control_files(root)
            original = (root / "catalogue.json").read_bytes()
            (root / "profiles.json").unlink()
            report = ensure_control_files(root)
            self.assertEqual(report["created"], [])
            self.assertEqual(report["skipped"], ["catalogue.json"])
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["missing"], ["profiles.json"])
            self.assertEqual((root / "catalogue.json").read_bytes(), original)

    def test_bootstrap_reports_profiles_only_incomplete_pair_without_rewriting(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ensure_control_files(root)
            original = (root / "profiles.json").read_bytes()
            (root / "catalogue.json").unlink()
            report = ensure_control_files(root)
            self.assertEqual(report["created"], [])
            self.assertEqual(report["skipped"], ["profiles.json"])
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["missing"], ["catalogue.json"])
            self.assertEqual((root / "profiles.json").read_bytes(), original)

    def test_explicit_migration_stops_before_state_import_for_incomplete_pair(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ensure_control_files(root)
            (root / "profiles.json").unlink()
            report = migrate_all(legacy_config=root / "absent-legacy", new_config=root, legacy_state=root / "absent-state", db=root / "ledger.sqlite3")
            self.assertEqual(report["config"]["control_files"]["status"], "incomplete")
            self.assertEqual(report["state"]["skipped"], "incomplete catalogue/profiles pair")
            self.assertFalse((root / "ledger.sqlite3").exists())


if __name__ == "__main__":
    unittest.main()
