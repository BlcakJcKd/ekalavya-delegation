import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ekalavya.ledger import clear_observability, connect, delete_feedback, normalize_cutoff, record_quota_snapshot, record_run, record_run_observability, record_safe_request_metric, record_resolution, set_feedback
import ekalavya.ledger as ledger_module
from ekalavya.schema import RunIntent, normalize_task
from ekalavya.usage import build_insights, build_usage, export_usage, period_bounds
from ekalavya.quota import hosted_quota_statuses
from ekalavya.telemetry import assert_safe_analytics_payload, persist_execution_observability


class UsageObservabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.conn = connect(Path(self.temp.name) / "ledger.sqlite3")

    def tearDown(self):
        self.conn.close(); self.temp.cleanup()

    def add_event(self, run_id="r1", **values):
        record_run(self.conn, run_id, {"profile": values.get("profile", "reviewer"), "provider": values.get("provider", "claude")}, evaluation_class="unknown")
        record_resolution(self.conn, run_id, {"profile": "reviewer"}, {"state": "resolved", "resolved": {"provider_model_id": values.get("requested_model", "claude-sonnet")}})
        record_run_observability(self.conn, run_id, {"task": values.get("task", "review"), "requested_profile": values.get("profile", "reviewer"), "primary_provider": values.get("provider", "claude"), "requested_provider_model_id": values.get("requested_model", "claude-sonnet"), "provider_reported_model_id": values.get("effective_model"), "effective_identity_status": "provider_reported" if values.get("effective_model") else "unavailable", "execution_status": values.get("status", "success"), "total_tokens": values.get("tokens", 10), "total_tokens_provenance": "provider_reported", "wall_seconds": values.get("wall", 2.0), "telemetry_status": "complete", "token_telemetry_status": "complete"})

    def test_task_contract_rejects_free_text_and_preserves_unspecified(self):
        self.assertEqual(normalize_task(None), "unspecified")
        self.assertEqual(normalize_task("scientific-critique"), "scientific-critique")
        for value in ("Review this prompt", "A" * 65, "../secret", "review\ntext", ""):
            if value == "":
                self.assertEqual(normalize_task(value), "unspecified")
            else:
                with self.assertRaises(ValueError): normalize_task(value)
        self.assertEqual(RunIntent("p", task="coding").task, "coding")

    def test_requested_and_effective_model_are_distinct(self):
        self.add_event(effective_model="claude-sonnet-4")
        row = self.conn.execute("SELECT requested_provider_model_id,provider_reported_model_id FROM run_observability").fetchone()
        self.assertEqual(tuple(row), ("claude-sonnet", "claude-sonnet-4"))

    def test_safe_metric_rejects_private_payload_and_stores_empty_metadata(self):
        record_run(self.conn, "r", {"profile": "p"})
        with self.assertRaises(ValueError): record_safe_request_metric(self.conn, "r", {"total_tokens": 4, "prompt": "secret"})
        record_safe_request_metric(self.conn, "r", {"total_tokens": 4, "total_tokens_provenance": "provider_reported"})
        self.assertEqual(self.conn.execute("SELECT metadata_json,observability_owned FROM request_metrics").fetchone()[0:2], ("{}", 1))

    def test_real_post_execution_projection_uses_canonical_cache_read_field(self):
        record_run(self.conn, "real", {"profile": "p"}, provider="vllm", identity_key="requested")
        persist_execution_observability(self.conn, "real", run_data={"task": "coding", "requested_profile": "p", "primary_provider": "vllm", "requested_provider_model_id": "requested"}, execution={"state": "completed", "provider_reported_usage": {"input_tokens": 100, "output_tokens": 20, "cache_read_tokens": 40, "total_tokens": 120}, "provider_reported_model_id": "served", "wall_seconds": 1.0})
        row = self.conn.execute("SELECT cache_read_tokens,cache_read_tokens_provenance,metadata_json FROM request_metrics").fetchone()
        self.assertEqual(tuple(row), (40, "provider_reported", "{}"))
        self.assertEqual(self.conn.execute("SELECT cache_read_tokens FROM run_observability").fetchone()[0], 40)
        persist_execution_observability(self.conn, "real", run_data={"task": "coding", "requested_profile": "p", "primary_provider": "vllm"}, execution={"state": "completed", "provider_reported_usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}})
        self.assertIsNone(self.conn.execute("SELECT cache_read_tokens FROM request_metrics ORDER BY id DESC LIMIT 1").fetchone()[0])
        self.assertNotIn("provider_reported_usage", export_usage(self.conn, fmt="json", period="all"))
        self.assertNotIn("provider_reported_usage", export_usage(self.conn, fmt="csv", period="all"))

    def test_harness_reported_usage_preserves_provenance_and_nulls(self):
        record_run(self.conn, "claude", {"profile": "p"}, provider="claude", identity_key="requested")
        persist_execution_observability(
            self.conn, "claude", run_data={"task": "coding", "requested_profile": "p", "primary_provider": "claude"},
            execution={"state": "completed", "provider_reported_usage": {"input_tokens": 120, "output_tokens": 34, "cache_read_tokens": 18, "cache_write_tokens": 6}, "usage_provenance": "harness_reported"},
        )
        row = self.conn.execute("SELECT input_tokens,output_tokens,cache_read_tokens,reasoning_tokens,provider_reported_model_id,input_tokens_provenance,total_tokens,total_tokens_provenance FROM run_observability").fetchone()
        self.assertEqual(tuple(row), (120, 34, 18, None, None, "harness_reported", 154, "derived"))
        metric = self.conn.execute("SELECT cache_write_tokens,cache_write_tokens_provenance FROM request_metrics").fetchone()
        self.assertEqual(tuple(metric), (6, "harness_reported"))
        exported = export_usage(self.conn, fmt="json", period="all")
        self.assertNotIn("sanitized response content", exported)
        self.assertNotIn("modelUsage", exported)

    def test_safe_model_usage_rows_preserve_per_model_metrics_without_fake_effective_model(self):
        record_run(self.conn, "claude-model-usage", {"profile": "haiku"}, provider="claude", identity_key="requested")
        persist_execution_observability(
            self.conn, "claude-model-usage",
            run_data={"task": "review", "requested_profile": "haiku", "primary_provider": "claude"},
            execution={
                "state": "completed",
                "provider_reported_usage_by_model": [
                    {"model": "claude-haiku-4-5-20251001", "input_tokens": 120, "output_tokens": 34, "cache_read_tokens": 18, "cache_write_tokens": 6},
                    {"model": "claude-sonnet-5", "input_tokens": 12, "output_tokens": 8},
                ],
                "usage_provenance": "harness_reported",
            },
        )
        run = self.conn.execute("SELECT input_tokens,output_tokens,cache_read_tokens,total_tokens,provider_reported_model_id FROM run_observability").fetchone()
        self.assertEqual(tuple(run), (132, 42, 18, 174, None))
        metrics = self.conn.execute("SELECT ordinal,model,input_tokens,output_tokens,cache_read_tokens,cache_write_tokens,reasoning_tokens,metadata_json FROM request_metrics ORDER BY ordinal").fetchall()
        self.assertEqual([tuple(row) for row in metrics], [
            (1, "claude-haiku-4-5-20251001", 120, 34, 18, 6, None, "{}"),
            (2, "claude-sonnet-5", 12, 8, None, None, None, "{}"),
        ])

    def test_invalid_model_usage_projection_is_not_persisted(self):
        record_run(self.conn, "invalid-model-usage", {"profile": "haiku"}, provider="claude", identity_key="requested")
        persist_execution_observability(
            self.conn, "invalid-model-usage",
            run_data={"task": "review", "requested_profile": "haiku", "primary_provider": "claude"},
            execution={
                "state": "completed",
                "provider_reported_usage_by_model": [{"model": "claude-haiku-4-5-20251001", "input_tokens": 1, "raw_payload": "private"}],
                "usage_provenance": "harness_reported",
            },
        )
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM request_metrics").fetchone()[0], 0)
        row = self.conn.execute("SELECT input_tokens,provider_reported_model_id FROM run_observability").fetchone()
        self.assertEqual(tuple(row), (None, None))

    def test_exported_safety_gate_rejects_nested_and_identity_secrets(self):
        assert_safe_analytics_payload({"input_tokens": 1, "cache_read_tokens": 0})
        for payload in ({"chain_of_thought": "private"}, {"api_key": "secret"}, {"safe": {"prompt": "private"}}, {"raw_payload": []}):
            with self.assertRaises(ValueError): assert_safe_analytics_payload(payload)

    def test_ambiguous_and_benchmark_rows_are_not_usage_events(self):
        self.add_event("ordinary")
        record_run(self.conn, "bench", {"profile": "p"}, evaluation_class="public_characterization")
        payload = build_usage(self.conn, period="all")
        self.assertEqual(payload["summary"]["runs"], 1)

    def test_migration_backfills_only_canonical_delegate_evidence(self):
        record_run(self.conn, "observed", {"profile": "p", "task": "review"}, provider="claude", identity_key="catalogue-key", status="completed", raw_evidence_path="/private/delegate_runs/x")
        record_run(self.conn, "imported", {"profile": "p"}, provider="claude", identity_key="catalogue-key", status="imported", raw_evidence_path="/private/delegate_runs/y")
        self.conn.execute("DELETE FROM schema_versions")
        self.conn.execute("INSERT INTO schema_versions VALUES(2,'2026-01-01T00:00:00+00:00','old')")
        self.conn.commit()
        self.conn.close()
        self.conn = connect(Path(self.temp.name) / "ledger.sqlite3")
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM run_observability").fetchone()[0], 1)
        row = self.conn.execute("SELECT task,total_tokens,token_telemetry_status FROM run_observability").fetchone()
        self.assertEqual(tuple(row), ("review", None, "unavailable"))

    def test_coverage_and_low_sample_latency_guardrails_are_deterministic(self):
        self.add_event(tokens=None, wall=1.0)
        payload = build_usage(self.conn, period="all")
        self.assertIsNone(payload["summary"]["reported_tokens"]["value"])
        self.assertIsNone(payload["summary"]["latency_seconds"]["median"])
        self.assertIsNone(payload["summary"]["latency_seconds"]["p90"])

    def test_controls_preserve_canonical_history(self):
        self.add_event()
        self.conn.execute("INSERT INTO cost_observations(run_id,billing_mode,cost_source) VALUES('r1','subscription','unavailable')")
        self.conn.commit()
        result = clear_observability(self.conn)
        self.assertGreaterEqual(result["run_observability"], 1)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0], 1)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM resolution_decisions").fetchone()[0], 1)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM cost_observations").fetchone()[0], 1)

    def test_prune_normalizes_offset_and_rejects_invalid_cutoffs_before_delete(self):
        record_run(self.conn, "old", {"profile": "p"}, provider="claude", identity_key="old", started_at="2026-01-01T00:00:00+00:00")
        record_resolution(self.conn, "old", {}, {"state": "resolved"})
        record_run_observability(self.conn, "old", {"task": "review", "execution_status": "success", "created_at": "2026-01-01T00:00:00+00:00"})
        record_quota_snapshot(self.conn, {"provider": "claude", "scope_kind": "account", "resource_kind": "quota", "capability": "unavailable", "observed_at": "2026-01-01T00:00:00+00:00"})
        set_feedback(self.conn, "old", "useful")
        self.assertEqual(normalize_cutoff("2026-01-02T05:30:00+05:30"), "2026-01-02T00:00:00+00:00")
        for cutoff in ("not-a-date", "2026-01-02T00:00:00"):
            with self.assertRaises(ValueError): clear_observability(self.conn, before=cutoff)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM run_observability").fetchone()[0], 1)
        result = clear_observability(self.conn, before="2026-01-02T05:30:00+05:30")
        self.assertEqual(result["run_observability"], 1)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0], 1)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM resolution_decisions").fetchone()[0], 1)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM user_feedback").fetchone()[0], 0)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM quota_snapshots").fetchone()[0], 0)

    def test_v2_to_v3_migration_failure_rolls_back_and_successful_reopen_does_not_rescan(self):
        path = Path(self.temp.name) / "migration.sqlite3"
        conn = connect(path)
        record_run(conn, "legacy", {"profile": "p"}, provider="claude", identity_key="identity", status="completed", raw_evidence_path="/x/delegate_runs/run")
        conn.execute("DELETE FROM schema_versions")
        conn.execute("INSERT INTO schema_versions VALUES(2,'2026-01-01T00:00:00+00:00','old')")
        conn.commit(); conn.close()
        with patch.object(ledger_module, "_backfill_unambiguous_observability", side_effect=RuntimeError("injected")):
            with self.assertRaises(RuntimeError): connect(path)
        raw = __import__("sqlite3").connect(path)
        self.assertEqual(raw.execute("SELECT MAX(version) FROM schema_versions").fetchone()[0], 2)
        self.assertEqual(raw.execute("SELECT COUNT(*) FROM run_observability").fetchone()[0], 0)
        raw.close()
        migrated = connect(path)
        self.assertEqual(migrated.execute("SELECT MAX(version) FROM schema_versions").fetchone()[0], 3)
        with patch.object(ledger_module, "_backfill_unambiguous_observability", side_effect=AssertionError("must not rescan")):
            reopened = connect(path)
        self.assertEqual(reopened.execute("SELECT COUNT(*) FROM run_observability").fetchone()[0], 1)

    def test_period_uses_utc_and_rejects_ambiguous_forms(self):
        start, end, label = period_bounds("7d")
        self.assertEqual(start.tzinfo, end.tzinfo)
        self.assertEqual(label, "7d")
        with self.assertRaises(ValueError): period_bounds("week")

    def test_insights_are_local_deterministic_and_evidence_aware(self):
        self.add_event()
        first = build_insights(self.conn, period="all")
        second = build_insights(self.conn, period="all")
        self.assertEqual(first["summary"], second["summary"])
        self.assertTrue(any(item["kind"] == "descriptive" for item in first["insights"]))

    def test_feedback_replaces_or_deletes_without_mutating_routing_state(self):
        self.add_event()
        set_feedback(self.conn, "r1", "useful")
        set_feedback(self.conn, "r1", "mixed")
        self.assertEqual(self.conn.execute("SELECT outcome FROM user_feedback WHERE run_id='r1'").fetchone()[0], "mixed")
        delete_feedback(self.conn, "r1")
        self.assertIsNone(self.conn.execute("SELECT outcome FROM user_feedback WHERE run_id='r1'").fetchone())
        with self.assertRaises(ValueError): set_feedback(self.conn, "missing", "useful")

    def test_feedback_rate_requires_three_ratings(self):
        self.add_event("f0")
        self.assertIsNone(build_usage(self.conn, period="all")["summary"]["feedback"]["useful_rate"])
        set_feedback(self.conn, "f0", "useful")
        self.assertIsNone(build_usage(self.conn, period="all")["summary"]["feedback"]["useful_rate"])
        self.add_event("f1"); set_feedback(self.conn, "f1", "mixed")
        self.assertIsNone(build_usage(self.conn, period="all")["summary"]["feedback"]["useful_rate"])
        self.add_event("f2"); set_feedback(self.conn, "f2", "useful")
        self.assertEqual(build_usage(self.conn, period="all")["summary"]["feedback"]["useful_rate"], 2 / 3)

    def test_hosted_quota_states_are_honest_and_scoped(self):
        snapshots = hosted_quota_statuses()
        self.assertEqual({row["provider"] for row in snapshots}, {"codex", "claude", "gemini", "deepseek", "minimax"})
        self.assertTrue(all(row["scope_kind"] == "account" for row in snapshots))
        self.assertTrue(all(row["percentage"] is None for row in snapshots))


if __name__ == "__main__":
    unittest.main()
