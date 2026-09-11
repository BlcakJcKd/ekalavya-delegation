"""Fixture-driven Claude structured-usage projection tests; no CLI invocation."""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from delegation.core import _claude_usage_projection, run_consultation


FIXTURE = Path(__file__).with_name("fixtures") / "claude_result_usage.json"
MODEL_USAGE_FIXTURE = Path(__file__).with_name("fixtures") / "claude_result_model_usage.json"
MODEL_USAGE_MULTI_FIXTURE = Path(__file__).with_name("fixtures") / "claude_result_model_usage_multiple.json"


class ClaudeUsageProjectionTests(unittest.TestCase):
    def test_exact_top_level_usage_projection_is_allowlisted(self):
        payload = FIXTURE.read_text(encoding="utf-8")
        projection, model_rows, warning = _claude_usage_projection(payload)
        self.assertIsNone(warning)
        self.assertIsNone(model_rows)
        self.assertEqual(projection, {"input_tokens": 120, "output_tokens": 34, "cache_read_tokens": 18, "cache_write_tokens": 6, "reasoning_tokens": 9})

    def test_jsonl_or_conflicting_records_are_not_guessed(self):
        payload = FIXTURE.read_text(encoding="utf-8") + "\n" + FIXTURE.read_text(encoding="utf-8")
        projection, model_rows, warning = _claude_usage_projection(payload)
        self.assertIsNone(projection)
        self.assertIsNone(model_rows)
        self.assertIn("ambiguous", warning)
        projection, model_rows, warning = _claude_usage_projection('{"type":"result","usage":{"input_tokens":1,"input_tokens":2}}')
        self.assertIsNone(projection)
        self.assertIsNone(model_rows)
        self.assertIn("ambiguous", warning)

    def test_top_level_usage_wins_over_model_usage_without_double_counting(self):
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        payload["modelUsage"] = {"claude-haiku-4-5-20251001": {"inputTokens": 9999, "outputTokens": 9999, "costUSD": 42}}
        projection, model_rows, warning = _claude_usage_projection(json.dumps(payload), fresh_one_shot=True)
        self.assertIsNone(warning)
        self.assertIsNone(model_rows)
        self.assertEqual(projection["input_tokens"], 120)
        self.assertEqual(projection["output_tokens"], 34)

    def test_fresh_one_shot_model_usage_projects_one_safe_metric_row(self):
        projection, model_rows, warning = _claude_usage_projection(MODEL_USAGE_FIXTURE.read_text(encoding="utf-8"), fresh_one_shot=True)
        self.assertIsNone(warning)
        self.assertIsNone(projection)
        self.assertEqual(model_rows, [{"model": "claude-haiku-4-5-20251001", "input_tokens": 120, "output_tokens": 34, "cache_read_tokens": 18, "cache_write_tokens": 6}])

    def test_multiple_model_usage_entries_remain_separate_safe_rows(self):
        projection, model_rows, warning = _claude_usage_projection(MODEL_USAGE_MULTI_FIXTURE.read_text(encoding="utf-8"), fresh_one_shot=True)
        self.assertIsNone(warning)
        self.assertIsNone(projection)
        self.assertEqual([row["model"] for row in model_rows], ["claude-haiku-4-5-20251001", "claude-sonnet-5"])
        self.assertEqual([row["output_tokens"] for row in model_rows], [34, 8])

    def test_model_usage_is_rejected_without_proven_fresh_session_or_when_malformed(self):
        payload = MODEL_USAGE_FIXTURE.read_text(encoding="utf-8")
        projection, model_rows, warning = _claude_usage_projection(payload, fresh_one_shot=False)
        self.assertIsNone(projection)
        self.assertIsNone(model_rows)
        self.assertIn("freshness", warning)
        malformed = json.loads(payload)
        malformed["modelUsage"]["claude-haiku-4-5-20251001"]["inputTokens"] = "120"
        projection, model_rows, warning = _claude_usage_projection(json.dumps(malformed), fresh_one_shot=True)
        self.assertIsNone(projection)
        self.assertIsNone(model_rows)
        self.assertIn("malformed", warning)

    def test_consultation_persists_only_safe_usage_projection(self):
        with TemporaryDirectory() as temp:
            root = Path(temp); workspace = root / "scope"; workspace.mkdir()
            (workspace / ".delegation-scope.json").write_text('{"mode":"read-only"}')
            completed = subprocess.CompletedProcess(["claude"], 0, stdout=FIXTURE.read_text(encoding="utf-8"), stderr="")
            with patch("delegation.core.shutil.which", return_value="/fake/claude"):
                _, record_dir = run_consultation("haiku", workspace, "task", log_root=root / "logs", run=lambda *args, **kwargs: completed)
            record = json.loads((record_dir / "execution.json").read_text())
            self.assertEqual(record["provider_reported_usage"], {"cache_read_tokens": 18, "cache_write_tokens": 6, "input_tokens": 120, "output_tokens": 34, "reasoning_tokens": 9})
            self.assertEqual(record["usage_provenance"], "harness_reported")
            self.assertNotIn("modelUsage", record)
            self.assertNotIn("result", record)
            self.assertNotIn("total_cost_usd", record)

    def test_consultation_retains_only_model_usage_token_projection(self):
        with TemporaryDirectory() as temp:
            root = Path(temp); workspace = root / "scope"; workspace.mkdir()
            (workspace / ".delegation-scope.json").write_text('{"mode":"read-only"}')
            completed = subprocess.CompletedProcess(["claude"], 0, stdout=MODEL_USAGE_FIXTURE.read_text(encoding="utf-8"), stderr="")
            with patch("delegation.core.shutil.which", return_value="/fake/claude"):
                _, record_dir = run_consultation("haiku", workspace, "task", log_root=root / "logs", run=lambda *args, **kwargs: completed)
            record = json.loads((record_dir / "execution.json").read_text())
            self.assertEqual(record["usage_provenance"], "harness_reported")
            self.assertEqual(record["provider_reported_usage_by_model"][0]["input_tokens"], 120)
            serialized = json.dumps(record, sort_keys=True)
            for forbidden in ("costUSD", "contextWindow", "maxOutputTokens", "webSearchRequests", "private result", "private prompt"):
                self.assertNotIn(forbidden, serialized)


if __name__ == "__main__":
    unittest.main()
