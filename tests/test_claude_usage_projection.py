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


class ClaudeUsageProjectionTests(unittest.TestCase):
    def test_exact_top_level_usage_projection_is_allowlisted(self):
        payload = FIXTURE.read_text(encoding="utf-8")
        projection, warning = _claude_usage_projection(payload)
        self.assertIsNone(warning)
        self.assertEqual(projection, {"input_tokens": 120, "output_tokens": 34, "cache_read_tokens": 18, "cache_write_tokens": 6, "reasoning_tokens": 9})

    def test_jsonl_or_conflicting_records_are_not_guessed(self):
        payload = FIXTURE.read_text(encoding="utf-8") + "\n" + FIXTURE.read_text(encoding="utf-8")
        projection, warning = _claude_usage_projection(payload)
        self.assertIsNone(projection)
        self.assertIn("ambiguous", warning)
        projection, warning = _claude_usage_projection('{"type":"result","usage":{"input_tokens":1,"input_tokens":2}}')
        self.assertIsNone(projection)
        self.assertIn("ambiguous", warning)

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


if __name__ == "__main__":
    unittest.main()
