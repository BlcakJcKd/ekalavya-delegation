"""No-model dry run of the Claude Code orchestration flow.

Simulates: Claude Code primary -> run_consultation("flash", ...) -> a mocked
Antigravity response -> the resulting log consumed by the primary. The
subprocess call is fully replaced by a fake; Antigravity (`agy`) is never
invoked. This exists as one explicit, self-contained scenario test alongside
the narrower unit tests in test_delegation.py.
"""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from delegation.core import DELEGATION_DEPTH_ENV, run_consultation

MOCKED_ANTIGRAVITY_RESPONSE = json.dumps({
    "result": "Likely cause: off-by-one in fixtures/debug_package/pkg/util.py:14; "
              "smallest check: `python -m unittest tests.test_util -k off_by_one`.",
})

TASK = "Read-only reconnaissance: identify the likely cause of the failing test."


class ClaudeOrchestrationDryRunTests(unittest.TestCase):
    def setUp(self):
        self._which = patch("delegation.core.shutil.which", side_effect=lambda name: f"/fake/{name}")
        self._which.start()
        self.addCleanup(self._which.stop)

    def test_full_dry_run_flow_matches_every_required_property(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "scoped-copy"
            workspace.mkdir()
            (workspace / ".delegation-scope.json").write_text(json.dumps({"mode": "read-only"}))
            (workspace / "notes.md").write_text("synthetic evidence")
            log_root = root / "delegate_runs"
            before_listing = sorted(p.relative_to(workspace) for p in workspace.rglob("*"))

            captured = {}

            def mocked_antigravity(argv, **kwargs):
                captured["argv"] = argv
                captured["kwargs"] = kwargs
                return subprocess.CompletedProcess(
                    argv, 0, stdout=MOCKED_ANTIGRAVITY_RESPONSE, stderr="",
                )

            exit_code, record_dir = run_consultation(
                "flash", workspace, TASK,
                timeout_seconds=42, log_root=log_root, caller="claude-code",
                run=mocked_antigravity,
            )

            # correct argv
            argv = captured["argv"]
            self.assertEqual(argv[0], "agy")
            self.assertEqual(argv[argv.index("--mode") + 1], "plan")
            self.assertIn("--sandbox", argv)
            self.assertEqual(argv[argv.index("--model") + 1], "gemini-3.7-flash-medium")
            self.assertEqual(argv[argv.index("--effort") + 1], "medium")
            self.assertEqual(argv[-2], "-p")
            self.assertIn(TASK, argv[-1])

            # correct cwd
            self.assertEqual(captured["kwargs"]["cwd"], workspace)

            # correct environment depth
            self.assertEqual(captured["kwargs"]["env"][DELEGATION_DEPTH_ENV], "1")

            # correct timeout
            self.assertEqual(captured["kwargs"]["timeout"], 42)

            # response captured
            self.assertEqual(exit_code, 0)
            self.assertEqual((record_dir / "stdout.txt").read_text(), MOCKED_ANTIGRAVITY_RESPONSE)

            record = json.loads((record_dir / "execution.json").read_text())
            self.assertEqual(record["requested_model"], "gemini-3.7-flash-medium")
            self.assertEqual(record["requested_effort"], "medium")
            self.assertEqual(record["timeout_seconds"], 42)
            self.assertEqual(record["caller"], "claude-code")
            self.assertFalse(record["timed_out"])

            # no write to scoped workspace
            after_listing = sorted(p.relative_to(workspace) for p in workspace.rglob("*"))
            self.assertEqual(before_listing, after_listing)

            # log generated outside workspace
            self.assertTrue(record_dir.is_relative_to(log_root))
            self.assertFalse(record_dir.is_relative_to(workspace))

            # primary consumes the log exactly as a file, no special handling
            consumed_result = json.loads((record_dir / "stdout.txt").read_text())
            self.assertIn("off-by-one", consumed_result["result"])


if __name__ == "__main__":
    unittest.main()
