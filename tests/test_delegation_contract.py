"""Regression coverage for the retained Ekalavya consultation evidence."""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from delegation.core import run_consultation


REPO_ROOT = Path(__file__).resolve().parents[1]


class AuditResponseStatusTests(unittest.TestCase):
    def setUp(self):
        self._which = patch("delegation.core.shutil.which", side_effect=lambda name: f"/fake/{name}")
        self._which.start()
        self.addCleanup(self._which.stop)

    def _scope(self, root: Path) -> Path:
        workspace = root / "scope"
        workspace.mkdir()
        (workspace / ".delegation-scope.json").write_text(json.dumps({"mode": "read-only"}))
        return workspace

    def test_audit_marks_textual_response_without_parsing_or_requiring_a_file(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = self._scope(root)

            def fake_run(argv, **kwargs):
                return subprocess.CompletedProcess(argv, 0, stdout="nothing material to add", stderr="")

            code, record = run_consultation("flash", workspace, "task", model="gemini-3.8-flash-medium", log_root=root / "logs", run=fake_run)
            metadata = json.loads((record / "execution.json").read_text())
            self.assertEqual(code, 0)
            self.assertEqual(metadata["response_status"], "text-returned")
            self.assertEqual((record / "stdout.txt").read_text(), "nothing material to add")


class CanonicalGuidanceTests(unittest.TestCase):
    def test_skill_is_ekalavya_only(self):
        skill = (REPO_ROOT / "skills/delegation/SKILL.md").read_text()
        for required in (
            "eka status --primary codex", "eka profiles", "eka models", "eka run <profile>", "eka config",
            "Primary owns routing", "same-provider-native-required", "Codex/OpenAI → native Codex agents",
            "Claude/Anthropic → native Claude subagents", "Gemini → native Gemini facilities",
            "--provider", "--family", "--model", "--reasoning", "--harness", "previous", "candidate",
            "terminal or tool yield is not a delegate timeout", "Do not blindly retry",
            "Persistent Ekalavya configuration is user-owned",
            "Missing token, cost, latency, or resource values remain null",
        ):
            self.assertIn(required, skill)

    def test_policy_contains_current_contract(self):
        policy = (REPO_ROOT / "docs/DELEGATION_POLICY.md").read_text()
        self.assertIn("Use native facilities for same-provider work", policy)
        self.assertIn("yield is not a timeout", policy)


if __name__ == "__main__":
    unittest.main()
