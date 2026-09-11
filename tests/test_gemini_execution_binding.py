from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from delegation.core import DELEGATES, build_argv, run_consultation
from ekalavya.executor import execute
from ekalavya.readiness import binding_preflight


def _candidate(model: str = "gemini-3.8-flash-medium", generation: str = "3.8") -> dict:
    return {
        "identity_key": "flash-key",
        "provider": "gemini",
        "family": "flash",
        "provider_model_id": model,
        "generation": generation,
        "execution_route": "flash",
        "legacy_route": "flash",
        "harness": "agy",
        "transport": "agy",
        "capabilities": {"reasoning_values": ["low", "medium", "high"], "harness_values": ["agy"]},
        "runtime_variants": [
            {"provider_model_id": f"gemini-{generation}-flash-low", "reasoning": "low"},
            {"provider_model_id": f"gemini-{generation}-flash-medium", "reasoning": "medium"},
            {"provider_model_id": f"gemini-{generation}-flash-high", "reasoning": "high"},
        ],
    }


class GeminiExecutionBindingTests(unittest.TestCase):
    def setUp(self):
        self._which = patch("delegation.core.shutil.which", return_value="/fake/agy")
        self._which.start()
        self.addCleanup(self._which.stop)

    def test_argv_uses_resolved_model_and_user_effort(self):
        argv = build_argv(DELEGATES["flash"], Path("/scope"), "review", model="gemini-3.8-flash-low", effort="low")
        self.assertEqual(argv[argv.index("--model") + 1], "gemini-3.8-flash-low")
        self.assertEqual(argv[argv.index("--effort") + 1], "low")
        self.assertIsNone(DELEGATES["flash"].model)

    def test_missing_or_contradictory_route_fails_closed(self):
        candidate = _candidate()
        with patch("ekalavya.readiness.shutil.which", return_value="/fake/agy"):
            missing = binding_preflight(candidate, None, "agy", model=candidate["provider_model_id"], reasoning="medium")
            stale = binding_preflight({**candidate, "provider_model_id": "gemini-3.7-flash-medium"}, "flash", "agy", model="gemini-3.7-flash-medium", reasoning="medium")
            unsupported = binding_preflight({**candidate, "runtime_variants": [{"provider_model_id": "gemini-3.8-flash-low", "reasoning": "low"}]}, "flash", "agy", model="gemini-3.8-flash-medium", reasoning="medium")
        self.assertFalse(missing["ok"])
        self.assertFalse(stale["ok"])
        self.assertFalse(unsupported["ok"])

    def test_execute_passes_resolved_model_and_effort_to_wrapper(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            prompt = root / "prompt.md"
            prompt.write_text("review")
            candidate = _candidate(model="gemini-3.8-flash-low")
            resolution = {"resolved": {**candidate, "reasoning": "low", "execution_route": "flash", "harness": "agy"}, "resolved_reasoning": "low"}
            captured = {}

            def fake_consultation(*args, **kwargs):
                captured.update(kwargs)
                evidence = root / "evidence"
                evidence.mkdir()
                (evidence / "execution.json").write_text(json.dumps({"requested_model": kwargs["model"], "requested_effort": kwargs["effort"]}))
                return 0, evidence

            with patch("ekalavya.executor.binding_preflight", return_value={"ok": True}), patch("ekalavya.executor.run_consultation", side_effect=fake_consultation):
                result = execute(resolution, prompt, root)
            self.assertEqual(result["state"], "completed")
            self.assertEqual(captured["model"], "gemini-3.8-flash-low")
            self.assertEqual(captured["effort"], "low")

    def test_agy_usage_is_projected_without_response_or_payload_fields(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "scope"
            workspace.mkdir()
            (workspace / ".delegation-scope.json").write_text('{"mode":"read-only"}')
            result = {"status": "success", "response": "private answer", "usage": {
                "input_tokens": 11, "output_tokens": 7, "thinking_tokens": 3,
                "cache_read_tokens": 2, "total_tokens": 18,
                "prompt": "must not persist",
            }}

            def fake_run(argv, **kwargs):
                return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(result), stderr="")

            _, evidence = run_consultation(
                "flash", workspace, "review", model="gemini-3.8-flash-low", effort="low",
                log_root=root / "logs", run=fake_run,
            )
            metadata = json.loads((evidence / "execution.json").read_text())
            self.assertEqual(metadata["provider_reported_usage"], {
                "input_tokens": 11, "output_tokens": 7, "reasoning_tokens": 3,
                "cache_read_tokens": 2, "total_tokens": 18,
            })
            self.assertEqual(metadata["usage_provenance"], "harness_reported")
            self.assertNotIn("response", metadata)
            self.assertNotIn("prompt", metadata)
            self.assertNotIn("raw_payload", metadata)


if __name__ == "__main__":
    unittest.main()
