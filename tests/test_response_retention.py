"""Zero-model tests for the durable external-delegate response contract."""

from __future__ import annotations

import json
import os
import subprocess
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from delegation.core import run_consultation
from delegation.retention import ResponseRetentionError, read_response


TASK = "synthetic consultation task"


def _scope(root: Path) -> Path:
    workspace = root / "scope"
    workspace.mkdir()
    (workspace / ".delegation-scope.json").write_text('{"mode":"read-only"}\n')
    return workspace


class DurableCoreResponseTests(unittest.TestCase):
    def _run(self, root: Path, stdout: str, *, stderr: str = "", **kwargs):
        workspace = _scope(root)

        def fake_run(argv, **run_kwargs):
            return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr=stderr)

        with patch("delegation.core.shutil.which", return_value="/usr/bin/fake-delegate"):
            return run_consultation(
                "flash", workspace, TASK, model="gemini-3.8-flash-medium", log_root=root / "logs", run=fake_run, **kwargs,
            )

    def test_success_is_retained_before_success_metadata_and_replayed_exactly(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            code, record_dir = self._run(root, "findings\n")
            metadata = json.loads((record_dir / "execution.json").read_text())

            self.assertEqual(code, 0)
            self.assertEqual(metadata["response_status"], "text-returned")
            self.assertTrue(metadata["response_recorded"])
            self.assertEqual(metadata["response_file"], "stdout.txt")
            self.assertEqual(metadata["response_length_bytes"], len("findings\n".encode()))
            self.assertEqual(read_response(record_dir), "findings\n")
            self.assertEqual((record_dir / "stdout.txt").read_text(), "findings\n")
            self.assertEqual(os.stat(record_dir).st_mode & 0o777, 0o700)
            self.assertEqual(os.stat(record_dir / "stdout.txt").st_mode & 0o777, 0o600)

    def test_slow_success_is_one_invocation_and_remains_recoverable(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = _scope(root)
            calls = []

            def slow_run(argv, **kwargs):
                calls.append(argv)
                time.sleep(0.05)
                return subprocess.CompletedProcess(argv, 0, stdout="slow result", stderr="")

            with patch("delegation.core.shutil.which", return_value="/usr/bin/fake-delegate"):
                code, record_dir = run_consultation(
                    "flash", workspace, TASK, model="gemini-3.8-flash-medium", log_root=root / "logs", run=slow_run,
                )

            self.assertEqual(code, 0)
            self.assertEqual(len(calls), 1)
            self.assertEqual(read_response(record_dir), "slow result")
            self.assertTrue(json.loads((record_dir / "execution.json").read_text())["response_recorded"])

    def test_lost_display_does_not_lose_response(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            code, record_dir = self._run(root, "display-independent result")
            self.assertEqual(code, 0)
            # Simulate a terminal/UI consumer dropping stdout entirely: the
            # existing evidence directory still provides exact recovery.
            self.assertEqual(read_response(record_dir), "display-independent result")

    def test_persistence_failure_is_not_a_provider_failure_or_success(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = _scope(root)

            def fake_run(argv, **kwargs):
                return subprocess.CompletedProcess(argv, 0, stdout="provider result", stderr="")

            with patch("delegation.core.shutil.which", return_value="/usr/bin/fake-delegate"), \
                 patch("delegation.core.persist_response", side_effect=OSError("disk full")):
                code, record_dir = run_consultation(
                    "flash", workspace, TASK, model="gemini-3.8-flash-medium", log_root=root / "logs", run=fake_run,
                )

            metadata = json.loads((record_dir / "execution.json").read_text())
            self.assertNotEqual(code, 0)
            self.assertEqual(metadata["response_status"], "response-retention-failure")
            self.assertEqual(metadata["error_category"], "response-retention")
            self.assertFalse(metadata["response_recorded"])
            self.assertTrue(metadata["provider_success"])
            self.assertTrue(metadata["inference_occurred"])
            self.assertFalse((record_dir / "stdout.txt").exists())
            self.assertIn("response-retention failure", (record_dir / "stderr.txt").read_text())

    def test_empty_response_contract_is_preserved(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            code, record_dir = self._run(root, " \n")
            metadata = json.loads((record_dir / "execution.json").read_text())
            self.assertEqual(code, 0)
            self.assertEqual(metadata["response_status"], "empty-response")
            self.assertFalse(metadata["response_recorded"])
            self.assertEqual(read_response(record_dir), " \n")

    def test_timeout_retains_partial_output_and_keeps_timeout_status(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = _scope(root)

            def timeout_run(argv, **kwargs):
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"], output="partial", stderr="quiet")

            with patch("delegation.core.shutil.which", return_value="/usr/bin/fake-delegate"):
                code, record_dir = run_consultation(
                    "flash", workspace, TASK, timeout_seconds=1,
                    model="gemini-3.8-flash-medium", log_root=root / "logs", run=timeout_run,
                )

            metadata = json.loads((record_dir / "execution.json").read_text())
            self.assertEqual(code, 124)
            self.assertTrue(metadata["timed_out"])
            self.assertEqual(metadata["response_status"], "text-returned")
            self.assertTrue(metadata["response_recorded"])
            self.assertEqual(read_response(record_dir), "partial")

    def test_text_returned_without_record_is_rejected_for_recovery(self):
        with TemporaryDirectory() as temp:
            record_dir = Path(temp) / "record"
            record_dir.mkdir(mode=0o700)
            (record_dir / "execution.json").write_text(json.dumps({
                "response_status": "text-returned",
                "response_recorded": False,
                "response_file": "stdout.txt",
            }))
            with self.assertRaisesRegex(ResponseRetentionError, "without a retained response"):
                read_response(record_dir)

    def test_log_root_inside_repository_is_rejected_before_launch(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            repository = root / "repository"
            repository.mkdir()
            (repository / ".git").mkdir()
            workspace = repository / "scope"
            workspace.mkdir()
            (workspace / ".delegation-scope.json").write_text('{"mode":"read-only"}\n')
            calls = []

            def fake_run(argv, **kwargs):
                calls.append(argv)
                return subprocess.CompletedProcess(argv, 0, stdout="result", stderr="")

            with patch("delegation.core.shutil.which", return_value="/usr/bin/fake-delegate"):
                with self.assertRaisesRegex(ValueError, "outside a repository"):
                    run_consultation(
                        "flash", workspace, TASK, model="gemini-3.8-flash-medium", log_root=repository / "runs", run=fake_run,
                    )
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
