"""Unhappy-path validation of the delegation infrastructure itself.

This is NOT a model benchmark: every case here proves the wrapper fails
closed (rejects before launching a delegate, terminates cleanly on timeout,
surfaces a missing executable without falling back) using synthetic
workspaces and, where needed, a real but harmless subprocess (``sleep``).
No real delegate CLI (agy/claude) is invoked in this file.
"""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import delegation.core as core
from delegation.core import DELEGATES, run_consultation

TASK = "synthetic unhappy-path probe; do not modify anything"


def _snapshot(workspace: Path) -> list[str]:
    return sorted(str(p.relative_to(workspace)) for p in workspace.rglob("*"))


class CaseAInvalidScopeTests(unittest.TestCase):
    """A. malformed/invalid scope marker must reject before any subprocess."""

    def _run_spy(self):
        calls = []

        def fake_run(argv, **kwargs):
            calls.append(argv)
            return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

        return calls, fake_run

    def test_missing_scope_marker_rejects_before_subprocess(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "unscoped"
            workspace.mkdir()
            (workspace / "notes.txt").write_text("evidence")
            before = _snapshot(workspace)
            calls, fake_run = self._run_spy()

            with self.assertRaisesRegex(ValueError, "scope marker"):
                run_consultation("flash", workspace, TASK, log_root=root / "logs", run=fake_run)

            self.assertEqual(calls, [])
            self.assertEqual(_snapshot(workspace), before)

    def test_malformed_json_scope_marker_rejects_before_subprocess(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "scope"
            workspace.mkdir()
            (workspace / ".delegation-scope.json").write_text("{not valid json")
            before = _snapshot(workspace)
            calls, fake_run = self._run_spy()

            with self.assertRaisesRegex(ValueError, "invalid scope marker JSON"):
                run_consultation("flash", workspace, TASK, log_root=root / "logs", run=fake_run)

            self.assertEqual(calls, [])
            self.assertEqual(_snapshot(workspace), before)

    def test_wrong_mode_scope_marker_rejects_before_subprocess(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "scope"
            workspace.mkdir()
            (workspace / ".delegation-scope.json").write_text(json.dumps({"mode": "read-write"}))
            before = _snapshot(workspace)
            calls, fake_run = self._run_spy()

            with self.assertRaisesRegex(ValueError, r'mode.*read-only'):
                run_consultation("flash", workspace, TASK, log_root=root / "logs", run=fake_run)

            self.assertEqual(calls, [])
            self.assertEqual(_snapshot(workspace), before)

    def test_scope_marker_that_is_not_a_json_object_rejects_before_subprocess(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "scope"
            workspace.mkdir()
            (workspace / ".delegation-scope.json").write_text(json.dumps(["read-only"]))
            calls, fake_run = self._run_spy()

            with self.assertRaisesRegex(ValueError, r'mode.*read-only'):
                run_consultation("flash", workspace, TASK, log_root=root / "logs", run=fake_run)

            self.assertEqual(calls, [])


class CaseBTimeoutTests(unittest.TestCase):
    """B. a delegate that outlives its timeout must be killed, not retried."""

    def setUp(self):
        self._which = patch.object(core.shutil, "which", side_effect=lambda name: f"/fake/{name}")
        self._which.start()
        self.addCleanup(self._which.stop)

    def _scope(self, root: Path) -> Path:
        workspace = root / "scope"
        workspace.mkdir()
        (workspace / ".delegation-scope.json").write_text(json.dumps({"mode": "read-only"}))
        return workspace

    def test_real_subprocess_that_outlives_timeout_is_killed_and_classified_as_timeout(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = self._scope(root)
            before = _snapshot(workspace)
            call_count = []

            def sleepy_run(argv, **kwargs):
                call_count.append(argv)
                # Real subprocess, real OS-level timeout/kill path; ignores the
                # constructed delegate argv and runs a harmless long sleep instead.
                return subprocess.run(
                    ["sleep", "60"], cwd=kwargs["cwd"], text=True, capture_output=True,
                    timeout=kwargs["timeout"], env=kwargs.get("env"),
                )

            exit_code, record_dir = run_consultation(
                "flash", workspace, TASK, timeout_seconds=1, log_root=root / "logs", run=sleepy_run,
            )

            self.assertEqual(exit_code, 124)
            self.assertEqual(len(call_count), 1, "no automatic retry after a timeout")
            record = json.loads((record_dir / "execution.json").read_text())
            self.assertTrue(record["timed_out"])
            self.assertEqual(record["exit_code"], 124)
            self.assertEqual(_snapshot(workspace), before, "timeout must not modify the scoped workspace")

            # The killed child must not remain running (real termination, not orphaned).
            leftover = subprocess.run(
                ["pgrep", "-f", r"(^|/)sleep 60$"], text=True, capture_output=True,
            )
            self.assertEqual(leftover.returncode, 1, "no leftover 'sleep 60' process after timeout kill")

    def test_timeout_still_records_child_delegation_depth_and_is_logged_outside_workspace(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = self._scope(root)
            log_root = root / "logs"

            def timeout_run(argv, **kwargs):
                self.assertEqual(kwargs["env"].get(core.DELEGATION_DEPTH_ENV), "1")
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

            exit_code, record_dir = run_consultation(
                "flash", workspace, TASK, timeout_seconds=1, log_root=log_root, run=timeout_run,
            )
            self.assertEqual(exit_code, 124)
            self.assertTrue(record_dir.is_relative_to(log_root))
            self.assertFalse(record_dir.is_relative_to(workspace))


class CaseCMissingExecutableTests(unittest.TestCase):
    """C. an unavailable delegate executable must fail clearly, with no fallback."""

    def _scope(self, root: Path) -> Path:
        workspace = root / "scope"
        workspace.mkdir()
        (workspace / ".delegation-scope.json").write_text(json.dumps({"mode": "read-only"}))
        return workspace

    def test_missing_executable_is_surfaced_without_launching_or_falling_back(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = self._scope(root)
            before = _snapshot(workspace)
            calls = []

            def fake_run(argv, **kwargs):
                calls.append(argv)
                return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

            # Only 'flash' -> 'agy' is made unavailable; this does not touch the
            # real installed binary, it only fakes resolution for this call.
            with patch.object(core.shutil, "which", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "delegate executable is unavailable: agy"):
                    run_consultation("flash", workspace, TASK, log_root=root / "logs", run=fake_run)

            self.assertEqual(calls, [], "no subprocess should launch when the executable is missing")
            self.assertEqual(_snapshot(workspace), before)

    def test_missing_executable_does_not_substitute_a_different_delegate(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = self._scope(root)
            observed_executables = []
            real_which = core.shutil.which

            def selective_which(name):
                observed_executables.append(name)
                if name == "agy":
                    return None
                return real_which(name)

            with patch.object(core.shutil, "which", side_effect=selective_which):
                with self.assertRaises(RuntimeError):
                    run_consultation("flash", workspace, TASK, log_root=root / "logs")

            # Only the requested delegate's executable was ever probed; the
            # implementation never tries a different DELEGATES entry as a fallback.
            self.assertEqual(observed_executables, ["agy"])


if __name__ == "__main__":
    unittest.main()
