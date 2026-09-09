import json
import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from delegation.core import (
    DELEGATES,
    DELEGATION_CALLER_ENV,
    DELEGATION_DEPTH_ENV,
    DelegateSpec,
    build_argv,
    run_consultation,
)
from delegation.preflight import REQUIRED_HELP, check


TASK = 'Inspect "quoted" values\non two lines; do not modify anything.'


class DelegationArgvTests(unittest.TestCase):
    def test_flash_read_only_argv_is_pinned_sandboxed_and_prompt_is_atomic(self):
        command = build_argv(DELEGATES["flash"], Path("/tmp/scoped workspace"), TASK)
        self.assertEqual(command[0], "agy")
        self.assertEqual(command[command.index("--mode") + 1], "plan")
        self.assertIn("--sandbox", command)
        self.assertEqual(command[command.index("--model") + 1], "gemini-3.7-flash-medium")
        self.assertEqual(command[command.index("--effort") + 1], "medium")
        self.assertEqual(command[-2:], ["-p", command[-1]])
        self.assertIn(TASK, command[-1])
        self._assert_no_dangerous_or_recursive_surface(command)

    def test_claude_read_only_argv_has_only_read_tools_and_pinned_models(self):
        for delegate, expected_model in (("haiku", "claude-haiku-4-5-20251001"), ("sonnet", "claude-sonnet-5")):
            command = build_argv(DELEGATES[delegate], Path("/tmp/scoped workspace"), TASK)
            self.assertEqual(command[0], "claude")
            self.assertEqual(command[command.index("--permission-mode") + 1], "plan")
            self.assertEqual(command[command.index("--tools") + 1], "Read,Glob,Grep")
            self.assertEqual(command[command.index("--allowedTools") + 1], "Read,Glob,Grep")
            self.assertEqual(command[command.index("--model") + 1], expected_model)
            self.assertEqual(command[command.index("--effort") + 1], "medium")
            self.assertEqual(command[-2:], ["-p", command[-1]])
            self.assertIn(TASK, command[-1])
            self._assert_no_dangerous_or_recursive_surface(command)
            self.assertFalse(any("Write" in item or "Edit" in item or "Bash" in item for item in command))

    def test_codex_reference_argv_uses_read_only_sandbox_not_recursive_cli_delegation(self):
        spec = DelegateSpec("codex", "codex", "gpt-5.6-luna", "medium")
        command = build_argv(spec, Path("/tmp/scoped workspace"), TASK)
        self.assertEqual(command[command.index("--sandbox") + 1], "read-only")
        self.assertEqual(command[command.index("--model") + 1], "gpt-5.6-luna")
        self.assertNotIn("--approve-for-me", command)
        self._assert_no_dangerous_or_recursive_surface(command)

    def _assert_no_dangerous_or_recursive_surface(self, command):
        rendered = " ".join(command)
        self.assertNotIn("danger-full-access", rendered)
        self.assertNotIn("dangerously", rendered)
        self.assertNotIn("bypassPermissions", rendered)
        self.assertNotIn("--add-dir", command)
        self.assertIn("Do not invoke, request, or delegate to another agent", command[-1])


class DelegationExecutionTests(unittest.TestCase):
    def setUp(self):
        self._which = patch("delegation.core.shutil.which", side_effect=lambda name: f"/fake/{name}")
        self._which.start()
        self.addCleanup(self._which.stop)

    def _scope(self, root: Path) -> Path:
        workspace = root / "scope"
        workspace.mkdir()
        (workspace / ".delegation-scope.json").write_text(json.dumps({"mode": "read-only"}))
        (workspace / "input.txt").write_text("evidence")
        return workspace

    def test_execution_logs_outside_workspace_without_environment_capture(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = self._scope(root)
            log_root = root / "logs"

            def fake_run(argv, **kwargs):
                self.assertEqual(kwargs["cwd"], workspace)
                self.assertTrue(kwargs["capture_output"])
                self.assertIn("env", kwargs)
                self.assertEqual(kwargs["env"].get(DELEGATION_DEPTH_ENV), "1")
                self.assertEqual(kwargs["env"].get("PATH"), os.environ.get("PATH"))
                return subprocess.CompletedProcess(argv, 0, stdout="consultation", stderr="")

            code, record_dir = run_consultation("haiku", workspace, TASK, log_root=log_root, run=fake_run)
            self.assertEqual(code, 0)
            self.assertTrue(record_dir.is_relative_to(log_root))
            self.assertFalse((workspace / ".delegate-runs").exists())
            record = json.loads((record_dir / "execution.json").read_text())
            self.assertEqual(record["requested_model"], "claude-haiku-4-5-20251001")
            self.assertEqual(record["requested_effort"], "medium")
            self.assertFalse(record["environment_captured"])
            self.assertFalse(record["recursive_delegation_enabled"])
            self.assertEqual(record["child_delegation_depth"], 1)
            self.assertEqual(record["caller"], "unknown")
            self.assertEqual((record_dir / "stdout.txt").read_text(), "consultation")
            self.assertEqual((record_dir / "prompt.md").read_text(), build_argv(DELEGATES["haiku"], workspace, TASK)[-1])

    def test_caller_metadata_prefers_argument_then_env_then_unknown(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = self._scope(root)

            def fake_run(argv, **kwargs):
                return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

            code, record_dir = run_consultation(
                "haiku", workspace, TASK, log_root=root / "logs", run=fake_run, caller="claude-code",
            )
            record = json.loads((record_dir / "execution.json").read_text())
            self.assertEqual(record["caller"], "claude-code")

            with patch.dict(os.environ, {DELEGATION_CALLER_ENV: "codex"}, clear=False):
                code, record_dir = run_consultation(
                    "haiku", workspace, TASK, log_root=root / "logs2", run=fake_run,
                )
            record = json.loads((record_dir / "execution.json").read_text())
            self.assertEqual(record["caller"], "codex")

    def test_timeout_is_logged_as_exit_124(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = self._scope(root)

            def timeout_run(argv, **kwargs):
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"], output="partial", stderr="timeout")

            code, record_dir = run_consultation("flash", workspace, TASK, timeout_seconds=1, log_root=root / "logs", run=timeout_run)
            self.assertEqual(code, 124)
            record = json.loads((record_dir / "execution.json").read_text())
            self.assertTrue(record["timed_out"])
            self.assertEqual(record["exit_code"], 124)
            self.assertEqual((record_dir / "stdout.txt").read_text(), "partial")

    def test_scope_marker_is_required_and_implementation_is_not_an_option(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "unscoped"
            workspace.mkdir()
            with self.assertRaises(ValueError):
                run_consultation("flash", workspace, TASK, log_root=root / "logs")
            with self.assertRaises(ValueError):
                run_consultation("not-a-delegate", workspace, TASK, log_root=root / "logs")

    def test_scope_rejects_symlinks_and_logs_inside_consulted_workspace(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = self._scope(root)
            (workspace / "linked-input").symlink_to(workspace / "input.txt")
            with self.assertRaisesRegex(ValueError, "symlinks"):
                run_consultation("flash", workspace, TASK, log_root=root / "logs")
            (workspace / "linked-input").unlink()
            with self.assertRaisesRegex(ValueError, "outside"):
                run_consultation("flash", workspace, TASK, log_root=workspace / "logs")


class DelegationRecursionGuardTests(unittest.TestCase):
    def setUp(self):
        self._which = patch("delegation.core.shutil.which", side_effect=lambda name: f"/fake/{name}")
        self._which.start()
        self.addCleanup(self._which.stop)

    def _scope(self, root: Path) -> Path:
        workspace = root / "scope"
        workspace.mkdir()
        (workspace / ".delegation-scope.json").write_text(json.dumps({"mode": "read-only"}))
        return workspace

    def test_root_invocation_succeeds_when_depth_marker_absent(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = self._scope(root)
            called = []

            def fake_run(argv, **kwargs):
                called.append(argv)
                return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop(DELEGATION_DEPTH_ENV, None)
                code, _ = run_consultation("flash", workspace, TASK, log_root=root / "logs", run=fake_run)
            self.assertEqual(code, 0)
            self.assertEqual(len(called), 1)

    def test_delegated_context_invocation_is_rejected_and_no_process_is_launched(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = self._scope(root)
            called = []

            def fake_run(argv, **kwargs):
                called.append(argv)
                return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

            for depth in ("1", "2", "5"):
                with patch.dict(os.environ, {DELEGATION_DEPTH_ENV: depth}, clear=False):
                    with self.assertRaisesRegex(ValueError, "recursive delegation rejected"):
                        run_consultation("flash", workspace, TASK, log_root=root / "logs", run=fake_run)
            self.assertEqual(called, [], "no model process should be launched on a rejected call")

    def test_malformed_depth_is_rejected_safely_without_launching_a_process(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = self._scope(root)
            called = []

            def fake_run(argv, **kwargs):
                called.append(argv)
                return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

            with patch.dict(os.environ, {DELEGATION_DEPTH_ENV: "not-a-number"}, clear=False):
                with self.assertRaisesRegex(ValueError, "malformed"):
                    run_consultation("flash", workspace, TASK, log_root=root / "logs", run=fake_run)
            self.assertEqual(called, [])

    def test_zero_depth_is_treated_as_outside_a_delegated_context(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = self._scope(root)
            called = []

            def fake_run(argv, **kwargs):
                called.append(argv)
                return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

            with patch.dict(os.environ, {DELEGATION_DEPTH_ENV: "0"}, clear=False):
                code, _ = run_consultation("flash", workspace, TASK, log_root=root / "logs", run=fake_run)
            self.assertEqual(code, 0)
            self.assertEqual(len(called), 1)

    def test_spawned_delegate_always_receives_depth_one_not_an_escalated_value(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = self._scope(root)
            seen_env = {}

            def fake_run(argv, **kwargs):
                seen_env.update(kwargs["env"])
                return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop(DELEGATION_DEPTH_ENV, None)
                run_consultation("flash", workspace, TASK, log_root=root / "logs", run=fake_run)
            self.assertEqual(seen_env[DELEGATION_DEPTH_ENV], "1")


class DelegationPreflightTests(unittest.TestCase):
    def test_read_only_flag_requirements_are_specific_to_each_wrapper(self):
        self.assertIn("plan", REQUIRED_HELP["flash"])
        self.assertIn("--sandbox", REQUIRED_HELP["flash"])
        for name in ("haiku", "sonnet"):
            self.assertIn("--tools", REQUIRED_HELP[name])
            self.assertIn("--allowedTools", REQUIRED_HELP[name])
            self.assertIn("--permission-mode", REQUIRED_HELP[name])

    def test_preflight_report_has_redacted_atomic_prompts(self):
        # Stub only subprocess calls; no model is contacted.
        import delegation.preflight as preflight
        from unittest.mock import patch

        def fake_capture(command):
            if command[-1] == "--version":
                return 0, "version"
            if command[-1] == "models":
                return 0, "gemini-3.7-flash-medium\tFlash"
            required = REQUIRED_HELP["flash"]
            return 0, " ".join(required)

        with patch.object(preflight.shutil, "which", return_value="/fake/agy"), patch.object(preflight, "_capture", side_effect=fake_capture):
            report = check(["flash"])
        self.assertTrue(report["ok"])
        argv_check = next(item for item in report["checks"] if item["name"] == "flash: read-only argv")
        self.assertEqual(argv_check["argv"][-1], "<PROMPT>")
        self.assertNotIn("dangerously", " ".join(argv_check["argv"]))
