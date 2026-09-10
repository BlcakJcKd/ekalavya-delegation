"""Tests for the experimental PAYG delegates (deepseek-pro, deepseek-flash,
minimax-m3): exact model/transport pinning, read-only codex-exec argv shape,
the self-provider guard across the new provider/transport distinction, and
that no credential is ever serialized into a log. No real delegate CLI
(codex-deepseek/codex-minimax) is invoked anywhere in this file -- every
case uses a mocked ``run``, matching the pattern in test_delegation.py and
test_self_provider_guard.py for the existing delegates.
"""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from delegation.core import DELEGATES, DELEGATION_DEPTH_ENV, build_argv, run_consultation
from delegation.preflight import HELP_ARGS, REQUIRED_HELP

TASK = "PAYG delegate probe; do not modify anything"


def _scope(root: Path) -> Path:
    workspace = root / "scope"
    workspace.mkdir()
    (workspace / ".delegation-scope.json").write_text(json.dumps({"mode": "read-only"}))
    return workspace


def _stub_executables():
    # run_consultation checks shutil.which(spec.executable) before ever
    # calling the injected `run`, so these tests -- which never invoke a
    # real codex-deepseek/codex-minimax -- must not depend on those
    # launchers actually being installed on the machine running the suite.
    return patch("delegation.core.shutil.which", return_value="/usr/bin/true")


def _fake_run(calls):
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")
    return run


class DelegateSpecPinningTests(unittest.TestCase):
    def test_deepseek_pro_is_pinned_to_the_pro_slug_via_the_codex_deepseek_launcher(self):
        spec = DELEGATES["deepseek-pro"]
        self.assertEqual(spec.executable, "codex-deepseek")
        self.assertEqual(spec.model, "deepseek-v4-pro")
        self.assertEqual(spec.effort, "high")

    def test_deepseek_flash_is_pinned_to_the_flash_slug_via_the_codex_deepseek_launcher(self):
        spec = DELEGATES["deepseek-flash"]
        self.assertEqual(spec.executable, "codex-deepseek")
        self.assertEqual(spec.model, "deepseek-flash")
        self.assertEqual(spec.effort, "high")

    def test_minimax_m3_is_pinned_via_the_codex_minimax_launcher(self):
        spec = DELEGATES["minimax-m3"]
        self.assertEqual(spec.executable, "codex-minimax")
        self.assertEqual(spec.model, "MiniMax-M3")
        self.assertEqual(spec.effort, "high")

    def test_user_config_cannot_redefine_which_model_a_route_resolves_to(self):
        # Route identity is fixed in code (see delegation.config's module
        # docstring); there is no config field that maps onto DELEGATES.
        # This is a structural guard against accidentally adding one.
        from delegation.config import _ALLOWED_ENTRY_KEYS
        self.assertEqual(_ALLOWED_ENTRY_KEYS, {"enabled", "reason"})


class BuildArgvCodexTransportTests(unittest.TestCase):
    def _assert_read_only_codex_transport_shape(self, command, executable, model):
        self.assertEqual(command[0], executable)
        self.assertEqual(command[1], "exec")
        self.assertIn("--ephemeral", command)
        self.assertIn("--skip-git-repo-check", command)
        self.assertEqual(command[command.index("--sandbox") + 1], "read-only")
        self.assertEqual(command[command.index("--model") + 1], model)
        self.assertIn("--json", command)
        config_flag = command[command.index("--config") + 1]
        self.assertIn('model_reasoning_effort="high"', config_flag)
        self.assertIn(TASK, command[-1])
        rendered = " ".join(command)
        self.assertNotIn("danger-full-access", rendered)
        self.assertNotIn("dangerously", rendered)
        self.assertNotIn("--add-dir", command)
        self.assertIn("Do not invoke, request, or delegate to another agent", command[-1])

    def test_deepseek_pro_argv_is_read_only_sandboxed_and_pinned(self):
        command = build_argv(DELEGATES["deepseek-pro"], Path("/tmp/scoped workspace"), TASK)
        self._assert_read_only_codex_transport_shape(command, "codex-deepseek", "deepseek-v4-pro")

    def test_deepseek_flash_argv_is_read_only_sandboxed_and_pinned(self):
        command = build_argv(DELEGATES["deepseek-flash"], Path("/tmp/scoped workspace"), TASK)
        self._assert_read_only_codex_transport_shape(command, "codex-deepseek", "deepseek-flash")

    def test_minimax_m3_argv_is_read_only_sandboxed_and_pinned(self):
        command = build_argv(DELEGATES["minimax-m3"], Path("/tmp/scoped workspace"), TASK)
        self._assert_read_only_codex_transport_shape(command, "codex-minimax", "MiniMax-M3")

    def test_prompt_is_a_single_atomic_argv_item(self):
        command = build_argv(DELEGATES["deepseek-pro"], Path("/tmp/scoped workspace"), TASK)
        self.assertEqual(command[-1].count(TASK), 1)
        # The prompt is the final positional argument, not split/escaped.
        self.assertNotIn(TASK, command[:-1])

    def test_workspace_is_passed_via_cd_not_baked_into_the_prompt(self):
        workspace = Path("/tmp/a scoped workspace")
        command = build_argv(DELEGATES["minimax-m3"], workspace, TASK)
        self.assertEqual(command[command.index("--cd") + 1], str(workspace))


class SelfProviderGuardForNewProvidersTests(unittest.TestCase):
    def test_claude_primary_may_externally_call_deepseek_and_minimax(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = _scope(root)
            calls = []
            with _stub_executables():
                for delegate in ("deepseek-pro", "deepseek-flash", "minimax-m3"):
                    code, _ = run_consultation(
                        delegate, workspace, TASK, log_root=root / "logs",
                        run=_fake_run(calls), primary="claude-code",
                    )
                    self.assertEqual(code, 0)
            self.assertEqual(len(calls), 3)

    def test_codex_primary_may_externally_call_deepseek_and_minimax(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = _scope(root)
            calls = []
            with _stub_executables():
                for delegate in ("deepseek-pro", "deepseek-flash", "minimax-m3"):
                    code, _ = run_consultation(
                        delegate, workspace, TASK, log_root=root / "logs",
                        run=_fake_run(calls), primary="codex",
                    )
                    self.assertEqual(code, 0)
            self.assertEqual(len(calls), 3)

    def test_deepseek_primary_calling_a_deepseek_route_is_rejected_without_launching(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = _scope(root)
            for delegate in ("deepseek-pro", "deepseek-flash"):
                calls = []
                with self.assertRaisesRegex(ValueError, "same-provider external delegation disabled"):
                    run_consultation(
                        delegate, workspace, TASK, log_root=root / "logs",
                        run=_fake_run(calls), primary="deepseek",
                    )
                self.assertEqual(calls, [])

    def test_minimax_primary_calling_the_minimax_route_is_rejected_without_launching(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = _scope(root)
            calls = []
            with self.assertRaisesRegex(ValueError, "same-provider external delegation disabled"):
                run_consultation(
                    "minimax-m3", workspace, TASK, log_root=root / "logs",
                    run=_fake_run(calls), primary="minimax",
                )
            self.assertEqual(calls, [])

    def test_deepseek_primary_may_still_call_minimax_and_vice_versa(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = _scope(root)
            calls = []
            with _stub_executables():
                code, _ = run_consultation(
                    "minimax-m3", workspace, TASK, log_root=root / "logs",
                    run=_fake_run(calls), primary="deepseek",
                )
                self.assertEqual(code, 0)
                code, _ = run_consultation(
                    "deepseek-pro", workspace, TASK, log_root=root / "logs",
                    run=_fake_run(calls), primary="minimax",
                )
                self.assertEqual(code, 0)
            self.assertEqual(len(calls), 2)

    def test_a_claude_transport_deepseek_primary_alias_is_still_a_deepseek_provider(self):
        # "claude-deepseek" is DeepSeek inference fronted by the Claude CLI
        # transport; the guard must key off the provider, not the transport.
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = _scope(root)
            calls = []
            with self.assertRaisesRegex(ValueError, "same-provider external delegation disabled"):
                run_consultation(
                    "deepseek-flash", workspace, TASK, log_root=root / "logs",
                    run=_fake_run(calls), primary="claude-deepseek",
                )
            self.assertEqual(calls, [])

    def test_recursion_guard_applies_uniformly_to_the_new_delegates(self):
        import os
        from unittest.mock import patch

        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = _scope(root)
            calls = []
            with patch.dict(os.environ, {DELEGATION_DEPTH_ENV: "1"}, clear=False):
                with self.assertRaisesRegex(ValueError, "recursive delegation rejected"):
                    run_consultation(
                        "minimax-m3", workspace, TASK, log_root=root / "logs",
                        run=_fake_run(calls), primary="claude-code",
                    )
            self.assertEqual(calls, [])


class NoCredentialSerializationTests(unittest.TestCase):
    """The credential itself is retrieved by codex-deepseek/codex-minimax
    from the keyring, entirely outside this package's process; this suite
    only proves this package never writes a plausible key-shaped value into
    its own auditable record."""

    def test_execution_record_never_mentions_the_provider_env_var_names(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = _scope(root)
            calls = []
            with _stub_executables():
                code, record_dir = run_consultation(
                    "deepseek-pro", workspace, TASK, log_root=root / "logs",
                    run=_fake_run(calls), primary="claude-code",
                )
            self.assertEqual(code, 0)
            record_text = (record_dir / "execution.json").read_text()
            for forbidden in ("DEEPSEEK_API_KEY", "MINIMAX_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
                self.assertNotIn(forbidden, record_text)
            record = json.loads(record_text)
            self.assertFalse(record["environment_captured"])

    def test_run_consultation_never_writes_a_serialized_environment_file(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = _scope(root)
            calls = []
            with _stub_executables():
                code, record_dir = run_consultation(
                    "minimax-m3", workspace, TASK, log_root=root / "logs", run=_fake_run(calls),
                )
            self.assertEqual(code, 0)
            written = {p.name for p in record_dir.iterdir()}
            self.assertEqual(written, {"prompt.md", "stdout.txt", "stderr.txt", "execution.json"})


class PreflightHelpArgsTests(unittest.TestCase):
    def test_new_delegates_check_help_under_the_exec_subcommand(self):
        for name in ("deepseek-pro", "deepseek-flash", "minimax-m3"):
            self.assertEqual(HELP_ARGS[name], ("exec", "--help"))

    def test_new_delegates_require_the_read_only_sandbox_flags(self):
        for name in ("deepseek-pro", "deepseek-flash", "minimax-m3"):
            self.assertIn("--sandbox", REQUIRED_HELP[name])
            self.assertIn("read-only", REQUIRED_HELP[name])
            self.assertIn("--json", REQUIRED_HELP[name])


if __name__ == "__main__":
    unittest.main()
