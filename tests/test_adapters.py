import unittest
from pathlib import Path

from benchmark.adapters import (
    ADAPTERS,
    AntigravityAdapter,
    ClaudeAdapter,
    CommandAgentAdapter,
    CodexAdapter,
    DeepSeekAdapter,
    MiniMaxAdapter,
    CLAUDE_TASK_ALLOWED_TOOLS,
    configured_adapters,
)
from benchmark.preflight import _adapter_command_problems


PROMPT = 'First line with spaces\nThen "quoted" text and $literal.'
WORKSPACE = Path("/tmp/benchmark workspace")
RESULT = Path("/tmp/benchmark result")


class AdapterArgvTests(unittest.TestCase):
    def test_command_agent_uses_argv_cwd_and_final_prompt_contract(self):
        adapter = CommandAgentAdapter(
            name="local-coding",
            command_argv=("coding-agent",),
            fixed_args=("--non-interactive",),
        )
        command = adapter.command(WORKSPACE, PROMPT, RESULT, task_id="research_python")
        self.assertEqual(command, ["coding-agent", "--non-interactive", PROMPT])
        self.assertEqual(adapter.executable, "coding-agent")
        self.assertEqual(adapter.describe()["adapter"], "command-agent")
        self.assertEqual(adapter.describe()["prompt_delivery"], "final argv item")

    def test_codex_uses_directly_supported_exec_flags(self):
        command = CodexAdapter(model="chosen-codex", reasoning_effort="high").command(WORKSPACE, PROMPT, RESULT)
        self.assertEqual(command[-1], PROMPT)
        self.assertNotIn("--approve-for-me", command)
        self.assertNotIn("--ask-for-approval", command)
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", command)
        self.assertNotIn("danger-full-access", command)
        self.assertEqual(command[command.index("--sandbox") + 1], "workspace-write")
        self.assertEqual(command[command.index("--model") + 1], "chosen-codex")
        self.assertEqual(command[command.index("--config") + 1], 'model_reasoning_effort="high"')
        self.assertEqual(command[command.index("--cd") + 1], str(WORKSPACE))
        self.assertIn("--json", command)
        self.assertEqual(command[command.index("--output-last-message") + 1], str(RESULT / "last_message.txt"))
        self.assertEqual(_adapter_command_problems(CodexAdapter(model="chosen-codex", reasoning_effort="high")), [])

    def _assert_claude_task_allowlist(self, task_id, expected):
        command = ClaudeAdapter(model="chosen-claude", reasoning_effort="medium").command(
            WORKSPACE, PROMPT, RESULT, task_id=task_id,
        )
        self.assertEqual(command[-1], PROMPT)
        self.assertEqual(command[-2], "-p")
        self.assertEqual(command[command.index("--permission-mode") + 1], "auto")
        self.assertNotIn("bypassPermissions", command)
        self.assertNotIn("--dangerously-skip-permissions", command)
        self.assertNotIn("Bash", command)
        self.assertEqual(command[command.index("--model") + 1], "chosen-claude")
        self.assertEqual(command[command.index("--effort") + 1], "medium")
        self.assertEqual(command[command.index("--allowedTools") + 1], ",".join(expected))
        self.assertIn("Read", expected)
        self.assertIn("Write", expected)
        self.assertIn("Edit", expected)
        self.assertIn("Bash(python *)", expected)
        self.assertIn("Bash(python3 *)", expected)
        self.assertNotIn("Bash(*)", expected)
        self.assertNotIn("Bash(git *)", expected)
        self.assertNotIn("Bash(rm *)", expected)
        self.assertEqual(
            _adapter_command_problems(ClaudeAdapter(model="chosen-claude", reasoning_effort="medium"), task_id), [],
        )

    def test_claude_research_allowlist_is_preapproved_and_prompt_is_atomic(self):
        self._assert_claude_task_allowlist("research_python", CLAUDE_TASK_ALLOWED_TOOLS["research_python"])

    def test_claude_diagnostic_plot_allowlist_is_preapproved(self):
        self._assert_claude_task_allowlist("diagnostic_plot", CLAUDE_TASK_ALLOWED_TOOLS["diagnostic_plot"])

    def test_claude_debug_allowlist_includes_pytest_only_for_debugging(self):
        expected = CLAUDE_TASK_ALLOWED_TOOLS["debug_package"]
        self._assert_claude_task_allowlist("debug_package", expected)
        self.assertIn("Bash(pytest *)", expected)

    def test_claude_patterns_cover_common_python_commands_without_unrestricted_bash(self):
        research = CLAUDE_TASK_ALLOWED_TOOLS["research_python"]
        debug = CLAUDE_TASK_ALLOWED_TOOLS["debug_package"]
        self.assertIn("Bash(python *)", research)   # python analysis.py; python -m py_compile ...
        self.assertIn("Bash(python3 *)", research)  # python3 analysis.py; python3 -m py_compile ...
        self.assertIn("Bash(python *)", debug)      # python -m pytest
        self.assertIn("Bash(python3 *)", debug)     # python3 -m pytest
        self.assertIn("Bash(pytest *)", debug)
        self.assertNotIn("Bash(*)", research + debug)

    def test_agy_places_every_option_before_print_and_keeps_prompt_atomic(self):
        command = AntigravityAdapter(model="gemini-3.7-flash-high").command(WORKSPACE, PROMPT, RESULT)
        print_index = command.index("-p")
        self.assertEqual(print_index, len(command) - 2)
        self.assertEqual(command[-1], PROMPT)
        self.assertLess(command.index("--output-format"), print_index)
        self.assertLess(command.index("--mode"), print_index)
        self.assertLess(command.index("--sandbox"), print_index)
        self.assertLess(command.index("--model"), print_index)
        self.assertEqual(command[command.index("--add-dir") + 1], str(WORKSPACE))
        self.assertNotIn("--dangerously-skip-permissions", command)
        self.assertEqual(_adapter_command_problems(AntigravityAdapter(model="gemini-3.7-flash-high")), [])

    def test_agy_reasoning_is_explicit_and_unsupported_values_fail_locally(self):
        command = AntigravityAdapter(model="gemini-3.7-flash-high", reasoning_effort="high").command(WORKSPACE, PROMPT, RESULT)
        self.assertEqual(command[command.index("--effort") + 1], "high")
        with self.assertRaisesRegex(ValueError, "unsupported AGY reasoning effort"):
            AntigravityAdapter(model="gemini-3.7-flash-high", reasoning_effort="xhigh").command(WORKSPACE, PROMPT, RESULT)

    def _assert_codex_transport_payg_shape(self, adapter, executable, model):
        command = adapter.command(WORKSPACE, PROMPT, RESULT)
        self.assertEqual(command[0], executable)
        self.assertEqual(command[-1], PROMPT)
        self.assertNotIn("--approve-for-me", command)
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", command)
        self.assertNotIn("danger-full-access", command)
        self.assertEqual(command[command.index("--sandbox") + 1], "workspace-write")
        self.assertEqual(command[command.index("--model") + 1], model)
        self.assertEqual(command[command.index("--config") + 1], 'model_reasoning_effort="high"')
        self.assertEqual(command[command.index("--cd") + 1], str(WORKSPACE))
        self.assertIn("--json", command)
        self.assertEqual(command[command.index("--output-last-message") + 1], str(RESULT / "last_message.txt"))
        self.assertEqual(_adapter_command_problems(adapter), [])

    def test_deepseek_pro_uses_the_codex_deepseek_launcher_pinned_to_high_effort(self):
        self._assert_codex_transport_payg_shape(
            DeepSeekAdapter(name="deepseek-pro", model="deepseek-v4-pro"), "codex-deepseek", "deepseek-v4-pro",
        )

    def test_deepseek_flash_uses_the_codex_deepseek_launcher_pinned_to_high_effort(self):
        self._assert_codex_transport_payg_shape(
            DeepSeekAdapter(name="deepseek-flash", model="deepseek-flash"), "codex-deepseek", "deepseek-flash",
        )

    def test_minimax_m3_uses_the_codex_minimax_launcher_pinned_to_high_effort(self):
        self._assert_codex_transport_payg_shape(MiniMaxAdapter(model="MiniMax-M3"), "codex-minimax", "MiniMax-M3")

    def test_deepseek_and_minimax_adapters_are_never_the_openai_codex_executable(self):
        # The self-provider-guard-equivalent distinction for the benchmark
        # harness: these must never resolve to plain "codex" (normal OpenAI
        # Codex), only to their own provider-profile launcher.
        self.assertNotEqual(DeepSeekAdapter().executable, "codex")
        self.assertNotEqual(MiniMaxAdapter().executable, "codex")

    def test_payg_adapters_describe_their_provider_transport_and_billing(self):
        deepseek = DeepSeekAdapter(name="deepseek-pro", model="deepseek-v4-pro").describe()
        self.assertEqual(deepseek["provider"], "deepseek")
        self.assertEqual(deepseek["transport"], "codex")
        self.assertEqual(deepseek["billing"], "payg")
        self.assertEqual(deepseek["maturity"], "experimental")
        minimax = MiniMaxAdapter(model="MiniMax-M3").describe()
        self.assertEqual(minimax["provider"], "minimax")
        self.assertEqual(minimax["transport"], "codex")
        self.assertEqual(minimax["billing"], "payg")


class RegistrationTests(unittest.TestCase):
    def test_global_adapters_include_the_three_payg_agent_keys(self):
        for key in ("deepseek-pro", "deepseek-flash", "minimax-m3"):
            self.assertIn(key, ADAPTERS)

    def test_configured_adapters_pins_model_per_payg_agent_key(self):
        adapters = configured_adapters({
            "deepseek-pro": "deepseek-v4-pro",
            "deepseek-flash": "deepseek-v4-flash",
            "minimax-m3": "MiniMax-M3",
        })
        self.assertEqual(adapters["deepseek-pro"].model, "deepseek-v4-pro")
        self.assertEqual(adapters["deepseek-pro"].name, "deepseek-pro")
        self.assertEqual(adapters["deepseek-flash"].model, "deepseek-v4-flash")
        self.assertEqual(adapters["deepseek-flash"].name, "deepseek-flash")
        self.assertEqual(adapters["minimax-m3"].model, "MiniMax-M3")

    def test_configured_adapters_without_payg_models_leaves_them_unset(self):
        adapters = configured_adapters({"codex": "some-model"})
        self.assertIsNone(adapters["deepseek-pro"].model)
        self.assertIsNone(adapters["minimax-m3"].model)
