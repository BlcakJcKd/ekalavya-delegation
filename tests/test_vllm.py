"""Hermetic tests for the bounded direct OpenAI-compatible vLLM adapter."""

from __future__ import annotations

import json
import socket
import urllib.error
from unittest.mock import MagicMock
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from delegation import vllm


class RecordingTransport:
    def __init__(self, status: int = 200, body: bytes | str = b'{"choices":[{"message":{"content":"answer"}}]}'):
        self.status = status
        self.body = body.encode() if isinstance(body, str) else body
        self.calls: list[tuple[str, dict[str, str], bytes, int]] = []

    def __call__(self, url: str, headers: dict[str, str], body: bytes, timeout: int) -> tuple[int, bytes]:
        self.calls.append((url, headers, body, timeout))
        return self.status, self.body


class VLLMTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        (self.workspace / ".delegation-scope.json").write_text('{"mode":"read-only"}')
        self.config = self.root / "vllm.toml"
        self.config.write_text(
            "[providers.example]\n"
            'model = "example-model"\n'
            'base_url = "http://127.0.0.1:9000/v1"\n'
            'credential_source = "env:UNIT_TEST_CREDENTIAL"\n'
            "shared_compute = true\n"
            "max_concurrency = 1\n"
            "thinking_default = false\n"
            "max_tokens = 64\n"
        )
        self.logs = self.root / "logs"
        self.issues = self.root / "issues.jsonl"
        self.lock = self.root / "vllm.lock"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def run_call(self, transport: RecordingTransport, **kwargs):
        return vllm.run_vllm_consultation(
            "example", self.workspace, "minimal task", config_path=self.config,
            log_root=self.logs, issue_path=self.issues, lock_path=self.lock,
            transport=transport, credential_loader=lambda _: "fixture", **kwargs,
        )

    def issue(self) -> dict:
        return json.loads(self.issues.read_text().splitlines()[-1])


class VLLMConfigTests(VLLMTestBase):
    def test_missing_config_has_no_named_routes(self):
        self.assertEqual(vllm.load_vllm_config(self.root / "missing.toml"), {})

    def test_shared_provider_must_have_one_concurrent_request(self):
        self.config.write_text(self.config.read_text().replace("max_concurrency = 1", "max_concurrency = 2"))
        with self.assertRaisesRegex(ValueError, "max_concurrency = 1"):
            vllm.load_vllm_config(self.config)

    def test_credential_reference_is_not_a_secret_field(self):
        config = vllm.load_vllm_config(self.config)
        self.assertEqual(config["example"].credential_source, "env:UNIT_TEST_CREDENTIAL")
        self.assertNotIn("fixture", self.config.read_text())

    def test_invalid_url_with_embedded_credentials_is_rejected(self):
        self.config.write_text(self.config.read_text().replace("http://127.0.0.1:9000/v1", "https://user:pass@example.invalid/v1"))
        with self.assertRaisesRegex(ValueError, "credentials"):
            vllm.load_vllm_config(self.config)

    def test_malformed_config_is_a_safe_configuration_error(self):
        self.config.write_text("[providers.example\n")
        with self.assertRaisesRegex(ValueError, "could not be parsed"):
            vllm.load_vllm_config(self.config)

    def test_legacy_max_tokens_maps_to_default_and_cap_without_widening(self):
        provider = vllm.load_vllm_config(self.config)["example"]
        self.assertEqual(provider.default_max_tokens, 64)
        self.assertEqual(provider.max_tokens_cap, 64)
        self.assertEqual(provider.max_tokens, 64)

    def test_new_default_and_cap_are_loaded_and_default_request_is_used(self):
        text = self.config.read_text().replace("max_tokens = 64", "default_max_tokens = 128\nmax_tokens_cap = 256")
        self.config.write_text(text)
        provider = vllm.load_vllm_config(self.config)["example"]
        self.assertEqual(provider.default_max_tokens, 128)
        self.assertEqual(provider.max_tokens_cap, 256)
        transport = RecordingTransport()
        outcome = self.run_call(transport)
        self.assertEqual(outcome[0], 0)
        self.assertEqual(json.loads(transport.calls[0][2])["max_tokens"], 128)

    def test_default_cannot_exceed_cap(self):
        self.config.write_text(
            self.config.read_text().replace("max_tokens = 64", "default_max_tokens = 256\nmax_tokens_cap = 128")
        )
        with self.assertRaisesRegex(ValueError, "default_max_tokens must not exceed max_tokens_cap"):
            vllm.load_vllm_config(self.config)

    def test_legacy_and_new_budget_fields_cannot_be_mixed(self):
        self.config.write_text(self.config.read_text().replace("max_tokens = 64", "max_tokens = 64\nmax_tokens_cap = 128"))
        with self.assertRaisesRegex(ValueError, "either legacy max_tokens"):
            vllm.load_vllm_config(self.config)

class VLLMTransportTests(VLLMTestBase):
    def test_http_transport_maps_connection_timeout_without_exposing_details(self):
        with patch.object(vllm.urllib.request, "urlopen", side_effect=urllib.error.URLError(socket.timeout("private"))):
            with self.assertRaises(vllm.VLLMFailure) as ctx:
                vllm._http_post("https://example.invalid/v1/chat/completions", {}, b"{}", 1)
        self.assertEqual(ctx.exception.category, "request-timeout")
        self.assertTrue(ctx.exception.timed_out)
        self.assertNotIn("private", str(ctx.exception))

    def test_http_transport_maps_read_timeout(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.read.side_effect = socket.timeout("private")
        with patch.object(vllm.urllib.request, "urlopen", return_value=response):
            with self.assertRaises(vllm.VLLMFailure) as ctx:
                vllm._http_post("https://example.invalid/v1/chat/completions", {}, b"{}", 1)
        self.assertEqual(ctx.exception.category, "request-timeout")
        self.assertTrue(ctx.exception.timed_out)

    def test_success_defaults_to_non_thinking_and_returns_text(self):
        transport = RecordingTransport()
        outcome = self.run_call(transport)
        code, record_dir = outcome
        self.assertEqual(code, 0)
        self.assertEqual(outcome.text, "answer")
        self.assertEqual(len(transport.calls), 1)
        url, headers, body, timeout = transport.calls[0]
        self.assertEqual(url, "http://127.0.0.1:9000/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer fixture")
        request = json.loads(body)
        self.assertFalse(request["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(request["max_tokens"], 64)
        self.assertEqual(timeout, 300)
        self.assertFalse(self.issues.exists(), "successful runs belong in delegate_runs, not the issue log")
        evidence = json.loads((record_dir / "execution.json").read_text())
        self.assertNotIn("fixture", evidence)
        self.assertTrue(evidence["response_recorded"])
        self.assertEqual(evidence["response_file"], "stdout.txt")
        self.assertEqual((record_dir / "stdout.txt").read_text(), "answer")
        self.assertEqual((record_dir / "stdout.txt").stat().st_mode & 0o777, 0o600)

    def test_provider_reported_model_identifier_has_bounded_telemetry_identity(self):
        transport = RecordingTransport(body=json.dumps({"model": "x" * 257, "choices": [{"message": {"content": "answer"}}]}))
        outcome = self.run_call(transport)
        evidence = json.loads((outcome.record_dir / "execution.json").read_text())
        self.assertIsNone(evidence["provider_reported_model_id"])

    def test_thinking_override_is_explicit(self):
        transport = RecordingTransport()
        self.assertEqual(self.run_call(transport, thinking=True)[0], 0)
        request = json.loads(transport.calls[0][2])
        self.assertTrue(request["chat_template_kwargs"]["enable_thinking"])

    def test_request_above_local_cap_is_rejected_before_http_or_issue_log(self):
        transport = RecordingTransport()
        with self.assertRaisesRegex(
            vllm.VLLMConfigurationError,
            r"requested max_tokens=65 exceeds local route cap=64; request rejected before model inference",
        ):
            self.run_call(transport, max_tokens=65)
        self.assertEqual(transport.calls, [])
        self.assertFalse(self.logs.exists())
        self.assertFalse(self.issues.exists())

    def test_invalid_output_budget_is_local_validation_error(self):
        transport = RecordingTransport()
        with self.assertRaisesRegex(vllm.VLLMConfigurationError, "local route default=64, cap=64"):
            self.run_call(transport, max_tokens=0)
        self.assertEqual(transport.calls, [])

    def test_authorization_is_never_in_issue_or_stderr(self):
        transport = RecordingTransport(status=401, body=b"private server detail")
        code, record_dir = self.run_call(transport)
        self.assertEqual(code, 1)
        for path in (self.issues, record_dir / "stderr.txt", record_dir / "execution.json"):
            content = path.read_text()
            self.assertNotIn("fixture", content)
            self.assertNotIn("Authorization", content)
            self.assertNotIn("private server detail", content)

    def test_http_failure_categories_have_no_fallback_or_retry(self):
        cases = {
            401: "authentication-failure", 403: "authentication-failure",
            404: "api-compatibility-failure", 405: "api-compatibility-failure",
            429: "rate-limited", 500: "server-failure", 503: "server-failure",
        }
        for status, category in cases.items():
            with self.subTest(status=status):
                transport = RecordingTransport(status=status, body=b"not logged")
                code, record_dir = self.run_call(transport)
                self.assertEqual(code, 1)
                self.assertEqual(len(transport.calls), 1)
                execution = json.loads((record_dir / "execution.json").read_text())
                self.assertEqual(execution["error_category"], category)
                self.assertFalse(execution["retry"])
                self.assertIsNone(execution["fallback"])

    def test_connection_timeout_is_exit_124_and_logged(self):
        def timed_out(*args):
            raise vllm.VLLMFailure("request-timeout", "vLLM request timed out", timed_out=True)

        code, record_dir = self.run_call(timed_out)
        self.assertEqual(code, 124)
        execution = json.loads((record_dir / "execution.json").read_text())
        self.assertTrue(execution["timed_out"])
        self.assertEqual(execution["error_category"], "request-timeout")
        self.assertTrue(self.issue()["timeout"])

    def test_connection_error_is_clear_and_not_a_model_failure(self):
        def unavailable(*args):
            raise vllm.VLLMFailure("connection-error", "private-marker must not be reflected")

        code, record_dir = self.run_call(unavailable)
        self.assertEqual(code, 1)
        execution = json.loads((record_dir / "execution.json").read_text())
        self.assertEqual(execution["error_category"], "connection-error")
        self.assertFalse(execution["timed_out"])
        self.assertNotIn("private-marker", (record_dir / "stderr.txt").read_text())

    def test_malformed_empty_and_refusal_responses_are_classified(self):
        cases = [
            (b"not-json", "malformed-response"),
            (b'{"choices":[]}', "empty-response"),
            (b'{"choices":[{"message":{"refusal":"no"}}]}', "model-refusal"),
        ]
        for body, category in cases:
            with self.subTest(category=category):
                transport = RecordingTransport(body=body)
                code, record_dir = self.run_call(transport)
                self.assertEqual(code, 1)
                execution = json.loads((record_dir / "execution.json").read_text())
                self.assertEqual(execution["error_category"], category)
                self.assertEqual(execution["response_status"], "model/response-failure")

    def test_top_level_model_error_is_classified_without_recording_error_body(self):
        transport = RecordingTransport(body=b'{"error":{"message":"private model detail"}}')
        code, record_dir = self.run_call(transport)
        self.assertEqual(code, 1)
        execution = json.loads((record_dir / "execution.json").read_text())
        self.assertEqual(execution["error_category"], "model-response-failure")
        self.assertEqual(execution["response_status"], "model/response-failure")
        self.assertNotIn("private model detail", (record_dir / "stderr.txt").read_text())

    def test_response_persistence_failure_is_distinct_and_not_issue_logged(self):
        transport = RecordingTransport()
        with patch("delegation.vllm.persist_response", side_effect=OSError("disk full")):
            outcome = self.run_call(transport)
        code, record_dir = outcome
        self.assertNotEqual(code, 0)
        self.assertEqual(outcome.text, "")
        self.assertFalse(outcome.response_recorded)
        execution = json.loads((record_dir / "execution.json").read_text())
        self.assertEqual(execution["response_status"], "response-retention-failure")
        self.assertEqual(execution["error_category"], "response-retention")
        self.assertTrue(execution["provider_success"])
        self.assertTrue(execution["inference_occurred"])
        self.assertFalse(execution["response_recorded"])
        self.assertIn("response-retention failure", (record_dir / "stderr.txt").read_text())
        self.assertFalse(self.issues.exists())

    def test_manual_termination_is_an_incomplete_infrastructure_run(self):
        def interrupted(*args):
            raise KeyboardInterrupt

        outcome = self.run_call(interrupted)
        code, record_dir = outcome
        self.assertEqual(code, 130)
        self.assertEqual(outcome.text, "")
        execution = json.loads((record_dir / "execution.json").read_text())
        self.assertEqual(execution["response_status"], "incomplete-infrastructure-run")
        self.assertEqual(execution["error_category"], "manual-termination")
        self.assertEqual(self.issue()["result_state"], "incomplete-infrastructure-run")

    def test_recursion_guard_rejects_vllm_before_any_request(self):
        transport = RecordingTransport()
        with patch.dict("os.environ", {"AGENT_DELEGATION_DEPTH": "1"}):
            with self.assertRaisesRegex(ValueError, "recursive delegation rejected"):
                self.run_call(transport)
        self.assertEqual(transport.calls, [])

    def test_lock_is_non_parallel_and_released_after_scope_exit(self):
        transport = RecordingTransport()
        with vllm._request_lock(self.lock):
            code, record_dir = self.run_call(transport)
            self.assertEqual(code, 1)
            self.assertEqual(json.loads((record_dir / "execution.json").read_text())["error_category"], "concurrency-busy")
            self.assertEqual(transport.calls, [])
        code, _ = self.run_call(transport)
        self.assertEqual(code, 0)
        self.assertEqual(len(transport.calls), 1)
        self.assertTrue(self.lock.exists(), "lock file may remain; the kernel lock must be released")

    def test_issue_log_is_local_structured_and_contains_no_prompt_or_response(self):
        transport = RecordingTransport(status=503, body=b"private server detail")
        self.run_call(transport)
        record = self.issue()
        self.assertEqual(record["adapter"], "openai-compatible-vllm")
        self.assertEqual(record["route"], "example")
        self.assertIn("timestamp", record)
        self.assertIn("machine_label", record)
        self.assertNotIn("minimal task", json.dumps(record))
        self.assertNotIn("answer", json.dumps(record))


if __name__ == "__main__":
    unittest.main()
