from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from smart_traffic_agent.llm import (
    LlmClient,
    LlmConfig,
    TransientLlmError,
    load_dotenv_file,
    parse_json_object_response,
)


class LlmClientTests(unittest.TestCase):
    def test_default_api_key_environment_is_deepseek(self) -> None:
        self.assertEqual(LlmConfig().api_key_env, "DEEPSEEK_API_KEY")

    def test_disabled_config_returns_disabled_client(self) -> None:
        client = LlmClient.from_config(LlmConfig())
        self.assertFalse(client.enabled)

    def test_tokenhub_requires_api_key_env(self) -> None:
        env_name = "MISSING_TOKENHUB_API_KEY_FOR_TEST"
        os.environ.pop(env_name, None)
        with self.assertRaisesRegex(RuntimeError, env_name):
            LlmClient.from_config(LlmConfig(provider="tokenhub", api_key_env=env_name))

    @patch("requests.post")
    def test_http_timeout_is_transient_llm_error_after_bounded_retries(self, post: Mock) -> None:
        env_name = "TEST_TRANSIENT_TIMEOUT_KEY"
        os.environ[env_name] = "test-key"
        import requests

        post.side_effect = requests.exceptions.ReadTimeout("slow provider")
        client = LlmClient.from_config(
            LlmConfig(
                provider="openai_compatible",
                model="gpt-5.6-sol",
                base_url="https://gateway.example/v1",
                api_key_env=env_name,
                transport="http",
                max_attempts=2,
            )
        )

        with self.assertRaises(TransientLlmError) as raised:
            client.invoke_text("system", "user", agent_name="PlanningAgent")

        self.assertIn("transient", raised.exception.category)
        self.assertEqual(post.call_count, 2)

    @patch("requests.post")
    def test_http_5xx_is_transient_llm_error(self, post: Mock) -> None:
        env_name = "TEST_TRANSIENT_5XX_KEY"
        os.environ[env_name] = "test-key"
        response = Mock(status_code=503, text="upstream unavailable", headers={})
        post.return_value = response
        client = LlmClient.from_config(
            LlmConfig(
                provider="openai_compatible",
                model="gpt-5.6-sol",
                base_url="https://gateway.example/v1",
                api_key_env=env_name,
                transport="http",
                max_attempts=1,
            )
        )

        with self.assertRaises(TransientLlmError):
            client.invoke_text("system", "user", agent_name="PlanningAgent")

    def test_loads_project_dotenv_without_overriding_existing_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                "LOCAL_ONLY_KEY=from_dotenv\n"
                "EXISTING_KEY=from_dotenv\n",
                encoding="utf-8",
            )
            os.environ.pop("LOCAL_ONLY_KEY", None)
            os.environ["EXISTING_KEY"] = "from_environment"

            load_dotenv_file(path)

            self.assertEqual(os.environ["LOCAL_ONLY_KEY"], "from_dotenv")
            self.assertEqual(os.environ["EXISTING_KEY"], "from_environment")

    def test_parse_json_object_response_extracts_json_from_text(self) -> None:
        payload = parse_json_object_response(
            'text before\n```json\n{"next_stage":"planning"}\n```\ntext after'
        )

        self.assertEqual(payload["next_stage"], "planning")

    @patch("requests.post")
    def test_http_transport_uses_openai_compatible_chat_endpoint(self, post: Mock) -> None:
        env_name = "TEST_HTTP_LLM_API_KEY"
        os.environ[env_name] = "test-key"
        response = Mock(status_code=200, text="")
        response.json.return_value = {
            "choices": [{"message": {"content": '{"next_stage":"planning"}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }
        post.return_value = response
        client = LlmClient.from_config(
            LlmConfig(
                provider="openai_compatible",
                model="gpt-5.5",
                base_url="https://gateway.example/v1/",
                api_key_env=env_name,
                transport="http",
            )
        )

        payload = client.invoke_json("system", "user")

        self.assertEqual(payload["next_stage"], "planning")
        self.assertEqual(client.last_usage["total_tokens"], 12)
        self.assertEqual(post.call_args.args[0], "https://gateway.example/v1/chat/completions")
        self.assertNotIn("test-key", str(post.call_args.kwargs["json"]))

    @patch("requests.post")
    def test_deepseek_chat_uses_official_endpoint_model_and_thinking(self, post: Mock) -> None:
        os.environ["DEEPSEEK_API_KEY"] = "deepseek-test-key"
        response = Mock(status_code=200, text="")
        response.json.return_value = {
            "choices": [{"message": {"content": '{"ok":true}'}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }
        post.return_value = response
        client = LlmClient.from_config(LlmConfig(
            provider="openai_compatible", model="deepseek-flash",
            base_url="https://api.deepseek.com", transport="http",
            reasoning_effort="high",
        ))

        self.assertTrue(client.invoke_json("system", "user")["ok"])
        self.assertEqual(post.call_args.args[0], "https://api.deepseek.com/chat/completions")
        request = post.call_args.kwargs["json"]
        self.assertEqual(request["model"], "deepseek-flash")
        self.assertEqual(request["thinking"], {"type": "enabled"})
        self.assertEqual(request["reasoning_effort"], "high")

    @patch("requests.post")
    def test_responses_http_400_retries_with_minimal_body(self, post: Mock) -> None:
        env_name = "TEST_RESPONSES_LLM_API_KEY"
        os.environ[env_name] = "test-key"
        rejected = Mock(status_code=400, text="", headers={})
        accepted = Mock(status_code=200, text="", headers={})
        accepted.json.return_value = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": '{"next_stage":"planning"}'}],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        }
        post.side_effect = [rejected, accepted]
        client = LlmClient.from_config(
            LlmConfig(
                provider="openai_compatible",
                model="deepseek-v4-flash",
                base_url="https://api.deepseek.com",
                api_key_env=env_name,
                transport="http",
                wire_api="responses",
                reasoning_effort="low",
            )
        )

        payload = client.invoke_json("system", "user")

        self.assertEqual(payload["next_stage"], "planning")
        self.assertEqual(post.call_count, 2)
        self.assertEqual(
            post.call_args.kwargs["json"],
            {"model": "deepseek-v4-flash", "instructions": "system", "input": "user"},
        )

    @patch("requests.post")
    def test_responses_http_error_reports_safe_request_metadata(self, post: Mock) -> None:
        env_name = "TEST_RESPONSES_ERROR_API_KEY"
        os.environ[env_name] = "secret-test-key"
        response = Mock(
            status_code=400,
            text="",
            headers={"x-request-id": "request-123", "Content-Type": "application/json"},
        )
        post.side_effect = [response, response]
        client = LlmClient.from_config(
            LlmConfig(
                provider="openai_compatible",
                model="deepseek-v4-flash",
                base_url="https://api.deepseek.com",
                api_key_env=env_name,
                transport="http",
                wire_api="responses",
            )
        )

        with self.assertRaisesRegex(RuntimeError, "request_id=request-123") as raised:
            client.invoke_text("system", "user")

        self.assertNotIn("secret-test-key", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
