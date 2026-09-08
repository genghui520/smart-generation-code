from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from smart_traffic_agent.llm import LlmClient, LlmConfig, load_dotenv_file, parse_json_object_response


class LlmClientTests(unittest.TestCase):
    def test_disabled_config_returns_disabled_client(self) -> None:
        client = LlmClient.from_config(LlmConfig())
        self.assertFalse(client.enabled)

    def test_tokenhub_requires_api_key_env(self) -> None:
        env_name = "MISSING_TOKENHUB_API_KEY_FOR_TEST"
        os.environ.pop(env_name, None)
        with self.assertRaisesRegex(RuntimeError, env_name):
            LlmClient.from_config(LlmConfig(provider="tokenhub", api_key_env=env_name))

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


if __name__ == "__main__":
    unittest.main()
