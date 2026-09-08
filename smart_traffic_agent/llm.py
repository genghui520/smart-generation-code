from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage


@dataclass(slots=True)
class LlmConfig:
    provider: str = "disabled"
    model: str = ""
    base_url: str = ""
    api_key_env: str = "LLM_API_KEY"
    temperature: float = 0.0
    transport: str = "sdk"
    wire_api: str = "chat_completions"
    timeout_seconds: float = 120.0
    reasoning_effort: str | None = None
    max_output_tokens: int | None = None


class LlmClient:
    """Small LangChain wrapper used by agents.

    The workflow can run without an LLM. When configured, this class provides a
    single place to call an OpenAI-compatible chat model through LangChain.
    """

    def __init__(
        self,
        chat_model: BaseChatModel | None = None,
        http_config: LlmConfig | None = None,
    ) -> None:
        self.chat_model = chat_model
        self.http_config = http_config
        self.last_usage: dict[str, Any] = {}
        self.usage_history: list[dict[str, Any]] = []

    @property
    def enabled(self) -> bool:
        return self.chat_model is not None or self.http_config is not None

    @classmethod
    def from_config(cls, config: LlmConfig | None) -> "LlmClient":
        if config is None or config.provider == "disabled":
            return cls()
        if config.provider == "tokenhub":
            api_key_env = config.api_key_env
            if api_key_env == "LLM_API_KEY":
                api_key_env = "TOKENHUB_API_KEY"
            config = LlmConfig(
                provider="openai_compatible",
                model=config.model or "glm-5.2",
                base_url=config.base_url or "https://api.tokenhub.market/v1",
                api_key_env=api_key_env or "TOKENHUB_API_KEY",
                temperature=config.temperature,
                transport=config.transport,
                timeout_seconds=config.timeout_seconds,
                reasoning_effort=config.reasoning_effort,
                max_output_tokens=config.max_output_tokens,
            )
            return cls.from_config(config)
        if config.provider in {"openai_compatible", "openai"}:
            if config.wire_api == "responses" and config.transport == "sdk":
                raise ValueError("Responses API currently requires http transport")
            if config.transport == "http":
                load_dotenv_file(Path(".env"))
                if not os.getenv(config.api_key_env):
                    raise RuntimeError(f"Missing API key environment variable: {config.api_key_env}")
                return cls(http_config=config)
            if config.transport != "sdk":
                raise ValueError(f"Unsupported LLM transport: {config.transport}")
            return cls(load_openai_compatible_model(config))
        raise ValueError(f"Unsupported LLM provider: {config.provider}")

    def invoke_text(self, system_prompt: str, user_prompt: str) -> str:
        if self.http_config is not None:
            return self._invoke_http_text(system_prompt, user_prompt)
        if self.chat_model is None:
            raise RuntimeError("LLM is not configured")
        result = self.chat_model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        content = result.content
        if isinstance(content, str):
            return content
        return str(content)

    def _invoke_http_text(self, system_prompt: str, user_prompt: str) -> str:
        """Call an OpenAI-compatible endpoint without SDK-specific headers."""
        config = self.http_config
        if config is None:
            raise RuntimeError("HTTP LLM transport is not configured")
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("HTTP LLM transport requires requests") from exc

        load_dotenv_file(Path(".env"))
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key environment variable: {config.api_key_env}")
        endpoint = "/responses" if config.wire_api == "responses" else "/chat/completions"
        url = config.base_url.rstrip("/") + endpoint
        payload: dict[str, Any] = {
            "model": config.model,
            "stream": False,
        }
        if config.wire_api == "responses":
            payload["instructions"] = system_prompt
            payload["input"] = user_prompt
            payload["store"] = False
            if config.reasoning_effort:
                payload["reasoning"] = {"effort": config.reasoning_effort}
            if config.max_output_tokens:
                payload["max_output_tokens"] = config.max_output_tokens
        else:
            payload["messages"] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            payload["temperature"] = config.temperature
            if config.reasoning_effort:
                payload["reasoning_effort"] = config.reasoning_effort
            if config.max_output_tokens:
                payload["max_tokens"] = config.max_output_tokens
        diagnostics = os.getenv("SMPAGENT_LLM_DIAGNOSTICS") == "1"
        started = time.perf_counter()
        if diagnostics:
            print(
                f"[LLM] request endpoint={endpoint} model={config.model} "
                f"system_chars={len(system_prompt)} user_chars={len(user_prompt)}",
                flush=True,
            )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        response = None
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=config.timeout_seconds,
                    allow_redirects=True,
                )
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                if attempt == max_attempts:
                    raise RuntimeError(f"HTTP LLM request failed after {attempt} attempts: {type(exc).__name__}: {exc}") from exc
                time.sleep(2 ** (attempt - 1))
                continue
            if response.status_code not in {502, 503, 504} or attempt == max_attempts:
                break
            time.sleep(2 ** (attempt - 1))
        assert response is not None
        if diagnostics:
            print(
                f"[LLM] response status={response.status_code} "
                f"elapsed_seconds={time.perf_counter() - started:.2f} "
                f"response_chars={len(response.text)}",
                flush=True,
            )
        if response.status_code >= 400:
            detail = response.text[:500].replace("\n", " ")
            raise RuntimeError(f"HTTP LLM request returned {response.status_code}: {detail}")
        try:
            data = response.json()
        except ValueError as exc:
            preview = response.text[:500].replace("\n", " ")
            content_type = response.headers.get("Content-Type", "unknown")
            raise RuntimeError(
                f"HTTP LLM returned a non-JSON response content_type={content_type} preview={preview!r}"
            ) from exc
        if config.wire_api == "responses":
            content = _response_text(data)
            usage = data.get("usage", {}) if isinstance(data, dict) else {}
            self.last_usage = _normalize_response_usage(usage)
        else:
            choices = data.get("choices") if isinstance(data, dict) else None
            if not isinstance(choices, list) or not choices:
                raise RuntimeError("HTTP LLM response did not contain choices")
            message = choices[0].get("message", {})
            content = message.get("content", "") if isinstance(message, dict) else ""
            if isinstance(content, list):
                content = "".join(
                    item.get("text", "") for item in content if isinstance(item, dict)
                )
            self.last_usage = data.get("usage", {}) if isinstance(data, dict) else {}
        if self.last_usage:
            self.usage_history.append(dict(self.last_usage))
        return content if isinstance(content, str) else str(content)

    def invoke_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        text = self.invoke_text(system_prompt, user_prompt)
        try:
            return parse_json_object_response(text)
        except ValueError:
            repair_prompt = (
                "Convert the model output below into one valid JSON object only. "
                "Do not add Markdown, comments, or explanations. If the output is empty, "
                "return a JSON object that follows the original task schema as closely as possible.\n\n"
                "Original system prompt:\n"
                f"{system_prompt}\n\n"
                "Original user prompt:\n"
                f"{user_prompt}\n\n"
                "Model output to repair:\n"
                f"{text}"
            )
            repaired = self.invoke_text(
                "You are a strict JSON repair assistant. Return JSON only.",
                repair_prompt,
            )
            return parse_json_object_response(repaired)


def _response_text(data: Any) -> str:
    if not isinstance(data, dict):
        raise RuntimeError("Responses API returned a non-object response")
    parts: list[str] = []
    for item in data.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                text = content.get("text", "")
                if isinstance(text, str):
                    parts.append(text)
    if not parts and isinstance(data.get("output_text"), str):
        return data["output_text"]
    if not parts:
        raise RuntimeError("Responses API response did not contain output text")
    return "".join(parts)


def _normalize_response_usage(usage: Any) -> dict[str, Any]:
    if not isinstance(usage, dict):
        return {}
    return {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        **usage,
    }

def strip_markdown_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_json_object_response(text: str) -> dict[str, Any]:
    cleaned = strip_markdown_fence(text.strip())
    if not cleaned:
        raise ValueError("LLM returned empty text when JSON was required.")

    decoder = json.JSONDecoder()
    candidates = [cleaned]
    candidates.extend(cleaned[index:] for index, char in enumerate(cleaned) if char == "{")
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            data, _ = decoder.raw_decode(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(data, dict):
            raise ValueError("LLM output must be a JSON object.")
        return data

    preview = cleaned[:240].replace("\n", "\\n")
    raise ValueError(f"LLM output was not valid JSON. preview={preview!r}") from last_error


def load_openai_compatible_model(config: LlmConfig) -> BaseChatModel:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "langchain-openai is required for OpenAI-compatible models. "
            "Install it with: pip install langchain-openai"
        ) from exc

    load_dotenv_file(Path(".env"))

    api_key = os.getenv(config.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {config.api_key_env}")

    kwargs: dict[str, Any] = {
        "model": config.model,
        "api_key": api_key,
        "temperature": config.temperature,
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return ChatOpenAI(**kwargs)


def load_dotenv_file(path: Path) -> None:
    """Load project-local .env values without overriding real env vars."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
