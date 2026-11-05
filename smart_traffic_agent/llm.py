from __future__ import annotations

import os
import json
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage


# Use the project-local DeepSeek credential by default. Callers may still
# override the environment-variable name for another compatible gateway.
DEFAULT_API_KEY_ENV = "DEEPSEEK_API_KEY"


class TransientLlmError(RuntimeError):
    """A retryable provider/network failure, not a planning-quality failure."""

    category = "transient_llm_error"


@dataclass(slots=True)
class LlmConfig:
    provider: str = "disabled"
    model: str = ""
    base_url: str = ""
    api_key_env: str = DEFAULT_API_KEY_ENV
    temperature: float = 0.0
    transport: str = "sdk"
    wire_api: str = "chat_completions"
    timeout_seconds: float = 120.0
    reasoning_effort: str | None = None
    max_output_tokens: int | None = None
    # Agent-level retry owns transient failures; the HTTP transport sends one
    # request per invocation so retry counts remain truthful.
    max_attempts: int = 1


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
        self.last_json_text: str = ""
        self.last_json_payload: dict[str, Any] = {}
        self.last_json_parse_error: str = ""

    @property
    def enabled(self) -> bool:
        return self.chat_model is not None or self.http_config is not None

    @classmethod
    def from_config(cls, config: LlmConfig | None) -> "LlmClient":
        if config is None or config.provider == "disabled":
            return cls()
        if config.provider == "tokenhub":
            api_key_env = config.api_key_env
            if api_key_env in {"LLM_API_KEY", "TOKENHUB_API_KEY"}:
                api_key_env = DEFAULT_API_KEY_ENV
            config = LlmConfig(
                provider="openai_compatible",
                model=config.model or "glm-5.2",
                base_url=config.base_url or "https://api.tokenhub.market/v1",
                api_key_env=api_key_env or DEFAULT_API_KEY_ENV,
                temperature=config.temperature,
                transport=config.transport,
                timeout_seconds=config.timeout_seconds,
                reasoning_effort=config.reasoning_effort,
                max_output_tokens=config.max_output_tokens,
                max_attempts=config.max_attempts,
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

    def invoke_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        agent_name: str = "unknown",
        diagnostics_context: dict[str, Any] | None = None,
    ) -> str:
        if self.http_config is not None:
            return self._invoke_http_text(system_prompt, user_prompt, agent_name=agent_name, diagnostics_context=diagnostics_context)
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

    def _invoke_http_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        agent_name: str = "unknown",
        diagnostics_context: dict[str, Any] | None = None,
    ) -> str:
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
                if "api.deepseek.com" in config.base_url.lower():
                    payload["thinking"] = {"type": "enabled"}
                payload["reasoning_effort"] = config.reasoning_effort
            if config.max_output_tokens:
                payload["max_tokens"] = config.max_output_tokens
        diagnostics = os.getenv("SMPAGENT_LLM_DIAGNOSTICS") == "1"
        input_tokens_estimate = max(1, (len(system_prompt) + len(user_prompt)) // 4)
        body_bytes = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        started = time.perf_counter()
        if diagnostics:
            print(
                f"[LLM] agent={agent_name} attempt=1 endpoint={endpoint} model={config.model} "
                f"input_tokens_est={input_tokens_estimate} body_bytes={body_bytes} "
                f"temperature={config.temperature} reasoning={config.reasoning_effort!r} "
                f"max_output_tokens={config.max_output_tokens} timeout={config.timeout_seconds} context={diagnostics_context or {}}",
                flush=True,
            )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        response = None
        max_attempts = max(1, int(config.max_attempts))
        heartbeat_stop = threading.Event()

        def heartbeat() -> None:
            while not heartbeat_stop.wait(10.0):
                elapsed = time.perf_counter() - started
                print(
                    f"[LLM] agent={agent_name} still waiting; elapsed={elapsed:.0f}s "
                    f"attempt_in_progress; timeout_window={config.timeout_seconds}s",
                    flush=True,
                )

        heartbeat_thread = threading.Thread(target=heartbeat, name="smpagent-llm-heartbeat", daemon=True)
        heartbeat_thread.start()
        try:
            for attempt in range(1, max_attempts + 1):
                if diagnostics and attempt > 1:
                    print(
                        f"[LLM] agent={agent_name} attempt={attempt}/{max_attempts} "
                        f"input_tokens_est={input_tokens_estimate} body_bytes={body_bytes} "
                        f"timeout={config.timeout_seconds}",
                        flush=True,
                    )
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
                        elapsed = time.perf_counter() - started
                        if diagnostics:
                            print(
                                f"[LLM] agent={agent_name} attempt={attempt} elapsed={elapsed:.2f}s "
                                f"timeout={config.timeout_seconds} failure_category=transient_llm_error",
                                flush=True,
                            )
                        raise TransientLlmError(
                            f"HTTP LLM request failed after {attempt} attempts: {type(exc).__name__}: {exc} "
                            f"[agent={agent_name}, input_tokens_est={input_tokens_estimate}, "
                            f"body_bytes={body_bytes}, timeout={config.timeout_seconds}]"
                        ) from exc
                    time.sleep(2 ** (attempt - 1))
                    continue
                if not 500 <= response.status_code <= 599 or attempt == max_attempts:
                    break
                time.sleep(2 ** (attempt - 1))
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1.0)
        assert response is not None
        compatibility_retry = False
        if response.status_code == 400 and config.wire_api == "responses":
            # Some OpenAI-compatible deployments advertise /responses while
            # rejecting optional Codex fields with an empty HTTP 400. Retry
            # once with the minimum documented Responses body. The prompts,
            # model, and evaluation task are unchanged.
            minimal_payload = {
                "model": config.model,
                "instructions": system_prompt,
                "input": user_prompt,
            }
            compatibility_retry = True
            print(
                "[LLM] Responses compatibility retry after HTTP 400 "
                "(optional fields removed)",
                flush=True,
            )
            response = requests.post(
                url,
                headers=headers,
                json=minimal_payload,
                timeout=config.timeout_seconds,
                allow_redirects=True,
            )
        if diagnostics:
            print(
                f"[LLM] response status={response.status_code} "
                f"elapsed_seconds={time.perf_counter() - started:.2f} "
                f"response_chars={len(response.text)}",
                flush=True,
            )
        if response.status_code >= 400:
            detail = response.text[:500].replace("\n", " ")
            request_id = (
                response.headers.get("x-request-id")
                or response.headers.get("request-id")
                or response.headers.get("cf-ray")
                or "unavailable"
            )
            content_type = response.headers.get("Content-Type", "unknown")
            error_text = (
                f"HTTP LLM request returned {response.status_code}: {detail} "
                f"[agent={agent_name}, endpoint={endpoint}, model={config.model}, "
                f"request_id={request_id}, content_type={content_type}, "
                f"input_tokens_est={input_tokens_estimate}, body_bytes={body_bytes}, "
                f"temperature={config.temperature}, reasoning={config.reasoning_effort!r}, "
                f"max_output_tokens={config.max_output_tokens}, timeout={config.timeout_seconds}, "
                f"compatibility_retry={compatibility_retry}]"
            )
            if 500 <= response.status_code <= 599:
                raise TransientLlmError(error_text)
            raise RuntimeError(error_text)
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

    def invoke_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        agent_name: str = "unknown",
        diagnostics_context: dict[str, Any] | None = None,
        repair_invalid_json: bool = True,
    ) -> dict[str, Any]:
        text = self.invoke_text(system_prompt, user_prompt, agent_name=agent_name, diagnostics_context=diagnostics_context)
        self.last_json_text = text if isinstance(text, str) else str(text)
        self.last_json_payload = {}
        self.last_json_parse_error = ""
        try:
            payload = parse_json_object_response(text)
            self.last_json_payload = dict(payload)
            return payload
        except ValueError as exc:
            self.last_json_parse_error = str(exc)
            if not repair_invalid_json:
                raise
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
                agent_name=agent_name,
                diagnostics_context=diagnostics_context,
            )
            self.last_json_text = repaired if isinstance(repaired, str) else str(repaired)
            payload = parse_json_object_response(repaired)
            self.last_json_payload = dict(payload)
            self.last_json_parse_error = ""
            return payload


@contextmanager
def temporary_llm_timeout(client: LlmClient, timeout_seconds: float):
    """Temporarily adjust a shared client's HTTP timeout for one agent."""
    config = client.http_config
    if config is None:
        yield
        return
    previous = config.timeout_seconds
    config.timeout_seconds = float(timeout_seconds)
    try:
        yield
    finally:
        config.timeout_seconds = previous


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
    load_dotenv_file(Path(".env"))

    api_key = os.getenv(config.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {config.api_key_env}")

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "langchain-openai is required for OpenAI-compatible models. "
            "Install it with: pip install langchain-openai"
        ) from exc

    kwargs: dict[str, Any] = {
        "model": config.model,
        "api_key": api_key,
        "temperature": config.temperature,
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.reasoning_effort and "api.deepseek.com" in config.base_url.lower():
        kwargs["extra_body"] = {
            "thinking": {"type": "enabled"},
            "reasoning_effort": config.reasoning_effort,
        }
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
