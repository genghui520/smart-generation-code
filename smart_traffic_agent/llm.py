from __future__ import annotations

import os
import json
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


class LlmClient:
    """Small LangChain wrapper used by agents.

    The workflow can run without an LLM. When configured, this class provides a
    single place to call an OpenAI-compatible chat model through LangChain.
    """

    def __init__(self, chat_model: BaseChatModel | None = None) -> None:
        self.chat_model = chat_model

    @property
    def enabled(self) -> bool:
        return self.chat_model is not None

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
            )
            return cls(load_openai_compatible_model(config))
        if config.provider in {"openai_compatible", "openai"}:
            return cls(load_openai_compatible_model(config))
        raise ValueError(f"Unsupported LLM provider: {config.provider}")

    def invoke_text(self, system_prompt: str, user_prompt: str) -> str:
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
