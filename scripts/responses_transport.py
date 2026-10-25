"""Small OpenAI Responses API transport shared by RQ1 baseline adapters."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

import requests


@dataclass(frozen=True)
class ResponsesResult:
    text: str
    usage: dict[str, int]
    response_id: str | None
    raw: dict[str, Any]


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)
    return str(value) if value is not None else ""


def messages_to_responses_input(messages: Iterable[Any]) -> tuple[str, str]:
    """Map chat-style messages to Responses instructions/input text."""
    instructions: list[str] = []
    turns: list[str] = []
    for message in messages:
        if isinstance(message, dict):
            role = message.get("role")
            content = _content_text(message.get("content", ""))
        else:
            role = getattr(message, "type", None) or getattr(message, "role", None)
            content = _content_text(getattr(message, "content", message))
        if role in {"system", "developer"}:
            instructions.append(content)
        else:
            label = "User" if role in {"human", "user"} else "Assistant"
            turns.append(f"{label}:\n{content}")
    return "\n\n".join(instructions), "\n\n".join(turns)


def _extract_output_text(data: dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    parts: list[str] = []
    for item in data.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if isinstance(content, dict) and content.get("type") == "output_text":
                if isinstance(content.get("text"), str):
                    parts.append(content["text"])
    if not parts:
        raise RuntimeError("Responses API response did not contain output text")
    return "".join(parts)


def responses_request(
    *,
    api_key: str,
    base_url: str,
    model: str,
    instructions: str,
    input_text: str,
    reasoning_effort: str = "low",
    timeout: float = 300.0,
) -> ResponsesResult:
    endpoint = base_url.rstrip("/") + "/responses"
    payload: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": input_text,
        "store": False,
        "reasoning": {"effort": reasoning_effort},
    }
    response = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json=payload,
        timeout=timeout,
        allow_redirects=True,
    )
    if response.status_code >= 400:
        detail = response.text[:1000].replace("\r", " ").replace("\n", " ")
        raise RuntimeError(
            f"Responses API request returned HTTP {response.status_code}: {detail}"
        )
    try:
        data = response.json()
    except (ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("Responses API returned non-JSON content") from exc
    if not isinstance(data, dict):
        raise RuntimeError("Responses API returned a non-object response")
    usage_raw = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    usage = {
        "input_tokens": int(usage_raw.get("input_tokens", 0) or 0),
        "output_tokens": int(usage_raw.get("output_tokens", 0) or 0),
        "total_tokens": int(usage_raw.get("total_tokens", 0) or 0),
    }
    return ResponsesResult(
        text=_extract_output_text(data),
        usage=usage,
        response_id=data.get("id") if isinstance(data.get("id"), str) else None,
        raw=data,
    )
