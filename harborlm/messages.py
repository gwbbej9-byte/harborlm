# language: Python 3.11, file: harborlm/messages.py, target: Linux / Docker Desktop
"""OpenAI-compatible chat message + request JSON builders."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, MutableSequence, Sequence

Role = str
Message = dict[str, str]

ALLOWED_ROLES = frozenset({"system", "user", "assistant", "tool"})


def make_message(role: str, content: str) -> Message:
    role = role.strip().lower()
    if role not in ALLOWED_ROLES:
        raise ValueError(f"unknown role: {role!r}")
    if not isinstance(content, str):
        raise TypeError("content must be str")
    return {"role": role, "content": content}


def normalize_messages(messages: Iterable[Mapping[str, Any]]) -> list[Message]:
    out: list[Message] = []
    for item in messages:
        if not isinstance(item, Mapping):
            raise TypeError("each message must be a mapping")
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("message needs string role and content")
        out.append(make_message(role, content))
    return out


def append_user(messages: MutableSequence[Mapping[str, str]], content: str) -> Message:
    text = content.strip()
    if not text:
        raise ValueError("empty user message")
    msg = make_message("user", text)
    messages.append(msg)
    return msg


def append_assistant(messages: MutableSequence[Mapping[str, str]], content: str) -> Message:
    msg = make_message("assistant", content)
    messages.append(msg)
    return msg


def build_chat_completions(
    model: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    stream: bool = True,
    temperature: float = 0.7,
    max_tokens: int | None = 256,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """POST /v1/chat/completions body (OpenAI + llama.cpp server)."""
    if not model or not str(model).strip():
        raise ValueError("model is required")
    payload: dict[str, Any] = {
        "model": str(model).strip(),
        "messages": normalize_messages(messages),
        "stream": bool(stream),
        "temperature": float(temperature),
    }
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    if extra:
        for key, value in extra.items():
            if key in payload:
                raise ValueError(f"extra collides with field {key!r}")
            payload[key] = value
    return payload


def extract_choice_text(body: Mapping[str, Any]) -> str:
    """Pull assistant text from a non-stream chat.completions JSON body."""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("no choices in completion body")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise ValueError("choice is not an object")
    message = first.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
        if isinstance(content, str):
            return content
    text = first.get("text")
    if isinstance(text, str):
        return text
    raise ValueError("choice has no message.content")


def extract_delta_text(chunk: Mapping[str, Any]) -> str:
    """Pull a streaming delta from one SSE JSON object."""
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, Mapping):
        return ""
    delta = first.get("delta")
    if isinstance(delta, Mapping):
        content = delta.get("content")
        if isinstance(content, str):
            return content
    message = first.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
        if isinstance(content, str):
            return content
    text = first.get("text")
    return text if isinstance(text, str) else ""


def extract_usage(body: Mapping[str, Any]) -> dict[str, int]:
    usage = body.get("usage")
    if not isinstance(usage, Mapping):
        return {}
    out: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        val = usage.get(key)
        if isinstance(val, int):
            out[key] = val
        elif isinstance(val, float):
            out[key] = int(val)
    return out
