# language: Python 3.11, file: tests/test_api.py, target: pytest
from __future__ import annotations

import httpx
import pytest

from harborlm.api import ChatError, LlamaClient


def _oneshot_body(text: str = "token") -> dict[str, object]:
    return {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }


def test_oneshot_retries_once_on_connect_error() -> None:
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        if hits["n"] == 1:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json=_oneshot_body("ok"))

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as http:
        client = LlamaClient("http://127.0.0.1:8080/v1", client=http, retry_backoff=0)
        text, usage = client.chat("tinyllama", [{"role": "user", "content": "hi"}], stream=False)
    assert text == "ok"
    assert usage["total_tokens"] == 3
    assert hits["n"] == 2


def test_oneshot_gives_up_after_one_retry() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as http:
        client = LlamaClient("http://127.0.0.1:8080/v1", client=http, retry_backoff=0)
        with pytest.raises(ChatError, match="connect failed"):
            client.chat("tinyllama", [{"role": "user", "content": "hi"}], stream=False)


def test_non_2xx_is_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "busy"}})

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as http:
        client = LlamaClient("http://127.0.0.1:8080/v1", client=http, retry_backoff=0)
        with pytest.raises(ChatError, match="http 503"):
            client.chat("tinyllama", [{"role": "user", "content": "hi"}], stream=False)


def test_sse_skips_bad_lines() -> None:
    payload = (
        b"data: not-json\n\n"
        b"data: {broken\n\n"
        b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n'
        b": keep-alive\n\n"
        b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        b"data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=payload,
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as http:
        client = LlamaClient("http://127.0.0.1:8080/v1", client=http, retry_backoff=0)
        text, _usage = client.chat("tinyllama", [{"role": "user", "content": "hi"}], stream=True)
    assert text == "hello"


def test_timeout_must_be_positive() -> None:
    with pytest.raises(ValueError, match="timeout"):
        LlamaClient("http://127.0.0.1:8080/v1", timeout=0)
