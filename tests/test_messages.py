# language: Python 3.11, file: tests/test_messages.py, target: pytest
from __future__ import annotations

import json

import pytest

from harborlm.api import completions_url
from harborlm.messages import (
    append_user,
    build_chat_completions,
    extract_choice_text,
    extract_delta_text,
    extract_usage,
    make_message,
    normalize_messages,
)


def test_build_chat_completions_stream_body() -> None:
    messages = [make_message("user", "ping the quay")]
    payload = build_chat_completions("tinyllama", messages, stream=True, temperature=0.2)
    assert payload == {
        "model": "tinyllama",
        "messages": [{"role": "user", "content": "ping the quay"}],
        "stream": True,
        "temperature": 0.2,
        "max_tokens": 256,
    }


def test_build_chat_completions_oneshot_and_extra() -> None:
    payload = build_chat_completions(
        "tinyllama",
        [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}],
        stream=False,
        max_tokens=64,
        extra={"stop": ["\n"]},
    )
    assert payload["stream"] is False
    assert payload["max_tokens"] == 64
    assert payload["stop"] == ["\n"]
    assert len(payload["messages"]) == 2


def test_builder_rejects_empty_model() -> None:
    with pytest.raises(ValueError, match="model"):
        build_chat_completions("  ", [{"role": "user", "content": "x"}])


def test_normalize_and_append_user() -> None:
    msgs = normalize_messages([{"role": "System", "content": "sys"}])
    assert msgs[0]["role"] == "system"
    append_user(msgs, "  hello  ")
    assert msgs[-1] == {"role": "user", "content": "hello"}
    with pytest.raises(ValueError, match="empty"):
        append_user(msgs, "   ")


def test_extract_choice_and_delta() -> None:
    body = {
        "choices": [{"message": {"role": "assistant", "content": "token"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }
    assert extract_choice_text(body) == "token"
    assert extract_usage(body)["total_tokens"] == 4
    chunk = {"choices": [{"delta": {"content": "tok"}}]}
    assert extract_delta_text(chunk) == "tok"
    assert extract_delta_text({"choices": []}) == ""


def test_completions_url_normalization() -> None:
    assert completions_url("http://127.0.0.1:8080/v1") == (
        "http://127.0.0.1:8080/v1/chat/completions"
    )
    assert completions_url("http://127.0.0.1:8080/v1/") == (
        "http://127.0.0.1:8080/v1/chat/completions"
    )
    assert completions_url("http://127.0.0.1:8080") == (
        "http://127.0.0.1:8080/v1/chat/completions"
    )
    assert completions_url("http://x/v1/chat/completions") == "http://x/v1/chat/completions"


def test_once_json_cli() -> None:
    from harborlm.cli import main
    from harborlm.demo import CANNED

    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(["chat", "--demo", "--once", "ping the quay", "--json"])
    assert code == 0
    payload = json.loads(buf.getvalue())
    assert payload["ok"] is True
    assert payload["demo"] is True
    assert payload["reply"] == CANNED["ping the quay"]
    assert payload["usage"]["completion_tokens"] >= 1
