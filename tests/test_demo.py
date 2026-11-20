# language: Python 3.11, file: tests/test_demo.py, target: pytest
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import httpx

from harborlm.demo import CANNED, DEMO_TOK_S
from harborlm.session import ChatSession, format_once

ROOT = Path(__file__).resolve().parents[1]


def test_demo_chat_turn_appends_both_roles() -> None:
    sess = ChatSession.demo_session()
    user = "ping the quay"
    reply = sess.send(user)
    assert reply == CANNED[user]
    turns = sess.visible_turns()
    assert turns[-2] == {"role": "user", "content": user}
    assert turns[-1] == {"role": "assistant", "content": reply}
    assert sess.usage.tokens_per_sec == DEMO_TOK_S
    assert sess.usage.completion_tokens >= 1
    assert sess.last_error == ""


def test_demo_three_turn_session_matches_screenshot() -> None:
    sess = ChatSession.demo_session()
    lines = [
        "ping the quay",
        "what model is this?",
        "say a token",
    ]
    replies = [sess.send(t) for t in lines]
    assert replies[0] == CANNED["ping the quay"]
    assert replies[1] == CANNED["what model is this?"]
    assert replies[2] == CANNED["say a token"]
    assert [m["role"] for m in sess.visible_turns()] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_demo_never_touches_http(monkeypatch) -> None:
    def boom(*_a, **_k):  # noqa: ANN002
        raise AssertionError("demo must not open sockets")

    monkeypatch.setattr(httpx.Client, "send", boom)
    monkeypatch.setattr(httpx.Client, "request", boom)
    sess = ChatSession.demo_session()
    assert "demo" in sess.send("hello").lower()


def test_demo_once_cli() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, "-m", "harborlm", "chat", "--demo", "--once", "ping the quay"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "you  ping the quay" in proc.stdout
    assert CANNED["ping the quay"] in proc.stdout
    assert "tinyllama · demo" in proc.stdout
    assert "24 tok/s" in proc.stdout
    assert "+" in proc.stdout.split("tinyllama · demo")[-1]


def test_format_once_status_line() -> None:
    sess = ChatSession.demo_session()
    block = format_once(sess, "say a token")
    assert "you  say a token" in block
    assert "llm  token. 24 tok/s on the demo clock." in block
    assert "tinyllama · demo" in block
    assert "24 tok/s" in block
    assert "tok ·" in block
