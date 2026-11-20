# language: Python 3.11, file: tests/test_session.py, target: pytest
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from harborlm.cli import main
from harborlm.demo import CANNED
from harborlm.session import ChatSession, load_session_jsonl


def test_session_jsonl_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "chat.jsonl"
    sess = ChatSession.demo_session(session_path=path)
    sess.send("ping the quay")
    sess.send("hello")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4
    rows = [json.loads(line) for line in lines]
    assert rows[0] == {"role": "user", "content": "ping the quay"}
    assert rows[1]["role"] == "assistant"
    assert rows[1]["content"] == CANNED["ping the quay"]

    again = ChatSession.demo_session(session_path=path)
    turns = again.visible_turns()
    assert turns[0]["content"] == "ping the quay"
    assert turns[-1]["role"] == "assistant"


def test_session_jsonl_skips_bad_lines(tmp_path: Path) -> None:
    path = tmp_path / "chat.jsonl"
    path.write_text(
        "not json\n"
        '{"role":"user","content":"hi"}\n'
        '{"nope":1}\n'
        '{"role":"assistant","content":"quay"}\n',
        encoding="utf-8",
    )
    loaded = load_session_jsonl(path)
    assert loaded == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "quay"},
    ]


def test_session_path_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="directory"):
        ChatSession.demo_session(session_path=tmp_path)


def test_demo_session_cli_never_http(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("demo must not open sockets")

    monkeypatch.setattr(httpx.Client, "send", boom)
    monkeypatch.setattr(httpx.Client, "request", boom)
    path = tmp_path / "s.jsonl"
    code = main(["chat", "--demo", "--session", str(path), "--once", "ping the quay"])
    assert code == 0
    assert path.exists()
