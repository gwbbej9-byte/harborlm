# language: Python 3.11, file: harborlm/session.py, target: Linux / Docker Desktop
"""Chat session: demo (offline) or live llama.cpp OpenAI API."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harborlm.api import ChatError, LlamaClient
from harborlm.demo import DEMO_MODEL, DEMO_TOK_S, DEMO_URL, SYSTEM, demo_reply, estimate_tokens
from harborlm.messages import ALLOWED_ROLES, append_assistant, append_user, make_message


@dataclass
class UsageSnap:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_per_sec: float = 0.0

    def status_bit(self) -> str:
        tok = f"{self.prompt_tokens}+{self.completion_tokens} tok"
        if self.tokens_per_sec > 0:
            return f"{tok} · {self.tokens_per_sec:.0f} tok/s"
        return tok

    def as_dict(self) -> dict[str, float | int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "tokens_per_sec": self.tokens_per_sec,
        }


def load_session_jsonl(path: Path) -> list[dict[str, str]]:
    """Load role/content lines. Skip blank and malformed JSON."""
    if not path.exists():
        return []
    if path.is_dir():
        raise ValueError(f"session path is a directory: {path}")
    out: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            role = obj.get("role")
            content = obj.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                continue
            role_n = role.strip().lower()
            if role_n not in ALLOWED_ROLES:
                continue
            out.append(make_message(role_n, content))
    return out


def append_session_jsonl(path: Path, message: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {"role": message["role"], "content": message["content"]},
        ensure_ascii=False,
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def resolve_session_path(path: str | Path) -> Path:
    p = Path(path)
    if p.exists() and p.is_dir():
        raise ValueError(f"session path is a directory: {p}")
    return p


@dataclass
class ChatSession:
    model: str
    url: str
    demo: bool = False
    stream: bool = True
    messages: list[dict[str, str]] = field(default_factory=list)
    usage: UsageSnap = field(default_factory=UsageSnap)
    last_error: str = ""
    session_path: Path | None = None
    timeout: float = 300.0
    connect_timeout: float = 5.0
    _client: LlamaClient | None = None

    @classmethod
    def live(
        cls,
        url: str,
        model: str,
        *,
        stream: bool = True,
        client: LlamaClient | None = None,
        session_path: str | Path | None = None,
        timeout: float = 300.0,
        connect_timeout: float = 5.0,
    ) -> ChatSession:
        if not url or not str(url).strip():
            raise ValueError("url is required")
        if not model or not str(model).strip():
            raise ValueError("model is required")
        sess = cls(
            model=model,
            url=url,
            demo=False,
            stream=stream,
            timeout=timeout,
            connect_timeout=connect_timeout,
            _client=client,
        )
        sess.messages = [make_message("system", SYSTEM)]
        sess._bind_session_file(session_path)
        return sess

    @classmethod
    def demo_session(
        cls,
        model: str = DEMO_MODEL,
        *,
        session_path: str | Path | None = None,
    ) -> ChatSession:
        sess = cls(model=model, url=DEMO_URL, demo=True, stream=True)
        sess.messages = [make_message("system", SYSTEM)]
        sess.usage.tokens_per_sec = DEMO_TOK_S
        sess._bind_session_file(session_path)
        return sess

    def _bind_session_file(self, session_path: str | Path | None) -> None:
        if session_path is None or str(session_path) == "":
            return
        path = resolve_session_path(session_path)
        self.session_path = path
        loaded = load_session_jsonl(path)
        if not loaded:
            return
        systems = [m for m in loaded if m["role"] == "system"]
        rest = [m for m in loaded if m["role"] != "system"]
        if systems:
            self.messages = systems + rest
        else:
            keep = [m for m in self.messages if m["role"] == "system"]
            self.messages = keep + rest

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def client(self) -> LlamaClient:
        if self.demo:
            raise ChatError("demo session has no HTTP client")
        if self._client is None:
            self._client = LlamaClient(
                self.url,
                timeout=self.timeout,
                connect_timeout=self.connect_timeout,
            )
        return self._client

    def send(self, user_text: str) -> str:
        parts: list[str] = []
        for kind, payload in self.send_iter(user_text):
            if kind == "delta":
                parts.append(payload)
        return "".join(parts)

    def send_iter(self, user_text: str) -> Iterator[tuple[str, str]]:
        """Yield ('delta', piece) then optionally ('status', ''). Updates messages."""
        text = user_text.strip()
        if not text:
            raise ValueError("empty user message")
        append_user(self.messages, text)
        self.last_error = ""
        t0 = time.perf_counter()
        if self.demo:
            reply = demo_reply(text)
            append_assistant(self.messages, reply)
            completion = estimate_tokens(reply)
            prompt = estimate_tokens(text)
            self.usage = UsageSnap(
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=prompt + completion,
                tokens_per_sec=DEMO_TOK_S,
            )
            self._persist_turn(self.messages[-2], self.messages[-1])
            yield ("delta", reply)
            yield ("status", "")
            return
        # live
        try:
            assembled: list[str] = []
            usage: dict[str, int] = {}
            for kind, payload in self.client().chat_events(
                self.model,
                self.messages,
                stream=self.stream,
            ):
                if kind == "delta":
                    assembled.append(str(payload))
                    yield ("delta", str(payload))
                elif kind == "usage":
                    usage.update(payload)  # type: ignore[arg-type]
            reply = "".join(assembled)
            append_assistant(self.messages, reply)
            elapsed = max(time.perf_counter() - t0, 1e-6)
            completion = usage.get("completion_tokens") or estimate_tokens(reply)
            prompt = usage.get("prompt_tokens") or estimate_tokens(text)
            total = usage.get("total_tokens") or (prompt + completion)
            tok_s = completion / elapsed if elapsed > 0 else 0.0
            self.usage = UsageSnap(
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=total,
                tokens_per_sec=tok_s,
            )
            self._persist_turn(self.messages[-2], self.messages[-1])
            yield ("status", "")
        except ChatError as exc:
            self.last_error = str(exc)
            # roll back the user turn so a retry can resend
            if self.messages and self.messages[-1]["role"] == "user":
                self.messages.pop()
            yield ("error", str(exc))

    def _persist_turn(self, user: dict[str, str], assistant: dict[str, str]) -> None:
        if self.session_path is None:
            return
        append_session_jsonl(self.session_path, user)
        append_session_jsonl(self.session_path, assistant)

    def visible_turns(self) -> list[dict[str, str]]:
        return [m for m in self.messages if m["role"] in {"user", "assistant"}]


def format_once(session: ChatSession, user_text: str) -> str:
    """Non-TUI one turn: two labeled lines + status."""
    lines = [f"you  {user_text.strip()}"]
    chunks: list[str] = []
    err = ""
    for kind, payload in session.send_iter(user_text):
        if kind == "delta":
            chunks.append(payload)
        elif kind == "error":
            err = payload
    if err:
        lines.append(f"err  {err}")
    else:
        lines.append(f"llm  {''.join(chunks)}")
    mode = "demo" if session.demo else session.url
    lines.append(f"{session.model} · {mode} · {session.usage.status_bit()}")
    return "\n".join(lines)


def format_once_json(session: ChatSession, user_text: str) -> str:
    """Machine JSON for --once --json. One object, one line."""
    chunks: list[str] = []
    err = ""
    for kind, payload in session.send_iter(user_text):
        if kind == "delta":
            chunks.append(payload)
        elif kind == "error":
            err = payload
    body: dict[str, Any] = {
        "ok": not err,
        "model": session.model,
        "demo": session.demo,
        "url": session.url,
        "user": user_text.strip(),
        "reply": "".join(chunks),
        "error": err,
        "usage": session.usage.as_dict(),
    }
    return json.dumps(body, ensure_ascii=False)
