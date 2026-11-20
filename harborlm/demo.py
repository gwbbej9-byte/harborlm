# language: Python 3.11, file: harborlm/demo.py, target: Linux / Docker Desktop
"""Offline canned assistant. No sockets, no weights."""

from __future__ import annotations

DEMO_MODEL = "tinyllama"
DEMO_URL = "demo://offline"
DEMO_TOK_S = 24.0

# Screenshot + README session. Keys are stripped lowercased user lines.
CANNED: dict[str, str] = {
    "ping the quay": "harborlm demo · backend offline · canned replies only",
    "what model is this?": (
        "tinyllama stand-in. mount a GGUF with compose + fetch-tiny.sh"
    ),
    "say a token": "token. 24 tok/s on the demo clock.",
    "hello": "quay is quiet. local GGUF when compose is up; this is demo.",
    "hi": "quay is quiet. local GGUF when compose is up; this is demo.",
    "hey": "quay is quiet. local GGUF when compose is up; this is demo.",
}

SYSTEM = (
    "You are harborlm, a local GGUF chat on llama.cpp. "
    "Short answers. No cloud."
)


def demo_reply(user_text: str) -> str:
    key = " ".join(user_text.strip().split()).lower()
    if key in CANNED:
        return CANNED[key]
    if "model" in key:
        return CANNED["what model is this?"]
    if key in {"ping", "health", "/health"}:
        return CANNED["ping the quay"]
    clipped = user_text.strip()[:240]
    return f"demo echo · {clipped}"


def estimate_tokens(text: str) -> int:
    parts = text.split()
    return max(1, len(parts)) if text.strip() else 0
