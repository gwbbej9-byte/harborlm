# language: Python 3.11, file: harborlm/cli.py, target: Linux / Docker Desktop
"""CLI: python -m harborlm chat|ping."""

from __future__ import annotations

import argparse
import sys

from harborlm import __version__
from harborlm.demo import DEMO_MODEL
from harborlm.ping import ping
from harborlm.session import ChatSession, format_once, format_once_json
from harborlm.tui import run_tui

DEFAULT_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = "tinyllama"


def _positive_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harborlm",
        description="Local GGUF chat — llama.cpp OpenAI-compatible TUI.",
    )
    parser.add_argument("--version", action="version", version=f"harborlm {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    chat = sub.add_parser("chat", help="TUI chat (or --once / --demo)")
    chat.add_argument("--url", default=DEFAULT_URL, help="OpenAI base, default %(default)s")
    chat.add_argument("--model", default=DEFAULT_MODEL, help="model name / alias")
    chat.add_argument("--demo", action="store_true", help="offline canned replies, no network")
    chat.add_argument("--once", metavar="TEXT", default=None, help="one turn, print, exit")
    chat.add_argument("--json", action="store_true", help="machine JSON (requires --once)")
    chat.add_argument("--no-stream", action="store_true", help="force one-shot completions")
    chat.add_argument("--session", metavar="PATH", default=None, help="jsonl transcript persist")
    chat.add_argument(
        "--timeout",
        type=_positive_float,
        default=300.0,
        help="per-call read timeout seconds",
    )
    chat.add_argument(
        "--connect-timeout",
        type=_positive_float,
        default=5.0,
        help="TCP connect timeout seconds",
    )

    p = sub.add_parser("ping", help="GET /v1/models or /health")
    p.add_argument("--url", default=DEFAULT_URL, help="server base, default %(default)s")
    p.add_argument("--timeout", type=_positive_float, default=3.0)
    p.add_argument("--json", action="store_true", help="machine JSON")
    return parser


def _force_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except (OSError, ValueError):
                pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8()
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "ping":
        result = ping(args.url, timeout=args.timeout)
        if args.json:
            sys.stdout.write(result.format_json() + "\n")
        else:
            sys.stdout.write(result.format_lines() + "\n")
        return 0 if result.ok else 1
    if args.cmd == "chat":
        if args.json and args.once is None:
            parser.error("--json requires --once")
        if args.demo:
            session = ChatSession.demo_session(
                model=args.model or DEMO_MODEL,
                session_path=args.session,
            )
        else:
            session = ChatSession.live(
                url=args.url,
                model=args.model,
                stream=not args.no_stream,
                session_path=args.session,
                timeout=args.timeout,
                connect_timeout=args.connect_timeout,
            )
        if args.once is not None:
            try:
                if args.json:
                    sys.stdout.write(format_once_json(session, args.once) + "\n")
                else:
                    sys.stdout.write(format_once(session, args.once) + "\n")
                return 0 if not session.last_error else 1
            finally:
                session.close()
        return run_tui(session)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
