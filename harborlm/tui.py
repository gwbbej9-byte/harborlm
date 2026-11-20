# language: Python 3.11, file: harborlm/tui.py, target: Linux / Docker Desktop
"""prompt_toolkit TUI: history pane, input line, status (model, tok/s, url)."""

from __future__ import annotations

import asyncio
import sys
import textwrap
from dataclasses import dataclass, field

from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea

from harborlm.session import ChatSession

STYLE = Style.from_dict(
    {
        "title": "#5eead4 bold",
        "muted": "#64748b",
        "user-role": "#38bdf8 bold",
        "user": "#e2e8f0",
        "llm-role": "#a78bfa bold",
        "llm": "#e5e7eb",
        "err-role": "#f87171 bold",
        "err": "#fca5a5",
        "sys": "#64748b",
        "prompt": "#5eead4 bold",
        "input": "#f8fafc",
        "status": "#94a3b8",
        "status.key": "#34d399",
        "bar": "bg:#0b1220",
        "frame": "#1e3a4c",
    }
)


@dataclass
class HistoryLine:
    role: str
    text: str


@dataclass
class HistoryBuffer:
    lines: list[HistoryLine] = field(default_factory=list)
    scroll: int = 0
    busy: bool = False

    def add(self, role: str, text: str) -> None:
        self.lines.append(HistoryLine(role, text))
        self.scroll = 0

    def append_to_last(self, piece: str) -> None:
        if not self.lines:
            self.lines.append(HistoryLine("assistant", piece))
        else:
            last = self.lines[-1]
            last.text = last.text + piece
        self.scroll = 0

    def clear(self) -> None:
        self.lines.clear()
        self.scroll = 0


def _wrap(text: str, width: int) -> list[str]:
    width = max(16, width)
    if not text:
        return [""]
    out: list[str] = []
    for para in text.split("\n"):
        if not para:
            out.append("")
            continue
        out.extend(textwrap.wrap(para, width=width) or [""])
    return out or [""]


def _role_style(role: str) -> tuple[str, str]:
    if role == "user":
        return "class:user-role", "class:user"
    if role == "error":
        return "class:err-role", "class:err"
    if role == "system":
        return "class:sys", "class:sys"
    return "class:llm-role", "class:llm"


def _role_label(role: str) -> str:
    return {"user": "you", "assistant": "llm", "error": "err", "system": "sys"}.get(role, role)[:3]


class HarborTUI:
    def __init__(self, session: ChatSession) -> None:
        self.session = session
        self.history = HistoryBuffer()
        self._input_history = InMemoryHistory()
        self.input = TextArea(
            height=1,
            prompt=lambda: [("class:prompt", " > ")],
            style="class:input",
            multiline=False,
            wrap_lines=False,
            history=self._input_history,
            accept_handler=self._on_accept,
        )
        self._title = Window(
            FormattedTextControl(self._title_fragments),
            height=1,
            style="class:bar",
        )
        self._rule_top = Window(char="─", height=1, style="class:frame")
        self._rule_mid = Window(char="─", height=1, style="class:frame")
        self._rule_bot = Window(char="─", height=1, style="class:frame")
        self._hist_control = FormattedTextControl(
            self._history_fragments,
            focusable=False,
            show_cursor=False,
        )
        self._hist_window = Window(
            self._hist_control,
            wrap_lines=False,
            always_hide_cursor=True,
            right_margins=[ScrollbarMargin(display_arrows=True)],
        )
        self._status = Window(
            FormattedTextControl(self._status_fragments),
            height=1,
            style="class:bar",
        )
        self._keys = Window(
            FormattedTextControl(self._keys_fragments),
            height=1,
            style="class:bar",
        )
        kb = self._bindings()
        layout = Layout(
            HSplit(
                [
                    self._title,
                    self._rule_top,
                    self._hist_window,
                    self._rule_mid,
                    self.input,
                    self._rule_bot,
                    self._status,
                    self._keys,
                ]
            ),
            focused_element=self.input,
        )
        self.app: Application[None] = Application(
            layout=layout,
            key_bindings=kb,
            style=STYLE,
            full_screen=True,
            mouse_support=False,
        )
        self.history.add(
            "system",
            "harborlm · Enter send · Ctrl+Q quit · demo" if session.demo else "harborlm · waiting on llama.cpp",
        )

    def _bindings(self) -> KeyBindings:
        kb = KeyBindings()
        busy = Condition(lambda: self.history.busy)

        @kb.add("c-q")
        @kb.add("c-d")
        @kb.add("c-c")
        def _(event: KeyPressEvent) -> None:
            event.app.exit()

        @kb.add("c-l")
        def _clear(event: KeyPressEvent) -> None:
            if self.history.busy:
                return
            self.history.clear()
            self.session.messages = [m for m in self.session.messages if m["role"] == "system"]
            self.history.add("system", "history cleared")

        @kb.add("pageup")
        def _up(event: KeyPressEvent) -> None:
            self.history.scroll += 4

        @kb.add("pagedown")
        def _down(event: KeyPressEvent) -> None:
            self.history.scroll = max(0, self.history.scroll - 4)

        @kb.add("escape")
        def _esc(event: KeyPressEvent) -> None:
            self.input.buffer.reset()

        @kb.add("c-p", filter=busy)
        def _noop(event: KeyPressEvent) -> None:
            pass

        return kb

    def _on_accept(self, buff: Buffer) -> bool:
        text = buff.text.strip()
        if not text or self.history.busy:
            return True
        buff.reset(append_to_history=True)
        self.app.create_background_task(self._run_turn(text))
        return True

    async def _run_turn(self, text: str) -> None:
        self.history.busy = True
        self.history.add("user", text)
        self.history.add("assistant", "")
        self.app.invalidate()
        loop = asyncio.get_running_loop()

        def consume() -> None:
            for kind, payload in self.session.send_iter(text):
                def apply(k=kind, p=payload) -> None:
                    if k == "delta":
                        self.history.append_to_last(p)
                    elif k == "error":
                        if self.history.lines and self.history.lines[-1].role == "assistant" and not self.history.lines[-1].text:
                            self.history.lines.pop()
                        self.history.add("error", p)
                    self.app.invalidate()

                loop.call_soon_threadsafe(apply)

        try:
            await loop.run_in_executor(None, consume)
        finally:
            self.history.busy = False
            self.app.invalidate()

    def _cols(self) -> int:
        try:
            return max(40, get_app().output.get_size().columns)
        except (LookupError, OSError, RuntimeError):
            return 80

    def _rows(self) -> int:
        try:
            return max(12, get_app().output.get_size().rows)
        except (LookupError, OSError, RuntimeError):
            return 24

    def _title_fragments(self) -> StyleAndTextTuples:
        cols = self._cols()
        left = " harborlm"
        right = "demo" if self.session.demo else "llama.cpp"
        mid = max(1, cols - len(left) - len(right) - 2)
        return [
            ("class:title", left),
            ("class:muted", " " * mid),
            ("class:muted", right + " "),
        ]

    def _keys_fragments(self) -> StyleAndTextTuples:
        return [
            ("class:muted", " Enter send · Ctrl+L clear · PgUp/PgDn scroll · Ctrl+Q quit")
        ]

    def _status_fragments(self) -> StyleAndTextTuples:
        sess = self.session
        mode = "demo" if sess.demo else sess.url
        left = f" {sess.model} · {mode} · {sess.usage.status_bit()}"
        if sess.last_error:
            left = f" {sess.model} · err · {sess.last_error[:48]}"
        url = "" if sess.demo else f" {sess.url}"
        cols = self._cols()
        pad = max(1, cols - len(left) - len(url))
        return [
            ("class:status.key", left),
            ("class:status", " " * pad),
            ("class:muted", url),
        ]

    def _history_fragments(self) -> StyleAndTextTuples:
        cols = self._cols()
        body_width = max(20, cols - 8)
        rendered: list[tuple[str, str, str, str]] = []
        for line in self.history.lines:
            label = f" {_role_label(line.role):<3} "
            rs, ts = _role_style(line.role)
            wrapped = _wrap(line.text, body_width)
            for i, piece in enumerate(wrapped):
                prefix = label if i == 0 else " " * len(label)
                pstyle = rs if i == 0 else "class:muted"
                rendered.append((pstyle, prefix, ts, piece))

        rows_budget = max(4, self._rows() - 7)
        if self.history.scroll > max(0, len(rendered) - rows_budget):
            self.history.scroll = max(0, len(rendered) - rows_budget)
        end = len(rendered) - self.history.scroll
        start = max(0, end - rows_budget)

        fragments: StyleAndTextTuples = []
        for pstyle, prefix, tstyle, piece in rendered[start:end]:
            fragments.append((pstyle, prefix))
            fragments.append((tstyle, piece + "\n"))
        if not fragments:
            fragments.append(("class:muted", " (empty)\n"))
        return fragments

    def run(self) -> int:
        self.app.run()
        self.session.close()
        return 0


def run_tui(session: ChatSession) -> int:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        sys.stderr.write("harborlm: no TTY. use --once TEXT or run in a terminal.\n")
        return 2
    return HarborTUI(session).run()
