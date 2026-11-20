# language: Python 3.11, file: scripts/render_ui.py, target: Windows / Linux
# Pillow still: 920x560 dark TUI, three-turn demo, status tinyllama · demo · 24 tok/s
"""Render screenshots/ui.png — no live TTY required."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "screenshots" / "ui.png"
W, H = 920, 560

BG = (11, 16, 24)
PANEL = (13, 20, 32)
CHROME = (18, 26, 38)
BORDER = (30, 58, 76)
RULE = (36, 68, 88)
TITLE = (94, 234, 212)
MUTED = (100, 116, 139)
USER = (56, 189, 248)
LLM = (167, 139, 250)
TEXT = (226, 232, 240)
DIM = (148, 163, 184)
STATUS = (52, 211, 153)
INPUT = (248, 250, 252)
ERR = (248, 113, 113)

TURNS = [
    ("you", "ping the quay"),
    ("llm", "harborlm demo · backend offline · canned replies only"),
    ("you", "what model is this?"),
    ("llm", "tinyllama stand-in. mount a GGUF with compose + fetch-tiny.sh"),
    ("you", "say a token"),
    ("llm", "token. 24 tok/s on the demo clock."),
]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = []
    if bold:
        names += [
            r"C:\Windows\Fonts\cascadiamonob.ttf",
            r"C:\Windows\Fonts\CascadiaMono-Bold.ttf",
            r"C:\Windows\Fonts\consolab.ttf",
            r"C:\Windows\Fonts\lucon.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSansMono-Bold.ttf",
        ]
    names += [
        r"C:\Windows\Fonts\cascadiamono.ttf",
        r"C:\Windows\Fonts\CascadiaMono.ttf",
        r"C:\Windows\Fonts\CascadiaCode.ttf",
        r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\lucon.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    for path in names:
        p = Path(path)
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _text_w(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_w: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = word if not cur else f"{cur} {word}"
        if _text_w(draw, trial, font) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines or [""]


def render() -> Path:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    title_font = _font(22, bold=True)
    body_font = _font(16)
    role_font = _font(16, bold=True)
    status_font = _font(15, bold=True)
    key_font = _font(13)
    chrome_font = _font(14, bold=True)

    # outer terminal chrome
    draw.rounded_rectangle((16, 14, W - 16, H - 14), radius=14, fill=CHROME, outline=BORDER, width=2)
    draw.rounded_rectangle((16, 14, W - 16, 52), radius=14, fill=(15, 23, 34), outline=BORDER, width=2)
    draw.rectangle((16, 38, W - 16, 52), fill=(15, 23, 34))
    for cx, color in ((36, (239, 68, 68)), (56, (245, 158, 11)), (76, (34, 197, 94))):
        draw.ellipse((cx, 26, cx + 12, 38), fill=color)
    draw.text((100, 24), "harborlm", font=chrome_font, fill=TITLE)
    draw.text((W - 78, 24), "demo", font=key_font, fill=MUTED)

    # inner panel
    x0, y0, x1, y1 = 32, 64, W - 32, H - 28
    draw.rounded_rectangle((x0, y0, x1, y1), radius=10, fill=PANEL, outline=RULE, width=1)

    # title row inside
    draw.text((x0 + 18, y0 + 12), "harborlm", font=title_font, fill=TITLE)
    sub = "local GGUF chat  ·  llama.cpp  ·  OpenAI-compatible"
    draw.text((x0 + 18, y0 + 42), sub, font=key_font, fill=MUTED)

    hist_top = y0 + 72
    y = hist_top + 16
    label_x = x0 + 28
    text_x = x0 + 78
    max_text_w = (x1 - 28) - text_x
    # measure first so the history box hugs the three turns
    cursor = y
    for _role, content in TURNS:
        wrapped = _wrap(draw, content, body_font, max_text_w)
        cursor += 22 * max(0, len(wrapped) - 1) + 24 + 6
    hist_bot = cursor + 10
    draw.rounded_rectangle((x0 + 12, hist_top, x1 - 12, hist_bot), radius=8, fill=(9, 14, 22), outline=RULE, width=1)

    for role, content in TURNS:
        role_fill = USER if role == "you" else LLM
        lines = _wrap(draw, content, body_font, max_text_w)
        draw.text((label_x, y), role, font=role_font, fill=role_fill)
        for i, line in enumerate(lines):
            draw.text((text_x, y), line, font=body_font, fill=TEXT)
            y += 24 if i == len(lines) - 1 else 22
        y += 6

    # prompt box
    py0, py1 = hist_bot + 12, hist_bot + 54
    draw.rounded_rectangle((x0 + 12, py0, x1 - 12, py1), radius=8, fill=(9, 14, 22), outline=RULE, width=1)
    draw.text((x0 + 26, py0 + 10), ">", font=role_font, fill=TITLE)
    # caret
    caret_x = x0 + 48
    draw.rectangle((caret_x, py0 + 12, caret_x + 8, py0 + 30), fill=TITLE)

    # status
    status = "tinyllama · demo · 24 tok/s"
    draw.text((x0 + 18, py1 + 14), status, font=status_font, fill=STATUS)
    keys = "Enter send  ·  Ctrl+L clear  ·  PgUp/PgDn  ·  Ctrl+Q quit"
    draw.text((x0 + 18, py1 + 38), keys, font=key_font, fill=MUTED)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, "PNG")
    return OUT


if __name__ == "__main__":
    path = render()
    print(f"wrote {path} ({W}x{H})")
