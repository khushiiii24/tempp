"""Render the 1200x630 Open Graph card for Recoup, from the exported snapshot.

Both reference briefs name the same weakness in the sites they tear down: no metadata, so
a shared link renders as a bare URL. For a submission that will be pasted into WhatsApp and
a judging form, the link preview *is* the first impression.

The card is generated rather than designed once in an image editor, for the same reason
every other figure on the site is: a headline number typed into a PNG goes stale the moment
a bug is fixed underneath it, and nobody re-opens the PNG. Run this after `export-web`.

    python web/scripts/make_og.py
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "web" / "src" / "data"
OUT = ROOT / "web" / "public" / "og.png"

W, H = 1200, 630
INK = (5, 6, 10)
PAPER = (233, 230, 222)
MUTED = (120, 126, 138)
GOLD = (232, 178, 76)
JADE = (71, 192, 138)
LEAK = (226, 83, 74)
RULE = (28, 31, 39)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Whatever the machine has. Segoe UI on Windows, DejaVu elsewhere."""
    for name in (
        ("seguisb.ttf", "segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf")
        if bold
        else ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf")
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _mono(size: int) -> ImageFont.FreeTypeFont:
    for name in ("consola.ttf", "cour.ttf", "DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def inr(paise: int) -> str:
    """Indian grouping, 2-2-3. `f"{n:,}"` gives 1,220,331 where we need 12,20,331."""
    n = abs(round(paise / 100))
    s = str(n)
    if len(s) <= 3:
        body = s
    else:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        body = ",".join(parts) + "," + tail
    return "Rs " + body


def main() -> None:
    leak = json.loads((DATA / "leak.json").read_text("utf-8"))
    board = json.loads((DATA / "scoreboard.json").read_text("utf-8"))
    agent = next(p for p in board["policies"] if p["policy"] == "agent")
    b1 = next((p for p in board["policies"] if p["policy"] == "b1"), None)

    img = Image.new("RGB", (W, H), INK)
    d = ImageDraw.Draw(img)

    # Ledger rules, the site's own texture, so the card belongs to the page it links to.
    for y in range(0, H, 32):
        d.line([(0, y), (W, y)], fill=RULE, width=1)

    # The hero in one static frame: a gold stream that reaches a gate and forks three ways.
    # Slate rises (money that was legitimately the buyer's), jade runs on (addressed),
    # crimson falls away (left on the table).
    GATE, MID = 690, 300
    SLATE = (109, 149, 189)

    def dot(x: float, y: float, colour: tuple[int, int, int], r: float = 1.5) -> None:
        d.ellipse([x - r, y - r, x + r, y + r], fill=colour)

    for i in range(300):
        x = 40 + i * (GATE - 40) / 300
        dot(x, MID + ((i * 7) % 5) - 2, GOLD)
    d.line([(GATE, MID - 120), (GATE, MID + 120)], fill=GOLD, width=1)

    for i in range(150):
        t = i / 150
        x = GATE + t * (W - GATE)
        dot(x, MID - (t * t * 3 if t < 1 else 1) * 0 - min(1.0, t * 2.2) ** 0.6 * 165, SLATE)
    for i in range(70):
        t = i / 70
        x = GATE + t * (W - GATE)
        dot(x, MID + ((i * 5) % 4) - 2, JADE)
    for i in range(110):
        t = i / 110
        x = GATE + t * (W - GATE)
        dot(x, MID + min(1.0, t * 1.9) ** 1.7 * 300, LEAK)

    # A gradient scrim rather than a rectangle. A hard vertical edge across the card reads
    # as a rendering mistake at thumbnail size.
    scrim = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(scrim)
    for x in range(W):
        a = 246 if x < 560 else max(0, int(246 * (1 - (x - 560) / 300)))
        sd.line([(x, 0), (x, H)], fill=(5, 6, 10, a))
    img = Image.alpha_composite(img.convert("RGBA"), scrim).convert("RGB")
    d = ImageDraw.Draw(img)

    d.text((64, 62), "RECOUP  ·  B2B RECEIVABLES", font=_mono(17), fill=GOLD)
    d.text((64, 100), "Recoup", font=_font(72, bold=True), fill=PAPER)
    d.text(
        (64, 190),
        "Every invoice was paid.\nNot every rupee arrived.",
        font=_font(36, bold=True),
        fill=PAPER,
        spacing=10,
    )

    d.line([(64, 322), (640, 322)], fill=RULE, width=2)

    cols = [
        (inr(leak["short_paid_paise"]), "never came back, across 400 invoices", GOLD),
        (inr(agent["money"]["addressed_paise"]), "handled: collected or escalated with evidence", JADE),
        (
            f"{agent['harm']['false_chase_contacts']} vs "
            f"{b1['harm']['false_chase_contacts'] if b1 else '-'}",
            "wrong letters: Recoup vs writing to everyone",
            LEAK,
        ),
    ]
    for i, (value, label, tone) in enumerate(cols):
        y = 352 + i * 72
        d.text((64, y), value, font=_mono(32), fill=tone)
        d.text((64, y + 40), label, font=_font(17), fill=MUTED)

    d.text(
        (64, H - 46),
        f"seed {board['seed']} · {board['days']}-day clock · local inference, cache committed",
        font=_mono(15),
        fill=MUTED,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, "PNG", optimize=True)
    print(f"{OUT}  {OUT.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
