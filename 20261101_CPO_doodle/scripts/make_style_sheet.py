"""Render the style-draft stills and stitch the comparison sheet in outputs/style_drafts/.

Usage: uv run --no-project --with playwright --with pillow python scripts/make_style_sheet.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "style_drafts"
NAMES = {"A": "A marker", "B": "B crayon", "C": "C manga tone", "D": "D picture-book cel", "E": "E chalk"}
W, H = 960, 540


def main() -> int:
    jobs = [f"{scene}:{k}" for k in NAMES for scene in ("oe", "line")] + ["variants:A"]
    subprocess.run([sys.executable, str(ROOT / "scripts" / "render_stills.py"), *jobs], check=True, cwd=ROOT / "scripts")
    sheet = Image.new("RGB", (W * 2, H * len(NAMES)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(str(ROOT / "lib" / "fonts" / "xkcd-script.ttf"), 44)
    for i, (k, name) in enumerate(NAMES.items()):
        sheet.paste(Image.open(OUT / f"oe_{k}.png").convert("RGB").resize((W, H), Image.LANCZOS), (0, i * H))
        sheet.paste(Image.open(OUT / f"line_{k}.png").convert("RGB").resize((W, H), Image.LANCZOS), (W, i * H))
        draw.text((24, i * H + 16), name, fill="#EFEBDD" if k == "E" else "#1E1A18", font=font)
    sheet.save(OUT / "style_sheet.png")
    print(OUT / "style_sheet.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
