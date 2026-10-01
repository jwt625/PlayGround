"""Regression tests for scripts/extract_source.py image handling (gray+alpha and CMYK embedded images)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from extract_source import extract  # noqa: E402


def make_pdf(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Fig. 1 test caption\n" + "text " * 500)
    gray_alpha = pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.IRect(0, 0, 200, 200), True)
    gray_alpha.clear_with(128)
    page.insert_image(pymupdf.Rect(72, 200, 272, 400), pixmap=gray_alpha)
    cmyk = pymupdf.Pixmap(pymupdf.csCMYK, pymupdf.IRect(0, 0, 200, 200), False)
    cmyk.clear_with(60)
    page.insert_image(pymupdf.Rect(300, 200, 500, 400), pixmap=cmyk)
    doc.save(path)


def test_gray_alpha_and_cmyk_images_do_not_abort_extraction(tmp_path: Path) -> None:
    pdf = tmp_path / "t.pdf"
    make_pdf(pdf)
    out = tmp_path / "out"
    info = extract(pdf, out, {"paper_id": "t2026"})
    assert (out / "text.md").exists() and (out / "source.json").exists()
    figs = json.loads((out / "figures" / "figures.json").read_text())
    assert not [f for f in figs if f["kind"] == "embedded_failed"], figs
    assert len([f for f in figs if f["kind"] == "embedded"]) == 2
    assert info["pages"] == 1
