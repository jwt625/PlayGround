#!/usr/bin/env python3
"""Build the small static data index for the OFC paper browser."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "OFC_2026_high_speed_materials_papers.md"
OUT = ROOT / "ofc_browser" / "papers.json"
METADATA = ROOT / "output" / "full_metadata" / "ofc_full_metadata.json"


CODE_RE = re.compile(r"`?([A-Z][a-z]?\d+[A-Z]\.\d+)`?")


def rel(path: Path | None) -> str:
    if not path:
        return ""
    return path.relative_to(ROOT).as_posix()


def first_match(folder: str, code: str, suffix: str) -> Path | None:
    matches = sorted((ROOT / folder).glob(f"{code}-*{suffix}"))
    return matches[0] if matches else None


def metadata_by_code() -> dict[str, dict]:
    if not METADATA.exists():
        return {}

    records = json.loads(METADATA.read_text(encoding="utf-8"))
    return {
        record.get("presentation_code", ""): record
        for record in records
        if record.get("presentation_code")
    }


def parse_markdown_tables(markdown: str, metadata: dict[str, dict]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    section = ""
    headers: list[str] = []
    in_table = False

    for raw in markdown.splitlines():
        line = raw.strip()
        if line.startswith("## "):
            section = line[3:].strip()
            in_table = False
            headers = []
            continue

        if not line.startswith("|"):
            in_table = False
            headers = []
            continue

        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not in_table:
            headers = cells
            in_table = True
            continue

        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue

        if not headers or len(cells) < len(headers):
            continue

        row = dict(zip(headers, cells))
        code_match = CODE_RE.search(row.get("Code", ""))
        if not code_match:
            continue

        code = code_match.group(1)
        title = row.get("Presentation / paper", "").replace("`", "").strip()
        affiliation = row.get("Affiliation(s)", "").strip()
        note_key = next((h for h in headers if h not in {"Code", "Presentation / paper", "Affiliation(s)"}), "")
        note = row.get(note_key, "").strip()

        pdf = first_match(".cache/ofc/pdfs", code, ".pdf")
        md = first_match("extracted_text", code, ".md")
        txt = first_match("extracted_text", code, ".txt")
        meta = metadata.get(code, {})

        rows.append(
            {
                "code": code,
                "title": title,
                "affiliation": affiliation,
                "section": section,
                "noteLabel": note_key,
                "note": note,
                "pdfPath": rel(pdf),
                "markdownPath": rel(md),
                "textPath": rel(txt),
                "hasPdf": bool(pdf),
                "hasMarkdown": bool(md),
                "hasText": bool(txt),
                "abstract": meta.get("abstract_text", "").strip(),
                "description": meta.get("description_text", "").strip(),
            }
        )

    return rows


def main() -> None:
    markdown = SOURCE.read_text(encoding="utf-8")
    rows = parse_markdown_tables(markdown, metadata_by_code())
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "sourceMarkdownPath": SOURCE.relative_to(ROOT).as_posix(),
                "generatedFrom": SOURCE.name,
                "paperCount": len(rows),
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUT.relative_to(ROOT)} with {len(rows)} rows")


if __name__ == "__main__":
    main()
