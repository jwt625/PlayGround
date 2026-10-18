"""Ingest manually downloaded PDFs from references/_inbox/<paper_id>.pdf (names from data/manual_downloads.md):
move to references/<paper_id>/source.pdf, write source.json, extract text/figures. Never deletes an inbox file that
fails the %PDF check (it is left in place and reported).

Usage: uv run python scripts/ingest_inbox.py [--dry-run]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_source import extract  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INBOX = ROOT / "references" / "_inbox"


def wanted() -> dict[str, str]:
    md = (ROOT / "data" / "manual_downloads.md").read_text()
    out = {}
    for block in md.split("- paper_id: ")[1:]:
        pid = block.split("\n", 1)[0].strip()
        m = re.search(r"doi: (\S+)", block)
        out[pid] = m.group(1) if m else ""
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    want = wanted()
    for f in sorted(INBOX.glob("*.pdf")):
        pid = f.stem
        if pid not in want:
            print(f"UNKNOWN {f.name}: not in manual_downloads.md (left in inbox)")
            continue
        if f.read_bytes()[:5] != b"%PDF-":
            print(f"NOT A PDF {f.name} (left in inbox)")
            continue
        if a.dry_run:
            print(f"would ingest {f.name}")
            continue
        out = ROOT / "references" / pid
        out.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), out / "source.pdf")
        body = (out / "source.pdf").read_bytes()
        (out / "source.json").write_text(json.dumps({
            "paper_id": pid, "doi": want[pid], "url": f"https://doi.org/{want[pid]}", "retrieved_on": dt.date.today().isoformat(),
            "access": "manual download (local-only)", "redistribution": "restricted_local_only",
            "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}, indent=1))
        info = extract(pid)
        print(f"ingested {pid}: {info['pages']} pages, {info['text_chars']} chars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
