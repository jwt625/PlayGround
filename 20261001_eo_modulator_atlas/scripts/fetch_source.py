"""Polite downloader for open-access sources (arXiv PDFs and open-access publisher PDFs).

Rules: one request at a time, >= MIN_DELAY seconds between requests (plus jitter), a descriptive User-Agent,
stop immediately on HTTP 403/429 or an HTML page where a PDF was expected (anti-bot / paywall), JSONL log with
timestamps under logs/. Never use this for paywalled sources; those go on the manual download list.

Usage: uv run python scripts/fetch_source.py --paper-id lee2026 --url https://arxiv.org/pdf/2601.17385
Writes references/<paper_id>/source.pdf and prints the sha256. Then run scripts/extract_source.py.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
UA = "eo-modulator-atlas/0.1 (research; mailto:jwt625@gmail.com)"
MIN_DELAY = 5.0
STAMP = ROOT / "logs" / ".last_fetch"


def wait_turn() -> None:
    STAMP.parent.mkdir(exist_ok=True)
    if STAMP.exists():
        elapsed = time.time() - float(STAMP.read_text() or 0)
        need = MIN_DELAY + random.uniform(0, 3) - elapsed
        if need > 0:
            time.sleep(need)
    STAMP.write_text(str(time.time()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-id", required=True)
    ap.add_argument("--url", required=True)
    a = ap.parse_args()
    out = ROOT / "references" / a.paper_id
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "source.pdf"
    if dest.exists():
        print(f"skip-existing {dest}")
        return 0
    wait_turn()
    r = requests.get(a.url, headers={"User-Agent": UA}, timeout=60, allow_redirects=True)
    ctype = r.headers.get("content-type", "")
    ok = r.status_code == 200 and r.content[:5] == b"%PDF-"
    log = {
        "ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "paper_id": a.paper_id,
        "url": a.url,
        "status": r.status_code,
        "content_type": ctype,
        "bytes": len(r.content),
        "ok": ok,
    }
    logfile = ROOT / "logs" / f"fetch_{dt.date.today().isoformat()}.jsonl"
    with logfile.open("a") as f:
        f.write(json.dumps(log) + "\n")
    if not ok:
        print(f"FAILED status={r.status_code} type={ctype}; not a PDF. Stop and report (possible paywall / anti-bot).", file=sys.stderr)
        return 3
    dest.write_bytes(r.content)
    print(json.dumps({"path": str(dest), "sha256": hashlib.sha256(r.content).hexdigest(), **log}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
