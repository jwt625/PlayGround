"""Polite, serialized downloader for open-access sources (arXiv PDFs and open-access publisher PDFs).

Guarantees (tested in tests/test_ingest_fetch.py):
  - Cross-process serialization: an exclusive file lock (logs/.fetch.lock) is held for the whole spacing-wait + request,
    so any number of simultaneous callers produce requests at least MIN_DELAY (+ jitter) seconds apart.
  - Per-host circuit breaker: HTTP 403/429/503, an HTML page where a PDF was expected, or a network error writes
    logs/.fetch_blocked_<host>.json (cooldown, default 30 min). While it is active, no caller requests that host; they exit 4.
  - Atomic cache: the download goes to source.pdf.part and is moved to source.pdf only after the %PDF magic, a minimum
    size and (when the server sent Content-Length) the full length are verified. Partial files never become cache hits.
  - JSONL log with ISO timestamps in logs/fetch_<date>.jsonl.
Never use this for paywalled sources; those go on the manual download list (data/manual_downloads.md).

Usage: uv run python scripts/fetch_source.py --paper-id lee2026 --url https://arxiv.org/pdf/2601.17385
Exit codes: 0 ok / already cached, 3 not a PDF or HTTP error (breaker armed when applicable), 4 breaker active.
Then run scripts/extract_source.py.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = Path(os.environ.get("EO_ATLAS_ROOT", Path(__file__).resolve().parent.parent))
UA = "eo-modulator-atlas/0.1 (research; mailto:jwt625@gmail.com)"
MIN_DELAY = float(os.environ.get("EO_ATLAS_MIN_DELAY", "5.0"))
JITTER = float(os.environ.get("EO_ATLAS_JITTER", "3.0"))
COOLDOWN_S = float(os.environ.get("EO_ATLAS_COOLDOWN_S", "1800"))
MIN_PDF_BYTES = int(os.environ.get("EO_ATLAS_MIN_PDF_BYTES", "20000"))
LOGS = ROOT / "logs"
LOCK = LOGS / ".fetch.lock"
STAMP = LOGS / ".last_fetch"


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def log(entry: dict[str, object]) -> None:
    LOGS.mkdir(exist_ok=True)
    with (LOGS / f"fetch_{dt.date.today().isoformat()}.jsonl").open("a") as f:
        f.write(json.dumps({"ts": now_iso(), **entry}) + "\n")


def valid_pdf(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(5)
        return head == b"%PDF-" and path.stat().st_size >= MIN_PDF_BYTES
    except OSError:
        return False


def breaker_path(url: str) -> Path:
    """Per-host breaker file: a block from one publisher must not stop downloads from an unrelated host."""
    host = (urlparse(url).hostname or "unknown").lower()
    return LOGS / f".fetch_blocked_{host}.json"


def breaker_active(url: str) -> float:
    """Seconds of cooldown remaining for this URL's host (0 when inactive)."""
    path = breaker_path(url)
    if not path.exists():
        return 0.0
    try:
        until = float(json.loads(path.read_text())["until"])
    except (ValueError, KeyError, json.JSONDecodeError):
        return 0.0
    return max(0.0, until - time.time())


def arm_breaker(reason: str, url: str) -> None:
    breaker_path(url).write_text(
        json.dumps({"until": time.time() + COOLDOWN_S, "reason": reason, "url": url, "armed_at": now_iso()})
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-id", required=True)
    ap.add_argument("--url", required=True)
    a = ap.parse_args()
    out = ROOT / "references" / a.paper_id
    dest = out / "source.pdf"
    if dest.exists() and valid_pdf(dest):
        print(f"skip-existing {dest}")
        return 0
    LOGS.mkdir(exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        remaining = breaker_active(a.url)
        if remaining > 0:
            log({"paper_id": a.paper_id, "url": a.url, "event": "blocked_by_breaker", "remaining_s": round(remaining)})
            print(
                f"BREAKER active ({remaining:.0f} s left): a previous request was blocked; not fetching.",
                file=sys.stderr,
            )
            return 4
        if STAMP.exists():
            try:
                elapsed = time.time() - float(STAMP.read_text() or 0)
            except ValueError:
                elapsed = 0.0
            need = MIN_DELAY + random.uniform(0, JITTER) - elapsed
            if need > 0:
                time.sleep(need)
        STAMP.write_text(str(time.time()))
        part = out / "source.pdf.part"
        try:
            r = requests.get(a.url, headers={"User-Agent": UA}, timeout=60, allow_redirects=True, stream=False)
        except requests.RequestException as exc:
            arm_breaker(f"network error: {exc}", a.url)
            log({"paper_id": a.paper_id, "url": a.url, "event": "network_error", "error": str(exc)})
            print(f"FAILED network error: {exc}", file=sys.stderr)
            return 3
        ctype = r.headers.get("content-type", "")
        declared = r.headers.get("content-length")
        complete = (
            declared is None
            or not declared.isdigit()
            or int(declared) == len(r.content)
            or r.headers.get("content-encoding")
        )
        ok = r.status_code == 200 and r.content[:5] == b"%PDF-" and len(r.content) >= MIN_PDF_BYTES and bool(complete)
        entry = {
            "paper_id": a.paper_id,
            "url": a.url,
            "event": "fetch",
            "status": r.status_code,
            "content_type": ctype,
            "bytes": len(r.content),
            "ok": ok,
        }
        log(entry)
        if not ok:
            if r.status_code in (403, 429, 503) or "html" in ctype.lower():
                arm_breaker(f"status={r.status_code} type={ctype}", a.url)
            part.unlink(missing_ok=True)
            print(
                f"FAILED status={r.status_code} type={ctype} bytes={len(r.content)}; not a complete PDF. "
                "Record it in needs_download.md and move on (no workarounds).",
                file=sys.stderr,
            )
            return 3
        part.write_bytes(r.content)
        os.replace(part, dest)
    print(json.dumps({"path": str(dest), "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(), **entry}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
