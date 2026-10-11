"""Wikimedia Commons search + download for the tree-intro bird cards (rate-limited, one request at a time).

Usage (from the project root):
  uv run python -I scripts/intro/fetch_bird_photos.py search "<query>" [n]
  uv run python -I scripts/intro/fetch_bird_photos.py info "File:<title>"
  uv run python -I scripts/intro/fetch_bird_photos.py get "File:<title>" <slug> [thumb_width]

Every HTTP request goes through _req(), which waits until >= MIN_GAP_S since the previous request
(timestamp file shared across invocations) and raises on any non-200 / non-image response.
`get` stores references/birds/raw/<slug>.<ext> (1280 px thumbnail or original if smaller) and appends a
record to references/birds/fetch_log.jsonl (file page, author, licence, URLs, SHA-256, size).
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "references" / "birds" / "raw"
LOG = ROOT / "references" / "birds" / "fetch_log.jsonl"
STAMP = ROOT / "references" / "birds" / ".last_request"
API = "https://commons.wikimedia.org/w/api.php"
UA = "pigeon-pluggable-intro/0.1 (personal non-commercial animation; one request at a time)"
MIN_GAP_S = 6.0
WIDTH = 1280
FREE = ("public domain", "pd", "cc0", "cc by", "cc-by")  # CC BY and CC BY-SA (prefix match), PD, CC0
NONFREE = ("nc", "nd", "fair use", "gfdl only")


def _req(url: str) -> bytes:
    RAW.mkdir(parents=True, exist_ok=True)
    if STAMP.exists():
        wait = MIN_GAP_S - (time.time() - STAMP.stat().st_mtime)
        if wait > 0:
            time.sleep(wait)
    STAMP.touch()
    r = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60)
    if r.status != 200:
        raise SystemExit(f"HTTP {r.status} for {url}")
    data = r.read()
    STAMP.touch()
    return data


def _api(**params: str) -> dict:
    params = {"format": "json", "formatversion": "2", **params}
    return json.loads(_req(API + "?" + urllib.parse.urlencode(params)))


def _strip(s: str | None) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", s or ""))).strip()


def _meta(page: dict) -> dict:
    ii = page["imageinfo"][0]
    em = ii.get("extmetadata", {})
    g = lambda k: _strip(em.get(k, {}).get("value"))
    return {
        "title": page["title"],
        "file_page": ii.get("descriptionurl"),
        "author": g("Artist"),
        "credit": g("Credit"),
        "licence": g("LicenseShortName"),
        "licence_url": g("LicenseUrl"),
        "usage_terms": g("UsageTerms"),
        "description": g("ImageDescription")[:300],
        "date": g("DateTimeOriginal"),
        "width": ii.get("width"),
        "height": ii.get("height"),
        "mime": ii.get("mime"),
        "original_url": ii.get("url"),
        "thumb_url": ii.get("thumburl"),
    }


def licence_ok(lic: str) -> bool:
    low = lic.lower()
    return any(low.startswith(f) for f in FREE) and not any(f" {n}" in low or f"-{n}" in low for n in NONFREE)


def cmd_search(query: str, n: int = 12) -> None:
    d = _api(action="query", generator="search", gsrsearch=f"filetype:bitmap {query}", gsrnamespace="6",
             gsrlimit=str(n), prop="imageinfo", iiprop="url|size|mime|extmetadata", iiurlwidth=str(WIDTH))
    for p in sorted(d.get("query", {}).get("pages", []), key=lambda p: p.get("index", 0)):
        m = _meta(p)
        ok = "OK " if licence_ok(m["licence"]) else "-- "
        print(f"{ok}{m['title']} | {m['width']}x{m['height']} | {m['licence']} | {m['author'][:60]} | {m['description'][:120]}")


def cmd_info(title: str, width: int = WIDTH) -> dict:
    d = _api(action="query", titles=title, prop="imageinfo", iiprop="url|size|mime|extmetadata", iiurlwidth=str(width))
    m = _meta(d["query"]["pages"][0])
    print(json.dumps(m, indent=1, ensure_ascii=False))
    return m


def cmd_get(title: str, slug: str, width: int = WIDTH) -> None:
    m = cmd_info(title, width)
    if not licence_ok(m["licence"]):
        raise SystemExit(f"licence not accepted: {m['licence']}")
    url = m["thumb_url"] or m["original_url"]
    data = _req(url)
    if not (data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n"):
        raise SystemExit("response is not a JPEG/PNG; stopping")
    ext = ".png" if data[:4] == b"\x89PNG" else ".jpg"
    out = RAW / f"{slug}{ext}"
    out.write_bytes(data)
    rec = {**m, "slug": slug, "downloaded_url": url, "local": str(out.relative_to(ROOT)),
           "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "date_fetched": time.strftime("%Y-%m-%d")}
    with LOG.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("saved", out.relative_to(ROOT), rec["sha256"], len(data))


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    if a[0] == "search":
        cmd_search(a[1], int(a[2]) if len(a) > 2 else 12)
    elif a[0] == "info":
        cmd_info(a[1])
    elif a[0] == "get":
        cmd_get(a[1], a[2], int(a[3]) if len(a) > 3 else WIDTH)
    else:
        raise SystemExit(__doc__)
