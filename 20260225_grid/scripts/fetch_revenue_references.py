#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import datetime as dt
import json
import os
import re
import ssl
import time
from dataclasses import dataclass
from html import unescape
from urllib.request import Request, urlopen

SOURCES = [
    (
        "revenue",
        "coreweave_q4_fy2025_results",
        "https://investors.coreweave.com/news/news-details/2026/CoreWeave-Reports-Strong-Fourth-Quarter-and-Fiscal-Year-2025-Results",
    ),
    (
        "revenue",
        "core_scientific_coreweave_update_2025_10_30",
        "https://investors.corescientific.com/sec-filings/all-sec-filings/content/0001140361-25-039864/ef20057996_ex99-1.htm",
    ),
    (
        "revenue",
        "nebius_q2_2025_results_ex99_1",
        "https://www.sec.gov/Archives/edgar/data/1513845/000110465925075028/tm2522866d1_ex99-1.htm",
    ),
    (
        "revenue",
        "nebius_q4_2025_results",
        "https://www.sec.gov/Archives/edgar/data/1513845/000110465925110831/tm2531034d1_ex99-1.htm",
    ),
    (
        "revenue",
        "nebius_q3_2025_shareholder_letter_ex99_2",
        "https://www.sec.gov/Archives/edgar/data/1513845/000110465925110831/tm2531034d1_ex99-2.htm",
    ),
]

RAW_DIR = "references/raw/revenue"
MD_DIR = "references/md/revenue"
MANIFEST = "references/manifest.revenue.json"


@dataclass
class Result:
    category: str
    key: str
    url: str
    status: str
    http_status: int | None
    source_type: str
    raw_path: str | None
    md_path: str | None
    error: str | None


def html_to_text(data: bytes) -> str:
    html = data.decode("utf-8", errors="ignore")
    html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
    html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.I)
    html = re.sub(r"</(p|div|h1|h2|h3|h4|h5|h6|li|tr|section|article|br)>", "\n", html, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", html)
    text = unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n\n".join(lines)


def fetch(url: str):
    err = None
    for attempt in range(3):
        headers = {
            "User-Agent": "wentaojiang research project contact: wentao@example.com",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=45, context=ssl.create_default_context()) as r:
                return getattr(r, "status", None), (r.headers.get("Content-Type", "") or "").lower(), r.geturl(), r.read()
        except Exception as e:
            err = e
            time.sleep(1 + attempt)
    raise err


def process(src):
    category, key, url = src
    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(MD_DIR, exist_ok=True)
    md_path = os.path.join(MD_DIR, f"{category}__{key}.md")
    try:
        status, ctype, final_url, data = fetch(url)
        raw_path = os.path.join(RAW_DIR, f"{key}.html")
        with open(raw_path, "wb") as f:
            f.write(data)
        text = html_to_text(data)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("---\n")
            f.write(f"category: {category}\n")
            f.write(f"key: {key}\n")
            f.write(f"url: {url}\n")
            f.write(f"final_url: {final_url}\n")
            f.write(f"retrieved_at_utc: {ts}\n")
            f.write("source_type: html\n")
            f.write(f"raw_path: {raw_path}\n")
            f.write("---\n\n")
            f.write((text or "Text extraction unavailable.") + "\n")
        return Result(category, key, url, "ok", status, "html", raw_path, md_path, None)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("---\n")
            f.write(f"category: {category}\nkey: {key}\nurl: {url}\nretrieved_at_utc: {ts}\nstatus: failed\nerror: {err}\n---\n")
        return Result(category, key, url, "failed", None, "unknown", None, md_path, err)


def main():
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        futures = [ex.submit(process, s) for s in SOURCES]
        for future in concurrent.futures.as_completed(futures):
            r = future.result()
            results.append(r)
            print(f"[{r.status}] {r.key}")
    results.sort(key=lambda x: x.key)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "total": len(results),
                "ok": sum(1 for x in results if x.status == "ok"),
                "failed": sum(1 for x in results if x.status != "ok"),
                "items": [x.__dict__ for x in results],
            },
            f,
            indent=2,
        )
    print("Manifest written:", MANIFEST)


if __name__ == "__main__":
    main()
