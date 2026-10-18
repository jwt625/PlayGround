"""Throttled IEEE Xplore full-text fetcher using a dedicated, persistent browser profile (institutional sign-in).

Decisions (DevLog-001, 2026-10-18): one PDF every 60-120 s, at most 60 per calendar day, priority 1 first; stop on any
captcha / access warning / HTML-instead-of-PDF (breaker file logs/.ieee_blocked.json, manual reset). After the daily
cap the process sleeps until the next day (run it in tmux). Each PDF is extracted right after download.

Usage:
  uv run python scripts/ieee_fetch.py --login        # opens a window; sign in (Institutional Sign In); waits for access
  uv run python scripts/ieee_fetch.py --run [--max N] [--dry-run]
Profile: cache/browser_profile (git-ignored). PDFs: references/<paper_id>/source.pdf (local-only; institutional stamp).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import BrowserContext, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_source import extract  # noqa: E402
from xplore_common import CHROMIUM  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROFILE = ROOT / "cache" / "browser_profile"
LOGS = ROOT / "logs"
BREAKER = LOGS / ".ieee_blocked.json"
BASE = "https://ieeexplore.ieee.org"
DAILY_CAP = 60
GAP_S = (60.0, 120.0)
PROBE_ARNUMBER = "10195565"  # CoWoS-L ECTC 2023 (subscription): used to test institutional access


def paper_id_of(doi: str) -> str:
    return doi.lower().split("/", 1)[1].replace("/", "_")


def log(entry: dict) -> None:
    LOGS.mkdir(exist_ok=True)
    with (LOGS / f"ieee_fetch_{dt.date.today().isoformat()}.jsonl").open("a") as f:
        f.write(json.dumps({"ts": dt.datetime.now().isoformat(timespec="seconds"), **entry}) + "\n")


def ok_today() -> int:
    p = LOGS / f"ieee_fetch_{dt.date.today().isoformat()}.jsonl"
    if not p.exists():
        return 0
    return sum(1 for ln in p.read_text().splitlines() if '"event": "saved"' in ln)


def open_ctx(p, headless: bool) -> BrowserContext:
    PROFILE.mkdir(parents=True, exist_ok=True)
    return p.chromium.launch_persistent_context(
        str(PROFILE), headless=headless, executable_path=str(CHROMIUM), accept_downloads=False,
        viewport={"width": 1280, "height": 900}, locale="en-US",
    )


def access_status(ctx: BrowserContext) -> dict:
    ieee = [p for p in ctx.pages if p.url.startswith(BASE)]
    page = ieee[-1] if ieee else (ctx.pages[0] if ctx.pages else ctx.new_page())
    r = page.evaluate(
        """async (u) => { const r = await fetch(u, {credentials: 'same-origin'});
                          return {status: r.status, text: await r.text()}; }""",
        f"{BASE}/rest/document/{PROBE_ARNUMBER}/abstract",
    )
    try:
        ui = json.loads(r["text"]).get("userInfo", {})
    except json.JSONDecodeError:
        return {"status": r["status"], "institute": None}
    return {"status": r["status"], "institute": ui.get("institute"), "individual": ui.get("individual"),
            "subscribed": ui.get("subscribedContent")}


def login() -> int:
    with sync_playwright() as p:
        ctx = open_ctx(p, headless=False)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(f"{BASE}/document/{PROBE_ARNUMBER}/", wait_until="domcontentloaded", timeout=90000)
        print("Sign in via 'Institutional Sign In' in the opened window. Waiting (checks every 15 s, up to 30 min)...",
              flush=True)
        for _ in range(120):
            time.sleep(15)
            try:
                if any(p.url.startswith(BASE) for p in ctx.pages):
                    st = access_status(ctx)
                    print(st, flush=True)
                    if st.get("institute") or st.get("individual"):
                        print("ACCESS OK; profile saved.", flush=True)
                        log({"event": "login_ok", **st})
                        ctx.close()
                        return 0
            except Exception as exc:  # noqa: BLE001  page may be mid-navigation during SSO
                print(f"(waiting: {str(exc)[:80]})", flush=True)
        ctx.close()
        return 1


def targets() -> list[dict]:
    rows = list(csv.DictReader((ROOT / "outputs" / "cache_list.csv").open()))
    pool = {}
    for ln in (ROOT / "cache" / "pool.jsonl").read_text().splitlines():
        r = json.loads(ln)
        if r.get("doi"):
            pool[r["doi"].lower()] = r.get("article_number")
    out = []
    for r in rows:
        doi = (r["doi"] or "").lower()
        if not doi.startswith(("10.1109/", "10.23919/")):
            continue
        an = pool.get(doi)
        if not an:
            m = re.search(r"ieeexplore\.ieee\.org/document/(\d+)", r.get("url") or "")
            an = m.group(1) if m else None
        out.append({"paper_id": paper_id_of(doi), "doi": doi, "arnumber": an, "priority": int(r["priority"]),
                    "bucket": r["primary_bucket"], "title": r["title"]})
    # priority first; within a priority, round-robin over buckets so early PDFs cover every bucket
    out.sort(key=lambda x: x["priority"])
    ordered: list[dict] = []
    for pr in sorted({x["priority"] for x in out}):
        group = [x for x in out if x["priority"] == pr]
        by_b: dict[str, list] = {}
        for x in group:
            by_b.setdefault(x["bucket"], []).append(x)
        while any(by_b.values()):
            for b in sorted(by_b):
                if by_b[b]:
                    ordered.append(by_b[b].pop(0))
    return ordered


def resolve_arnumber(ctx: BrowserContext, doi: str) -> str | None:
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    page.goto(f"https://doi.org/{doi}", wait_until="domcontentloaded", timeout=90000)
    m = re.search(r"/document/(\d+)", page.url)
    return m.group(1) if m else None


def fetch_one(ctx: BrowserContext, t: dict) -> str:
    out = ROOT / "references" / t["paper_id"]
    if (out / "source.pdf").exists():
        return "skip"
    an = t["arnumber"] or resolve_arnumber(ctx, t["doi"])
    if not an:
        log({"event": "no_arnumber", **t})
        return "no_arnumber"
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    stamp = f"{BASE}/stamp/stamp.jsp?tp=&arnumber={an}"
    page.goto(stamp, wait_until="domcontentloaded", timeout=90000)
    page.wait_for_timeout(4000)
    html = page.content()
    if "getPDF" not in html and (re.search(r"awsWafCookieDomainList|challenge-container|captcha", html, re.I)
                                 or len(html) < 5000):
        page.wait_for_timeout(8000)  # WAF JS challenge usually clears itself in a headed browser
        html = page.content()
        (LOGS / f"ieee_challenge_{an}.html").write_text(html[:20000])
    if "getPDF" not in html and re.search(r"awsWafCookieDomainList|challenge-container|captcha", html, re.I):
        BREAKER.write_text(json.dumps({"at": dt.datetime.now().isoformat(), "reason": "challenge page", "arnumber": an}))
        log({"event": "blocked", "arnumber": an, **t})
        return "blocked"
    m = re.search(r'src="([^"]*getPDF\.jsp[^"]*)"', html)
    pdf_url = (BASE + m.group(1).replace("&amp;", "&")) if m and m.group(1).startswith("/") else \
        (m.group(1).replace("&amp;", "&") if m else f"{BASE}/stampPDF/getPDF.jsp?tp=&arnumber={an}&ref=")
    r = ctx.request.get(pdf_url, headers={"Referer": stamp}, timeout=120000)
    body = r.body()
    if r.status != 200 or body[:5] != b"%PDF-" or len(body) < 20000:
        reason = f"status={r.status} type={r.headers.get('content-type')} bytes={len(body)}"
        log({"event": "not_pdf", "arnumber": an, "reason": reason, **t})
        if r.status in (403, 429, 503) or b"captcha" in body[:20000].lower():
            BREAKER.write_text(json.dumps({"at": dt.datetime.now().isoformat(), "reason": reason, "arnumber": an}))
            return "blocked"
        return "not_pdf"
    out.mkdir(parents=True, exist_ok=True)
    part = out / "source.pdf.part"
    part.write_bytes(body)
    part.replace(out / "source.pdf")
    (out / "source.json").write_text(json.dumps({
        "paper_id": t["paper_id"], "doi": t["doi"], "url": f"https://doi.org/{t['doi']}", "arnumber": an,
        "retrieved_on": dt.date.today().isoformat(), "access": "institutional subscription (local-only)",
        "redistribution": "restricted_local_only", "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest()}, indent=1))
    try:
        info = extract(t["paper_id"])
        log({"event": "saved", "arnumber": an, "pages": info["pages"], "text_chars": info["text_chars"], **t})
    except Exception as exc:  # noqa: BLE001
        log({"event": "saved", "arnumber": an, "extract_error": str(exc)[:200], **t})
    return "saved"


def run(max_n: int | None, dry: bool, headless: bool) -> int:
    todo = [t for t in targets() if not (ROOT / "references" / t["paper_id"] / "source.pdf").exists()]
    print(f"{len(todo)} IEEE-hosted papers to fetch (priority order)", flush=True)
    if dry:
        for t in todo[:15]:
            print(t["priority"], t["bucket"], t["paper_id"], t["title"][:70])
        return 0
    done = 0
    with sync_playwright() as p:
        ctx = open_ctx(p, headless=headless)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(BASE + "/Xplore/home.jsp", wait_until="domcontentloaded", timeout=90000)
        page.wait_for_timeout(5000)
        st = access_status(ctx)
        print("access:", st, flush=True)
        if not (st.get("institute") or st.get("individual")):
            print("NOT SIGNED IN: run --login first.", flush=True)
            ctx.close()
            return 2
        misses = 0
        for t in todo:
            if BREAKER.exists():
                print("BREAKER active; stopping.", flush=True)
                break
            while ok_today() >= DAILY_CAP:
                tomorrow = dt.datetime.combine(dt.date.today() + dt.timedelta(days=1), dt.time(8, 0))
                wait = (tomorrow - dt.datetime.now()).total_seconds() + random.uniform(0, 1800)
                print(f"daily cap {DAILY_CAP} reached; sleeping {wait / 3600:.1f} h", flush=True)
                time.sleep(max(60, wait))
            res = fetch_one(ctx, t)
            print(dt.datetime.now().strftime("%H:%M:%S"), res, t["paper_id"], flush=True)
            misses = misses + 1 if res == "not_pdf" else 0
            if misses >= 2:
                st = access_status(ctx)
                if not (st.get("institute") or st.get("individual")):
                    print(f"SESSION LOST {st}: stopping; run --login again.", flush=True)
                    log({"event": "session_lost", **st})
                    break
            if res == "blocked":
                break
            if res != "skip":
                done += 1
                if max_n and done >= max_n:
                    break
                time.sleep(random.uniform(*GAP_S))
        ctx.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--login", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--max", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--headed", action="store_true", help="keep the window visible during --run")
    a = ap.parse_args()
    if a.login:
        return login()
    if a.run:
        return run(a.max, a.dry_run, headless=not a.headed)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
