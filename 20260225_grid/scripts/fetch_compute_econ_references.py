#!/usr/bin/env python3
import concurrent.futures, datetime as dt, json, os, re, ssl, time
from dataclasses import dataclass
from html import unescape
from urllib.request import Request, urlopen

SOURCES = [
    ("compute", "coreweave_s1a1_2025", "https://www.sec.gov/Archives/edgar/data/2042022/000121390025066839/ea0249945-s1a1_white.htm"),
    ("compute", "coreweave_sec_comment_2025", "https://www.sec.gov/Archives/edgar/data/1769628/000095012325000509/filename1.htm"),
    ("compute", "corvex_s1_2025", "https://www.sec.gov/Archives/edgar/data/1734750/000121390025124017/ea0269747-s1_movano.htm"),
    ("compute", "nvidia_warranty", "https://www.nvidia.com/en-us/support/warranty/"),
    ("compute", "nvidia_enterprise_support", "https://www.nvidia.com/en-us/support/enterprise/"),
    ("compute", "google_dram_errors", "https://research.google/pubs/dram-errors-in-the-wild-a-large-scale-field-study/"),
    ("compute", "fail_slow_scale_arxiv", "https://arxiv.org/abs/2309.07242"),
]

RAW_DIR = "references/raw/subtech/compute"
MD_DIR = "references/md/subtech"
MANIFEST = "references/manifest.compute_econ.json"

@dataclass
class R:
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
    html = re.sub(r"<script[\\s\\S]*?</script>", " ", html, flags=re.I)
    html = re.sub(r"<style[\\s\\S]*?</style>", " ", html, flags=re.I)
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
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        if "sec.gov" in url:
            headers["User-Agent"] = "wentaojiang research project contact: wentao@example.com"
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=30, context=ssl.create_default_context()) as r:
                return getattr(r, "status", None), (r.headers.get("Content-Type", "") or "").lower(), r.geturl(), r.read()
        except Exception as e:
            err = e
            time.sleep(1 + attempt)
    raise err


def proc(item):
    cat, key, url = item
    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(MD_DIR, exist_ok=True)

    md_path = os.path.join(MD_DIR, f"{cat}__{key}.md")
    try:
        status, ctype, final_url, data = fetch(url)
        raw_path = os.path.join(RAW_DIR, f"{key}.html")
        with open(raw_path, "wb") as f:
            f.write(data)
        text = html_to_text(data)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("---\n")
            f.write(f"category: {cat}\n")
            f.write(f"key: {key}\n")
            f.write(f"url: {url}\n")
            f.write(f"final_url: {final_url}\n")
            f.write(f"retrieved_at_utc: {ts}\n")
            f.write("source_type: html\n")
            f.write(f"raw_path: {raw_path}\n")
            f.write("---\n\n")
            f.write((text or "Text extraction unavailable.") + "\n")
        return R(cat, key, url, "ok", status, "html", raw_path, md_path, None)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("---\n")
            f.write(f"category: {cat}\nkey: {key}\nurl: {url}\nretrieved_at_utc: {ts}\nstatus: failed\nerror: {err}\n---\n")
        return R(cat, key, url, "failed", None, "unknown", None, md_path, err)


def main():
    out = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(proc, s) for s in SOURCES]
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            out.append(r)
            print(f"[{r.status}] {r.key}")
    out.sort(key=lambda x: x.key)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "total": len(out),
            "ok": sum(1 for x in out if x.status == "ok"),
            "failed": sum(1 for x in out if x.status != "ok"),
            "items": [x.__dict__ for x in out],
        }, f, indent=2)
    print("Manifest written:", MANIFEST)


if __name__ == "__main__":
    main()
