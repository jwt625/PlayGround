#!/usr/bin/env python3
import concurrent.futures
import datetime as dt
import json
import os
import random
import re
import ssl
import sys
import time
from dataclasses import dataclass
from html import unescape
from urllib.parse import urlparse
from urllib.request import Request, urlopen

try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    PdfReader = None

SOURCES = [
    ("lbnl_2024_data_center_report", "https://eta.lbl.gov/publications/2024-lbnl-data-center-energy-usage-report"),
    ("doe_release_data_center_demand", "https://www.energy.gov/articles/doe-releases-new-report-evaluating-increase-electricity-demand-data-centers"),
    ("berkeley_queues_landing", "https://emp.lbl.gov/queues"),
    ("osti_queued_up_2025", "https://www.osti.gov/biblio/3008763"),
    ("ferc_order_2023_explainer", "https://www.ferc.gov/explainer-interconnection-final-rule"),
    ("ferc_order_2023a_explainer", "https://www.ferc.gov/explainer-interconnection-final-rule-2023-A"),
    ("pjm_2026_2027_bra", "https://insidelines.pjm.com/pjm-auction-procures-134311-mw-of-generation-resources-supply-responds-to-price-signal/"),
    ("pjm_2027_2028_bra", "https://insidelines.pjm.com/pjm-auction-procures-134479-mw-of-generation-resources/"),
    ("wecc_large_load_report", "https://www.wecc.org/wecc-document/19111"),
    ("ferc_nerc_ltra_2025_presentation", "https://www.ferc.gov/news-events/news/ferc-nerc-presentation-2025-long-term-reliability-assessment"),
    ("eia_steo", "https://www.eia.gov/steo"),
    ("eia_electricity_monthly", "https://www.eia.gov/electricity/monthly/index.php"),
    ("eia_aeo2025_lcoe_pdf", "https://www.eia.gov/outlooks/aeo/electricity_generation/pdf/AEO2025_LCOE_report.pdf"),
    ("treasury_45y_48e_release", "https://home.treasury.gov/news/press-releases/jy2787"),
    ("bloom_equinix_100mw", "https://investor.bloomenergy.com/press-releases/press-release-details/2025/Bloom-Energy-Expands-Data-Center-Power-Agreement-with-Equinix-Surpassing-100MW/default.aspx"),
    ("openai_stargate", "https://openai.com/blog/announcing-the-stargate-project/"),
]

RAW_DIR = "references/raw"
MD_DIR = "references/md"
MANIFEST = "references/manifest.json"


@dataclass
class Result:
    key: str
    url: str
    status: str
    http_status: int | None
    source_type: str
    raw_path: str | None
    md_path: str | None
    error: str | None


def slugify(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")
    return s[:120] if s else "source"


def fetch_url(url: str):
    user_agents = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7; rv:123.0) Gecko/20100101 Firefox/123.0",
        "Mozilla/5.0 (compatible; CodexFetcher/1.0)",
    ]
    last_err = None
    for attempt in range(3):
        req = Request(
            url,
            headers={
                "User-Agent": random.choice(user_agents),
                "Accept": "text/html,application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        ctx = ssl.create_default_context()
        try:
            with urlopen(req, timeout=40, context=ctx) as resp:
                data = resp.read()
                status = getattr(resp, "status", None)
                content_type = resp.headers.get("Content-Type", "").lower()
                final_url = resp.geturl()
                return status, content_type, final_url, data
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(1.2 * (attempt + 1))
    raise last_err


def html_to_markdown(html_bytes: bytes) -> str:
    html = html_bytes.decode("utf-8", errors="ignore")
    html = re.sub(r"<script[\\s\\S]*?</script>", " ", html, flags=re.IGNORECASE)
    html = re.sub(r"<style[\\s\\S]*?</style>", " ", html, flags=re.IGNORECASE)
    html = re.sub(r"</(p|div|h1|h2|h3|h4|h5|h6|li|tr|section|article|br)>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    text = unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return ""
    return "\n\n".join(lines)


def extract_pdf_text(pdf_path: str) -> str:
    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(pdf_path)
        chunks: list[str] = []
        for p in reader.pages:
            chunks.append((p.extract_text() or "").strip())
        text = "\n\n".join([c for c in chunks if c])
        return text
    except Exception:
        return ""


def write_md(key: str, url: str, final_url: str, source_type: str, retrieved_at: str, text: str, raw_path: str | None):
    md_path = os.path.join(MD_DIR, f"{key}.md")
    header = [
        "---",
        f"key: {key}",
        f"url: {url}",
        f"final_url: {final_url}",
        f"retrieved_at_utc: {retrieved_at}",
        f"source_type: {source_type}",
        f"raw_path: {raw_path or ''}",
        "---",
        "",
    ]
    if text.strip():
        body = text.strip() + "\n"
    else:
        body = "Text extraction unavailable; see raw artifact path above.\n"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + body)
    return md_path


def process_source(item):
    key, url = item
    retrieved_at = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        status, content_type, final_url, data = fetch_url(url)
        parsed = urlparse(final_url)
        ext = ".pdf" if ("application/pdf" in content_type or final_url.lower().endswith(".pdf")) else ".html"
        raw_name = f"{slugify(key)}{ext}"
        raw_path = os.path.join(RAW_DIR, raw_name)
        with open(raw_path, "wb") as f:
            f.write(data)

        source_type = "pdf" if ext == ".pdf" else "html"
        if source_type == "pdf":
            text = extract_pdf_text(raw_path)
        else:
            text = html_to_markdown(data)

        md_path = write_md(key, url, final_url, source_type, retrieved_at, text, raw_path)
        return Result(key, url, "ok", status, source_type, raw_path, md_path, None)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        md_path = os.path.join(MD_DIR, f"{key}.md")
        # Preserve previous successful extraction if present.
        if not os.path.exists(md_path):
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(
                    "---\n"
                    f"key: {key}\n"
                    f"url: {url}\n"
                    f"retrieved_at_utc: {retrieved_at}\n"
                    "status: failed\n"
                    f"error: {err}\n"
                    "---\n\n"
                )
        return Result(key, url, "failed", None, "unknown", None, md_path, err)


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(MD_DIR, exist_ok=True)

    results: list[Result] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(process_source, s) for s in SOURCES]
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            results.append(r)
            print(f"[{r.status}] {r.key} ({r.source_type})")

    results_sorted = sorted(results, key=lambda x: x.key)
    manifest = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "total": len(results_sorted),
        "ok": sum(1 for r in results_sorted if r.status == "ok"),
        "failed": sum(1 for r in results_sorted if r.status != "ok"),
        "items": [r.__dict__ for r in results_sorted],
    }
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written: {MANIFEST}")


if __name__ == "__main__":
    main()
