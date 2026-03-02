#!/usr/bin/env python3
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

try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    PdfReader = None

SOURCES = [
    # SOFC/SOEC
    ("sofc", "nrel_sam_fuel_cell", "https://samrepo.nrelcloud.org/help/fuelcell.html"),
    ("sofc", "nrel_fuel_cell_power_model", "https://research-hub.nrel.gov/en/publications/fuel-cell-power-model-version-2-startup-guide-system-designs-and-"),
    ("sofc", "netl_data_center_sofc_project", "https://www.netl.doe.gov/node/5959"),
    ("sofc", "bloom_equinix_100mw", "https://investor.bloomenergy.com/press-releases/press-release-details/2025/Bloom-Energy-Expands-Data-Center-Power-Agreement-with-Equinix-Surpassing-100MW/default.aspx"),
    ("sofc", "fuelcell_energy_fy2025", "https://investor.fce.com/press-releases/press-release-details/2025/FuelCell-Energy-Ends-FY2025-with-Revenue-Growth-and-a-Focus-on-Data-Center-Opportunities/default.aspx"),
    ("soec", "topsoe_soec_factory", "https://www.topsoe.com/news/topsoe-inaugurates-europes-largest-soec-manufacturing-facility"),

    # Turbines
    ("turbine", "siemens_energy_ir", "https://www.siemens-energy.com/global/en/home/investor-relations.html"),
    ("turbine", "siemens_q1_fy26_earnings_pdf", "https://p3.aprimocdn.net/siemensenergy/f8b7f2d2-e4f2-4fe5-8295-b3ee005765a4/2026-02-11_Earnings-Release-Q1-FY26-pdf_Original%20file.pdf"),
    ("turbine", "siemens_q1_fy26_analyst_pdf", "https://p3.aprimocdn.net/siemensenergy/d5e24758-5d19-44f7-b57d-b3ee0057e335/2026-02-11_Q1_Analyst_presentation-pdf_Original%20file.pdf"),
    ("turbine", "ge_vernova_ceo_letter", "https://www.gevernova.com/investors/annual-report/ceo-letter"),
    ("turbine", "mhi_mccoy_share", "https://power.mhi.com/news/240315.html"),
    ("turbine", "mhi_us_power_outlook", "https://power.mhi.com/regions/amer/insights/us-power-outlook-and-long-term-trends"),

    # BESS
    ("bess", "tesla_q4_2025_deployments", "https://ir.tesla.com/press-release/tesla-fourth-quarter-2025-production-deliveries-deployments"),
    ("bess", "tesla_megafactory", "https://www.tesla.com/megafactory"),
    ("bess", "fluence_q3_2025", "https://ir.fluenceenergy.com/news-releases/news-release-details/fluence-energy-inc-reports-third-quarter-2025-results-reaffirms/"),
    ("bess", "nrel_atb_electricity_2025", "https://atb.nrel.gov/electricity/2025/"),

    # Solar/Grid
    ("solar", "first_solar_2024_results_2025_guidance", "https://www.businesswire.com/news/home/20250225031936/en/First-Solar-Inc.-Announces-Fourth-Quarter-and-Full-Year-2024-Financial-Results-and-2025-Guidance"),
    ("solar", "trina_2023_annual", "https://www.trinasolar.com/us/resources/newsroom/2023-annual-report/"),
    ("solar", "canadian_solar_20f", "https://www.sec.gov/Archives/edgar/data/1375877/000141057825001046/csiq-20241231x20f.htm"),
    ("grid", "eia_electricity_monthly", "https://www.eia.gov/electricity/monthly/index.php"),
    ("grid", "eia_steo", "https://www.eia.gov/steo"),
    ("grid", "pjm_2027_2028_bra", "https://insidelines.pjm.com/pjm-auction-procures-134479-mw-of-generation-resources/"),
]

RAW_DIR = "references/raw/subtech"
MD_DIR = "references/md/subtech"
MANIFEST = "references/manifest.subtech.json"

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


def html_to_text(html_bytes: bytes) -> str:
    html = html_bytes.decode("utf-8", errors="ignore")
    html = re.sub(r"<script[\\s\\S]*?</script>", " ", html, flags=re.I)
    html = re.sub(r"<style[\\s\\S]*?</style>", " ", html, flags=re.I)
    html = re.sub(r"</(p|div|h1|h2|h3|h4|h5|h6|li|tr|section|article|br)>", "\n", html, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", html)
    text = unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n\n".join(lines)


def extract_pdf_text(path: str) -> str:
    if PdfReader is None:
        return ""
    try:
        rd = PdfReader(path)
        chunks = []
        for p in rd.pages:
            chunks.append((p.extract_text() or "").strip())
        return "\n\n".join([c for c in chunks if c])
    except Exception:
        return ""


def fetch(url: str):
    agents = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7; rv:123.0) Gecko/20100101 Firefox/123.0",
    ]
    err = None
    for i in range(2):
        try:
            req = Request(url, headers={
                "User-Agent": agents[i % len(agents)],
                "Accept": "text/html,application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            })
            with urlopen(req, timeout=20, context=ssl.create_default_context()) as r:
                return getattr(r, "status", None), (r.headers.get("Content-Type", "") or "").lower(), r.geturl(), r.read()
        except Exception as e:
            err = e
            time.sleep(1 + i)
    raise err


def process(item):
    category, key, url = item
    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    raw_dir = os.path.join(RAW_DIR, category)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(MD_DIR, exist_ok=True)

    try:
        status, ctype, final_url, data = fetch(url)
        is_pdf = ("application/pdf" in ctype) or final_url.lower().endswith(".pdf")
        ext = "pdf" if is_pdf else "html"
        raw_path = os.path.join(raw_dir, f"{key}.{ext}")
        with open(raw_path, "wb") as f:
            f.write(data)

        text = extract_pdf_text(raw_path) if is_pdf else html_to_text(data)
        md_path = os.path.join(MD_DIR, f"{category}__{key}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("---\n")
            f.write(f"category: {category}\n")
            f.write(f"key: {key}\n")
            f.write(f"url: {url}\n")
            f.write(f"final_url: {final_url}\n")
            f.write(f"retrieved_at_utc: {ts}\n")
            f.write(f"source_type: {'pdf' if is_pdf else 'html'}\n")
            f.write(f"raw_path: {raw_path}\n")
            f.write("---\n\n")
            f.write((text.strip() or "Text extraction unavailable.") + "\n")
        return Result(category, key, url, "ok", status, "pdf" if is_pdf else "html", raw_path, md_path, None)
    except Exception as e:
        md_path = os.path.join(MD_DIR, f"{category}__{key}.md")
        err = f"{type(e).__name__}: {e}"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("---\n")
            f.write(f"category: {category}\n")
            f.write(f"key: {key}\n")
            f.write(f"url: {url}\n")
            f.write(f"retrieved_at_utc: {ts}\n")
            f.write("status: failed\n")
            f.write(f"error: {err}\n")
            f.write("---\n")
        return Result(category, key, url, "failed", None, "unknown", None, md_path, err)


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(MD_DIR, exist_ok=True)
    out = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(process, s) for s in SOURCES]
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            out.append(r)
            print(f"[{r.status}] {r.category}/{r.key}")
    out.sort(key=lambda x: (x.category, x.key))
    manifest = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "total": len(out),
        "ok": sum(1 for x in out if x.status == "ok"),
        "failed": sum(1 for x in out if x.status != "ok"),
        "items": [x.__dict__ for x in out],
    }
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written: {MANIFEST}")


if __name__ == "__main__":
    main()
