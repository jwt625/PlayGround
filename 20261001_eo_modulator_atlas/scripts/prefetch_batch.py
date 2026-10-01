"""Single download owner: prefetch identity metadata and source PDFs for one batch, so distillers need no network.

For every paper in a batch CSV (data/_staging/batches/<tag>_NN.csv):
  1. references/<id>/crossref.json   Crossref work record (identity, license, authors, links); 2 s spacing
  2. references/<id>/source.pdf      first of: already cached | local corpus alias (git-ignored private mapping) |
                                     arXiv PDF | Crossref-listed open-access PDF link (CC license only)
  3. references/<id>/text.md + figures/   via scripts/extract_source.py
Status per paper goes to data/_staging/<batch>/prefetch_status.jsonl; papers without a source are written to
data/_staging/<batch>/needs_download.md (manual-download format). Network access goes through scripts/fetch_source.py
(file lock, spacing, circuit breaker). Run one prefetch at a time. Private source paths are never written to any output.

Usage: uv run python scripts/prefetch_batch.py data/_staging/batches/p1_02.csv [--no-network]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parent.parent
UA = "eo-modulator-atlas/0.1 (research; mailto:jwt625@gmail.com)"
PRIVATE_MAP = ROOT / "data" / "_staging" / "candidates_seed_local_paths_private.csv"
CROSSREF_DELAY_S = 2.0
# Publisher hosts known to serve open-access PDFs directly; others (Wiley, AIP, IEEE, Optica, ACS) challenge bots.
OA_PDF_HOSTS = {"www.nature.com", "nature.com", "www.mdpi.com", "mdpi-res.com"}


def load_private_map() -> dict[str, str]:
    if not PRIVATE_MAP.exists():
        return {}
    with PRIVATE_MAP.open(newline="") as f:
        return {r["paper_id"]: r["absolute_local_path"] for r in csv.DictReader(f)}


def crossref(pid: str, doi: str, refdir: Path) -> dict[str, object] | None:
    path = refdir / "crossref.json"
    if path.exists():
        return json.loads(path.read_text())["message"]  # type: ignore[no-any-return]
    if not doi:
        return None
    time.sleep(CROSSREF_DELAY_S)
    r = requests.get(f"https://api.crossref.org/works/{doi}", headers={"User-Agent": UA}, timeout=30)
    if r.status_code in (403, 429, 503):
        raise RuntimeError(f"Crossref blocked/throttled: HTTP {r.status_code}; stop and wait")
    if r.status_code != 200:
        return None
    refdir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(r.json(), indent=1))
    return r.json()["message"]  # type: ignore[no-any-return]


def cc_license(msg: dict[str, object] | None) -> str:
    for lic in (msg or {}).get("license", []) or []:  # type: ignore[attr-defined]
        url = str(lic.get("URL", ""))
        if "creativecommons.org" in url:
            return url
    return ""


def pdf_links(msg: dict[str, object] | None) -> list[str]:
    return [
        str(link["URL"])
        for link in (msg or {}).get("link", []) or []  # type: ignore[attr-defined]
        if "pdf" in str(link.get("content-type", "")).lower() or str(link.get("URL", "")).lower().endswith(".pdf")
    ]


def run_fetch(pid: str, url: str) -> int:
    p = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "fetch_source.py"), "--paper-id", pid, "--url", url],
        capture_output=True,
        text=True,
    )
    if p.returncode not in (0,):
        print(f"  fetch {pid} {url}: exit {p.returncode} {p.stderr.strip()[:160]}")
    return p.returncode


def extract(pid: str, refdir: Path, url: str, doi: str, lic: str) -> bool:
    if (refdir / "text.md").exists():
        return True
    p = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "extract_source.py"),
            "--paper-id",
            pid,
            "--pdf",
            str(refdir / "source.pdf"),
            "--url",
            url,
            "--doi",
            doi,
            "--license",
            lic,
            "--local-origin",
            "local_corpus",
        ],
        capture_output=True,
        text=True,
    )
    return p.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("batch_csv")
    ap.add_argument("--no-network", action="store_true")
    a = ap.parse_args()
    batch = Path(a.batch_csv)
    rows = list(csv.DictReader(batch.open(newline="")))
    out = ROOT / "data" / "_staging" / batch.stem
    out.mkdir(parents=True, exist_ok=True)
    private = load_private_map()
    status_rows: list[dict[str, str]] = []
    needs: list[dict[str, str]] = []
    for r in rows:
        pid, doi, ax = r["paper_id"], r["doi"], r["arxiv_id"]
        refdir = ROOT / "references" / pid
        refdir.mkdir(parents=True, exist_ok=True)
        st = {"paper_id": pid, "crossref": "", "source": "", "extracted": ""}
        msg = None
        try:
            if not a.no_network or (refdir / "crossref.json").exists():
                msg = crossref(pid, doi, refdir)
            st["crossref"] = "ok" if msg else "missing"
        except RuntimeError as exc:
            print(f"STOP: {exc}")
            st["crossref"] = "blocked"
            status_rows.append(st)
            break
        pdf = refdir / "source.pdf"
        if pdf.exists():
            st["source"] = "cached"
        elif r.get("local_source_path", "").startswith("@") and pid in private and Path(private[pid]).exists():
            src = Path(private[pid])
            if src.suffix.lower() == ".pdf":
                shutil.copyfile(src, pdf)
                st["source"] = "local_corpus"
            else:
                header = (
                    f"---\npaper_id: {pid}\nsource_url: {r.get('url', '')}\ndoi: {doi}\nlicense: \n"
                    f"extracted_on: {dt.date.today().isoformat()}\n"
                    "extraction_method: pre-extracted text copied from a local corpus; no PDF and no figure images available; "
                    "figures and tables must be treated as unavailable\n---\n\n"
                )
                (refdir / "text.md").write_text(header + src.read_text())
                st["source"] = "local_text_only"
        elif not a.no_network:
            tried = False
            if ax:
                tried = True
                run_fetch(pid, f"https://arxiv.org/pdf/{ax}")
            if not pdf.exists() and cc_license(msg):
                for link in [u for u in pdf_links(msg) if urlparse(u).hostname in OA_PDF_HOSTS][:1]:
                    tried = True
                    run_fetch(pid, link)
            st["source"] = "fetched" if pdf.exists() else ("failed" if tried else "no_open_source")
        else:
            st["source"] = "missing"
        if st["source"] == "local_text_only":
            st["extracted"] = "text_only"
        elif pdf.exists():
            lic = cc_license(msg) or r.get("license", "")
            ok = extract(pid, refdir, r.get("url", ""), doi, lic)
            st["extracted"] = "ok" if ok else "failed"
        else:
            needs.append(r)
        status_rows.append(st)
        print(f"{pid}: crossref={st['crossref']} source={st['source']} extracted={st['extracted']}")
    with (out / "prefetch_status.jsonl").open("a") as f:
        for st in status_rows:
            f.write(json.dumps({"ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), **st}) + "\n")
    if needs:
        with (out / "needs_download.md").open("a") as f:
            for r in needs:
                f.write(
                    f"- paper_id: {r['paper_id']}\n  title: {r['title']}\n  doi: {r['doi']}\n  publisher_url: {r['url']}\n"
                    f"  save_as: {r['paper_id']}.pdf\n  drop_folder: references/_inbox/\n"
                    "  why_needed: no open-access or local source retrievable by prefetch\n"
                )
    print(f"done: {len(rows)} papers, {len(needs)} need download")
    return 0


if __name__ == "__main__":
    sys.exit(main())
