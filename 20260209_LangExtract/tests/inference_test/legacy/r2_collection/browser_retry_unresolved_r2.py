#!/usr/bin/env python3
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options

BASE = Path("semiconductor_processing_dataset/raw_documents_R2")
PAPERS = BASE / "papers"
MANIFEST_PATH = BASE / "manifest_documents_r2.jsonl"
ATTEMPTS_PATH = BASE / "collection_attempts_r2.jsonl"
UNRESOLVED_PATH = BASE / "unresolved_references_r2.jsonl"
SUMMARY_PATH = BASE / "browser_retry_summary_r2.json"
CONFIG_PATH = Path(os.environ.get("BROWSER_CONFIG_PATH", "config.json"))
FIREFOX_BIN = "/Applications/Firefox.app/Contents/MacOS/firefox"


def nowz() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_manifest() -> Dict[str, dict]:
    m: Dict[str, dict] = {}
    for r in read_jsonl(MANIFEST_PATH):
        doc = r.get("document_id")
        if doc:
            m[doc] = r
    return m


def save_manifest(manifest: Dict[str, dict]) -> None:
    rows = [manifest[k] for k in sorted(manifest.keys())]
    write_jsonl(MANIFEST_PATH, rows)


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def normalize_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    d = doi.strip()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d, flags=re.I)
    return d.lower().strip().rstrip(".")


def doi_candidates(doi: str) -> List[str]:
    cands = [f"https://doi.org/{doi}"]
    if doi.startswith("10.1103/"):
        mapping = {
            "physrevlett": "prl",
            "physrevx": "prx",
            "physreva": "pra",
            "physrevb": "prb",
            "physrevc": "prc",
            "physrevd": "prd",
            "physreve": "pre",
            "physrevapplied": "prapplied",
            "revmodphys": "rmp",
            "physrev": "pr",
        }
        low = doi.lower()
        for k, v in mapping.items():
            if k in low:
                cands.append(f"https://journals.aps.org/{v}/pdf/{doi}")
                break
    if doi.startswith("10.1038/"):
        cands.append(f"https://www.nature.com/articles/{doi.split('/',1)[1]}.pdf")
    if doi.startswith("10.1126/"):
        cands.append(f"https://www.science.org/doi/pdf/{doi}")
    if doi.startswith("10.1063/"):
        cands.append(f"https://pubs.aip.org/aip/jap/article-pdf/doi/{doi}/{doi}.pdf")
    if doi.startswith("10.1109/"):
        cands.append(f"https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=")
    return cands


def wait_download(download_dir: Path, before: set, timeout: float = 18.0) -> Optional[Path]:
    end = time.time() + timeout
    last_file: Optional[Path] = None
    last_size = -1
    stable_count = 0
    while time.time() < end:
        current = set(download_dir.glob("*"))
        new_files = [p for p in current - before if p.is_file() and not p.name.endswith((".part", ".tmp", ".crdownload"))]
        if new_files:
            newest = max(new_files, key=lambda p: p.stat().st_mtime)
            size = newest.stat().st_size
            if newest == last_file and size == last_size and size > 1024:
                stable_count += 1
            else:
                stable_count = 0
            last_file, last_size = newest, size
            if stable_count >= 2:
                return newest
        time.sleep(0.7)
    return None


def try_click_pdfish(driver) -> None:
    # click links/buttons likely to trigger PDF
    xpaths = [
        "//a[contains(translate(., 'PDFDOWNLOADFULLTEXTVIEWARTICLE', 'pdfdownloadfulltextviewarticle'), 'pdf')]",
        "//a[contains(translate(., 'PDFDOWNLOADFULLTEXTVIEWARTICLE', 'pdfdownloadfulltextviewarticle'), 'download')]",
        "//button[contains(translate(., 'PDFDOWNLOADFULLTEXTVIEWARTICLE', 'pdfdownloadfulltextviewarticle'), 'pdf')]",
        "//button[contains(translate(., 'PDFDOWNLOADFULLTEXTVIEWARTICLE', 'pdfdownloadfulltextviewarticle'), 'download')]",
    ]
    for xp in xpaths:
        try:
            elems = driver.find_elements(By.XPATH, xp)
        except Exception:
            elems = []
        for e in elems[:5]:
            try:
                href = e.get_attribute("href") or ""
                if href and ("pdf" in href.lower() or href.lower().endswith(".pdf")):
                    driver.get(href)
                else:
                    driver.execute_script("arguments[0].click();", e)
                time.sleep(1.5)
            except Exception:
                continue


def collect_anchor_pdf_urls(driver) -> List[str]:
    urls: List[str] = []
    try:
        anchors = driver.find_elements(By.TAG_NAME, "a")
    except Exception:
        return urls
    for a in anchors:
        try:
            href = a.get_attribute("href") or ""
            if not href:
                continue
            h = href.lower()
            if any(k in h for k in [".pdf", "/pdf", "articlepdf", "download", "fulltextpdf", "stamp.jsp"]):
                urls.append(href)
        except Exception:
            continue
    dedup = []
    seen = set()
    for u in urls:
        if u not in seen:
            seen.add(u)
            dedup.append(u)
    return dedup


def setup_driver(download_dir: Path):
    cfg = json.loads(CONFIG_PATH.read_text())
    profile = cfg.get("firefox_profile_path")
    if not profile:
        raise RuntimeError("firefox_profile_path missing in config")
    prof_path = Path(profile)
    if not prof_path.exists():
        raise RuntimeError(f"profile path not found: {prof_path}")

    # Clone profile to avoid lock conflicts with existing interactive session.
    cloned = Path(tempfile.mkdtemp(prefix="fx_profile_clone_"))
    shutil.copytree(prof_path, cloned / "profile", dirs_exist_ok=True)
    run_profile = cloned / "profile"

    opts = Options()
    opts.binary_location = FIREFOX_BIN
    opts.add_argument(f"--profile={run_profile}")
    opts.add_argument("-headless")
    opts.add_argument("--width=1500")
    opts.add_argument("--height=1000")

    opts.set_preference("browser.download.folderList", 2)
    opts.set_preference("browser.download.dir", str(download_dir.resolve()))
    opts.set_preference("browser.download.useDownloadDir", True)
    opts.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/pdf,application/octet-stream,application/x-pdf,application/download")
    opts.set_preference("browser.download.manager.showWhenStarting", False)
    opts.set_preference("pdfjs.disabled", True)
    opts.set_preference("browser.download.alwaysOpenPanel", False)

    driver = webdriver.Firefox(options=opts)
    driver.set_page_load_timeout(35)
    return driver, cloned


def process_one(driver, row: dict, manifest: Dict[str, dict], download_dir: Path) -> dict:
    doc_id = row["document_id"]
    doi = normalize_doi(row.get("doi"))
    title = row.get("title") or ""
    prev = manifest.get(doc_id, {})
    prev_attempts = int(prev.get("attempt_count") or 0)
    start = time.time()

    if not doi:
        ts = nowz()
        ev = {
            "attempt_id": f"{ts}_{doc_id}_browser_no_doi_skipped",
            "timestamp": ts,
            "worker_id": "collector_browser_firefox",
            "document_id": doc_id,
            "action": "browser_manual_download_pdf",
            "source_url": None,
            "result": "failed",
            "http_status": None,
            "error_type": "no_doi_for_browser_retry",
            "error_message": "title-only item; skipped in DOI browser retry",
            "output_path": None,
            "sha256": None,
            "duration_sec": round(time.time() - start, 3),
            "notes": "requires manual title search"
        }
        append_jsonl(ATTEMPTS_PATH, ev)
        m = dict(prev)
        m.update({
            "document_id": doc_id,
            "status": "failed",
            "last_attempt_at": ts,
            "attempt_count": prev_attempts + 1,
            "last_error": "no_doi_for_browser_retry",
            "notes": "browser retry skipped: missing DOI",
        })
        manifest[doc_id] = m
        return {"document_id": doc_id, "status": "failed", "reason": "no_doi"}

    cands = doi_candidates(doi)

    for cand in cands:
        before = set(download_dir.glob("*"))
        try:
            driver.get(cand)
            time.sleep(2.0)
        except Exception:
            pass

        got = wait_download(download_dir, before, timeout=8.0)
        if not got:
            # Try anchors
            for u in collect_anchor_pdf_urls(driver)[:12]:
                before2 = set(download_dir.glob("*"))
                try:
                    driver.get(u)
                    time.sleep(1.5)
                except Exception:
                    continue
                got = wait_download(download_dir, before2, timeout=6.0)
                if got:
                    break

        if not got:
            before3 = set(download_dir.glob("*"))
            try_click_pdfish(driver)
            got = wait_download(download_dir, before3, timeout=8.0)

        if got:
            target = PAPERS / f"{doc_id}.pdf"
            PAPERS.mkdir(parents=True, exist_ok=True)
            if target.exists():
                target.unlink()
            shutil.move(str(got), str(target))
            sha = checksum(target)
            ts = nowz()
            ev = {
                "attempt_id": f"{ts}_{doc_id}_browser_download_succeeded",
                "timestamp": ts,
                "worker_id": "collector_browser_firefox",
                "document_id": doc_id,
                "action": "browser_manual_download_pdf",
                "source_url": driver.current_url,
                "result": "succeeded",
                "http_status": None,
                "error_type": None,
                "error_message": None,
                "output_path": str(target),
                "sha256": sha,
                "duration_sec": round(time.time() - start, 3),
                "notes": f"browser retry with DOI={doi}; seed={cand}"
            }
            append_jsonl(ATTEMPTS_PATH, ev)
            m = dict(prev)
            m.update({
                "document_id": doc_id,
                "url": driver.current_url,
                "source_path": str(target),
                "status": "succeeded",
                "last_attempt_at": ts,
                "attempt_count": prev_attempts + 1,
                "last_error": None,
                "quality_assessment": "browser_download_profile_auth",
                "notes": "succeeded via firefox profile browser retry",
            })
            manifest[doc_id] = m
            return {"document_id": doc_id, "status": "succeeded", "url": driver.current_url}

    ts = nowz()
    ev = {
        "attempt_id": f"{ts}_{doc_id}_browser_download_failed",
        "timestamp": ts,
        "worker_id": "collector_browser_firefox",
        "document_id": doc_id,
        "action": "browser_manual_download_pdf",
        "source_url": cands[0] if cands else None,
        "result": "failed",
        "http_status": None,
        "error_type": "browser_retry_exhausted",
        "error_message": "no downloaded PDF detected",
        "output_path": None,
        "sha256": None,
        "duration_sec": round(time.time() - start, 3),
        "notes": f"tried {len(cands)} DOI/publisher routes; title={title}"
    }
    append_jsonl(ATTEMPTS_PATH, ev)
    m = dict(prev)
    m.update({
        "document_id": doc_id,
        "status": "failed",
        "last_attempt_at": ts,
        "attempt_count": prev_attempts + 1,
        "last_error": "browser_retry_exhausted",
        "notes": "browser retry exhausted",
    })
    manifest[doc_id] = m
    return {"document_id": doc_id, "status": "failed"}


def main() -> None:
    unresolved = read_jsonl(UNRESOLVED_PATH)
    manifest = load_manifest()

    download_dir = BASE / "_tmp_browser_downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    for p in download_dir.glob("*"):
        if p.is_file():
            p.unlink()

    driver = None
    cloned_profile_root = None
    results: List[dict] = []

    try:
        driver, cloned_profile_root = setup_driver(download_dir)
        for idx, row in enumerate(unresolved, start=1):
            res = process_one(driver, row, manifest, download_dir)
            results.append(res)
            print(f"[{idx}/{len(unresolved)}] {row.get('document_id')} -> {res.get('status')}")
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass
        if cloned_profile_root is not None:
            try:
                shutil.rmtree(cloned_profile_root, ignore_errors=True)
            except Exception:
                pass

    save_manifest(manifest)

    still_failed = [manifest[k] for k in sorted(manifest.keys()) if manifest[k].get("status") == "failed"]
    write_jsonl(UNRESOLVED_PATH, still_failed)

    summary = {
        "timestamp": nowz(),
        "targets": len(unresolved),
        "succeeded": sum(1 for r in results if r.get("status") == "succeeded"),
        "failed": sum(1 for r in results if r.get("status") == "failed"),
        "remaining_unresolved": len(still_failed),
        "notes": "browser retry using Firefox persistent profile clone",
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
