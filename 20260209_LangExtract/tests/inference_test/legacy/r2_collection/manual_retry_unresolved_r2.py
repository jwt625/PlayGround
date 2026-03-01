#!/usr/bin/env python3
import concurrent.futures
import hashlib
import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus, quote

import requests

UA = "LangExtract-R2-ManualCollector/1.0"
TIMEOUT = 25

BASE = Path("semiconductor_processing_dataset/raw_documents_R2")
PAPERS = BASE / "papers"
MANIFEST_PATH = BASE / "manifest_documents_r2.jsonl"
ATTEMPTS_PATH = BASE / "collection_attempts_r2.jsonl"
UNRESOLVED_PATH = BASE / "unresolved_references_r2.jsonl"


def nowz() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_manifest(path: Path) -> Dict[str, dict]:
    m = {}
    for row in read_jsonl(path):
        doc_id = row.get("document_id")
        if doc_id:
            m[doc_id] = row
    return m


def save_manifest(path: Path, m: Dict[str, dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for k in sorted(m):
            f.write(json.dumps(m[k], ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def looks_pdf(content_type: str, first_chunk: bytes) -> bool:
    return "application/pdf" in (content_type or "").lower() or first_chunk.startswith(b"%PDF")


def fetch_pdf(session: requests.Session, url: str, out: Path) -> Tuple[bool, Optional[int], str, str]:
    try:
        r = session.get(url, timeout=TIMEOUT, allow_redirects=True, stream=True, headers={"User-Agent": UA, "Accept": "application/pdf,*/*;q=0.8"})
    except Exception as e:
        return False, None, f"request_error:{type(e).__name__}", url

    status = r.status_code
    final_url = r.url
    if status >= 400:
        return False, status, f"http_{status}", final_url

    it = r.iter_content(chunk_size=1024 * 256)
    try:
        first = next(it, b"")
    except Exception as e:
        return False, status, f"stream_error:{type(e).__name__}", final_url

    if not looks_pdf(r.headers.get("content-type", ""), first):
        return False, status, "non_pdf", final_url

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        if first:
            f.write(first)
        for chunk in it:
            if chunk:
                f.write(chunk)

    if out.stat().st_size < 1024:
        out.unlink(missing_ok=True)
        return False, status, "too_small", final_url
    return True, status, "ok", final_url


def normalize_title(t: str) -> str:
    t = (t or "").lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def score_title(a: str, b: str) -> float:
    sa = set(normalize_title(a).split())
    sb = set(normalize_title(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def doi_to_candidates(doi: Optional[str]) -> List[Tuple[str, str]]:
    if not doi:
        return []
    d = doi.lower().strip()
    out: List[Tuple[str, str]] = []

    out.append(("doi", f"https://doi.org/{d}"))

    if d.startswith("10.1103/"):
        # APS direct PDF patterns
        low = d
        mapping = {
            "physrevlett": "prl",
            "physrevx": "prx",
            "physreva": "pra",
            "physrevb": "prb",
            "physrevc": "prc",
            "physrevd": "prd",
            "physreve": "pre",
            "physrevapplied": "prapplied",
            "physrev": "pr",
        }
        for key, j in mapping.items():
            if key in low:
                out.append(("aps_direct", f"https://journals.aps.org/{j}/pdf/{d}"))
                break

    if d.startswith("10.1038/"):
        art = d.split("/", 1)[1]
        out.append(("nature_direct", f"https://www.nature.com/articles/{art}.pdf"))

    if d.startswith("10.1063/"):
        out.append(("aip_direct", f"https://pubs.aip.org/aip/jap/article-pdf/doi/{d}/{d}.pdf"))

    return out


def title_search_candidates(session: requests.Session, title: str, year: Optional[int]) -> List[Tuple[str, str]]:
    cands: List[Tuple[str, str]] = []

    # OpenAlex title search
    try:
        q = quote_plus(title)
        r = session.get(f"https://api.openalex.org/works?search={q}&per-page=8", timeout=TIMEOUT, headers={"User-Agent": UA})
        if r.status_code == 200:
            for w in (r.json().get("results") or []):
                t = w.get("display_name") or ""
                y = w.get("publication_year")
                if score_title(title, t) < 0.35:
                    continue
                if year and y and abs(int(year) - int(y)) > 3:
                    continue
                oa = w.get("open_access") or {}
                for u in [oa.get("oa_url"), (w.get("best_oa_location") or {}).get("pdf_url"), (w.get("best_oa_location") or {}).get("landing_page_url")]:
                    if u:
                        cands.append(("openalex_title", u))
                for loc in (w.get("locations") or [])[:5]:
                    for u in [loc.get("pdf_url"), loc.get("landing_page_url")]:
                        if u:
                            cands.append(("openalex_title", u))
    except Exception:
        pass

    # Semantic Scholar title search
    try:
        q = quote_plus(title)
        r = session.get(
            f"https://api.semanticscholar.org/graph/v1/paper/search?query={q}&limit=8&fields=title,year,openAccessPdf,externalIds,url",
            timeout=TIMEOUT,
            headers={"User-Agent": UA},
        )
        if r.status_code == 200:
            for p in (r.json().get("data") or []):
                t = p.get("title") or ""
                y = p.get("year")
                if score_title(title, t) < 0.35:
                    continue
                if year and y and abs(int(year) - int(y)) > 3:
                    continue
                oap = (p.get("openAccessPdf") or {}).get("url")
                if oap:
                    cands.append(("s2_title", oap))
                if p.get("url"):
                    cands.append(("s2_title", p.get("url")))
                ext = p.get("externalIds") or {}
                arx = ext.get("ArXiv")
                if arx:
                    cands.append(("s2_title", f"https://arxiv.org/pdf/{arx}.pdf"))
    except Exception:
        pass

    # arXiv title search
    try:
        q = quote_plus(f'ti:"{title}"')
        r = session.get(f"http://export.arxiv.org/api/query?search_query={q}&start=0&max_results=5", timeout=TIMEOUT, headers={"User-Agent": UA})
        if r.status_code == 200:
            xml = r.text
            ids = re.findall(r"<id>https?://arxiv.org/abs/([^<]+)</id>", xml)
            for aid in ids[:5]:
                aid = aid.strip()
                cands.append(("arxiv_title", f"https://arxiv.org/pdf/{aid}.pdf"))
    except Exception:
        pass

    # dedupe
    seen = set()
    out = []
    for src, u in cands:
        if not u:
            continue
        if u in seen:
            continue
        seen.add(u)
        out.append((src, u))
    return out


def process_row(row: dict, manifest: Dict[str, dict], lock: threading.Lock) -> dict:
    doc_id = row["document_id"]
    doi = row.get("doi")
    title = row.get("title") or ""
    year = row.get("year")

    t0 = time.time()
    worker = f"collector_manual_{threading.current_thread().name.split('_')[-1]}"
    out_pdf = PAPERS / f"{doc_id}.pdf"

    session = requests.Session()
    session.headers.update({"User-Agent": UA})

    candidates = []
    candidates.extend(doi_to_candidates(doi))
    candidates.extend(title_search_candidates(session, title, year))

    prev = manifest.get(doc_id, {})
    prev_attempts = int(prev.get("attempt_count") or 0)

    for src, url in candidates:
        ok, status, msg, final = fetch_pdf(session, url, out_pdf)
        if ok:
            digest = sha256(out_pdf)
            ts = nowz()
            ev = {
                "attempt_id": f"{ts}_{doc_id}_manual_download_succeeded",
                "timestamp": ts,
                "worker_id": worker,
                "document_id": doc_id,
                "action": "manual_search_download_pdf",
                "source_url": final,
                "result": "succeeded",
                "http_status": status,
                "error_type": None,
                "error_message": None,
                "output_path": str(out_pdf),
                "sha256": digest,
                "duration_sec": round(time.time() - t0, 3),
                "notes": f"manual_retry source={src}"
            }
            with lock:
                append_jsonl(ATTEMPTS_PATH, ev)
                m = dict(prev)
                m.update({
                    "document_id": doc_id,
                    "url": final,
                    "source_path": str(out_pdf),
                    "status": "succeeded",
                    "last_attempt_at": ts,
                    "attempt_count": prev_attempts + 1,
                    "last_error": None,
                    "quality_assessment": "open_access_pdf_manual_retry",
                    "notes": f"manual retry succeeded via {src}",
                })
                manifest[doc_id] = m
            return {"document_id": doc_id, "status": "succeeded", "url": final, "source": src}

    ts = nowz()
    ev = {
        "attempt_id": f"{ts}_{doc_id}_manual_download_failed",
        "timestamp": ts,
        "worker_id": worker,
        "document_id": doc_id,
        "action": "manual_search_download_pdf",
        "source_url": candidates[0][1] if candidates else None,
        "result": "failed",
        "http_status": None,
        "error_type": "manual_no_open_pdf",
        "error_message": "manual retry exhausted",
        "output_path": None,
        "sha256": None,
        "duration_sec": round(time.time() - t0, 3),
        "notes": f"manual retry attempted {len(candidates)} URLs"
    }
    with lock:
        append_jsonl(ATTEMPTS_PATH, ev)
        m = dict(prev)
        m.update({
            "document_id": doc_id,
            "status": "failed",
            "last_attempt_at": ts,
            "attempt_count": prev_attempts + 1,
            "last_error": "manual_no_open_pdf",
            "notes": "manual retry exhausted; likely paywalled or metadata mismatch",
        })
        manifest[doc_id] = m
    return {"document_id": doc_id, "status": "failed"}


def main() -> None:
    unresolved = read_jsonl(UNRESOLVED_PATH)
    if not unresolved:
        print("No unresolved references found")
        return

    manifest = load_manifest(MANIFEST_PATH)
    lock = threading.Lock()
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=8, thread_name_prefix="manual") as ex:
        futs = [ex.submit(process_row, r, manifest, lock) for r in unresolved]
        for f in concurrent.futures.as_completed(futs):
            try:
                results.append(f.result())
            except Exception as e:
                results.append({"document_id": None, "status": "failed", "error": str(e)})

    save_manifest(MANIFEST_PATH, manifest)

    # rebuild unresolved file from manifest
    failed = [manifest[k] for k in sorted(manifest) if manifest[k].get("status") == "failed"]
    with UNRESOLVED_PATH.open("w", encoding="utf-8") as f:
        for r in failed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    out = {
        "timestamp": nowz(),
        "manual_retry_targets": len(unresolved),
        "succeeded": sum(1 for r in results if r.get("status") == "succeeded"),
        "failed": sum(1 for r in results if r.get("status") == "failed"),
        "remaining_unresolved": len(failed),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
