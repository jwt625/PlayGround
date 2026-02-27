#!/usr/bin/env python3
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import requests

USER_AGENT = "LangExtract-R2-Collector/1.0 (+https://github.com/wentaojiang)"
REQUEST_TIMEOUT = 25
MAX_CANDIDATES_PER_SOURCE = 6
DOC_PREFIX = "r2"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def jsonl_append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_manifest(path: Path) -> Dict[str, dict]:
    manifest: Dict[str, dict] = {}
    if not path.exists():
        return manifest
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = row.get("document_id")
            if doc_id:
                manifest[doc_id] = row
    return manifest


def save_manifest(path: Path, manifest: Dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for doc_id in sorted(manifest.keys()):
            f.write(json.dumps(manifest[doc_id], ensure_ascii=False) + "\n")


def sanitize_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    d = doi.strip()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d, flags=re.I)
    d = d.strip().rstrip(".")
    return d.lower() if d else None


def sanitize_arxiv(arxiv: Optional[str]) -> Optional[str]:
    if not arxiv:
        return None
    a = arxiv.strip()
    a = re.sub(r"^arxiv:\s*", "", a, flags=re.I)
    a = a.replace(" ", "")
    if not a:
        return None
    return a


def slugify(text: str, max_len: int = 90) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] if len(text) > max_len else text


def make_document_id(ref: dict, idx: int) -> str:
    doi = sanitize_doi(ref.get("doi"))
    arxiv = sanitize_arxiv(ref.get("arxiv"))
    year = str(ref.get("year") or "unk")
    if doi:
        base = slugify(doi.replace("/", "_"))
        if base:
            return f"{DOC_PREFIX}_{base}"
    if arxiv:
        base = slugify(arxiv.replace("/", "_"))
        if base:
            return f"{DOC_PREFIX}_arxiv_{base}"
    title = slugify(ref.get("title") or "untitled", max_len=60)
    digest = hashlib.sha1((ref.get("title") or str(idx)).encode("utf-8")).hexdigest()[:10]
    return f"{DOC_PREFIX}_{year}_{title}_{digest}"


def checksum_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def is_pdf_bytes(data: bytes) -> bool:
    return data.startswith(b"%PDF")


def try_download_pdf(session: requests.Session, url: str, out_path: Path) -> Tuple[bool, Optional[int], Optional[str], str]:
    """Returns (success, http_status, error, final_url)."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8"}
    try:
        resp = session.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True, stream=True)
    except Exception as e:
        return False, None, f"request_error:{type(e).__name__}:{e}", url

    final_url = resp.url
    status = resp.status_code
    if status >= 400:
        return False, status, f"http_{status}", final_url

    content_type = (resp.headers.get("content-type") or "").lower()
    if "application/pdf" not in content_type:
        try:
            first = next(resp.iter_content(chunk_size=512), b"")
        except Exception as e:
            return False, status, f"stream_error:{type(e).__name__}:{e}", final_url
        if not is_pdf_bytes(first):
            return False, status, f"non_pdf_content_type:{content_type}", final_url
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as f:
            f.write(first)
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        return True, status, None, final_url

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        first = b""
        for i, chunk in enumerate(resp.iter_content(chunk_size=1024 * 1024)):
            if not chunk:
                continue
            if i == 0:
                first = chunk[:16]
            f.write(chunk)
    if out_path.stat().st_size < 1024 or not first.startswith(b"%PDF"):
        try:
            out_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False, status, "invalid_pdf_signature_or_too_small", final_url
    return True, status, None, final_url


def oa_candidates_for_ref(session: requests.Session, doi: Optional[str], arxiv: Optional[str]) -> List[Tuple[str, str]]:
    candidates: List[Tuple[str, str]] = []

    if arxiv:
        arx = arxiv.replace("arXiv:", "").strip()
        if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", arx) or "/" in arx:
            candidates.append(("arxiv", f"https://arxiv.org/pdf/{arx}.pdf"))
            candidates.append(("arxiv_abs", f"https://arxiv.org/abs/{arx}"))

    if doi:
        doi_q = quote(doi, safe="")

        try:
            r = session.get(f"https://api.openalex.org/works/https://doi.org/{doi_q}", headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                oa = data.get("open_access") or {}
                for k in ["oa_url"]:
                    u = oa.get(k)
                    if u:
                        candidates.append(("openalex_oa", u))
                best = data.get("best_oa_location") or {}
                for k in ["pdf_url", "landing_page_url"]:
                    u = best.get(k)
                    if u:
                        candidates.append(("openalex_best", u))
                for loc in (data.get("locations") or [])[:MAX_CANDIDATES_PER_SOURCE]:
                    for k in ["pdf_url", "landing_page_url"]:
                        u = (loc or {}).get(k)
                        if u:
                            candidates.append(("openalex_loc", u))
        except Exception:
            pass

        try:
            r = session.get(f"https://api.unpaywall.org/v2/{doi_q}", params={"email": "langextract@example.com"}, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                best = data.get("best_oa_location") or {}
                for k in ["url_for_pdf", "url"]:
                    u = best.get(k)
                    if u:
                        candidates.append(("unpaywall_best", u))
                for loc in (data.get("oa_locations") or [])[:MAX_CANDIDATES_PER_SOURCE]:
                    for k in ["url_for_pdf", "url"]:
                        u = (loc or {}).get(k)
                        if u:
                            candidates.append(("unpaywall_loc", u))
        except Exception:
            pass

        candidates.append(("doi", f"https://doi.org/{doi}"))

    seen = set()
    deduped: List[Tuple[str, str]] = []
    for src, url in candidates:
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append((src, url))
    return deduped


def classify_source_type(ref: dict) -> str:
    title = (ref.get("title") or "").lower()
    if any(k in title for k in ["review", "state of play", "outlook"]):
        return "review_paper"
    return "journal_article"


def collect_one(
    ref: dict,
    idx: int,
    out_dir: Path,
    attempts_path: Path,
    manifest: Dict[str, dict],
    manifest_lock: threading.Lock,
    io_lock: threading.Lock,
    dry_run: bool = False,
) -> dict:
    start = time.time()
    thread_name = threading.current_thread().name
    worker_id = f"collector_papers_{thread_name.split('_')[-1]}"

    doi = sanitize_doi(ref.get("doi"))
    arxiv = sanitize_arxiv(ref.get("arxiv"))
    title = (ref.get("title") or "").strip()
    year = ref.get("year")
    citation_count = ref.get("citation_count")

    doc_id = make_document_id(ref, idx)
    pdf_path = out_dir / f"{doc_id}.pdf"

    with manifest_lock:
        prev = manifest.get(doc_id)
        prev_attempt_count = int((prev or {}).get("attempt_count") or 0)
    if pdf_path.exists() and pdf_path.stat().st_size > 1024:
        sha = checksum_sha256(pdf_path)
        ts = now_utc_iso()
        event = {
            "attempt_id": f"{ts}_{doc_id}_skip_existing",
            "timestamp": ts,
            "worker_id": worker_id,
            "document_id": doc_id,
            "action": "skip_existing_pdf",
            "source_url": None,
            "result": "skipped",
            "http_status": None,
            "error_type": None,
            "error_message": None,
            "output_path": str(pdf_path),
            "sha256": sha,
            "duration_sec": round(time.time() - start, 3),
            "notes": "existing file in raw_documents_R2"
        }
        with io_lock:
            jsonl_append(attempts_path, event)
            manifest[doc_id] = {
                "document_id": doc_id,
                "source_type": classify_source_type(ref),
                "title": title,
                "institution": None,
                "year": year,
                "doi": doi,
                "arxiv": arxiv,
                "citation_count": citation_count,
                "url": prev.get("url") if prev else None,
                "source_path": str(pdf_path),
                "status": "succeeded",
                "last_attempt_at": ts,
                "attempt_count": prev_attempt_count + 1,
                "last_error": None,
                "quality_assessment": "open_access_or_existing",
                "priority": "high" if (citation_count or 0) >= 30 else "medium",
                "notes": "existing file reused"
            }
        return {"document_id": doc_id, "status": "succeeded", "url": None, "path": str(pdf_path)}

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    candidates = oa_candidates_for_ref(session, doi=doi, arxiv=arxiv)

    ts = now_utc_iso()
    with io_lock:
        manifest[doc_id] = {
            "document_id": doc_id,
            "source_type": classify_source_type(ref),
            "title": title,
            "institution": None,
            "year": year,
            "doi": doi,
            "arxiv": arxiv,
            "citation_count": citation_count,
            "url": None,
            "source_path": None,
            "status": "attempted",
            "last_attempt_at": ts,
            "attempt_count": prev_attempt_count + 1,
            "last_error": None,
            "quality_assessment": None,
            "priority": "high" if (citation_count or 0) >= 30 else "medium",
            "notes": f"queued with {len(candidates)} candidates"
        }

    for src, url in candidates:
        if dry_run:
            continue
        ok, http_status, err, final_url = try_download_pdf(session, url, pdf_path)
        event_ts = now_utc_iso()
        if ok:
            sha = checksum_sha256(pdf_path)
            event = {
                "attempt_id": f"{event_ts}_{doc_id}_download_pdf_succeeded",
                "timestamp": event_ts,
                "worker_id": worker_id,
                "document_id": doc_id,
                "action": "download_pdf",
                "source_url": final_url,
                "result": "succeeded",
                "http_status": http_status,
                "error_type": None,
                "error_message": None,
                "output_path": str(pdf_path),
                "sha256": sha,
                "duration_sec": round(time.time() - start, 3),
                "notes": f"source={src}; open-access candidate"
            }
            with io_lock:
                jsonl_append(attempts_path, event)
                manifest[doc_id] = {
                    "document_id": doc_id,
                    "source_type": classify_source_type(ref),
                    "title": title,
                    "institution": None,
                    "year": year,
                    "doi": doi,
                    "arxiv": arxiv,
                    "citation_count": citation_count,
                    "url": final_url,
                    "source_path": str(pdf_path),
                    "status": "succeeded",
                    "last_attempt_at": event_ts,
                    "attempt_count": prev_attempt_count + 1,
                    "last_error": None,
                    "quality_assessment": "open_access_pdf",
                    "priority": "high" if (citation_count or 0) >= 30 else "medium",
                    "notes": f"downloaded via {src}"
                }
            return {"document_id": doc_id, "status": "succeeded", "url": final_url, "path": str(pdf_path)}

    fail_ts = now_utc_iso()
    event = {
        "attempt_id": f"{fail_ts}_{doc_id}_download_pdf_failed",
        "timestamp": fail_ts,
        "worker_id": worker_id,
        "document_id": doc_id,
        "action": "download_pdf",
        "source_url": candidates[0][1] if candidates else None,
        "result": "failed",
        "http_status": None,
        "error_type": "no_open_access_pdf_found",
        "error_message": "all candidates failed or produced non-pdf content",
        "output_path": None,
        "sha256": None,
        "duration_sec": round(time.time() - start, 3),
        "notes": f"tried {len(candidates)} candidate URLs"
    }
    with io_lock:
        jsonl_append(attempts_path, event)
        manifest[doc_id] = {
            "document_id": doc_id,
            "source_type": classify_source_type(ref),
            "title": title,
            "institution": None,
            "year": year,
            "doi": doi,
            "arxiv": arxiv,
            "citation_count": citation_count,
            "url": None,
            "source_path": None,
            "status": "failed",
            "last_attempt_at": fail_ts,
            "attempt_count": prev_attempt_count + 1,
            "last_error": "no_open_access_pdf_found",
            "quality_assessment": None,
            "priority": "high" if (citation_count or 0) >= 30 else "medium",
            "notes": "No OA PDF resolved; may require manual access"
        }
    return {"document_id": doc_id, "status": "failed", "url": None, "path": None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect OA PDFs for external refs cited > threshold")
    parser.add_argument("--input", default="tests/inference_test/output/external_refs_cited_gt10.json")
    parser.add_argument("--output-dir", default="semiconductor_processing_dataset/raw_documents_R2")
    parser.add_argument("--tag", default="r2", help="Round tag used for output file names and doc_id prefix")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    global DOC_PREFIX
    DOC_PREFIX = (args.tag or "r2").strip().lower()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    papers_dir = out_dir / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)

    attempts_path = out_dir / f"collection_attempts_{DOC_PREFIX}.jsonl"
    manifest_path = out_dir / f"manifest_documents_{DOC_PREFIX}.jsonl"
    summary_path = out_dir / f"collection_summary_{DOC_PREFIX}.json"

    with input_path.open("r", encoding="utf-8") as f:
        refs = json.load(f)

    manifest = load_manifest(manifest_path)
    manifest_lock = threading.Lock()
    io_lock = threading.Lock()

    results: List[dict] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="worker") as ex:
        futures = [
            ex.submit(
                collect_one,
                ref,
                idx,
                papers_dir,
                attempts_path,
                manifest,
                manifest_lock,
                io_lock,
                args.dry_run,
            )
            for idx, ref in enumerate(refs, start=1)
        ]
        for fut in concurrent.futures.as_completed(futures):
            try:
                res = fut.result()
            except Exception as e:
                res = {"document_id": None, "status": "failed", "error": f"worker_exception:{type(e).__name__}:{e}"}
            results.append(res)

    save_manifest(manifest_path, manifest)

    succeeded = sum(1 for r in results if r.get("status") == "succeeded")
    failed = sum(1 for r in results if r.get("status") == "failed")
    skipped = sum(1 for r in results if r.get("status") == "skipped")

    summary = {
        "timestamp": now_utc_iso(),
        "input": str(input_path),
        "output_dir": str(out_dir),
        "total_refs": len(refs),
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "manifest_path": str(manifest_path),
        "attempts_path": str(attempts_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
