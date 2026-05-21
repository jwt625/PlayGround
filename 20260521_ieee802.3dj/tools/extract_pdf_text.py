#!/usr/bin/env python3
"""
Extract text from cached IEEE 802.3dj PDFs and build browser-ready metadata.

The script prefers Marker output when present, and can also run Marker per PDF.
It falls back to PyMuPDF text extraction if Marker is unavailable or fails.
Progress is append-only JSONL so interrupted runs can be resumed safely.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CHECKLIST = ROOT / "IEEE-802p3dj-cache-checklist.md"
SOURCE_ROOT = ROOT / "ieee802_3dj_cache"
OUTPUT_ROOT = ROOT / "ieee802_3dj_browser"
EXTRACTED_ROOT = OUTPUT_ROOT / "extracted_text"
METADATA_ROOT = OUTPUT_ROOT / "metadata"
PROGRESS_JSONL = METADATA_ROOT / "progress.jsonl"
DOCUMENTS_JSON = METADATA_ROOT / "documents.json"
TALKS_JSON = METADATA_ROOT / "talks.json"
PUBLIC_BASE_URL = "https://www.ieee802.org/3/dj/public/"
BR_MARK = "\u241e"


@dataclass
class Document:
    doc_id: str
    meeting: str
    filename: str
    stem: str
    title: str
    website_title: str
    title_cell_text: str
    presentation_code: str
    file_code: str
    presentation_date: str
    presentation_date_iso: str
    presenters: list[str]
    affiliations: list[str]
    presenter_affiliations: list[dict[str, str]]
    source_parent_url: str
    source_parent_path: str
    presentation_url: str
    source_url: str
    source_path: str
    source_size: int
    sha256: str
    markdown_path: str
    text_path: str
    marker_output_dir: str


class TableMetadataParser(HTMLParser):
    def __init__(self, parent_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.parent_url = parent_url
        self.rows: list[list[dict[str, Any]]] = []
        self._row: list[dict[str, Any]] | None = None
        self._cell: dict[str, Any] | None = None
        self._link: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._row = []
        elif tag == "td" and self._row is not None:
            self._cell = {"chunks": [], "links": []}
        elif tag == "br" and self._cell is not None:
            self._cell["chunks"].append(BR_MARK)
        elif tag == "a" and self._cell is not None:
            attrs_dict = dict(attrs)
            href = attrs_dict.get("href")
            if href:
                self._link = {"href": urllib.parse.urljoin(self.parent_url, href), "text": ""}

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell["chunks"].append(data)
        if self._link is not None:
            self._link["text"] += data

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "a" and self._cell is not None and self._link is not None:
            self._link["text"] = clean_inline(self._link["text"])
            self._cell["links"].append(self._link)
            self._link = None
        elif tag == "td" and self._row is not None and self._cell is not None:
            text = clean_multiline("".join(self._cell["chunks"]))
            self._row.append({"text": text, "links": self._cell["links"]})
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slug_doc_id(meeting: str, filename: str) -> str:
    stem = Path(filename).stem
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return f"{meeting}__{safe}"


def clean_inline(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\xa0", " ")).strip()


def clean_multiline(value: str) -> str:
    value = value.replace("\xa0", " ")
    lines = [clean_inline(line) for line in value.split(BR_MARK)]
    return "\n".join(line for line in lines if line)


def split_people(value: str) -> list[str]:
    if not value:
        return []
    return [clean_inline(part) for part in value.split("\n") if clean_inline(part)]


MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def parse_date_iso(value: str) -> str:
    match = re.search(r"(\d{1,2})[-\s]+([A-Za-z]+)[-\s]+(\d{2,4})", value)
    if not match:
        return ""
    day = int(match.group(1))
    month = MONTHS.get(match.group(2).lower())
    if month is None:
        return ""
    year = int(match.group(3))
    if year < 100:
        year += 2000
    try:
        return datetime(year, month, day).date().isoformat()
    except ValueError:
        return ""


def file_name_for_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return Path(urllib.parse.unquote(parsed.path)).name


def pair_presenters_affiliations(presenters: list[str], affiliations: list[str]) -> list[dict[str, str]]:
    paired: list[dict[str, str]] = []
    for index, presenter in enumerate(presenters):
        affiliation = affiliations[index] if index < len(affiliations) else ""
        paired.append({"presenter": presenter, "affiliation": affiliation})
    if not paired and affiliations:
        paired.extend({"presenter": "", "affiliation": affiliation} for affiliation in affiliations)
    return paired


def looks_like_file_code(value: str) -> bool:
    value = clean_inline(value)
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9]+_\d+[A-Za-z]?", value))


def empty_talk_metadata(meeting: str) -> dict[str, Any]:
    parent_path = SOURCE_ROOT / meeting / "index.html"
    parent_url = urllib.parse.urljoin(PUBLIC_BASE_URL, f"{meeting}/index.html")
    return {
        "website_title": "",
        "title_cell_text": "",
        "presentation_code": "",
        "file_code": "",
        "presentation_date": "",
        "presentation_date_iso": "",
        "presenters": [],
        "affiliations": [],
        "presenter_affiliations": [],
        "source_parent_url": parent_url,
        "source_parent_path": rel(parent_path) if parent_path.exists() else "",
        "presentation_url": "",
    }


def parse_meeting_metadata(meeting: str) -> dict[str, dict[str, Any]]:
    parent_path = SOURCE_ROOT / meeting / "index.html"
    parent_url = urllib.parse.urljoin(PUBLIC_BASE_URL, f"{meeting}/index.html")
    if not parent_path.exists():
        return {}
    parser = TableMetadataParser(parent_url)
    parser.feed(parent_path.read_text(encoding="utf-8", errors="replace"))
    records: dict[str, dict[str, Any]] = {}
    for row in parser.rows:
        for link_index, cell in enumerate(row):
            pdf_links = [
                link
                for link in cell["links"]
                if Path(urllib.parse.urlparse(link["href"]).path).suffix.lower() == ".pdf"
            ]
            for link in pdf_links:
                after = row[link_index + 1 :]
                before = row[:link_index]
                date_offset = next(
                    (index for index, item in enumerate(after) if parse_date_iso(item["text"])),
                    None,
                )
                date_text = after[date_offset]["text"] if date_offset is not None else ""
                date_iso = parse_date_iso(date_text)
                file_code = ""
                presenters: list[str] = []
                affiliations: list[str] = []
                if date_offset is not None:
                    file_code = clean_inline(" ".join(item["text"] for item in after[:date_offset]))
                    tail = [item["text"] for item in after[date_offset + 1 :] if item["text"]]
                    if not file_code and tail and looks_like_file_code(tail[0]):
                        file_code = tail.pop(0)
                    presenters = split_people(tail[0]) if tail else []
                    affiliations = split_people(tail[1]) if len(tail) > 1 else []
                elif after:
                    presenters = split_people(after[0]["text"])
                    affiliations = split_people(after[1]["text"]) if len(after) > 1 else []
                presentation_code = clean_inline(" ".join(item["text"] for item in before))
                title = clean_inline(link["text"]) or clean_inline(cell["text"])
                records[file_name_for_url(link["href"])] = {
                    "website_title": title,
                    "title_cell_text": clean_inline(cell["text"]),
                    "presentation_code": presentation_code,
                    "file_code": file_code,
                    "presentation_date": clean_inline(date_text),
                    "presentation_date_iso": date_iso,
                    "presenters": presenters,
                    "affiliations": affiliations,
                    "presenter_affiliations": pair_presenters_affiliations(presenters, affiliations),
                    "source_parent_url": parent_url,
                    "source_parent_path": rel(parent_path),
                    "presentation_url": link["href"],
                }
    return records


def load_talk_metadata() -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for index_path in SOURCE_ROOT.glob("*/index.html"):
        meeting = index_path.parent.name
        if meeting.startswith("_"):
            continue
        for filename, row in parse_meeting_metadata(meeting).items():
            metadata[f"{meeting}/{filename}"] = row
    return metadata


def parse_checklist() -> list[Document]:
    text = CHECKLIST.read_text(encoding="utf-8")
    pattern = re.compile(
        r"`(?P<path>ieee802_3dj_cache/(?P<meeting>[0-9_]+)/(?P<filename>[^`]+\.pdf))` "
        r"\((?P<size>[0-9,]+) bytes\) - "
        r"\[(?P<title>[^\]]*)\]\((?P<url>[^)]+)\)"
    )
    docs: list[Document] = []
    talk_metadata = load_talk_metadata()
    talk_metadata_by_title: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, row in talk_metadata.items():
        meeting = key.split("/", 1)[0]
        title = clean_inline(row.get("website_title", ""))
        if title:
            talk_metadata_by_title.setdefault((meeting, title), []).append(row)
    for match in pattern.finditer(text):
        source_path = ROOT / match.group("path")
        if not source_path.exists():
            continue
        meeting = match.group("meeting")
        filename = match.group("filename")
        matched_metadata = talk_metadata.get(f"{meeting}/{filename}", {})
        if not matched_metadata:
            title_matches = talk_metadata_by_title.get((meeting, clean_inline(match.group("title"))), [])
            if len(title_matches) == 1:
                matched_metadata = title_matches[0]
        metadata = empty_talk_metadata(meeting) | matched_metadata
        if not metadata["presentation_url"]:
            metadata["presentation_url"] = match.group("url")
        doc_id = slug_doc_id(meeting, filename)
        out_dir = EXTRACTED_ROOT / meeting
        marker_dir = out_dir / "_marker" / Path(filename).stem
        docs.append(
            Document(
                doc_id=doc_id,
                meeting=meeting,
                filename=filename,
                stem=Path(filename).stem,
                title=match.group("title"),
                website_title=metadata["website_title"],
                title_cell_text=metadata["title_cell_text"],
                presentation_code=metadata["presentation_code"],
                file_code=metadata["file_code"],
                presentation_date=metadata["presentation_date"],
                presentation_date_iso=metadata["presentation_date_iso"],
                presenters=metadata["presenters"],
                affiliations=metadata["affiliations"],
                presenter_affiliations=metadata["presenter_affiliations"],
                source_parent_url=metadata["source_parent_url"],
                source_parent_path=metadata["source_parent_path"],
                presentation_url=metadata["presentation_url"],
                source_url=match.group("url"),
                source_path=rel(source_path),
                source_size=source_path.stat().st_size,
                sha256=sha256_file(source_path),
                markdown_path=rel(out_dir / f"{Path(filename).stem}.md"),
                text_path=rel(out_dir / f"{Path(filename).stem}.txt"),
                marker_output_dir=rel(marker_dir),
            )
        )
    return docs


def load_done(method: str) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not PROGRESS_JSONL.exists():
        return done
    for line in PROGRESS_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") not in {"ok", "skipped"}:
            continue
        row_method = row.get("method")
        if method != "auto" and row_method != method:
            continue
        if method == "auto" or row_method == method:
            done[row["doc_id"]] = row
    return done


def append_progress(row: dict[str, Any]) -> None:
    METADATA_ROOT.mkdir(parents=True, exist_ok=True)
    with PROGRESS_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_documents(docs: list[Document], tool_info: dict[str, Any]) -> None:
    METADATA_ROOT.mkdir(parents=True, exist_ok=True)
    document_rows = [asdict(doc) for doc in docs]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_checklist": rel(CHECKLIST),
        "source_root": rel(SOURCE_ROOT),
        "output_root": rel(OUTPUT_ROOT),
        "document_count": len(docs),
        "tool_info": tool_info,
        "documents": document_rows,
    }
    DOCUMENTS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TALKS_JSON.write_text(
        json.dumps(
            {
                "generated_at": payload["generated_at"],
                "source": "cached IEEE meeting index.html pages",
                "document_count": len(docs),
                "talks": document_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def marker_command() -> str | None:
    return shutil.which("marker")


def run_marker(doc: Document, timeout: int) -> tuple[str | None, str | None]:
    marker = marker_command()
    if not marker:
        return None, "marker executable not found"

    source = ROOT / doc.source_path
    marker_out = ROOT / doc.marker_output_dir
    marker_parent = marker_out.parent
    marker_parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        marker,
        str(source),
        "--output_dir",
        str(marker_parent),
        "--output_format",
        "markdown",
        "--disable_image_extraction",
        "--workers",
        "1",
        "--disable_multiprocessing",
        "--disable_tqdm",
    ]
    started = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    elapsed = time.time() - started
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        return None, f"marker failed after {elapsed:.1f}s: {msg[-2000:]}"

    candidates = sorted(marker_parent.glob(f"{source.stem}/**/*.md")) + sorted(marker_parent.glob(f"{source.stem}*.md"))
    if not candidates:
        candidates = sorted(marker_parent.rglob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None, f"marker produced no markdown after {elapsed:.1f}s"
    return candidates[0].read_text(encoding="utf-8", errors="replace"), None


def run_pymupdf(doc: Document) -> tuple[str | None, str | None]:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        return None, f"PyMuPDF unavailable: {exc}"

    source = ROOT / doc.source_path
    try:
        pdf = fitz.open(source)
        parts = [f"# {doc.title or doc.stem}", ""]
        for index, page in enumerate(pdf, start=1):
            text = page.get_text("text", sort=True).strip()
            parts.extend([f"## Page {index}", "", text, ""])
        return "\n".join(parts).strip() + "\n", None
    except Exception as exc:
        return None, f"PyMuPDF failed: {exc}"


def markdown_to_text(markdown: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", markdown)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"[*_#>|~-]+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def extract_one(doc: Document, method: str, timeout: int) -> dict[str, Any]:
    started = time.time()
    md_path = ROOT / doc.markdown_path
    txt_path = ROOT / doc.text_path
    md_path.parent.mkdir(parents=True, exist_ok=True)
    chosen_method = method
    errors: list[str] = []

    markdown: str | None = None
    if method in {"marker", "auto"}:
        markdown, error = run_marker(doc, timeout)
        if error:
            errors.append(error)
    if markdown is None and method in {"pymupdf", "auto"}:
        chosen_method = "pymupdf" if method == "pymupdf" else "auto:pymupdf"
        markdown, error = run_pymupdf(doc)
        if error:
            errors.append(error)

    elapsed = time.time() - started
    if markdown is None:
        return {
            "doc_id": doc.doc_id,
            "status": "error",
            "method": method,
            "elapsed_seconds": round(elapsed, 3),
            "errors": errors,
            "source_path": doc.source_path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    md_path.write_text(markdown, encoding="utf-8")
    txt_path.write_text(markdown_to_text(markdown), encoding="utf-8")
    return {
        "doc_id": doc.doc_id,
        "status": "ok",
        "method": chosen_method,
        "elapsed_seconds": round(elapsed, 3),
        "source_path": doc.source_path,
        "markdown_path": doc.markdown_path,
        "text_path": doc.text_path,
        "markdown_bytes": md_path.stat().st_size,
        "text_bytes": txt_path.stat().st_size,
        "errors": errors,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["auto", "marker", "pymupdf"], default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    docs = parse_checklist()
    tool_info = {
        "python": sys.version,
        "method": args.method,
        "marker": marker_command(),
        "pymupdf_available": importlib.util.find_spec("fitz") is not None,
    }
    write_documents(docs, tool_info)
    if args.manifest_only:
        print(f"Wrote {DOCUMENTS_JSON} with {len(docs)} documents")
        return 0

    done = load_done(args.method) if args.resume else {}
    pending = [doc for doc in docs if doc.doc_id not in done]
    if args.limit:
        pending = pending[: args.limit]

    print(f"Documents: {len(docs)} total, {len(done)} already done, {len(pending)} pending")
    for index, doc in enumerate(pending, start=1):
        print(f"[{index}/{len(pending)}] {doc.meeting}/{doc.filename}", flush=True)
        row = extract_one(doc, args.method, args.timeout)
        append_progress(row)
        if row["status"] != "ok":
            print(f"  ERROR: {'; '.join(row.get('errors', []))}", flush=True)
        else:
            print(f"  OK via {row['method']} ({row['elapsed_seconds']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
