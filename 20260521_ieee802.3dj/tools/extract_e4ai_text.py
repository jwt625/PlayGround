#!/usr/bin/env python3
"""Extract text from cached IEEE 802.3 E4AI materials and update metadata."""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import re
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_JSON = ROOT / "ieee802_e4ai_metadata" / "documents.json"
PROGRESS_JSONL = ROOT / "ieee802_e4ai_metadata" / "progress.jsonl"
EXTRACTED_ROOT = ROOT / "ieee802_3dj_browser" / "extracted_text" / "e4ai"


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def clean_text(value: str) -> str:
    value = html.unescape(value)
    value = value.replace("\xa0", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def markdown_to_text(markdown: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", markdown)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"[*_#>|~-]+", " ", text)
    return clean_text(text) + "\n"


def load_documents() -> dict[str, Any]:
    return json.loads(DOCUMENTS_JSON.read_text(encoding="utf-8"))


def write_documents(payload: dict[str, Any]) -> None:
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    DOCUMENTS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def output_paths(row: dict[str, Any]) -> tuple[Path, Path]:
    page = row.get("page_slug") or "unknown"
    stem = Path(row.get("path") or row.get("url") or row["doc_id"]).stem
    out_dir = EXTRACTED_ROOT / page
    return out_dir / f"{stem}.md", out_dir / f"{stem}.txt"


def append_progress(row: dict[str, Any]) -> None:
    PROGRESS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_done(method: str) -> set[str]:
    done: set[str] = set()
    if not PROGRESS_JSONL.exists():
        return done
    for line in PROGRESS_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") == "ok" and (method == "auto" or row.get("method") == method):
            done.add(row.get("doc_id", ""))
    return done


def extract_pdf(source: Path, title: str) -> str:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"PyMuPDF unavailable: {exc}") from exc

    parts = [f"# {title}", ""]
    with fitz.open(source) as pdf:
        for index, page in enumerate(pdf, start=1):
            text = page.get_text("text", sort=True).strip()
            parts.extend([f"## Page {index}", "", text, ""])
    return clean_text("\n".join(parts)) + "\n"


def extract_pptx(source: Path, title: str) -> str:
    parts = [f"# {title}", ""]
    ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    with zipfile.ZipFile(source) as archive:
        slide_names = sorted(
            (name for name in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)),
            key=lambda value: int(re.search(r"slide(\d+)\.xml", value).group(1)),  # type: ignore[union-attr]
        )
        for index, name in enumerate(slide_names, start=1):
            root = ElementTree.fromstring(archive.read(name))
            texts = [node.text or "" for node in root.findall(".//a:t", ns)]
            body = clean_text("\n".join(texts))
            parts.extend([f"## Slide {index}", "", body, ""])
    return clean_text("\n".join(parts)) + "\n"


def extract_zip_listing(source: Path, title: str) -> str:
    parts = [f"# {title}", "", "## Archive Contents", ""]
    with zipfile.ZipFile(source) as archive:
        rows = []
        for info in sorted(archive.infolist(), key=lambda item: item.filename.lower()):
            if info.is_dir():
                continue
            rows.append((info.filename, info.file_size, info.compress_size))
    parts.append("| File | Size | Compressed |")
    parts.append("|---|---:|---:|")
    for filename, size, compressed in rows:
        parts.append(f"| `{filename}` | {size} | {compressed} |")
    return "\n".join(parts).strip() + "\n"


def extract_fallback(source: Path, title: str) -> str:
    try:
        text = source.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = source.read_text(encoding="latin-1", errors="replace")
    return f"# {title}\n\n```\n{text[:1_000_000]}\n```\n"


def extract_one(row: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    source = ROOT / row["path"]
    md_path, txt_path = output_paths(row)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower()
    title = row.get("title") or source.name
    method = {
        ".pdf": "pymupdf",
        ".pptx": "pptx-xml",
        ".zip": "zip-listing",
    }.get(suffix, "text")

    try:
        if suffix == ".pdf":
            markdown = extract_pdf(source, title)
        elif suffix == ".pptx":
            markdown = extract_pptx(source, title)
        elif suffix == ".zip":
            markdown = extract_zip_listing(source, title)
        else:
            markdown = extract_fallback(source, title)
        md_path.write_text(markdown, encoding="utf-8")
        txt_path.write_text(markdown_to_text(markdown), encoding="utf-8")
        row["markdown_path"] = rel(md_path)
        row["text_path"] = rel(txt_path)
        row["extraction_method"] = method
        row["extraction_status"] = "ok"
        row["extraction_error"] = ""
        status = "ok"
        error = ""
    except Exception as exc:
        row["markdown_path"] = ""
        row["text_path"] = ""
        row["extraction_method"] = method
        row["extraction_status"] = "error"
        row["extraction_error"] = str(exc)
        status = "error"
        error = str(exc)

    return {
        "doc_id": row["doc_id"],
        "status": status,
        "method": method,
        "elapsed_seconds": round(time.time() - started, 3),
        "source_path": row["path"],
        "markdown_path": row.get("markdown_path", ""),
        "text_path": row.get("text_path", ""),
        "error": error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    payload = load_documents()
    docs = payload.get("documents", [])
    for row in docs:
        md_path, txt_path = output_paths(row)
        if md_path.exists() and txt_path.exists():
            row["markdown_path"] = rel(md_path)
            row["text_path"] = rel(txt_path)
            row.setdefault("extraction_status", "ok")

    payload["tool_info"] = {
        "python": sys.version,
        "pymupdf_available": importlib.util.find_spec("fitz") is not None,
        "pptx_method": "stdlib zip/xml",
        "zip_method": "stdlib zipfile central directory listing",
    }
    write_documents(payload)
    if args.manifest_only:
        print(f"Wrote {DOCUMENTS_JSON} with {len(docs)} documents")
        return 0

    done = load_done("auto") if args.resume else set()
    pending = [row for row in docs if row["doc_id"] not in done]
    if args.limit:
        pending = pending[: args.limit]

    print(f"Documents: {len(docs)} total, {len(done)} already done, {len(pending)} pending")
    for index, row in enumerate(pending, start=1):
        print(f"[{index}/{len(pending)}] {row['page_slug']}/{Path(row['path']).name}", flush=True)
        progress = extract_one(row)
        append_progress(progress)
        if progress["status"] == "ok":
            print(f"  OK via {progress['method']} ({progress['elapsed_seconds']}s)", flush=True)
        else:
            print(f"  ERROR: {progress['error']}", flush=True)
    write_documents(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
