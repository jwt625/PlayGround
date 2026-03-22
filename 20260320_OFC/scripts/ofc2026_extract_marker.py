#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from pypdf import PdfReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract OFC 2026 PDFs with Marker.")
    parser.add_argument("--archive", required=True, help="Path to the source ZIP archive.")
    parser.add_argument("--pdf-dir", default="data/ofc2026_pdfs", help="Directory for extracted PDFs.")
    parser.add_argument("--output-dir", default="extracted_text", help="Directory for extracted markdown/text.")
    parser.add_argument("--index", default="paper_text_index.json", help="Output JSON index path.")
    parser.add_argument(
        "--fail-log",
        default="extraction_failures.jsonl",
        help="JSONL file for extraction failures.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of PDFs to process.",
    )
    parser.add_argument("--num-shards", type=int, default=1, help="Total shard count.")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index.")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_pdfs_unzipped(archive_path: Path, pdf_dir: Path) -> list[Path]:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_paths: list[Path] = []
    with zipfile.ZipFile(archive_path) as archive:
        for name in sorted(archive.namelist()):
            if not name.lower().endswith(".pdf"):
                continue
            if "__macosx/" in name.lower():
                continue
            target = pdf_dir / Path(name).name
            if not target.exists():
                with archive.open(name) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            pdf_paths.append(target)
    return pdf_paths


def load_index(index_path: Path) -> dict:
    if not index_path.exists():
        return {}
    return json.loads(index_path.read_text(encoding="utf-8"))


def write_index(index_path: Path, records: dict) -> None:
    payload = {
        "generated_at": utc_now(),
        "paper_count": len(records),
        "papers": records,
    }
    index_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_failure(log_path: Path, record: dict) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def markdown_to_text(markdown: str) -> str:
    text = markdown.replace("\u00a0", " ")
    text = re.sub(r"</?[^>]+>", " ", text)
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("*", " ")
    text = re.sub(r"^[#>*\-\s]+", "", text, flags=re.M)
    text = re.sub(r"\|", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_with_marker(converter: PdfConverter, pdf_path: Path) -> tuple[str, str]:
    rendered = converter(str(pdf_path))
    markdown = (rendered.markdown or "").strip()
    text = markdown_to_text(markdown)
    if not text:
        raise ValueError("Marker returned empty text")
    return markdown, text


def extract_with_pypdf(pdf_path: Path) -> tuple[str, str]:
    reader = PdfReader(str(pdf_path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    text = "\n\n".join(page for page in pages if page).strip()
    if not text:
        raise ValueError("PyPDF returned empty text")
    markdown = text
    return markdown, text


def main() -> None:
    args = parse_args()
    archive_path = Path(args.archive).resolve()
    pdf_dir = Path(args.pdf_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    index_path = Path(args.index).resolve()
    fail_log = Path(args.fail_log).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_paths = ensure_pdfs_unzipped(archive_path, pdf_dir)
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be within [0, num_shards)")
    if args.num_shards > 1:
        pdf_paths = [
            pdf_path
            for idx, pdf_path in enumerate(pdf_paths)
            if idx % args.num_shards == args.shard_index
        ]
    if args.limit is not None:
        pdf_paths = pdf_paths[: args.limit]

    existing_payload = load_index(index_path)
    records = existing_payload.get("papers", {}) if existing_payload else {}

    model_dict = create_model_dict()
    converter = PdfConverter(artifact_dict=model_dict)

    processed = 0
    skipped = 0
    failures = 0

    for pdf_path in pdf_paths:
        pdf_name = pdf_path.name
        stem = pdf_path.stem
        md_path = output_dir / f"{stem}.md"
        txt_path = output_dir / f"{stem}.txt"
        current = records.get(pdf_name, {})

        if (
            md_path.exists()
            and txt_path.exists()
        ):
            text_length = txt_path.stat().st_size
            records[pdf_name] = {
                "status": "success",
                "pdf_path": str(pdf_path),
                "markdown_path": str(md_path),
                "text_path": str(txt_path),
                "extraction_method": current.get("extraction_method", "marker"),
                "text_length_chars": current.get("text_length_chars", text_length),
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "updated_at": current.get("updated_at", utc_now()),
            }
            write_index(index_path, records)
            skipped += 1
            continue

        method = "marker"
        started_at = utc_now()
        try:
            markdown, text = extract_with_marker(converter, pdf_path)
        except Exception as marker_error:
            try:
                method = "pypdf_fallback"
                markdown, text = extract_with_pypdf(pdf_path)
            except Exception as fallback_error:
                failures += 1
                failure_record = {
                    "pdf_filename": pdf_name,
                    "pdf_path": str(pdf_path),
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                    "status": "failed",
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "marker_error": f"{type(marker_error).__name__}: {marker_error}",
                    "fallback_error": f"{type(fallback_error).__name__}: {fallback_error}",
                    "traceback": traceback.format_exc(),
                }
                records[pdf_name] = {
                    "status": "failed",
                    "pdf_path": str(pdf_path),
                    "markdown_path": str(md_path),
                    "text_path": str(txt_path),
                    "extraction_method": "failed",
                    "updated_at": utc_now(),
                }
                append_failure(fail_log, failure_record)
                write_index(index_path, records)
                print(f"FAIL {pdf_name}")
                continue

        md_path.write_text(markdown, encoding="utf-8")
        txt_path.write_text(text, encoding="utf-8")
        records[pdf_name] = {
            "status": "success",
            "pdf_path": str(pdf_path),
            "markdown_path": str(md_path),
            "text_path": str(txt_path),
            "extraction_method": method,
            "text_length_chars": len(text),
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "updated_at": utc_now(),
        }
        write_index(index_path, records)
        processed += 1
        print(f"OK   {pdf_name} [{method}]")

    print(
        json.dumps(
            {
                "archive": str(archive_path),
                "pdf_count": len(pdf_paths),
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "processed": processed,
                "skipped": skipped,
                "failures": failures,
                "index": str(index_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
