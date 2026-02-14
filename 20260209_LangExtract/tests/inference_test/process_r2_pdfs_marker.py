#!/usr/bin/env python3
"""
Concurrent PDF to Markdown Processor using Marker - R2 Batch

Processes the R2 papers batch (238 PDFs) using Marker.
Features:
- Processes all PDFs in papers_r2 directory
- Concurrent processing with configurable workers
- Safe stop/resume: tracks progress in JSONL file
- Graceful shutdown on Ctrl+C
"""

import json
import os
import signal
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import filelock

# Paths relative to this script
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RAW_DOCS_DIR = PROJECT_ROOT / "semiconductor_processing_dataset" / "raw_documents" / "papers_r2" / "papers"
OUTPUT_BASE = PROJECT_ROOT / "semiconductor_processing_dataset" / "processed_documents" / "text_extracted"
OUTPUT_DIR = OUTPUT_BASE / "marker_r2"
PROGRESS_FILE = OUTPUT_BASE / "marker_r2_processing_progress.jsonl"
LOCK_FILE = OUTPUT_BASE / "marker_r2_processing.lock"


@dataclass
class ProcessingResult:
    document_id: str
    source_path: str
    success: bool
    processing_time_seconds: float
    input_size_bytes: int
    output_size_bytes: Optional[int]
    page_count: Optional[int]
    error_message: Optional[str]
    timestamp: str


def load_completed_documents() -> set[str]:
    """Load set of already-processed document IDs from progress file."""
    completed = set()
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if record.get("success"):
                            completed.add(record["document_id"])
                    except json.JSONDecodeError:
                        continue
    return completed


def get_all_pdfs() -> list[tuple[str, Path]]:
    """Get all PDFs from the R2 papers directory."""
    pdfs = []
    for pdf_path in sorted(RAW_DOCS_DIR.glob("*.pdf")):
        doc_id = pdf_path.stem  # filename without extension
        pdfs.append((doc_id, pdf_path))
    return pdfs


def save_result(result: ProcessingResult):
    """Append a result to the progress file with file locking."""
    lock = filelock.FileLock(str(LOCK_FILE), timeout=30)
    with lock:
        with open(PROGRESS_FILE, "a") as f:
            f.write(json.dumps(asdict(result)) + "\n")


def convert_single_pdf(doc_id: str, pdf_path: str, gpu_id: int = 0) -> ProcessingResult:
    """Convert a single PDF using Marker. Runs in subprocess."""
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    pdf_path = Path(pdf_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"{doc_id}.md"

    input_size = pdf_path.stat().st_size if pdf_path.exists() else 0

    start_time = time.time()
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        model_dict = create_model_dict()
        converter = PdfConverter(artifact_dict=model_dict)
        result = converter(str(pdf_path))
        
        markdown_text = result.markdown
        page_count = len(result.pages) if hasattr(result, 'pages') else None
        
        output_file.write_text(markdown_text, encoding="utf-8")
        output_size = output_file.stat().st_size
        
        elapsed = time.time() - start_time
        return ProcessingResult(
            document_id=doc_id, source_path=str(pdf_path), success=True,
            processing_time_seconds=round(elapsed, 2),
            input_size_bytes=input_size, output_size_bytes=output_size,
            page_count=page_count, error_message=None,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        elapsed = time.time() - start_time
        return ProcessingResult(
            document_id=doc_id, source_path=str(pdf_path), success=False,
            processing_time_seconds=round(elapsed, 2),
            input_size_bytes=input_size, output_size_bytes=None,
            page_count=None, error_message=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            timestamp=datetime.now().isoformat()
        )


# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global shutdown_requested
    if shutdown_requested:
        print("\n\nForce quit requested. Exiting immediately...")
        sys.exit(1)
    shutdown_requested = True
    print("\n\nShutdown requested. Waiting for current jobs to finish...")
    print("Press Ctrl+C again to force quit.")


def run_processing(max_workers: int = 2, limit: int = None, dry_run: bool = False, num_gpus: int = 2):
    """Run the full processing pipeline."""
    global shutdown_requested

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("Scanning for PDFs...")
    all_pdfs = get_all_pdfs()
    completed = load_completed_documents()

    pending = [(doc_id, pdf_path) for doc_id, pdf_path in all_pdfs if doc_id not in completed]

    if limit:
        pending = pending[:limit]

    print(f"Total PDFs found: {len(all_pdfs)}")
    print(f"Already processed: {len(completed)}")
    print(f"Pending: {len(pending)}")

    if dry_run:
        print("\n[DRY RUN] Would process:")
        for doc_id, _ in pending[:20]:
            print(f"  - {doc_id}")
        if len(pending) > 20:
            print(f"  ... and {len(pending) - 20} more")
        return

    if not pending:
        print("\nAll documents already processed!")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting processing with {max_workers} workers across {num_gpus} GPUs...")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Progress file: {PROGRESS_FILE}")
    print("-" * 60)

    success_count = 0
    error_count = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_doc = {}
        for idx, (doc_id, pdf_path) in enumerate(pending):
            gpu_id = idx % num_gpus
            future = executor.submit(convert_single_pdf, doc_id, str(pdf_path), gpu_id)
            future_to_doc[future] = (doc_id, pdf_path)

        for future in as_completed(future_to_doc):
            if shutdown_requested:
                print("\nCancelling pending tasks...")
                for f in future_to_doc:
                    f.cancel()
                break

            doc_id, _ = future_to_doc[future]
            try:
                result = future.result()
                save_result(result)

                if result.success:
                    success_count += 1
                    print(f"✓ {result.document_id}: {result.processing_time_seconds}s, "
                          f"{result.output_size_bytes/1024:.1f}KB")
                else:
                    error_count += 1
                    error_short = result.error_message.split('\n')[0][:80] if result.error_message else "Unknown"
                    print(f"✗ {result.document_id}: {error_short}")
            except Exception as e:
                error_count += 1
                print(f"✗ {doc_id}: Future exception: {e}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Processing complete!")
    print(f"  Succeeded: {success_count}")
    print(f"  Failed: {error_count}")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    if (success_count + error_count) > 0:
        print(f"  Average: {elapsed/(success_count+error_count):.1f}s per document")


def show_status():
    """Show current processing status."""
    all_pdfs = get_all_pdfs()
    completed = load_completed_documents()

    results = []
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    successes = [r for r in results if r.get("success")]
    failures = [r for r in results if not r.get("success")]

    total_time = sum(r.get("processing_time_seconds", 0) for r in successes)
    total_output = sum(r.get("output_size_bytes", 0) or 0 for r in successes)

    print(f"Total PDFs: {len(all_pdfs)}")
    print(f"Completed successfully: {len(successes)}")
    print(f"Failed: {len(failures)}")
    print(f"Remaining: {len(all_pdfs) - len(completed)}")
    print(f"Total processing time: {total_time/3600:.1f} hours")
    print(f"Total output size: {total_output/1024/1024:.1f} MB")

    if failures:
        print(f"\nFailed documents:")
        for r in failures[:10]:
            print(f"  - {r['document_id']}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="R2 Batch PDF to Markdown Processor using Marker")
    parser.add_argument("--workers", "-w", type=int, default=2,
                        help="Number of parallel workers (default: 2)")
    parser.add_argument("--gpus", "-g", type=int, default=2,
                        help="Number of GPUs to use (default: 2)")
    parser.add_argument("--limit", "-l", type=int, default=None,
                        help="Limit number of documents to process")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Show what would be processed without doing it")
    parser.add_argument("--status", "-s", action="store_true",
                        help="Show current processing status")
    args = parser.parse_args()

    if args.status:
        show_status()
    else:
        run_processing(max_workers=args.workers, limit=args.limit, dry_run=args.dry_run, num_gpus=args.gpus)

