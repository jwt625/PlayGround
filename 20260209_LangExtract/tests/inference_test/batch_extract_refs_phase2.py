#!/usr/bin/env python3
"""Phase 2 Batch: Extract structured references from all documents.

Features:
- 5 concurrent API calls
- Resume-safe: skips already processed documents
- Progress logging
- CLI arguments for input/output file paths

Usage:
  # R1 (default):
  python batch_extract_refs_phase2.py

  # R2:
  python batch_extract_refs_phase2.py \
    --input-file tests/inference_test/output/reference_sections_r2.jsonl \
    --output-file tests/inference_test/output/phase2_extracted_refs_r2.jsonl \
    --progress-file tests/inference_test/output/phase2_progress_r2.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

load_dotenv()

API_URL = os.getenv("API_URL")
API_TOKEN = os.getenv("API_TOKEN")
MODEL_ID = os.getenv("MODEL_ID", "zai-org/GLM-4.7-FP8")

OUTPUT_DIR = Path(__file__).parent / "output"

# Default paths (R1)
DEFAULT_INPUT_FILE = OUTPUT_DIR / "reference_sections.jsonl"
DEFAULT_OUTPUT_FILE = OUTPUT_DIR / "phase2_extracted_refs.jsonl"
DEFAULT_PROGRESS_FILE = OUTPUT_DIR / "phase2_progress.json"

MAX_CONCURRENT = 5
MAX_REFS_PER_CHUNK = 10
MAX_RETRIES = 3
TIMEOUT = 600.0

# Import tool and prompt from test script
from test_extract_references_llm import EXTRACT_REFS_TOOL, SYSTEM_PROMPT


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: Extract structured references using LLM")
    parser.add_argument(
        "--input-file", "-i",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Input JSONL file with reference sections (default: reference_sections.jsonl)"
    )
    parser.add_argument(
        "--output-file", "-o",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Output JSONL file for extracted references (default: phase2_extracted_refs.jsonl)"
    )
    parser.add_argument(
        "--progress-file", "-p",
        type=Path,
        default=DEFAULT_PROGRESS_FILE,
        help="Progress JSON file for resume support (default: phase2_progress.json)"
    )
    parser.add_argument(
        "--max-concurrent", "-c",
        type=int,
        default=MAX_CONCURRENT,
        help=f"Maximum concurrent API calls (default: {MAX_CONCURRENT})"
    )
    return parser.parse_args()


def load_progress(progress_file: Path) -> set:
    """Load set of already processed document IDs."""
    if progress_file.exists():
        with open(progress_file) as f:
            data = json.load(f)
            return set(data.get("completed", []))
    return set()


def save_progress(completed: set, progress_file: Path):
    """Save progress to file."""
    with open(progress_file, "w") as f:
        json.dump({"completed": list(completed), "last_update": time.strftime("%Y-%m-%d %H:%M:%S")}, f)


def load_all_documents(input_file: Path) -> list:
    """Load all successful reference sections."""
    docs = []
    with open(input_file) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") == "success":
                docs.append(record)
    return docs


def chunk_references(text: str) -> list:
    """Split references into chunks by lines."""
    lines = [l for l in text.split('\n') if l.strip()]
    chunks = []
    for i in range(0, len(lines), MAX_REFS_PER_CHUNK):
        chunks.append('\n'.join(lines[i:i + MAX_REFS_PER_CHUNK]))
    return chunks


async def extract_chunk(client: httpx.AsyncClient, chunk: str, semaphore: asyncio.Semaphore) -> Optional[dict]:
    """Extract references from a single chunk with retry logic."""
    async with semaphore:
        payload = {
            "model": MODEL_ID,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Parse these references into structured data:\n\n{chunk}"}
            ],
            "tools": [EXTRACT_REFS_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "extract_references"}},
            "max_tokens": 8192,
            "temperature": 0.1
        }

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await client.post(f"{API_URL}/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
                tool_calls = data.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
                if tool_calls:
                    return json.loads(tool_calls[0]["function"]["arguments"])
            except json.JSONDecodeError as e:
                logger.warning(f"Chunk JSON error (attempt {attempt}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(1)  # Brief delay before retry
                    continue
                logger.error(f"Chunk failed after {MAX_RETRIES} retries: {e}")
            except Exception as e:
                logger.error(f"Chunk error: {e}")
                break  # Don't retry on non-JSON errors (network, etc.)
        return None


async def process_document(client: httpx.AsyncClient, doc: dict, semaphore: asyncio.Semaphore) -> dict:
    """Process a single document, extracting all references."""
    doc_id = doc["document_id"]
    text = doc["references_text"]
    chunks = chunk_references(text)
    
    all_refs = []
    for chunk in chunks:
        result = await extract_chunk(client, chunk, semaphore)
        if result:
            all_refs.extend(result.get("references", []))
    
    return {
        "document_id": doc_id,
        "references": all_refs,
        "ref_count": len(all_refs),
        "chunk_count": len(chunks),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }


async def main():
    args = parse_args()

    input_file = args.input_file
    output_file = args.output_file
    progress_file = args.progress_file
    max_concurrent = args.max_concurrent

    # Derive log file name from output file
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = output_file.parent / f"{output_file.stem}_run_{run_timestamp}.log"

    # Add file handler with timestamp
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 70)
    logger.info("Phase 2: Batch Reference Extraction")
    logger.info(f"Input file:    {input_file}")
    logger.info(f"Output file:   {output_file}")
    logger.info(f"Progress file: {progress_file}")
    logger.info(f"Log file:      {log_file}")
    logger.info("=" * 70)

    docs = load_all_documents(input_file)
    completed = load_progress(progress_file)
    pending = [d for d in docs if d["document_id"] not in completed]

    logger.info(f"Total documents: {len(docs)}")
    logger.info(f"Already completed: {len(completed)}")
    logger.info(f"Pending: {len(pending)}")
    logger.info(f"Concurrent calls: {max_concurrent}")
    logger.info("=" * 70)

    if not pending:
        logger.info("All documents already processed!")
        return

    semaphore = asyncio.Semaphore(max_concurrent)
    headers = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=TIMEOUT, headers=headers) as client:
        for i, doc in enumerate(pending):
            doc_id = doc["document_id"]
            logger.info(f"[{i+1}/{len(pending)}] Processing: {doc_id}")

            start = time.time()
            result = await process_document(client, doc, semaphore)
            elapsed = time.time() - start

            # Append to output file
            with open(output_file, "a") as f:
                f.write(json.dumps(result) + "\n")

            # Update progress
            completed.add(doc_id)
            save_progress(completed, progress_file)

            logger.info(f"  Extracted {result['ref_count']} refs in {elapsed:.1f}s")

    logger.info("=" * 70)
    logger.info(f"Phase 2 complete! Processed {len(pending)} documents.")
    logger.info(f"Output: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())

