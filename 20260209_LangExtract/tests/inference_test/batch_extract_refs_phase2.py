#!/usr/bin/env python3
"""Phase 2 Batch: Extract structured references from all documents.

Features:
- 5 concurrent API calls
- Resume-safe: skips already processed documents
- Progress logging
"""

from __future__ import annotations

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
REFERENCE_SECTIONS_FILE = OUTPUT_DIR / "reference_sections.jsonl"
PHASE2_OUTPUT_FILE = OUTPUT_DIR / "phase2_extracted_refs.jsonl"
PROGRESS_FILE = OUTPUT_DIR / "phase2_progress.json"

# Timestamped log file
RUN_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
LOG_FILE = OUTPUT_DIR / f"phase2_run_{RUN_TIMESTAMP}.log"

MAX_CONCURRENT = 5
MAX_REFS_PER_CHUNK = 10
MAX_RETRIES = 3
TIMEOUT = 600.0

# Import tool and prompt from test script
from test_extract_references_llm import EXTRACT_REFS_TOOL, SYSTEM_PROMPT


def load_progress() -> set:
    """Load set of already processed document IDs."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            data = json.load(f)
            return set(data.get("completed", []))
    return set()


def save_progress(completed: set):
    """Save progress to file."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"completed": list(completed), "last_update": time.strftime("%Y-%m-%d %H:%M:%S")}, f)


def load_all_documents() -> list:
    """Load all successful reference sections."""
    docs = []
    with open(REFERENCE_SECTIONS_FILE) as f:
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
    # Add file handler with timestamp
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 70)
    logger.info("Phase 2: Batch Reference Extraction")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info("=" * 70)

    docs = load_all_documents()
    completed = load_progress()
    pending = [d for d in docs if d["document_id"] not in completed]

    logger.info(f"Total documents: {len(docs)}")
    logger.info(f"Already completed: {len(completed)}")
    logger.info(f"Pending: {len(pending)}")
    logger.info(f"Concurrent calls: {MAX_CONCURRENT}")
    logger.info("=" * 70)

    if not pending:
        logger.info("All documents already processed!")
        return

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    headers = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=TIMEOUT, headers=headers) as client:
        for i, doc in enumerate(pending):
            doc_id = doc["document_id"]
            logger.info(f"[{i+1}/{len(pending)}] Processing: {doc_id}")
            
            start = time.time()
            result = await process_document(client, doc, semaphore)
            elapsed = time.time() - start
            
            # Append to output file
            with open(PHASE2_OUTPUT_FILE, "a") as f:
                f.write(json.dumps(result) + "\n")
            
            # Update progress
            completed.add(doc_id)
            save_progress(completed)
            
            logger.info(f"  Extracted {result['ref_count']} refs in {elapsed:.1f}s")

    logger.info("=" * 70)
    logger.info(f"Phase 2 complete! Processed {len(pending)} documents.")
    logger.info(f"Output: {PHASE2_OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())

