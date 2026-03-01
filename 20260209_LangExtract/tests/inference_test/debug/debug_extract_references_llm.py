#!/usr/bin/env python3
"""Phase 2 Test: Extract structured reference data from a single document using LLM.

Tests the GLM-4.7 model's ability to parse raw reference text into structured data
suitable for deduplication, graph building, and citation statistics.

Output fields required for downstream phases:
- Phase 3 (Dedup): doi, arxiv_id, title, year, authors (for matching)
- Phase 4 (Stats): All fields for ranking and download list generation
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL")
API_TOKEN = os.getenv("API_TOKEN")
MODEL_ID = os.getenv("MODEL_ID", "zai-org/GLM-4.7-FP8")

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
REFERENCE_SECTIONS_FILE = OUTPUT_DIR / "reference_sections.jsonl"

# Tool definition for structured reference extraction
EXTRACT_REFS_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_references",
        "description": "Parse raw reference text into structured bibliographic data",
        "parameters": {
            "type": "object",
            "properties": {
                "references": {
                    "type": "array",
                    "description": "List of parsed references",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ref_num": {
                                "type": "integer",
                                "description": "Reference number in the document (1, 2, 3, ...)"
                            },
                            "title": {
                                "type": "string",
                                "description": "Paper/book title (required for matching)"
                            },
                            "authors": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of author names (Last, First or F. Last format)"
                            },
                            "year": {
                                "type": "integer",
                                "description": "Publication year (4-digit)"
                            },
                            "journal": {
                                "type": "string",
                                "description": "Journal name or conference name"
                            },
                            "volume": {
                                "type": "string",
                                "description": "Volume number"
                            },
                            "pages": {
                                "type": "string",
                                "description": "Page numbers (e.g., '123-456' or '15023')"
                            },
                            "doi": {
                                "type": "string",
                                "description": "DOI identifier (e.g., '10.1038/ncomms12345')"
                            },
                            "arxiv_id": {
                                "type": "string",
                                "description": "arXiv identifier (e.g., '2104.12345' or 'quant-ph/0512345')"
                            },
                            "url": {
                                "type": "string",
                                "description": "URL if no DOI/arXiv available"
                            }
                        },
                        "required": ["ref_num", "title", "authors", "year"]
                    }
                },
                "total_count": {
                    "type": "integer",
                    "description": "Total number of references parsed"
                },
                "parse_notes": {
                    "type": "string",
                    "description": "Notes about parsing issues or unusual formats encountered"
                }
            },
            "required": ["references", "total_count"]
        }
    }
}

SYSTEM_PROMPT = """You are a bibliographic data extraction expert. Parse the raw reference text into structured data.

CRITICAL REQUIREMENTS:
1. Extract ALL references, do not skip any
2. For each reference, extract: ref_num, title, authors, year, journal, volume, pages, doi, arxiv_id
3. Authors: list of strings. If more than 5 authors, include first 3 and last 2 only (e.g., ["A. Smith", "B. Jones", "C. Lee", "Y. Wang", "Z. Chen"])
4. DOI format: just the identifier (e.g., "10.1038/ncomms12345"), not the full URL
5. arXiv format: just the ID (e.g., "2104.12345" or "quant-ph/0512345")
6. If a field is not present in the text, omit it or set to null
7. Handle varied citation styles: Nature, APS, IEEE, thesis bibliographies

After analyzing, you MUST call extract_references with your findings."""


def load_reference_section(document_id: str) -> Optional[dict]:
    """Load a specific document's reference section from the JSONL file."""
    if not REFERENCE_SECTIONS_FILE.exists():
        print(f"Error: {REFERENCE_SECTIONS_FILE} not found. Run phase1_extract_reference_sections.py first.")
        return None
    
    with open(REFERENCE_SECTIONS_FILE) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("document_id") == document_id:
                return record
    return None


def chunk_references(references_text: str, max_refs_per_chunk: int = 10) -> list:
    """Split references into chunks by lines. Each line is one reference."""
    lines = [l for l in references_text.split('\n') if l.strip()]
    chunks = []
    for i in range(0, len(lines), max_refs_per_chunk):
        chunk = '\n'.join(lines[i:i + max_refs_per_chunk])
        chunks.append(chunk)
    return chunks


def extract_references_single(references_text: str, timeout: float = 600.0) -> dict:
    """Send reference text to LLM for structured extraction (single chunk)."""
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Parse these references into structured data:\n\n{references_text}"}
        ],
        "tools": [EXTRACT_REFS_TOOL],
        "tool_choice": {"type": "function", "function": {"name": "extract_references"}},
        "max_tokens": 8192,
        "temperature": 0.1
    }

    start_time = time.time()

    with httpx.Client(timeout=timeout) as client:
        resp = client.post(
            f"{API_URL}/chat/completions",
            headers=headers,
            json=payload
        )

    elapsed = time.time() - start_time
    print(f"  Response received in {elapsed:.1f}s")

    resp.raise_for_status()
    return resp.json()


def extract_references(references_text: str, max_refs_per_chunk: int = 10, timeout: float = 600.0) -> dict:
    """Extract references, chunking if necessary for large reference lists.

    Always chunks into groups of max_refs_per_chunk (default 10) references.
    This ensures each call stays well within token limits.
    """
    text_len = len(references_text)

    print(f"Sending request to {API_URL}/chat/completions...")
    print(f"Model: {MODEL_ID}, Timeout: {timeout}s")
    print(f"Reference text: {text_len} chars")

    # Always chunk - 10 refs per chunk is safe for token limits
    chunks = chunk_references(references_text, max_refs_per_chunk)

    if len(chunks) == 1:
        # Single chunk - process directly
        return extract_references_single(references_text, timeout)

    print(f"Splitting into {len(chunks)} chunks (~{max_refs_per_chunk} refs each)")

    all_refs = []
    total_tokens = {"prompt": 0, "completion": 0, "total": 0}
    parse_notes = []

    for i, chunk in enumerate(chunks):
        print(f"  Processing chunk {i+1}/{len(chunks)}...")
        response = extract_references_single(chunk, timeout)
        parsed = parse_response(response)

        if parsed:
            all_refs.extend(parsed.get("references", []))
            if parsed.get("parse_notes"):
                parse_notes.append(f"Chunk {i+1}: {parsed['parse_notes']}")

        # Accumulate token usage
        usage = response.get("usage", {})
        total_tokens["prompt"] += usage.get("prompt_tokens", 0)
        total_tokens["completion"] += usage.get("completion_tokens", 0)
        total_tokens["total"] += usage.get("total_tokens", 0)

    # Construct combined response
    combined = {
        "choices": [{
            "message": {
                "tool_calls": [{
                    "function": {
                        "name": "extract_references",
                        "arguments": json.dumps({
                            "references": all_refs,
                            "total_count": len(all_refs),
                            "parse_notes": "; ".join(parse_notes) if parse_notes else None
                        })
                    }
                }]
            }
        }],
        "usage": total_tokens,
        "_chunked": True,
        "_num_chunks": len(chunks)
    }

    return combined


def parse_response(response: dict) -> Optional[dict]:
    """Parse the LLM response to extract structured references."""
    try:
        tool_calls = response.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
        if tool_calls:
            return json.loads(tool_calls[0]["function"]["arguments"])
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Parse error: {e}")
    return None


def list_available_documents(limit: int = 20) -> list:
    """List document IDs available for processing."""
    if not REFERENCE_SECTIONS_FILE.exists():
        return []

    docs = []
    with open(REFERENCE_SECTIONS_FILE) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") == "success":
                docs.append({
                    "id": record["document_id"],
                    "ref_count": record.get("ref_count_estimate", 0),
                    "chars": record.get("section_char_count", 0)
                })

    # Sort by ref_count for interesting test cases
    docs.sort(key=lambda x: x["ref_count"], reverse=True)
    return docs[:limit]


def main():
    """Main test function."""
    import sys

    print("=" * 70)
    print("Phase 2 Test: LLM-Based Structured Reference Extraction")
    print("=" * 70)

    # Get document ID from command line or use default
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list":
            print("\nAvailable documents (sorted by ref count):")
            for doc in list_available_documents(30):
                print(f"  {doc['id']}: ~{doc['ref_count']} refs, {doc['chars']} chars")
            return
        document_id = sys.argv[1]
    else:
        # Default: use a medium-sized document for testing
        document_id = "altoe_2022_localization_reduction_circuit_losses"

    print(f"\nDocument: {document_id}")

    # Load reference section
    record = load_reference_section(document_id)
    if not record:
        print(f"Error: Document '{document_id}' not found in reference_sections.jsonl")
        print("Run: python tests/inference_test/pipeline/phase1_extract_reference_sections.py first")
        print("Or use --list to see available documents")
        return

    if record.get("status") != "success":
        print(f"Error: Document has status '{record.get('status')}'")
        return

    references_text = record.get("references_text", "")
    print(f"Reference section: {len(references_text)} chars, ~{record.get('ref_count_estimate', 0)} refs")
    print(f"Detection method: {record.get('detection_method', 'unknown')}")
    print()

    # Preview first 500 chars
    print("Preview (first 500 chars):")
    print("-" * 40)
    print(references_text[:500])
    print("-" * 40)
    print()

    # Extract structured references
    response = extract_references(references_text)

    # Save raw response
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"test_refs_{document_id}.json"
    with open(output_file, "w") as f:
        json.dump(response, f, indent=2)
    print(f"Raw response saved to: {output_file}")

    # Parse and display results
    parsed = parse_response(response)

    if not parsed:
        print("Error: Failed to parse LLM response")
        print("Check raw response file for details")
        return

    # Save parsed output
    parsed_file = OUTPUT_DIR / f"test_refs_{document_id}_parsed.json"
    with open(parsed_file, "w") as f:
        json.dump({"document_id": document_id, **parsed}, f, indent=2)
    print(f"Parsed output saved to: {parsed_file}")

    print()
    print("=" * 70)
    print("EXTRACTION RESULTS")
    print("=" * 70)

    refs = parsed.get("references", [])
    print(f"Total references extracted: {parsed.get('total_count', len(refs))}")

    if parsed.get("parse_notes"):
        print(f"Parse notes: {parsed['parse_notes']}")

    # Summary statistics
    has_doi = sum(1 for r in refs if r.get("doi"))
    has_arxiv = sum(1 for r in refs if r.get("arxiv_id"))
    has_journal = sum(1 for r in refs if r.get("journal"))
    has_year = sum(1 for r in refs if r.get("year"))

    print(f"\nField coverage:")
    print(f"  - Has DOI: {has_doi}/{len(refs)} ({100*has_doi/len(refs):.0f}%)" if refs else "  - No refs")
    print(f"  - Has arXiv: {has_arxiv}/{len(refs)} ({100*has_arxiv/len(refs):.0f}%)" if refs else "")
    print(f"  - Has journal: {has_journal}/{len(refs)} ({100*has_journal/len(refs):.0f}%)" if refs else "")
    print(f"  - Has year: {has_year}/{len(refs)} ({100*has_year/len(refs):.0f}%)" if refs else "")

    # Show first 5 references
    print(f"\nFirst 5 references:")
    print("-" * 40)
    for ref in refs[:5]:
        print(f"\n[{ref.get('ref_num', '?')}] {ref.get('title', 'No title')}")
        if ref.get("authors"):
            authors = ref["authors"][:3]
            if len(ref["authors"]) > 3:
                authors.append("et al.")
            print(f"    Authors: {', '.join(authors)}")
        if ref.get("year"):
            print(f"    Year: {ref['year']}")
        if ref.get("journal"):
            print(f"    Journal: {ref['journal']}")
        if ref.get("doi"):
            print(f"    DOI: {ref['doi']}")
        if ref.get("arxiv_id"):
            print(f"    arXiv: {ref['arxiv_id']}")

    # Token usage
    usage = response.get("usage", {})
    if usage:
        print()
        print("-" * 40)
        print(f"Token usage: {usage.get('prompt_tokens', 0)} prompt + "
              f"{usage.get('completion_tokens', 0)} completion = "
              f"{usage.get('total_tokens', 0)} total")


if __name__ == "__main__":
    main()
