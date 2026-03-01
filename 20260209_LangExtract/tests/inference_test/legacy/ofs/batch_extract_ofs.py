#!/usr/bin/env python3
"""Batch extract SC qubit papers from all OFS blog posts.

Features:
- Incremental JSONL output (safe to kill and resume)
- Skips already processed URLs
- Robust error handling with retries
- Progress tracking
"""

import json
import os
import re
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL")
API_TOKEN = os.getenv("API_TOKEN")
MODEL_ID = os.getenv("MODEL_ID", "zai-org/GLM-4.7-FP8")

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"
OUTPUT_FILE = OUTPUT_DIR / "ofs_extractions.jsonl"
GITHUB_API = "https://api.github.com"
RAW_BASE = "https://raw.githubusercontent.com/jwt625/jwt625.github.io/master/_posts"

TOOL = {
    "type": "function",
    "function": {
        "name": "extract_papers",
        "description": "Extract papers about superconducting circuits and qubits",
        "parameters": {
            "type": "object",
            "properties": {
                "papers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "url": {"type": "string"},
                            "authors": {"type": "string"},
                            "year": {"type": "integer"},
                            "relevance": {"type": "string"}
                        },
                        "required": ["title", "url"]
                    }
                },
                "has_sc_qubit_content": {"type": "boolean"},
                "summary": {"type": "string"}
            },
            "required": ["papers", "has_sc_qubit_content"]
        }
    }
}


def load_processed() -> set[str]:
    """Load already processed filenames from JSONL."""
    processed = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        processed.add(record.get("filename", ""))
                    except json.JSONDecodeError:
                        continue
    return processed


def list_ofs_posts() -> list[str]:
    """List all OFS blog post filenames from GitHub."""
    url = f"{GITHUB_API}/repos/jwt625/jwt625.github.io/contents/_posts"
    resp = httpx.get(url, timeout=30.0)
    resp.raise_for_status()
    
    files = resp.json()
    # Filter for OFS posts (weekly-OFS-*.md pattern)
    ofs_files = [f["name"] for f in files if re.match(r".*weekly-OFS-\d+\.md$", f["name"])]
    return sorted(ofs_files)


def fetch_content(filename: str) -> str:
    """Fetch raw content of a blog post."""
    url = f"{RAW_BASE}/{filename}"
    resp = httpx.get(url, timeout=30.0)
    resp.raise_for_status()
    return resp.text


def extract_papers(content: str, timeout: float = 300.0) -> dict:
    """Send extraction request to the model."""
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert in superconducting quantum computing. "
                    "Extract ONLY papers that are DIRECTLY about superconducting circuits and qubits "
                    "(Josephson junctions, transmon, fluxonium, cQED, qubit coherence, etc). "
                    "Do NOT include papers that are merely tangentially related. "
                    "If no SC qubit papers exist, set has_sc_qubit_content to false and return empty papers array."
                )
            },
            {
                "role": "user", 
                "content": f"Extract SC qubit related papers from this blog post:\n\n{content}"
            }
        ],
        "tools": [TOOL],
        "tool_choice": {"type": "function", "function": {"name": "extract_papers"}},
        "max_tokens": 8192,
        "temperature": 0.1
    }
    
    resp = httpx.post(
        f"{API_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout
    )
    resp.raise_for_status()
    return resp.json()


def parse_response(response: dict) -> dict | None:
    """Parse the model response to extract papers."""
    try:
        tool_calls = response.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
        if tool_calls:
            return json.loads(tool_calls[0]["function"]["arguments"])
    except (json.JSONDecodeError, KeyError, IndexError):
        pass
    return None


def append_result(record: dict):
    """Append a result to the JSONL file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def process_file(filename: str, max_retries: int = 3) -> dict:
    """Process a single file with retries."""
    record = {
        "filename": filename,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "pending",
        "papers": [],
        "has_sc_qubit_content": False,
        "summary": None,
        "error": None,
        "duration_seconds": None,
        "tokens": None
    }

    for attempt in range(max_retries):
        try:
            start = time.time()

            # Fetch content
            content = fetch_content(filename)
            record["content_length"] = len(content)

            # Extract papers
            response = extract_papers(content)
            elapsed = time.time() - start
            record["duration_seconds"] = round(elapsed, 2)

            # Parse response
            parsed = parse_response(response)
            if parsed:
                record["papers"] = parsed.get("papers", [])
                record["has_sc_qubit_content"] = parsed.get("has_sc_qubit_content", False)
                record["summary"] = parsed.get("summary")
                record["status"] = "success"
            else:
                record["status"] = "parse_error"
                record["error"] = "Failed to parse model response"

            # Token usage
            usage = response.get("usage", {})
            if usage:
                record["tokens"] = {
                    "prompt": usage.get("prompt_tokens", 0),
                    "completion": usage.get("completion_tokens", 0),
                    "total": usage.get("total_tokens", 0)
                }

            return record

        except httpx.TimeoutException as e:
            record["error"] = f"Timeout (attempt {attempt + 1}): {str(e)}"
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))  # Backoff
        except httpx.HTTPStatusError as e:
            record["error"] = f"HTTP {e.response.status_code}: {str(e)}"
            if e.response.status_code >= 500 and attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                break
        except Exception as e:
            record["error"] = f"{type(e).__name__}: {str(e)}"
            break

    record["status"] = "error"
    return record


def main():
    """Main batch processing function."""
    print("=" * 60)
    print("OFS Blog SC Qubit Paper Extraction")
    print("=" * 60)

    # Load already processed
    processed = load_processed()
    print(f"Already processed: {len(processed)} files")

    # List all OFS posts
    print("Fetching OFS blog post list...")
    all_files = list_ofs_posts()
    print(f"Found {len(all_files)} OFS blog posts")

    # Filter to unprocessed
    to_process = [f for f in all_files if f not in processed]
    print(f"To process: {len(to_process)} files")
    print()

    if not to_process:
        print("Nothing to process. All done!")
        return

    # Process each file
    total_papers = 0
    for i, filename in enumerate(to_process, 1):
        print(f"[{i}/{len(to_process)}] Processing: {filename}")

        record = process_file(filename)
        append_result(record)

        # Report
        if record["status"] == "success":
            n_papers = len(record["papers"])
            total_papers += n_papers
            has_content = "✓" if record["has_sc_qubit_content"] else "✗"
            print(f"    {has_content} {n_papers} papers, {record['duration_seconds']}s")
        else:
            print(f"    ✗ Error: {record['error']}")

        # Small delay between requests
        if i < len(to_process):
            time.sleep(1)

    print()
    print("=" * 60)
    print(f"Batch complete. Extracted {total_papers} papers from {len(to_process)} posts.")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
