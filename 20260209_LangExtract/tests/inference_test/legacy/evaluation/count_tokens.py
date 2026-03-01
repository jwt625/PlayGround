#!/usr/bin/env python3
"""
Count tokens in all R1 and R2 markdown files.

Uses tiktoken locally for fast counting (cl100k_base encoding, similar to GLM).
"""

import json
from pathlib import Path

import tiktoken

# Directories (relative to script location)
R1_DIR = Path("../../semiconductor_processing_dataset/processed_documents/text_extracted/marker")
R2_DIR = Path("../../semiconductor_processing_dataset/processed_documents_R2/text_extracted/marker_r2")


def count_tokens(text: str, enc: tiktoken.Encoding) -> int:
    """Count tokens using tiktoken."""
    return len(enc.encode(text))


def process_files(files: list[Path], enc: tiktoken.Encoding, label: str) -> tuple[int, int, list[dict]]:
    """Process all files and return (total_tokens, total_chars, per_file_stats)."""
    total_tokens = 0
    total_chars = 0
    per_file = []

    for i, fpath in enumerate(files, 1):
        text = fpath.read_text(encoding="utf-8", errors="replace")
        char_count = len(text)
        tokens = count_tokens(text, enc)

        total_tokens += tokens
        total_chars += char_count
        per_file.append({
            "file": fpath.name,
            "tokens": tokens,
            "chars": char_count
        })

        if i % 50 == 0 or i == len(files):
            print(f"  [{label}] {i}/{len(files)} files, {total_tokens:,} tokens so far...")

    return total_tokens, total_chars, per_file


def main():
    script_dir = Path(__file__).resolve().parents[2]
    r1_dir = (script_dir / R1_DIR).resolve()
    r2_dir = (script_dir / R2_DIR).resolve()

    print(f"R1 directory: {r1_dir}")
    print(f"R2 directory: {r2_dir}")

    r1_files = sorted(r1_dir.glob("*.md"))
    r2_files = sorted(r2_dir.glob("*.md"))

    print(f"\nFound {len(r1_files)} R1 files, {len(r2_files)} R2 files")
    print(f"Total: {len(r1_files) + len(r2_files)} markdown files\n")

    # Use cl100k_base (GPT-4/ChatGPT tokenizer, similar token density to GLM)
    enc = tiktoken.get_encoding("cl100k_base")
    print(f"Using tiktoken encoding: cl100k_base\n")

    # Process R1
    print("Processing R1 files...")
    r1_tokens, r1_chars, r1_stats = process_files(r1_files, enc, "R1")

    # Process R2
    print("\nProcessing R2 files...")
    r2_tokens, r2_chars, r2_stats = process_files(r2_files, enc, "R2")

    # Summary
    total_tokens = r1_tokens + r2_tokens
    total_chars = r1_chars + r2_chars
    total_files = len(r1_files) + len(r2_files)

    print("\n" + "=" * 60)
    print("TOKEN COUNT SUMMARY (tiktoken cl100k_base)")
    print("=" * 60)
    print(f"R1: {len(r1_files):,} files, {r1_tokens:,} tokens, {r1_chars:,} chars")
    print(f"R2: {len(r2_files):,} files, {r2_tokens:,} tokens, {r2_chars:,} chars")
    print(f"TOTAL: {total_files:,} files, {total_tokens:,} tokens, {total_chars:,} chars")
    print(f"Avg tokens/file: {total_tokens // total_files:,}")
    print(f"Avg chars/token: {total_chars / total_tokens:.2f}" if total_tokens else "N/A")

    # Save detailed stats
    output = {
        "encoding": "cl100k_base",
        "r1": {"files": len(r1_files), "tokens": r1_tokens, "chars": r1_chars},
        "r2": {"files": len(r2_files), "tokens": r2_tokens, "chars": r2_chars},
        "total": {"files": total_files, "tokens": total_tokens, "chars": total_chars},
        "r1_per_file": r1_stats,
        "r2_per_file": r2_stats
    }

    output_path = Path(__file__).resolve().parents[2] / "output" / "token_counts.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed stats saved to: {output_path}")


if __name__ == "__main__":
    main()
