#!/usr/bin/env python3
"""Phase 1: Extract reference sections from markdown files using regex.

This script identifies and extracts the references/bibliography section from
each markdown file, estimates the reference count, and outputs structured data
for downstream LLM processing.

Improved version with:
- More header patterns (span-embedded, bold combinations)
- Fallback detection for files without headers (pattern clustering)
- Better reference line detection
"""

from __future__ import annotations

import json
import re
import html
import argparse
from pathlib import Path
from typing import Optional, Tuple

# Paths
MARKER_DIR = Path(__file__).parent.parent.parent / "semiconductor_processing_dataset" / "processed_documents" / "text_extracted" / "marker"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_FILE = OUTPUT_DIR / "reference_sections.jsonl"

# Patterns to identify reference section headers
REFERENCE_HEADERS = [
    # Standard markdown headers
    r"^#{1,6}\s*References?\s*(and\s+Notes?)?\s*$",
    r"^#{1,6}\s*Bibliography\s*$",
    r"^#{1,6}\s*REFERENCES?\s*$",
    r"^#{1,6}\s*Literature\s+Cited\s*$",
    r"^#{1,6}\s*Works\s+Cited\s*$",
    r"^#{1,6}\s*Cited\s+Literature\s*$",
    # Bold headers
    r"^\*\*References?\*\*\s*$",
    r"^\*\*Bibliography\*\*\s*$",
    r"^References?\s*$",
    r"^Bibliography\s*$",
    r"^\*\*\s*References?\s*\*\*\s*$",
    r"^\*\*\s*Bibliography\s*\*\*\s*$",
    # Hash + bold combination: # **References** or ## **Bibliography**
    r"^#{1,6}\s*\*\*\s*References?\s*\*\*\s*$",
    r"^#{1,6}\s*\*\*\s*Bibliography\s*\*\*\s*$",
    # With span tags INSIDE header: ## <span...>**Bibliography**
    r"^#{1,6}\s*<span[^>]*>\s*\*\*\s*References?\s*\*\*",
    r"^#{1,6}\s*<span[^>]*>\s*\*\*\s*Bibliography\s*\*\*",
    r"^#{1,6}\s*<span[^>]*>\s*References?",
    r"^#{1,6}\s*<span[^>]*>\s*Bibliography",
    # Span at start of line
    r"^<span[^>]*>\s*#{0,6}\s*\*\*\s*References?",
    r"^<span[^>]*>\s*#{0,6}\s*\*\*\s*Bibliography",
    r"^<span[^>]*>\s*#{0,6}\s*References?",
    r"^<span[^>]*>\s*#{0,6}\s*Bibliography",
    # Subheadings with ####
    r"^####\s*References?\s*$",
    r"^####\s*Supplemental\s+References?\s*$",
    # Numbered headers: ## 10. References / # **10. References**
    r"^#{1,6}\s*\*{0,2}\s*\d+\.?\s*References?\s*\*{0,2}\s*$",
    r"^#{1,6}\s*\*{0,2}\s*\d+\.?\s*Bibliography\s*\*{0,2}\s*$",
    # Underline style headers
    r"^References?\s*\n[=-]+$",
    r"^Bibliography\s*\n[=-]+$",
]

# Patterns to identify end of reference section (next major section)
SECTION_END_PATTERNS = [
    r"^#{1,6}\s+(?!References|Bibliography|REFERENCES|Supplemental\s+Ref)",  # New heading (not refs)
    r"^#{1,6}\s*Appendix",
    r"^#{1,6}\s*Supplementary\s+(?!Ref)",  # Supplementary but not Supplementary References
    r"^#{1,6}\s*Acknowledgements?",
    r"^#{1,6}\s*Author\s+Contributions?",
    r"^#{1,6}\s*Figure\s+S",  # Supplementary figures section
    r"^\*\*\s*Appendix",
    r"^\*\*\s*Acknowledgements?",
]

# Patterns to identify reference lines (for counting and fallback detection)
REF_LINE_PATTERNS = [
    r"^\s*-\s*<span[^>]*>\s*\[\d+\]",    # - <span>[1] style (most common in Marker output)
    r"^\s*-\s*<span[^>]*>\s*[A-Z]",      # - <span>Author style
    r"^\s*-\s*\[\d+\]",                   # - [1] style
    r"^\s*-\s*\d+\.\s",                   # - 1. style
    r"^\s*\[\d+\]\s*[A-Z]",               # [1] Author at line start
    r"^\s*\d+\.\s+[A-Z]",                 # 1. Author style
    r"^<sup>\d+</sup>",                   # <sup>1</sup> style
    r"^\s*<span[^>]*>\s*\[\d+\]",         # <span>[1] style (no dash)
    # Marker variants with closing span before content
    r"^\s*-\s*<span[^>]*>.*?</span>\s*(\[\d+\]|\d+[.)]|[A-Z])",
    r"^\s*<span[^>]*>.*?</span>\s*(\[\d+\]|\d+[.)]|[A-Z])",
    # Escaped/malformed superscript variants
    r"^\s*-\s*<sup>.*?</sup>\s*[A-Z]",
    r"^\s*<sup>.*?</sup>\s*[A-Z]",
]

# Minimum consecutive reference lines to trigger fallback detection
MIN_REF_CLUSTER_SIZE = 3

# Patterns to extract reference number from a line (for sequential detection)
# Try in order: more specific patterns first
REF_NUM_PATTERNS = [
    re.compile(r'\[(\d+)\]'),                     # [1], [2], etc.
    re.compile(r'<sup>(\d+)</sup>'),              # <sup>1</sup>
    re.compile(r'^\s*-?\s*(\d+)\.\s'),            # 1. or - 1. at line start
    re.compile(r'(?:^|[\s\-])(\d+)(?:[\.\]\:\s]|$)'),  # any number at word boundary
]


def is_reference_line(line: str) -> bool:
    """Check if a line looks like a reference entry."""
    stripped = line.strip()
    if not stripped:
        return False
    normalized = html.unescape(stripped)
    normalized = normalized.replace("</span>", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    for pattern in REF_LINE_PATTERNS:
        if re.match(pattern, stripped) or re.match(pattern, normalized):
            return True
    return False


def get_ref_number(line: str) -> Optional[int]:
    """Extract reference number from a line - try multiple patterns."""
    line_norm = html.unescape(line)
    line_norm = re.sub(r"<[^>]+>", " ", line_norm)
    line_norm = re.sub(r"\s+", " ", line_norm).strip()
    for pattern in REF_NUM_PATTERNS:
        match = pattern.search(line_norm)
        if match:
            return int(match.group(1))
    return None


def find_sequential_refs(lines: list, min_sequential: int = 4) -> Tuple[Optional[int], Optional[int]]:
    """Find a section with sequential reference numbers [1], [2], [3], [4]... or 1., 2., 3., 4....

    This is a very robust fallback: if we see 4+ lines with incrementing
    numbers, it's almost certainly a reference section.
    Allows every other row (up to 2 lines gap between references).

    Returns:
        (section_start, section_end) or (None, None) if not found
    """
    # Build list of (line_index, ref_number) for lines that have [N]
    ref_lines = []
    for i, line in enumerate(lines):
        num = get_ref_number(line)
        if num is not None:
            ref_lines.append((i, num))

    if len(ref_lines) < min_sequential:
        return None, None

    # Look for sequences of incrementing numbers on nearby lines
    best_start = None
    best_end = None
    best_count = 0

    i = 0
    while i < len(ref_lines):
        # Try to build a sequence starting from ref_lines[i]
        seq_start_idx = ref_lines[i][0]
        expected_num = ref_lines[i][1]
        seq_count = 1
        last_line_idx = ref_lines[i][0]

        for j in range(i + 1, len(ref_lines)):
            line_idx, num = ref_lines[j]
            # Check if this is the next expected number
            # Allow gaps for blank lines between refs (max 5 lines gap)
            if num == expected_num + 1 and line_idx - last_line_idx <= 5:
                expected_num = num
                seq_count += 1
                last_line_idx = line_idx
            elif num > expected_num + 1 and num <= expected_num + 3 and line_idx - last_line_idx <= 5:
                # Skipped a number or two (maybe combined refs)
                expected_num = num
                seq_count += 1
                last_line_idx = line_idx

        if seq_count >= min_sequential and seq_count > best_count:
            best_count = seq_count
            best_start = seq_start_idx
            best_end = last_line_idx

        i += 1

    if best_start is not None:
        # Extend to end of file or next major section
        for k in range(best_end + 1, len(lines)):
            line = lines[k].strip()
            # Stop if we hit a new section header
            if re.match(r'^#{1,6}\s+[A-Z]', line) and not re.search(r'References?|Bibliography', line, re.IGNORECASE):
                best_end = k
                break
            # Keep extending if line has a ref number or is blank/continuation
            if get_ref_number(lines[k]) is not None or not line or line.startswith('-'):
                best_end = k
        else:
            best_end = len(lines)

        return best_start, best_end

    return None, None


def find_reference_cluster(lines: list, start_search: int = 0) -> Tuple[Optional[int], Optional[int]]:
    """Find a cluster of consecutive reference lines.

    Searches from start_search to end of file for clusters of MIN_REF_CLUSTER_SIZE
    or more consecutive reference-like lines.

    Returns:
        (cluster_start, cluster_end) or (None, None) if not found
    """
    cluster_start = None
    consecutive_refs = 0

    for i in range(start_search, len(lines)):
        if is_reference_line(lines[i]):
            if cluster_start is None:
                cluster_start = i
            consecutive_refs += 1
        else:
            # Allow up to 2 blank/non-ref lines within a cluster
            if cluster_start is not None:
                # Check if next few lines continue the pattern
                lookahead = min(3, len(lines) - i - 1)
                found_more = False
                for j in range(1, lookahead + 1):
                    if i + j < len(lines) and is_reference_line(lines[i + j]):
                        found_more = True
                        break

                if not found_more:
                    # End of cluster
                    if consecutive_refs >= MIN_REF_CLUSTER_SIZE:
                        return cluster_start, i
                    # Reset and keep searching
                    cluster_start = None
                    consecutive_refs = 0

    # Check if we ended with a valid cluster
    if cluster_start is not None and consecutive_refs >= MIN_REF_CLUSTER_SIZE:
        return cluster_start, len(lines)

    return None, None


def find_reference_section(content: str) -> Tuple[Optional[int], Optional[int], Optional[str], str]:
    """Find the start and end of the reference section.

    Uses two strategies:
    1. Header-based: Look for Reference/Bibliography headers
    2. Fallback: Look for clusters of reference-like lines near end of file

    Returns:
        (start_line, end_line, section_text, method) or (None, None, None, "none")
        method is one of: "header", "fallback", "none"
    """
    lines = content.split("\n")
    ref_start = None
    ref_end = None

    # Strategy 1: Find reference section by header
    for i, line in enumerate(lines):
        for pattern in REFERENCE_HEADERS:
            if re.match(pattern, line.strip(), re.IGNORECASE):
                ref_start = i
                break
        if ref_start is not None:
            break

    if ref_start is not None:
        # Find reference section end (next major section or EOF)
        for i in range(ref_start + 1, len(lines)):
            line = lines[i].strip()
            for pattern in SECTION_END_PATTERNS:
                if re.match(pattern, line, re.IGNORECASE):
                    ref_end = i
                    break
            if ref_end is not None:
                break

        # If no end found, use end of file
        if ref_end is None:
            ref_end = len(lines)

        section_text = "\n".join(lines[ref_start:ref_end])
        return ref_start, ref_end, section_text, "header"

    # Strategy 2: Sequential [1], [2], [3], [4] detection (most robust)
    seq_start, seq_end = find_sequential_refs(lines, min_sequential=4)
    if seq_start is not None:
        section_text = "\n".join(lines[seq_start:seq_end])
        return seq_start, seq_end, section_text, "sequential"

    # Strategy 3: Fallback - search for reference clusters in the last half of the file
    search_start = int(len(lines) * 0.5)
    cluster_start, cluster_end = find_reference_cluster(lines, search_start)

    if cluster_start is not None:
        section_text = "\n".join(lines[cluster_start:cluster_end])
        return cluster_start, cluster_end, section_text, "cluster"

    return None, None, None, "none"


def estimate_ref_count(section_text: str) -> int:
    """Estimate the number of references in the section."""
    if not section_text:
        return 0

    count = 0
    for line in section_text.split("\n"):
        if is_reference_line(line):
            count += 1

    # Fallback: count lines that look like references (start with - or number)
    if count == 0:
        for line in section_text.split("\n"):
            line = line.strip()
            if line and (line.startswith("-") or re.match(r"^\d+\.", line)):
                count += 1

    return count


def process_file(md_path: Path) -> dict:
    """Process a single markdown file."""
    document_id = md_path.stem

    try:
        content = md_path.read_text(encoding="utf-8")
    except Exception as e:
        return {
            "document_id": document_id,
            "status": "error",
            "error": f"Failed to read file: {e}",
        }

    start_line, end_line, section_text, method = find_reference_section(content)
    
    if section_text is None:
        return {
            "document_id": document_id,
            "status": "no_references",
            "detection_method": "none",
            "ref_section_start": None,
            "ref_section_end": None,
            "ref_count_estimate": 0,
            "references_text": None,
        }

    ref_count = estimate_ref_count(section_text)

    return {
        "document_id": document_id,
        "status": "success",
        "detection_method": method,
        "ref_section_start": start_line,
        "ref_section_end": end_line,
        "ref_count_estimate": ref_count,
        "section_char_count": len(section_text),
        "references_text": section_text,
    }


def main():
    """Process all markdown files."""
    parser = argparse.ArgumentParser(description="Extract reference sections from markdown files")
    parser.add_argument("--input-dir", type=Path, default=MARKER_DIR)
    parser.add_argument("--output-file", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()

    marker_dir = args.input_dir
    output_file = args.output_file
    output_dir = output_file.parent

    print("=" * 60)
    print("Phase 1: Reference Section Extraction (v2 - improved)")
    print("=" * 60)

    # Find all markdown files
    md_files = sorted(marker_dir.glob("*.md"))
    print(f"Found {len(md_files)} markdown files")

    # Process each file
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "success": 0,
        "no_references": 0,
        "error": 0,
        "total_refs": 0,
        "by_header": 0,
        "by_sequential": 0,
        "by_cluster": 0,
    }

    with open(output_file, "w") as f:
        for i, md_path in enumerate(md_files, 1):
            result = process_file(md_path)
            f.write(json.dumps(result) + "\n")

            stats[result["status"]] = stats.get(result["status"], 0) + 1
            if result.get("ref_count_estimate"):
                stats["total_refs"] += result["ref_count_estimate"]

            # Track detection method
            method = result.get("detection_method", "none")
            if method == "header":
                stats["by_header"] += 1
            elif method == "sequential":
                stats["by_sequential"] += 1
            elif method == "cluster":
                stats["by_cluster"] += 1

            if i % 50 == 0:
                print(f"  Processed {i}/{len(md_files)} files...")

    print()
    print("=" * 60)
    print("Summary:")
    print(f"  Success: {stats['success']} files ({100*stats['success']/len(md_files):.1f}%)")
    print(f"    - By header detection: {stats['by_header']}")
    print(f"    - By sequential [N] detection: {stats['by_sequential']}")
    print(f"    - By cluster detection: {stats['by_cluster']}")
    print(f"  No references found: {stats['no_references']} files")
    print(f"  Errors: {stats['error']} files")
    print(f"  Total references estimated: {stats['total_refs']}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
