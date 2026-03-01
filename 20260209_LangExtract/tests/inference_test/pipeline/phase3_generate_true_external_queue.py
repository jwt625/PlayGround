#!/usr/bin/env python3
"""Generate a clean true-external collection queue from canonical registry.

Rules:
- Use effective classification only (`external_effective == true`).
- Exclude references already downloaded in any round (strict).
- Keep high-signal items (DOI/arXiv present, or non-trivial title).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def has_useful_title(title: str) -> bool:
    t = (title or "").strip()
    if len(t) < 20:
        return False
    low = t.lower()
    noisy_prefixes = (
        "see supplemental material",
        "supplemental material",
        "nature (london)",
        "phys. rev.",
    )
    if any(low.startswith(p) for p in noisy_prefixes):
        return False
    if "http://" in low or "https://" in low:
        return False
    return True


def is_plausible_doi(doi: str | None) -> bool:
    if not doi:
        return False
    d = doi.strip().lower()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d, flags=re.I)
    d = re.sub(r"^doi:\s*", "", d, flags=re.I)
    if "unknown" in d:
        return False
    if not re.match(r"^10\.\d{4,9}/\S+$", d):
        return False
    # Common placeholder artifacts seen in noisy extraction.
    if any(tok in d for tok in ("12345", "xxxxx", "xxxx")):
        return False
    return True


def is_plausible_arxiv(arxiv_id: str | None) -> bool:
    if not arxiv_id:
        return False
    a = arxiv_id.strip().lower()
    a = re.sub(r"^arxiv:\s*", "", a, flags=re.I)
    if not re.match(r"^[a-z0-9.\-\/]+$", a):
        return False
    if re.match(r"^\d{4}\.\d{4,5}$", a):
        return True
    if re.match(r"^[a-z\-]+(?:\.[a-z\-]+)?\/\d{7}$", a):
        return True
    return False


def doi_to_url(doi: str | None) -> str | None:
    if not doi:
        return None
    d = doi.strip()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d, flags=re.I)
    d = re.sub(r"^doi:\s*", "", d, flags=re.I)
    return f"https://doi.org/{d}" if d else None


def arxiv_to_url(arxiv_id: str | None) -> str | None:
    if not arxiv_id:
        return None
    a = arxiv_id.strip()
    a = re.sub(r"^arxiv:\s*", "", a, flags=re.I)
    return f"https://arxiv.org/abs/{a}" if a else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate true-external queue from canonical registry")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("tests/inference_test/output/canonical_reference_registry.jsonl"),
        help="Input canonical registry JSONL",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of candidates to include",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("tests/inference_test/output/top_external_refs_for_r3_processing.md"),
        help="Output markdown path",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("tests/inference_test/output/r3_processing_input.json"),
        help="Output JSON queue path",
    )
    parser.add_argument(
        "--out-validation",
        type=Path,
        default=Path("tests/inference_test/output/true_external_queue_validation.json"),
        help="Output validation summary path",
    )
    args = parser.parse_args()

    rows = []
    with args.input.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    total = len(rows)
    violations = []
    candidates = []

    for r in rows:
        ref_id = r.get("ref_id")
        c = r.get("canonical_enriched") or r.get("canonical") or {}
        title = c.get("title")
        year = c.get("year")
        doi = c.get("doi")
        arxiv_id = c.get("arxiv_id")

        external_effective = bool(r.get("external_effective"))
        if not external_effective:
            continue

        # Regression guard: truly external must not be seen/in-dataset.
        if r.get("in_dataset_effective") or r.get("seen_in_any_round") or r.get("downloaded_any_round"):
            violations.append(
                {
                    "ref_id": ref_id,
                    "reason": "effective_external_invariant_violation",
                    "in_dataset_effective": r.get("in_dataset_effective"),
                    "seen_in_any_round": r.get("seen_in_any_round"),
                    "downloaded_any_round": r.get("downloaded_any_round"),
                }
            )
            continue

        # Keep references not yet downloaded (strict path to avoid retry noise).
        if bool(r.get("downloaded_any_round_strict")):
            continue

        signal_ok = bool(doi or arxiv_id or has_useful_title(title or ""))
        if not signal_ok:
            continue

        doi_ok = is_plausible_doi(doi)
        arxiv_ok = is_plausible_arxiv(arxiv_id)
        if not doi_ok and not arxiv_ok:
            continue

        if not has_useful_title(title or ""):
            continue

        doi = doi if doi_ok else None
        arxiv_id = arxiv_id if arxiv_ok else None
        url = doi_to_url(doi) or arxiv_to_url(arxiv_id)
        candidates.append(
            {
                "ref_id": ref_id,
                "citation_count": int(r.get("citation_count") or 0),
                "title": title,
                "year": year,
                "doi": doi,
                "arxiv": arxiv_id,
                "url": url,
            }
        )

    candidates.sort(key=lambda x: x["citation_count"], reverse=True)
    top = candidates[: args.top_n]

    # Build markdown
    md_lines = []
    md_lines.append("# Top External References For R3 Processing (Current)")
    md_lines.append("")
    md_lines.append("- Filter: `external_effective == true` and `downloaded_any_round_strict == false`")
    md_lines.append(f"- Total candidates (post-filter): {len(candidates)}")
    md_lines.append("")
    for i, x in enumerate(top, 1):
        title = x.get("title") or "Unknown title"
        year = x.get("year") or "Unknown"
        if x.get("url"):
            md_lines.append(f"{i}. **[{x['citation_count']} citations]** [{title}]({x['url']}) ({year})")
        else:
            md_lines.append(f"{i}. **[{x['citation_count']} citations]** {title} ({year})")
    args.out_md.write_text("\n".join(md_lines) + "\n")

    # Build collector input JSON
    queue = []
    for i, x in enumerate(top, 1):
        queue.append(
            {
                "rank": i,
                "citation_count": x["citation_count"],
                "title": x.get("title"),
                "year": x.get("year"),
                "doi": x.get("doi"),
                "arxiv": x.get("arxiv"),
                "url": x.get("url"),
                "ref_id": x.get("ref_id"),
            }
        )
    args.out_json.write_text(json.dumps(queue, indent=2) + "\n")

    summary = {
        "input": str(args.input),
        "total_registry_rows": total,
        "effective_external_candidates": len(candidates),
        "written_top_n": len(top),
        "validation_violations": len(violations),
        "output_markdown": str(args.out_md),
        "output_json": str(args.out_json),
    }
    if violations:
        summary["violations"] = violations[:50]
    args.out_validation.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    if violations:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
