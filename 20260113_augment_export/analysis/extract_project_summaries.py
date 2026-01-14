#!/usr/bin/env python3
"""
Extract project-level summary responses from all workspaces.
Uses heuristic scoring to filter out task completions, bug fixes, and other
non-project-context responses.

Output: JSON file per workspace with extracted summaries for later LLM consolidation.
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Tuple

# Minimum score threshold to be considered a project summary
MIN_SCORE_THRESHOLD = 0.3

# Positive signal patterns in trigger requests
TRIGGER_POSITIVE_PATTERNS = [
    r'\b(inspect|analyze|research|understand|explore|explain)\b.*\b(codebase|project|architecture|code\s*base)\b',
    r'\b(codebase|project|architecture)\b.*\b(inspect|analyze|research|understand|explore|explain)\b',
    r'\bwhat\s+(is|does)\s+(this|the)\s+(project|codebase|repo)\b',
    r'\b(project|codebase|repo)\s+(overview|structure|architecture)\b',
    r'\bhow\s+(is|does)\s+(this|the)\s+(project|app|system)\s+(work|structured)\b',
    r'\b(tech|technology)\s+stack\b',
    r'\bdirectory\s+structure\b',
]

# Positive signal patterns in response headers
RESPONSE_HEADER_POSITIVE = [
    r'^#{1,3}\s*(project\s+)?(overview|summary)',
    r'^#{1,3}\s*architecture',
    r'^#{1,3}\s*(core\s+)?components?',
    r'^#{1,3}\s*tech(nology)?\s+stack',
    r'^#{1,3}\s*directory\s+(structure|layout)',
    r'^#{1,3}\s*design\s+(decisions?|patterns?|philosophy)',
    r'^#{1,3}\s*system\s+(design|architecture)',
    r'^#{1,3}\s*key\s+(features?|modules?|components?)',
    r'^#{1,3}\s*data(base)?\s+(schema|model|architecture)',
    r'^#{1,3}\s*api\s+(design|overview|endpoints?)',
    r'^#{1,3}\s*(how|what)\s+.*\s+works?',
    r'^#{1,3}\s*executive\s+summary',
    r'^#{1,3}\s*codebase\s+(overview|analysis|structure)',
]

# Negative signal patterns in trigger requests (task-specific)
TRIGGER_NEGATIVE_PATTERNS = [
    r'\b(fix|debug|solve|resolve)\s+(this|the|that|a)\b',
    r'\bwrite\s+(a\s+)?(documentation|doc|readme)\b',
    r'\b(implement|add|create)\s+(a\s+)?(feature|function|endpoint)\b',
    r'\bwhat\s+(did|was)\s+(wrong|the\s+issue|the\s+bug)\b',
]

# Negative signal patterns in response content (task completion indicators)
RESPONSE_NEGATIVE_PATTERNS = [
    r'\bi\'ve\s+(successfully|now|just)?\s*(completed|implemented|fixed|updated|created|added)\b',
    r'\bhere\'s\s+what\s+(i\s+)?(did|changed|fixed|updated)\b',
    r'\bsummary\s+of\s+(the\s+)?(changes|fixes|updates|what\s+was)\b',
    r'\bphase\s+\d+\s+(is\s+)?(complete|done|finished)\b',
    r'\bthe\s+(bug|issue|problem|fix)\s+(is|has\s+been)\s+(complete|fixed|resolved)\b',
    r'\ball\s+(tests?\s+)?(pass|passing|passed)\b',
    r'\bsuccessfully\s+(deployed|merged|pushed|created)\b',
    r'\bi\s+(have\s+)?(created|wrote|generated)\s+(the\s+)?(documentation|doc|file)\b',
    r'\bhere\'s\s+the\s+(documentation|implementation|fix)\b',
    r'\bperfect!\s+(all|the|it)\b',
    r'\bexcellent!\s+(the|all|it)\b',
]

# Strong negative headers (documentation writing, not project overview)
RESPONSE_HEADER_NEGATIVE = [
    r'^#{1,3}\s*implementation\s+(details?|notes?|summary)',
    r'^#{1,3}\s*changes?\s+(made|summary|log)',
    r'^#{1,3}\s*what\s+(was|i)\s+(done|did|changed|fixed)',
    r'^#{1,3}\s*fix(es)?\s+(summary|applied|details?)',
    r'^#{1,3}\s*bug\s+fix',
    r'^#{1,3}\s*update\s+summary',
]


def extract_project_name(filepath: Path) -> str:
    """Extract project name from filepath."""
    # Format: hash__path_to_project.json
    # Include hash prefix to disambiguate duplicate project names
    parts = filepath.stem.split("__")
    hash_prefix = parts[0][:8] if parts else ""
    name = parts[-1] if len(parts) > 1 else filepath.stem

    # Take last component of path
    name_parts = name.split("_")
    # Find GitHub or Documents marker and take what follows
    for i, p in enumerate(name_parts):
        if p in ("GitHub", "Documents") and i + 1 < len(name_parts):
            project = "_".join(name_parts[i + 1:])
            return f"{project}_{hash_prefix}"
    return f"{name_parts[-1]}_{hash_prefix}" if name_parts else f"{name}_{hash_prefix}"


def find_trigger_request(exchanges: list, target_idx: int) -> dict:
    """Find the user request that triggered this response."""
    for j in range(target_idx, -1, -1):
        req = exchanges[j].get("request_message", "")
        if req:
            return {
                "text": req,
                "exchange_index": j,
            }
    return {"text": "", "exchange_index": -1}


def score_project_summary(
    response: str, trigger_request: str, min_length: int = 1500
) -> Tuple[bool, float, list]:
    """
    Score whether a response is a project-level summary.

    Returns:
        (is_summary, score, reasons) where:
        - is_summary: True if score >= MIN_SCORE_THRESHOLD
        - score: float between -1.0 and 1.0
        - reasons: list of (signal, weight) tuples explaining the score
    """
    # Basic length check
    if len(response) < min_length:
        return (False, 0.0, [("too_short", 0.0)])

    # Must have markdown headers
    if not re.search(r'^#{1,3}\s+', response, re.MULTILINE):
        return (False, 0.0, [("no_headers", 0.0)])

    score = 0.0
    reasons = []
    trigger_lower = trigger_request.lower()
    response_lower = response.lower()

    # --- Positive signals from trigger request ---
    for pattern in TRIGGER_POSITIVE_PATTERNS:
        if re.search(pattern, trigger_lower):
            score += 0.25
            reasons.append(("trigger_positive", 0.25))
            break  # Only count once

    # --- Positive signals from response headers ---
    header_matches = 0
    for pattern in RESPONSE_HEADER_POSITIVE:
        if re.search(pattern, response_lower, re.MULTILINE):
            header_matches += 1
    if header_matches >= 3:
        score += 0.4
        reasons.append(("strong_headers", 0.4))
    elif header_matches >= 1:
        score += 0.2
        reasons.append(("some_headers", 0.2))

    # --- Structural signals (multiple sections = more likely overview) ---
    h2_count = len(re.findall(r'^##\s+', response, re.MULTILINE))
    h3_count = len(re.findall(r'^###\s+', response, re.MULTILINE))
    if h2_count >= 4 or (h2_count >= 2 and h3_count >= 4):
        score += 0.15
        reasons.append(("good_structure", 0.15))

    # --- Negative signals from trigger request ---
    for pattern in TRIGGER_NEGATIVE_PATTERNS:
        if re.search(pattern, trigger_lower):
            score -= 0.2
            reasons.append(("trigger_negative", -0.2))
            break

    # --- Negative signals from response content ---
    neg_matches = 0
    for pattern in RESPONSE_NEGATIVE_PATTERNS:
        if re.search(pattern, response_lower):
            neg_matches += 1
    if neg_matches >= 2:
        score -= 0.4
        reasons.append(("completion_language", -0.4))
    elif neg_matches >= 1:
        score -= 0.2
        reasons.append(("some_completion", -0.2))

    # --- Negative signals from headers ---
    for pattern in RESPONSE_HEADER_NEGATIVE:
        if re.search(pattern, response_lower, re.MULTILINE):
            score -= 0.25
            reasons.append(("negative_header", -0.25))
            break

    # --- Length bonus (longer = more likely comprehensive overview) ---
    if len(response) > 8000:
        score += 0.1
        reasons.append(("very_long", 0.1))
    elif len(response) > 5000:
        score += 0.05
        reasons.append(("long", 0.05))

    is_summary = score >= MIN_SCORE_THRESHOLD
    return (is_summary, round(score, 3), reasons)


def extract_summaries_from_workspace(filepath: Path) -> dict:
    """Extract project-level summaries from a workspace using heuristic scoring."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    project_name = extract_project_name(filepath)
    folder_path = data.get("folder_path", "")

    summaries = []
    rejected_count = 0

    for conv in data.get("conversations", []):
        conv_id = conv.get("conversation_id", conv.get("conversationId", ""))
        exchanges = conv.get("exchanges", [])

        for i, ex in enumerate(exchanges):
            resp = ex.get("response_text", "")
            trigger = find_trigger_request(exchanges, i)

            is_summary, score, reasons = score_project_summary(
                resp, trigger["text"]
            )

            if is_summary:
                summaries.append({
                    "conversation_id": conv_id,
                    "exchange_index": i,
                    "trigger_request": trigger["text"][:500],
                    "trigger_exchange_index": trigger["exchange_index"],
                    "response_length": len(resp),
                    "score": score,
                    "score_reasons": reasons,
                    "response_text": resp,
                    "timestamp": ex.get("timestamp", ""),
                })
            elif len(resp) >= 1500 and re.search(r'^#{1,3}\s+', resp, re.MULTILINE):
                # Count rejections that would have passed old criteria
                rejected_count += 1

    # Sort by score (highest first), then by length
    summaries.sort(key=lambda x: (x["score"], x["response_length"]), reverse=True)

    return {
        "workspace_id": filepath.stem.split("__")[0],
        "project_name": project_name,
        "folder_path": folder_path,
        "total_summaries": len(summaries),
        "rejected_by_heuristics": rejected_count,
        "summaries": summaries,
    }


def main():
    data_dir = Path(__file__).parent.parent / "augment_conversations_export_leveldb"
    output_dir = Path(__file__).parent / "project_summaries"
    output_dir.mkdir(exist_ok=True)

    print(f"Scanning {data_dir}...")
    print(f"Using score threshold: {MIN_SCORE_THRESHOLD}\n")

    all_stats = []
    total_rejected = 0

    for filepath in sorted(data_dir.glob("*.json")):
        if "extraction_summary" in filepath.name:
            continue

        result = extract_summaries_from_workspace(filepath)
        rejected = result.get("rejected_by_heuristics", 0)
        total_rejected += rejected

        # Save per-workspace
        out_file = output_dir / f"{result['project_name']}_summaries.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        all_stats.append({
            "project_name": result["project_name"],
            "folder_path": result["folder_path"],
            "total_summaries": result["total_summaries"],
            "rejected_by_heuristics": rejected,
            "top_3_scores": [s["score"] for s in result["summaries"][:3]],
            "top_3_lengths": [s["response_length"] for s in result["summaries"][:3]],
        })

        kept = result["total_summaries"]
        print(f"  {result['project_name']}: {kept} kept, {rejected} rejected")

    # Save overall stats
    total_kept = sum(s["total_summaries"] for s in all_stats)
    stats_file = output_dir / "_extraction_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump({
            "extracted_at": datetime.now().isoformat(),
            "score_threshold": MIN_SCORE_THRESHOLD,
            "total_workspaces": len(all_stats),
            "total_summaries": total_kept,
            "total_rejected": total_rejected,
            "reduction_percent": round(
                100 * total_rejected / (total_kept + total_rejected), 1
            ) if (total_kept + total_rejected) > 0 else 0,
            "workspaces": all_stats,
        }, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Results: {total_kept} kept, {total_rejected} rejected")
    print(f"Reduction: {total_rejected}/{total_kept + total_rejected} = "
          f"{100 * total_rejected / (total_kept + total_rejected):.1f}%")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()

