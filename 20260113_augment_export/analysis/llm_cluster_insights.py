#!/usr/bin/env python3
"""
LLM-based incremental clustering of insights.
For each cluster, iteratively present insights to LLM and ask it to:
- Assign to existing subcluster, OR
- Create a new subcluster

Uses Llama4 for fast classification with managed memory of subclusters.
"""

import json
import os
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

# Config
API_KEY = os.getenv("LAMBDA_API_KEY")
API_BASE = os.getenv("LAMBDA_API_BASE")
LLAMA4_MODEL = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"

BATCH_SIZE = 10  # Process insights in batches of 10
MAX_SUBCLUSTERS = 20  # Max subclusters per parent cluster before forcing merge

# Logging setup
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

def setup_logging():
    logger = logging.getLogger("llm_cluster")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%H:%M:%S")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(LOG_DIR / f"llm_cluster_{timestamp}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

logger = setup_logging()


@dataclass
class Subcluster:
    """A subcluster within a parent cluster."""
    id: int
    summary: str  # LLM-generated description
    representative_samples: list[str] = field(default_factory=list)  # 3-5 examples
    member_contents: list[str] = field(default_factory=list)  # All member insight contents
    count: int = 0


CLASSIFY_SYSTEM_PROMPT = """You are classifying user preference insights into semantic subclusters.

You will be given:
1. EXISTING SUBCLUSTERS: Each has an ID, summary description, and 3-5 example insights
2. NEW INSIGHTS: A batch of insights to classify

For EACH new insight, decide:
- If it fits an existing subcluster, assign it there
- If it's genuinely different from ALL existing subclusters, create a NEW subcluster

Output JSON:
{
  "assignments": [
    {"insight_index": 0, "subcluster_id": 1, "reason": "matches existing pattern"},
    {"insight_index": 1, "subcluster_id": "NEW", "new_summary": "brief description of new pattern", "reason": "different from all existing"}
  ]
}

Be CONSERVATIVE about creating new subclusters. Only create new if truly distinct.
Focus on SEMANTIC meaning, not surface phrasing differences."""


def format_subclusters_for_prompt(subclusters: list[Subcluster]) -> str:
    """Format existing subclusters for the LLM prompt."""
    if not subclusters:
        return "No existing subclusters yet. You may create the first one."
    
    lines = []
    for sc in subclusters:
        lines.append(f"[Subcluster {sc.id}] {sc.summary}")
        lines.append(f"  Count: {sc.count} insights")
        lines.append("  Examples:")
        for ex in sc.representative_samples[:3]:
            lines.append(f"    - {ex[:100]}")
        lines.append("")
    return "\n".join(lines)


def format_insights_for_prompt(insights: list[str]) -> str:
    """Format new insights for classification."""
    lines = []
    for i, content in enumerate(insights):
        lines.append(f"[{i}] {content}")
    return "\n".join(lines)


async def call_llm_async(client: httpx.AsyncClient, messages: list[dict], max_tokens: int = 2048) -> str:
    """Call the LLM API."""
    url = f"{API_BASE}/chat/completions"
    payload = {
        "model": LLAMA4_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    response = await client.post(url, json=payload, headers={"Authorization": f"Bearer {API_KEY}"})
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()
    return content


def parse_json_response(content: str) -> dict:
    """Extract JSON from LLM response."""
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        parts = content.split("```")
        if len(parts) >= 2:
            content = parts[1]
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(content[start:end])
        raise ValueError(f"No valid JSON in response: {content[:200]}")


async def cluster_insights_incrementally(
    cluster_id: int,
    insights: list[str],
    client: httpx.AsyncClient
) -> list[Subcluster]:
    """
    Incrementally cluster insights using LLM.
    Process in batches, maintaining subcluster memory.
    """
    subclusters: list[Subcluster] = []
    next_subcluster_id = 0

    total_batches = (len(insights) + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(f"Cluster {cluster_id}: Processing {len(insights)} insights in {total_batches} batches")

    for batch_idx in range(0, len(insights), BATCH_SIZE):
        batch = insights[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1

        # Build prompt
        user_prompt = f"""EXISTING SUBCLUSTERS:
{format_subclusters_for_prompt(subclusters)}

NEW INSIGHTS TO CLASSIFY:
{format_insights_for_prompt(batch)}

Classify each insight. Output JSON with "assignments" array."""

        messages = [
            {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = await call_llm_async(client, messages)
            parsed = parse_json_response(response)
            assignments = parsed.get("assignments", [])

            for assignment in assignments:
                idx = assignment.get("insight_index", 0)
                if idx >= len(batch):
                    continue

                insight_content = batch[idx]
                sc_id = assignment.get("subcluster_id")

                if sc_id == "NEW":
                    # Create new subcluster
                    new_summary = assignment.get("new_summary", insight_content[:80])
                    new_sc = Subcluster(
                        id=next_subcluster_id,
                        summary=new_summary,
                        representative_samples=[insight_content],
                        member_contents=[insight_content],
                        count=1
                    )
                    subclusters.append(new_sc)
                    next_subcluster_id += 1
                else:
                    # Assign to existing subcluster
                    for sc in subclusters:
                        if sc.id == sc_id:
                            sc.member_contents.append(insight_content)
                            sc.count += 1
                            if len(sc.representative_samples) < 5:
                                sc.representative_samples.append(insight_content)
                            break

            logger.info(f"  Batch {batch_num}/{total_batches}: {len(subclusters)} subclusters")

        except Exception as e:
            logger.error(f"  Batch {batch_num} error: {e}")
            # On error, create individual subclusters for each insight
            for insight_content in batch:
                new_sc = Subcluster(
                    id=next_subcluster_id,
                    summary=insight_content[:80],
                    representative_samples=[insight_content],
                    member_contents=[insight_content],
                    count=1
                )
                subclusters.append(new_sc)
                next_subcluster_id += 1

    logger.info(f"Cluster {cluster_id}: Final {len(subclusters)} subclusters")
    return subclusters


async def process_all_clusters(input_file: str, output_file: str, max_clusters: int = None):
    """Process all clusters from embedding-based clustering."""
    script_dir = Path(__file__).parent
    input_path = script_dir / "classification_results" / input_file
    output_path = script_dir / "classification_results" / output_file

    logger.info(f"Loading clusters from {input_path}")
    with open(input_path) as f:
        data = json.load(f)

    clusters = data.get("clusters", [])
    if max_clusters:
        clusters = clusters[:max_clusters]

    logger.info(f"Processing {len(clusters)} clusters")

    results = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        for i, cluster in enumerate(clusters):
            cluster_id = cluster.get("cluster_id", i)
            members = cluster.get("all_members", [])
            insights = [m["content"] for m in members]

            if len(insights) < 3:
                # Too small to subcluster
                logger.info(f"Cluster {cluster_id}: Skipping (only {len(insights)} insights)")
                results.append({
                    "parent_cluster_id": cluster_id,
                    "parent_size": len(insights),
                    "subclusters": [{
                        "id": 0,
                        "summary": insights[0] if insights else "",
                        "count": len(insights),
                        "members": insights
                    }]
                })
                continue

            subclusters = await cluster_insights_incrementally(cluster_id, insights, client)

            results.append({
                "parent_cluster_id": cluster_id,
                "parent_size": len(insights),
                "num_subclusters": len(subclusters),
                "subclusters": [asdict(sc) for sc in subclusters]
            })

            # Save incrementally
            with open(output_path, 'w') as f:
                json.dump({"results": results}, f, indent=2)

    logger.info(f"Done! Results saved to {output_path}")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="insight_clusters.json", help="Input cluster file")
    parser.add_argument("--output", default="llm_subclusters.json", help="Output file")
    parser.add_argument("--max-clusters", type=int, default=None, help="Max clusters to process")
    parser.add_argument("--test", type=int, default=0, help="Test on N clusters")
    args = parser.parse_args()

    if args.test > 0:
        args.max_clusters = args.test

    asyncio.run(process_all_clusters(args.input, args.output, args.max_clusters))


if __name__ == "__main__":
    main()

