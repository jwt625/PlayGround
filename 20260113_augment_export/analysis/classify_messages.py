#!/usr/bin/env python3
"""
Two-stage message classification pipeline.
Stage 1: Fast filter with Llama-4 to identify high-value messages
Stage 2: Deep classification with GLM-4.6 for multi-label types and generalizability scoring
"""

import json
import hashlib
import time
import os
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

# Configure logging
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

def setup_logging(log_name: str = "classify_messages") -> logging.Logger:
    """Set up logging with both console and file handlers."""
    logger = logging.getLogger(log_name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Format with timestamp
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (DEBUG and above) - new file per run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"{log_name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logging to {log_file}")

    return logger

# Initialize logger
logger = setup_logging()

# LLM Configuration - use same env vars as generate_missing_summaries.py
API_KEY = os.getenv("LAMBDA_API_KEY")
API_BASE = os.getenv("LAMBDA_API_BASE")  # e.g. https://api.lambda...../v1
LLAMA4_MODEL = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
GLM_MODEL = "zai-org/GLM-4.6-FP8"

# Stage 1 prompt - Fast filtering with Llama-4
STAGE1_SYSTEM_PROMPT = """You are a message classifier for analyzing user messages in coding assistant conversations.

Your task: Determine if a user message is HIGH_VALUE or LOW_VALUE for extracting development preferences and patterns.

HIGH_VALUE messages contain:
- Explicit preferences (e.g., "always use uv", "I prefer pytest")
- Corrections or complaints about LLM behavior
- Decisions about architecture, tools, or approaches
- Feature requests or requirements statements
- Frustration or strong opinions
- Clarifications that reveal preferences

LOW_VALUE messages contain:
- Simple acknowledgments ("ok", "thanks", "got it")
- Questions without opinions embedded
- Pure error logs or terminal output without commentary
- Requests for information without preference signals
- Conversational filler

Respond with ONLY a JSON object:
{"value": "HIGH" or "LOW", "reason": "brief one-line explanation"}"""

STAGE1_USER_TEMPLATE = """Classify this user message:

MESSAGE:
{message}

CONTEXT (what the assistant said before):
{prev_response}"""

# Stage 2 prompt - Deep classification with GLM-4.6
STAGE2_SYSTEM_PROMPT = """You are an expert analyst extracting development preferences and patterns from user messages in coding assistant conversations.

For each message, provide:

1. CLASSIFICATION (multi-label, select all that apply):
   - preference_statement: User states a preference for tools, patterns, or approaches
   - decision: User makes a choice between alternatives
   - correction: User corrects the assistant's behavior or output
   - constraint: User states something that must ALWAYS or NEVER be done
   - frustration: User expresses annoyance or dissatisfaction
   - clarification: User provides additional context or requirements
   - feature_request: User asks for new functionality
   - approval: User approves or accepts something
   - bug_report: User reports an error or unexpected behavior

2. GENERALIZABILITY SCORE (0.0 to 1.0):
   - 0.0-0.2: Highly task-specific, only applies to this exact situation
   - 0.3-0.5: Context-dependent, applies to similar project types
   - 0.6-0.8: Broadly applicable to this tech stack or domain
   - 0.9-1.0: Universal preference, applies across all development work

3. EXTRACTED INSIGHTS (if any):
   - Tool preferences (e.g., "prefers uv over pip")
   - Workflow patterns (e.g., "wants tests after each feature")
   - Quality standards (e.g., "dislikes verbose output")
   - Hard constraints (e.g., "NEVER use emojis in code")
   - Communication preferences (e.g., "prefers concise responses")

Respond with ONLY a JSON object:
{
  "labels": ["label1", "label2"],
  "generalizability": 0.7,
  "insights": [
    {"type": "tool_preference", "content": "prefers uv for Python package management", "confidence": 0.9},
    {"type": "constraint", "content": "never include emojis in responses", "confidence": 0.95}
  ],
  "reasoning": "Brief explanation of classification"
}

If the message has no extractable insights, return:
{"labels": ["conversational"], "generalizability": 0.0, "insights": [], "reasoning": "No actionable preferences"}"""

STAGE2_USER_TEMPLATE = """Analyze this user message for development preferences and patterns:

PROJECT CONTEXT:
{project_summary}

PREVIOUS ASSISTANT RESPONSE:
{prev_response}

USER MESSAGE:
{message}

ASSISTANT'S RESPONSE TO THIS MESSAGE:
{next_response}"""


@dataclass
class MessageContext:
    """Context for a single user message."""
    # Message identification
    workspace_id: str
    folder_path: str
    conversation_id: str
    exchange_index: int
    message_id: str
    timestamp: str
    
    # The message itself
    user_message: str
    
    # Context
    project_summary: str
    prev_assistant_response: str
    next_assistant_response: str
    
    # Metadata for deduplication
    message_hash: str = field(default="")
    
    def __post_init__(self):
        if not self.message_hash:
            self.message_hash = hashlib.sha256(self.user_message.encode()).hexdigest()[:16]


def call_llm(
    messages: list[dict],
    model: str,
    max_tokens: int = 1024,
    temperature: float = 0.1
) -> str:
    """Call the inference API and return the response content."""
    url = f"{API_BASE}/chat/completions"

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        response.raise_for_status()
        data = response.json()

    content = data["choices"][0]["message"]["content"]

    # Handle GLM thinking tokens
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()

    return content


async def call_llm_async(
    client: httpx.AsyncClient,
    messages: list[dict],
    model: str,
    max_tokens: int = 1024,
    temperature: float = 0.1
) -> str:
    """Async version of call_llm for concurrent requests."""
    url = f"{API_BASE}/chat/completions"

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    response = await client.post(
        url,
        json=payload,
        headers={"Authorization": f"Bearer {API_KEY}"}
    )
    response.raise_for_status()
    data = response.json()

    content = data["choices"][0]["message"]["content"]

    # Handle GLM thinking tokens
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()

    return content


def parse_json_response(content: str, debug: bool = False) -> dict:
    """Parse JSON from LLM response, handling common issues."""
    original_content = content

    # Try to extract JSON from markdown code blocks
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        parts = content.split("```")
        if len(parts) >= 2:
            content = parts[1]

    content = content.strip()

    # First try direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the content
    start = content.find("{")
    end = content.rfind("}") + 1
    if start >= 0 and end > start:
        json_str = content[start:end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Try to find in original content (before code block extraction)
    start = original_content.find("{")
    end = original_content.rfind("}") + 1
    if start >= 0 and end > start:
        json_str = original_content[start:end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            if debug:
                logger.debug(f"Failed to parse JSON: {e}")
                logger.debug(f"Content: {json_str[:500]}")
            raise

    raise ValueError(f"No valid JSON found in response: {content[:200]}")


async def classify_stage1_batch_async(
    messages: list[MessageContext],
    output_path: str,
    concurrency: int = 10
) -> list[dict]:
    """
    Stage 1: Fast async classification with Llama-4 to filter high-value messages.
    Uses concurrent requests for significant speedup (~9x with concurrency=20).
    Saves incrementally to allow resuming interrupted runs.
    """
    results = []
    output_file = Path(output_path)

    # Load existing results if resuming
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            existing = json.load(f)
            results = existing.get("results", [])
            processed_hashes = {r["message_hash"] for r in results}
            messages = [m for m in messages if m.message_hash not in processed_hashes]
            logger.info(f"Stage 1: Resuming with {len(results)} already done, {len(messages)} remaining")

    if not messages:
        logger.info("Stage 1: No messages to process")
        return results

    total_batches = (len(messages) + concurrency - 1) // concurrency
    logger.info(f"Stage 1: Classifying {len(messages)} messages with Llama-4 ({concurrency} concurrent)...")

    async with httpx.AsyncClient(timeout=120.0) as client:
        for batch_idx in range(0, len(messages), concurrency):
            batch = messages[batch_idx:batch_idx + concurrency]
            batch_num = batch_idx // concurrency + 1

            async def classify_one(msg):
                try:
                    user_prompt = STAGE1_USER_TEMPLATE.format(
                        message=msg.user_message[:1500],
                        prev_response=msg.prev_assistant_response[:500]
                    )

                    llm_messages = [
                        {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ]

                    response = await call_llm_async(client, llm_messages, LLAMA4_MODEL, max_tokens=256)
                    parsed = parse_json_response(response)

                    is_high = parsed.get("value", "LOW").upper() == "HIGH"
                    reason = parsed.get("reason", "")

                    return {
                        "message_hash": msg.message_hash,
                        "is_high_value": is_high,
                        "reason": reason,
                        "message_preview": msg.user_message[:200]
                    }
                except Exception as e:
                    logger.error(f"Error classifying message: {e}")
                    return {
                        "message_hash": msg.message_hash,
                        "is_high_value": True,
                        "reason": f"Error: {e}",
                        "message_preview": msg.user_message[:200]
                    }

            # Process batch concurrently
            tasks = [classify_one(msg) for msg in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            # Save incrementally
            high_count = sum(1 for r in results if r["is_high_value"])
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "total_processed": len(results),
                    "high_value_count": high_count,
                    "results": results
                }, f, indent=2)

            logger.info(f"Batch {batch_num}/{total_batches}: {high_count}/{len(results)} high-value")

    high_value = sum(1 for r in results if r["is_high_value"])
    logger.info(f"Stage 1 complete: {high_value}/{len(results)} messages marked high-value ({high_value/len(results)*100:.1f}%)")

    return results


def classify_stage1_batch(
    messages: list[MessageContext],
    output_path: str,
    batch_size: int = 20
) -> list[dict]:
    """
    Stage 1: Fast classification with Llama-4 (synchronous version - deprecated).
    Use classify_stage1_batch_async for better performance.
    Saves incrementally to allow resuming interrupted runs.
    Returns list of result dicts.
    """
    results = []
    output_file = Path(output_path)
    processed_hashes = set()

    # Load existing results if resuming
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            existing = json.load(f)
            results = existing.get("results", [])
            processed_hashes = {r["message_hash"] for r in results}
            logger.info(f"Resuming Stage 1: {len(results)} already processed")

    # Filter to only unprocessed messages
    remaining = [m for m in messages if m.message_hash not in processed_hashes]

    if not remaining:
        logger.info(f"Stage 1: All {len(messages)} messages already processed")
        return results

    total_batches = (len(remaining) + batch_size - 1) // batch_size
    logger.info(f"Stage 1: Classifying {len(remaining)} messages with Llama-4 ({len(results)} already done)...")

    for batch_idx in range(0, len(remaining), batch_size):
        batch = remaining[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1

        for msg in batch:
            try:
                user_prompt = STAGE1_USER_TEMPLATE.format(
                    message=msg.user_message[:1500],
                    prev_response=msg.prev_assistant_response[:500]
                )

                llm_messages = [
                    {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]

                response = call_llm(llm_messages, LLAMA4_MODEL, max_tokens=256)
                parsed = parse_json_response(response)

                is_high = parsed.get("value", "LOW").upper() == "HIGH"
                reason = parsed.get("reason", "")

                results.append({
                    "message_hash": msg.message_hash,
                    "is_high_value": is_high,
                    "reason": reason,
                    "message_preview": msg.user_message[:200]
                })

            except Exception as e:
                logger.error(f"Error classifying message: {e}")
                # Default to high value on error to avoid losing data
                results.append({
                    "message_hash": msg.message_hash,
                    "is_high_value": True,
                    "reason": f"Error: {e}",
                    "message_preview": msg.user_message[:200]
                })

        # Save after each batch
        high_count = sum(1 for r in results if r["is_high_value"])
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "total_processed": len(results),
                "high_value_count": high_count,
                "results": results
            }, f, indent=2)

        logger.info(f"Batch {batch_num}/{total_batches}: {high_count}/{len(results)} high-value (saved)")

        # Small delay to avoid rate limiting
        time.sleep(0.1)

    high_value = sum(1 for r in results if r["is_high_value"])
    logger.info(f"Stage 1 complete: {high_value}/{len(results)} messages marked high-value ({high_value/len(results)*100:.1f}%)")

    return results


async def classify_single_message_async(
    client: httpx.AsyncClient,
    msg: MessageContext
) -> dict:
    """Classify a single message asynchronously."""
    try:
        user_prompt = STAGE2_USER_TEMPLATE.format(
            project_summary=msg.project_summary[:1500],
            prev_response=msg.prev_assistant_response[:800],
            message=msg.user_message[:2000],
            next_response=msg.next_assistant_response[:800]
        )

        llm_messages = [
            {"role": "system", "content": STAGE2_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        response = await call_llm_async(client, llm_messages, GLM_MODEL, max_tokens=4096)
        parsed = parse_json_response(response)

        return {
            "message_hash": msg.message_hash,
            "workspace_id": msg.workspace_id,
            "folder_path": msg.folder_path,
            "conversation_id": msg.conversation_id,
            "exchange_index": msg.exchange_index,
            "user_message": msg.user_message,
            "labels": parsed.get("labels", []),
            "generalizability": parsed.get("generalizability", 0.0),
            "insights": parsed.get("insights", []),
            "reasoning": parsed.get("reasoning", "")
        }

    except Exception as e:
        logger.error(f"Error classifying message: {e}")
        return {
            "message_hash": msg.message_hash,
            "workspace_id": msg.workspace_id,
            "folder_path": msg.folder_path,
            "user_message": msg.user_message,
            "error": str(e)
        }


async def classify_stage2_batch_async(
    messages: list[MessageContext],
    output_path: str,
    concurrency: int = 10
) -> list[dict]:
    """
    Stage 2: Deep classification with GLM-4.6 using concurrent requests.
    Saves incrementally after each batch to avoid data loss.
    """
    results = []
    output_file = Path(output_path)

    # Load existing results if resuming
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            existing = json.load(f)
            results = existing.get("results", [])
            processed_hashes = {r["message_hash"] for r in results}
            messages = [m for m in messages if m.message_hash not in processed_hashes]
            logger.info(f"Resuming: {len(results)} already processed, {len(messages)} remaining")

    if not messages:
        logger.info("No messages to process")
        return results

    total_batches = (len(messages) + concurrency - 1) // concurrency
    logger.info(f"Stage 2: Deep classification of {len(messages)} messages with GLM-4.6 ({concurrency} concurrent)...")

    async with httpx.AsyncClient(timeout=120.0) as client:
        for batch_idx in range(0, len(messages), concurrency):
            batch = messages[batch_idx:batch_idx + concurrency]
            batch_num = batch_idx // concurrency + 1

            # Process batch concurrently
            tasks = [classify_single_message_async(client, msg) for msg in batch]
            batch_results = await asyncio.gather(*tasks)

            results.extend(batch_results)

            # Save incrementally after each batch
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({"results": results}, f, indent=2)

            logger.info(f"Batch {batch_num}/{total_batches}: {len(results)} total processed")

    logger.info(f"Stage 2 complete: {len(results)} messages classified")
    return results


def classify_stage2_batch(
    messages: list[MessageContext],
    output_path: str,
    concurrency: int = 10
) -> list[dict]:
    """
    Stage 2: Deep classification with GLM-4.6.
    Wrapper that runs the async version.
    """
    return asyncio.run(classify_stage2_batch_async(messages, output_path, concurrency))


def load_project_summaries(summaries_path: str) -> dict[str, str]:
    """Load consolidated project summaries, keyed by folder_path."""
    with open(summaries_path, 'r', encoding='utf-8') as f:
        summaries = json.load(f)

    result = {}
    for item in summaries:
        folder_path = item.get('folder_path', '')
        summary = item.get('final_summary', '')
        if folder_path and summary:
            result[folder_path] = summary

    logger.info(f"Loaded {len(result)} project summaries")
    return result


def load_workspace_conversations(data_dir: str) -> list[dict]:
    """Load all workspace conversation files."""
    data_path = Path(data_dir)
    workspaces = []

    for json_file in sorted(data_path.glob("*.json")):
        if json_file.name == "extraction_summary.json":
            continue

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            workspaces.append(data)
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")

    logger.info(f"Loaded {len(workspaces)} workspaces")
    return workspaces


def extract_full_response(exchange: dict) -> str:
    """
    Extract full assistant response from response_nodes.
    Concatenates all text nodes (type 0) to build the complete response.
    """
    response_nodes = exchange.get('response_nodes', [])

    text_parts = []
    for node in response_nodes:
        node_type = node.get('type', -1)
        # Type 0 = text content
        if node_type == 0:
            content = node.get('content', '')
            if content:
                text_parts.append(content)

    full_response = '\n'.join(text_parts)

    # Fall back to response_text if no nodes found
    if not full_response:
        full_response = exchange.get('response_text', '')

    return full_response


def find_prev_user_exchange(exchanges: list[dict], current_index: int) -> tuple[int, str]:
    """
    Find the previous exchange that was a user message (not tool result).
    Returns (index, assistant_response_to_that_message).
    """
    # Search backwards for an exchange with a non-empty request_message
    for j in range(current_index - 1, -1, -1):
        req_msg = exchanges[j].get('request_message', '').strip()
        if req_msg and len(req_msg) >= 10:
            # Found the previous user message, return its assistant response
            return j, extract_full_response(exchanges[j])
    return -1, ""


def extract_messages_with_context(
    workspaces: list[dict],
    project_summaries: dict[str, str],
    min_message_length: int = 10
) -> list[MessageContext]:
    """
    Extract all user messages with their surrounding context.
    Deduplicates messages based on content hash.

    For context:
    - prev_assistant_response: Full response from the last REAL user message exchange
    - next_assistant_response: Full response to THIS user message
    """
    messages = []
    seen_hashes = set()

    for workspace in workspaces:
        workspace_id = workspace.get('workspace_id', '')
        folder_path = workspace.get('folder_path', '')
        project_summary = project_summaries.get(folder_path, 'No project summary available.')

        for conv in workspace.get('conversations', []):
            conv_id = conv.get('conversation_id', '')
            exchanges = conv.get('exchanges', [])

            for i, exchange in enumerate(exchanges):
                user_msg = exchange.get('request_message', '').strip()

                # Skip empty or very short messages
                if not user_msg or len(user_msg) < min_message_length:
                    continue

                # Compute hash for deduplication
                msg_hash = hashlib.sha256(user_msg.encode()).hexdigest()[:16]

                # Skip duplicates (same message content)
                if msg_hash in seen_hashes:
                    continue
                seen_hashes.add(msg_hash)

                # Get previous assistant response (from last real user message exchange)
                _, prev_response = find_prev_user_exchange(exchanges, i)

                # Get current assistant response (full response to this message)
                next_response = extract_full_response(exchange)

                msg_ctx = MessageContext(
                    workspace_id=workspace_id,
                    folder_path=folder_path,
                    conversation_id=conv_id,
                    exchange_index=i,
                    message_id=exchange.get('uuid', exchange.get('request_id', '')),
                    timestamp=exchange.get('timestamp', ''),
                    user_message=user_msg,
                    project_summary=project_summary,
                    prev_assistant_response=prev_response[:4000],  # Truncate for context limits
                    next_assistant_response=next_response[:4000],
                    message_hash=msg_hash
                )
                messages.append(msg_ctx)

    return messages


def main():
    """Main pipeline for message classification."""
    import argparse

    parser = argparse.ArgumentParser(description="Two-stage message classification pipeline")
    parser.add_argument("--test-context", action="store_true", help="Test context assembly only")
    parser.add_argument("--test-stage1", type=int, default=0, help="Test Stage 1 on N messages")
    parser.add_argument("--test-stage2", type=int, default=0, help="Test Stage 2 on N messages")
    parser.add_argument("--run-stage1", action="store_true", help="Run Stage 1 (fast filter) only")
    parser.add_argument("--run-stage2", action="store_true", help="Run Stage 2 only (requires Stage 1 results)")
    parser.add_argument("--run-full", action="store_true", help="Run full pipeline (Stage 1 + Stage 2)")
    parser.add_argument("--data-dir", default="../augment_conversations_export_leveldb")
    parser.add_argument("--summaries-path", default="project_summaries/_consolidated_summaries.json")
    parser.add_argument("--output-dir", default="classification_results")
    args = parser.parse_args()

    # Load data
    project_summaries = load_project_summaries(args.summaries_path)
    workspaces = load_workspace_conversations(args.data_dir)
    messages = extract_messages_with_context(workspaces, project_summaries)
    logger.info(f"Extracted {len(messages)} unique user messages with context")

    if args.test_context:
        # Show sample messages with FULL prev/next responses
        logger.info("=" * 80)
        logger.info("SAMPLE MESSAGES WITH FULL CONTEXT")
        logger.info("=" * 80)
        for i, msg in enumerate(messages[:5]):
            logger.info(f"{'='*80}")
            logger.info(f"MESSAGE {i+1}")
            logger.info(f"{'='*80}")
            logger.info(f"Workspace: {Path(msg.folder_path).name}")
            logger.info(f"Conversation: {msg.conversation_id}")
            logger.info(f"Exchange index: {msg.exchange_index}")
            logger.info(f"--- USER MESSAGE ({len(msg.user_message)} chars) ---")
            logger.info(msg.user_message)
            logger.info(f"--- PREV ASSISTANT RESPONSE ({len(msg.prev_assistant_response)} chars) ---")
            logger.info(msg.prev_assistant_response if msg.prev_assistant_response else "(none - first message in conversation)")
            logger.info(f"--- NEXT ASSISTANT RESPONSE ({len(msg.next_assistant_response)} chars) ---")
            logger.info(msg.next_assistant_response if msg.next_assistant_response else "(none)")

    elif args.test_stage1 > 0:
        # Test Stage 1 on a small sample
        sample = messages[:args.test_stage1]
        output_path = Path(args.output_dir) / "_test_stage1.json"
        output_path.parent.mkdir(exist_ok=True)

        results = classify_stage1_batch(sample, str(output_path))

        logger.info("=" * 80)
        logger.info("STAGE 1 RESULTS")
        logger.info("=" * 80)
        for r in results:
            status = "HIGH" if r["is_high_value"] else "LOW"
            logger.info(f"[{status}] {r['message_preview'][:100]}...")
            logger.info(f"  Reason: {r['reason']}")

    elif args.test_stage2 > 0:
        # Test Stage 2 on a small sample with debug output
        sample = messages[:args.test_stage2]
        output_path = Path(args.output_dir) / "_test_stage2.json"
        output_path.parent.mkdir(exist_ok=True)

        # Test with debug mode - show raw LLM output
        logger.info("=" * 80)
        logger.info("TESTING GLM-4.6 RESPONSES (DEBUG)")
        logger.info("=" * 80)

        for i, msg in enumerate(sample):
            logger.info(f"--- Testing message {i+1} ---")
            logger.info(f"Message: {msg.user_message[:150]}...")

            user_prompt = STAGE2_USER_TEMPLATE.format(
                project_summary=msg.project_summary[:1500],
                prev_response=msg.prev_assistant_response[:800],
                message=msg.user_message[:2000],
                next_response=msg.next_assistant_response[:800]
            )

            llm_messages = [
                {"role": "system", "content": STAGE2_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]

            try:
                response = call_llm(llm_messages, GLM_MODEL, max_tokens=4096)
                logger.debug(f"Raw response ({len(response)} chars):")
                logger.debug(response[:1500])

                parsed = parse_json_response(response, debug=True)
                logger.info(f"Parsed result:")
                logger.info(f"  Labels: {parsed.get('labels', [])}")
                logger.info(f"  Generalizability: {parsed.get('generalizability', 0)}")
                logger.info(f"  Insights: {parsed.get('insights', [])}")
            except Exception as e:
                logger.error(f"Error: {e}")

    elif args.run_stage1:
        # Run Stage 1 only
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        logger.info("=" * 80)
        logger.info("RUNNING STAGE 1: FAST FILTER (ASYNC)")
        logger.info("=" * 80)
        stage1_output = output_dir / "stage1_filter_results.json"
        stage1_results = asyncio.run(classify_stage1_batch_async(messages, str(stage1_output), concurrency=10))

        high_value = sum(1 for r in stage1_results if r["is_high_value"])
        logger.info(f"Stage 1 complete: {high_value}/{len(stage1_results)} high-value messages")
        logger.info(f"Results saved to {stage1_output}")
        logger.info(f"Run --run-stage2 to classify high-value messages with GLM-4.6")

    elif args.run_stage2:
        # Run Stage 2 only (requires Stage 1 results)
        output_dir = Path(args.output_dir)
        stage1_output = output_dir / "stage1_filter_results.json"

        if not stage1_output.exists():
            logger.error(f"Stage 1 results not found at {stage1_output}")
            logger.error("Run --run-stage1 first")
            return

        # Load Stage 1 results
        with open(stage1_output, 'r', encoding='utf-8') as f:
            stage1_data = json.load(f)
        stage1_results = stage1_data.get("results", [])
        high_value_hashes = {r["message_hash"] for r in stage1_results if r["is_high_value"]}

        # Filter to high-value messages
        high_value_messages = [msg for msg in messages if msg.message_hash in high_value_hashes]

        logger.info(f"Loaded Stage 1 results: {len(high_value_hashes)} high-value messages")

        logger.info("=" * 80)
        logger.info("RUNNING STAGE 2: DEEP CLASSIFICATION")
        logger.info("=" * 80)
        stage2_output = output_dir / "stage2_classification_results.json"
        classify_stage2_batch(high_value_messages, str(stage2_output))

        logger.info("Stage 2 complete!")

    elif args.run_full:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        # Stage 1: Filter
        logger.info("=" * 80)
        logger.info("RUNNING STAGE 1: FAST FILTER (ASYNC)")
        logger.info("=" * 80)
        stage1_output = output_dir / "stage1_filter_results.json"
        stage1_results = asyncio.run(classify_stage1_batch_async(messages, str(stage1_output), concurrency=10))
        logger.info(f"Stage 1 results saved to {stage1_output}")

        # Build lookup of high-value message hashes
        high_value_hashes = {r["message_hash"] for r in stage1_results if r["is_high_value"]}

        # Stage 2: Deep classification on high-value only
        high_value_messages = [msg for msg in messages if msg.message_hash in high_value_hashes]

        logger.info("=" * 80)
        logger.info("RUNNING STAGE 2: DEEP CLASSIFICATION")
        logger.info("=" * 80)
        stage2_output = output_dir / "stage2_classification_results.json"
        classify_stage2_batch(high_value_messages, str(stage2_output))

        logger.info("Pipeline complete!")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

