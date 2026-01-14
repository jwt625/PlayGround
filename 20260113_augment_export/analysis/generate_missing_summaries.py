#!/usr/bin/env python3
"""
Consolidate project summaries for all workspaces.

Workflow for each workspace:
1. If workspace has extracted summaries:
   a. Validate each with Llama-4
   b. If >1 valid: consolidate with GLM-4.6, then validate
   c. If 1 valid: use it directly
   d. If 0 valid: fall back to generation from scratch
2. If workspace has 0 extracted summaries:
   a. Sample top 5 longest LLM responses
   b. Extract file types and tech stack
   c. Generate summary with GLM-4.6
   d. Validate with Llama-4
"""

import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
import httpx

load_dotenv()

API_KEY = os.getenv("LAMBDA_API_KEY")
API_BASE = os.getenv("LAMBDA_API_BASE")

MODEL_GENERATOR = "zai-org/GLM-4.6-FP8"
MODEL_VALIDATOR = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"

DATA_DIR = Path(__file__).parent.parent / "augment_conversations_export_leveldb"
OUTPUT_DIR = Path(__file__).parent / "project_summaries"

# File extension patterns to detect
FILE_PATTERNS = [
    (r'\.py\b', 'Python'),
    (r'\.ts\b', 'TypeScript'),
    (r'\.tsx\b', 'React/TypeScript'),
    (r'\.js\b', 'JavaScript'),
    (r'\.jsx\b', 'React/JavaScript'),
    (r'\.go\b', 'Go'),
    (r'\.rs\b', 'Rust'),
    (r'\.java\b', 'Java'),
    (r'\.cpp\b|\.cc\b|\.cxx\b', 'C++'),
    (r'\.c\b', 'C'),
    (r'\.rb\b', 'Ruby'),
    (r'\.php\b', 'PHP'),
    (r'\.swift\b', 'Swift'),
    (r'\.kt\b', 'Kotlin'),
    (r'\.sql\b', 'SQL'),
    (r'\.yaml\b|\.yml\b', 'YAML'),
    (r'\.json\b', 'JSON'),
    (r'\.md\b', 'Markdown'),
    (r'\.html\b', 'HTML'),
    (r'\.css\b|\.scss\b|\.sass\b', 'CSS'),
    (r'Dockerfile', 'Docker'),
    (r'docker-compose', 'Docker Compose'),
    (r'requirements\.txt', 'Python/pip'),
    (r'package\.json', 'Node.js'),
    (r'Cargo\.toml', 'Rust/Cargo'),
    (r'go\.mod', 'Go modules'),
    (r'pyproject\.toml', 'Python/uv/poetry'),
]

# Tech stack patterns in content
TECH_PATTERNS = [
    (r'\b(FastAPI|fastapi)\b', 'FastAPI'),
    (r'\b(Django|django)\b', 'Django'),
    (r'\b(Flask|flask)\b', 'Flask'),
    (r'\b(React|react)\b', 'React'),
    (r'\b(Vue|vue\.js)\b', 'Vue.js'),
    (r'\b(Next\.js|nextjs)\b', 'Next.js'),
    (r'\b(PostgreSQL|postgres|psycopg)\b', 'PostgreSQL'),
    (r'\b(MongoDB|mongo)\b', 'MongoDB'),
    (r'\b(Redis|redis)\b', 'Redis'),
    (r'\b(Kubernetes|k8s)\b', 'Kubernetes'),
    (r'\b(AWS|Lambda|S3|EC2)\b', 'AWS'),
    (r'\b(Slack API|slack_sdk|SlackClient)\b', 'Slack'),
    (r'\b(OpenAI|GPT|ChatGPT)\b', 'OpenAI'),
    (r'\b(LangChain|langchain)\b', 'LangChain'),
    (r'\b(pytest|unittest)\b', 'pytest'),
    (r'\basync\s+def\b|await\b', 'async/await'),
]

SYSTEM_PROMPT_GENERATOR = """You are a technical writer creating concise project summaries.

Given context about a software project (folder path, file types, and conversation excerpts), write a clear ~200 word summary that covers:
1. What the project is (main purpose)
2. Key technologies used
3. Main components or features mentioned

Write in third person, present tense. Be factual based only on provided context.
Do NOT include meta-commentary or explanations about your process.
Output ONLY the summary text."""

SYSTEM_PROMPT_VALIDATOR = """You are evaluating if a project summary is valid and useful.

A valid summary should:
1. Describe a coherent software project
2. Mention specific technologies or features
3. Be factual (not vague or generic)
4. Be approximately 100-250 words

Respond with exactly one word: VALID or INVALID"""

SYSTEM_PROMPT_CONSOLIDATOR = """You are a technical writer consolidating multiple project summaries into one coherent summary.

Given multiple summaries about the same software project, create a single ~200 word summary that:
1. Combines the key information from all summaries
2. Removes redundancy and contradictions
3. Covers: project purpose, technologies used, main components/features
4. Is written in third person, present tense

Output ONLY the consolidated summary text, no meta-commentary."""


def call_llm(model: str, system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> str:
    """Call LLM and return content, parsing out thinking tokens for GLM."""
    resp = httpx.post(
        f"{API_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.3
        },
        timeout=120
    )
    
    if resp.status_code != 200:
        raise Exception(f"API error {resp.status_code}: {resp.text[:200]}")
    
    content = resp.json()["choices"][0]["message"]["content"]
    
    # Parse out thinking tokens from GLM-4.6
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()
    
    return content


def extract_file_types(text: str) -> set:
    """Extract mentioned file types from text."""
    found = set()
    for pattern, name in FILE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            found.add(name)
    return found


def extract_tech_stack(text: str) -> set:
    """Extract mentioned technologies from text."""
    found = set()
    for pattern, name in TECH_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            found.add(name)
    return found


def get_workspace_context(workspace_file: Path) -> dict:
    """Extract context from a workspace's conversation data."""
    with open(workspace_file) as f:
        data = json.load(f)

    folder_path = data.get("folder_path", "")
    conversations = data.get("conversations", [])

    # Collect all responses and requests
    all_responses = []
    all_requests = []
    all_text = ""

    for conv in conversations:
        for ex in conv.get("exchanges", []):
            resp = ex.get("response_text", "")
            req = ex.get("request_message", "")

            if resp:
                all_responses.append({"text": resp, "length": len(resp)})
                all_text += " " + resp
            if req:
                all_requests.append(req)
                all_text += " " + req

    # Sort by length and get top 5
    all_responses.sort(key=lambda x: x["length"], reverse=True)
    top_responses = [r["text"][:2000] for r in all_responses[:5]]  # Truncate each

    # Extract file types and tech stack
    file_types = extract_file_types(all_text)
    tech_stack = extract_tech_stack(all_text)

    # Sample user requests (first 10, truncated)
    sample_requests = [r[:200] for r in all_requests[:10]]

    return {
        "folder_path": folder_path,
        "num_conversations": len(conversations),
        "num_responses": len(all_responses),
        "top_responses": top_responses,
        "sample_requests": sample_requests,
        "file_types": list(file_types),
        "tech_stack": list(tech_stack)
    }


def build_generation_prompt(context: dict) -> str:
    """Build the prompt for generating a project summary."""
    parts = []

    parts.append(f"## Project Folder\n{context['folder_path']}")

    if context["file_types"]:
        parts.append(f"\n## File Types Detected\n{', '.join(context['file_types'])}")

    if context["tech_stack"]:
        parts.append(f"\n## Technologies Mentioned\n{', '.join(context['tech_stack'])}")

    if context["sample_requests"]:
        parts.append("\n## Sample User Requests")
        for i, req in enumerate(context["sample_requests"][:5], 1):
            parts.append(f"{i}. {req}")

    if context["top_responses"]:
        parts.append("\n## Excerpts from LLM Responses (longest ones)")
        for i, resp in enumerate(context["top_responses"][:3], 1):
            # Take first 800 chars of each
            excerpt = resp[:800].replace("\n", " ")
            parts.append(f"\n### Response {i}\n{excerpt}...")

    return "\n".join(parts)


def generate_summary_for_workspace(workspace_file: Path) -> dict:
    """Generate a summary for a workspace with no extracted summaries."""
    print(f"\nProcessing: {workspace_file.name}")

    context = get_workspace_context(workspace_file)
    print(f"  Folder: {context['folder_path']}")
    print(f"  Conversations: {context['num_conversations']}, Responses: {context['num_responses']}")
    print(f"  File types: {context['file_types']}")
    print(f"  Tech stack: {context['tech_stack']}")

    if context["num_responses"] == 0:
        print("  Skipping: no responses found")
        return None

    # Build prompt and generate
    user_prompt = build_generation_prompt(context)
    print(f"  Generating summary with {MODEL_GENERATOR}...")

    try:
        summary = call_llm(MODEL_GENERATOR, SYSTEM_PROMPT_GENERATOR, user_prompt, max_tokens=2000)
        print(f"  Generated {len(summary)} chars")

        # Validate with Llama-4
        print(f"  Validating with {MODEL_VALIDATOR}...")
        validation = call_llm(
            MODEL_VALIDATOR,
            SYSTEM_PROMPT_VALIDATOR,
            f"Project summary to evaluate:\n\n{summary}",
            max_tokens=10
        )
        is_valid = "VALID" in validation.upper()
        print(f"  Validation: {validation.strip()} -> {'PASS' if is_valid else 'FAIL'}")

        return {
            "folder_path": context["folder_path"],
            "final_summary": summary,
            "source": "generated",
            "validation": validation.strip(),
            "is_valid": is_valid,
            "context": {
                "file_types": context["file_types"],
                "tech_stack": context["tech_stack"],
                "num_conversations": context["num_conversations"],
                "num_responses": context["num_responses"]
            }
        }
    except Exception as e:
        print(f"  Error: {e}")
        return {"folder_path": context["folder_path"], "source": "generated", "error": str(e), "is_valid": False}


def validate_summary(summary_text: str) -> tuple[bool, str]:
    """Validate a summary using Llama-4. Returns (is_valid, raw_response)."""
    try:
        response = call_llm(
            MODEL_VALIDATOR,
            SYSTEM_PROMPT_VALIDATOR,
            f"Project summary to evaluate:\n\n{summary_text}",
            max_tokens=10
        )
        # Check for VALID but not INVALID
        resp_upper = response.upper().strip()
        is_valid = resp_upper.startswith("VALID") and not resp_upper.startswith("INVALID")
        return is_valid, response.strip()
    except Exception as e:
        return False, f"Error: {e}"


def consolidate_summaries(summaries: list[str], folder_path: str) -> str:
    """Consolidate multiple summaries into one using GLM-4.6."""
    prompt_parts = [f"## Project Folder\n{folder_path}\n"]
    prompt_parts.append("## Summaries to Consolidate\n")
    for i, s in enumerate(summaries, 1):
        # Truncate each summary to avoid token limits
        prompt_parts.append(f"### Summary {i}\n{s[:1500]}\n")

    return call_llm(
        MODEL_GENERATOR,
        SYSTEM_PROMPT_CONSOLIDATOR,
        "\n".join(prompt_parts),
        max_tokens=2000
    )


def process_workspace_with_summaries(summary_file: Path) -> dict:
    """Process a workspace that has extracted summaries."""
    with open(summary_file) as f:
        data = json.load(f)

    folder_path = data.get("folder_path", "")
    project_name = data.get("project_name", "")
    summaries = data.get("summaries", [])

    print(f"\nProcessing: {project_name}")
    print(f"  Folder: {folder_path}")
    print(f"  Extracted summaries: {len(summaries)}")

    if not summaries:
        return None

    # Step 1: Validate each extracted summary with Llama-4
    valid_summaries = []
    for i, s in enumerate(summaries):
        text = s.get("response_text", "")
        if not text:
            continue

        # Truncate for validation (first 2000 chars should be enough)
        is_valid, validation = validate_summary(text[:2000])
        print(f"    Summary {i+1} (score {s.get('score', 0):.2f}, {len(text)} chars): {validation}")

        if is_valid:
            valid_summaries.append(text)

    print(f"  Valid summaries: {len(valid_summaries)}/{len(summaries)}")

    # Step 2: Handle based on count
    if len(valid_summaries) == 0:
        # Fall back to generation
        print("  No valid summaries, falling back to generation...")
        # Find the raw data file
        hash_prefix = project_name.split("_")[-1]
        for f in DATA_DIR.glob("*.json"):
            if "extraction_summary" in f.name:
                continue
            if hash_prefix in f.stem:
                return generate_summary_for_workspace(f)
        return {"folder_path": folder_path, "error": "No data file found for generation"}

    elif len(valid_summaries) == 1:
        # Use the single valid summary directly
        print("  Using single valid summary directly")
        final_summary = valid_summaries[0]
        return {
            "folder_path": folder_path,
            "project_name": project_name,
            "final_summary": final_summary,
            "source": "single_valid",
            "validated_count": 1,
            "is_valid": True
        }

    else:
        # Consolidate multiple valid summaries
        print(f"  Consolidating {len(valid_summaries)} valid summaries with {MODEL_GENERATOR}...")
        # Take top 5 to avoid token limits
        to_consolidate = valid_summaries[:5]
        consolidated = consolidate_summaries(to_consolidate, folder_path)
        print(f"  Consolidated to {len(consolidated)} chars")

        # Validate consolidated summary
        is_valid, validation = validate_summary(consolidated)
        print(f"  Consolidated validation: {validation}")

        return {
            "folder_path": folder_path,
            "project_name": project_name,
            "final_summary": consolidated,
            "source": "consolidated",
            "validated_count": len(valid_summaries),
            "consolidated_from": len(to_consolidate),
            "validation": validation,
            "is_valid": is_valid
        }


def load_existing_results() -> dict:
    """Load existing results to avoid reprocessing."""
    existing = {}

    # Load from consolidated file
    consolidated_file = OUTPUT_DIR / "_consolidated_summaries.json"
    if consolidated_file.exists():
        with open(consolidated_file) as f:
            for r in json.load(f):
                existing[r.get("folder_path")] = r

    # Also load from generated file (for workspaces with 0 extracted summaries)
    generated_file = OUTPUT_DIR / "_generated_summaries.json"
    if generated_file.exists():
        with open(generated_file) as f:
            for r in json.load(f):
                folder_path = r.get("folder_path")
                if folder_path and folder_path not in existing:
                    # Convert old format to new format
                    existing[folder_path] = {
                        "folder_path": folder_path,
                        "final_summary": r.get("generated_summary", r.get("final_summary", "")),
                        "source": "generated",
                        "is_valid": r.get("is_valid", False),
                        "validation": r.get("validation", ""),
                        "context": r.get("context", {})
                    }

    return existing


def get_all_workspaces() -> list[dict]:
    """Get all workspaces with their summary files and stats."""
    stats_file = OUTPUT_DIR / "_extraction_stats.json"
    if not stats_file.exists():
        print("No extraction stats found. Run extract_project_summaries.py first.")
        return []

    with open(stats_file) as f:
        stats = json.load(f)

    workspaces = []
    for ws_info in stats.get("workspaces", []):
        project_name = ws_info.get("project_name", "")
        folder_path = ws_info.get("folder_path", "")
        total_summaries = ws_info.get("total_summaries", 0)
        hash_prefix = project_name.split("_")[-1]

        # Find summary file if exists
        summary_file = OUTPUT_DIR / f"{project_name}_summaries.json"

        # Find raw data file
        data_file = None
        for f in DATA_DIR.glob("*.json"):
            if "extraction_summary" in f.name:
                continue
            if hash_prefix in f.stem:
                data_file = f
                break

        workspaces.append({
            "project_name": project_name,
            "folder_path": folder_path,
            "total_summaries": total_summaries,
            "summary_file": summary_file if summary_file.exists() else None,
            "data_file": data_file
        })

    return workspaces


def main():
    print("=" * 60)
    print("Consolidate Project Summaries")
    print("=" * 60)

    # Load existing results to skip already processed
    existing = load_existing_results()
    print(f"Already processed: {len(existing)} workspaces")

    # Get all workspaces
    workspaces = get_all_workspaces()
    print(f"Total workspaces: {len(workspaces)}")

    results = list(existing.values())  # Start with existing

    for ws in workspaces:
        folder_path = ws["folder_path"]

        # Skip if already processed
        if folder_path in existing:
            print(f"\nSkipping (already done): {ws['project_name']}")
            continue

        if ws["total_summaries"] > 0 and ws["summary_file"]:
            # Has extracted summaries - validate and consolidate
            result = process_workspace_with_summaries(ws["summary_file"])
        elif ws["data_file"]:
            # No extracted summaries - generate from scratch
            print(f"\nGenerating for: {ws['project_name']} (0 extracted summaries)")
            result = generate_summary_for_workspace(ws["data_file"])
        else:
            print(f"\nSkipping (no data): {ws['project_name']}")
            continue

        if result:
            results.append(result)

            # Save after each to preserve progress
            output_file = OUTPUT_DIR / "_consolidated_summaries.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)

    # Final save
    output_file = OUTPUT_DIR / "_consolidated_summaries.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved {len(results)} consolidated summaries to {output_file}")

    # Stats
    valid_count = sum(1 for r in results if r.get("is_valid"))
    sources = {}
    for r in results:
        src = r.get("source", "generated")
        sources[src] = sources.get(src, 0) + 1

    print(f"Valid summaries: {valid_count}/{len(results)}")
    print(f"Sources: {sources}")


if __name__ == "__main__":
    main()

