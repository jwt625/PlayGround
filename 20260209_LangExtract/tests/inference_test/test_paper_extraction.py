#!/usr/bin/env python3
"""
Test script for extracting SC qubit-related papers from blog posts using GLM-4.7.

Uses tool-call framing for structured extraction, making it scalable and robust.
"""

import json
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_URL = os.getenv("API_URL")  # Required: set in .env
API_TOKEN = os.getenv("API_TOKEN")
MODEL_ID = os.getenv("MODEL_ID")  # Required: set in .env

# Tool definition for structured paper extraction
EXTRACT_PAPERS_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_papers",
        "description": "Extract academic papers related to superconducting circuits and qubits from text",
        "parameters": {
            "type": "object",
            "properties": {
                "papers": {
                    "type": "array",
                    "description": "List of extracted papers",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Paper title or description"
                            },
                            "url": {
                                "type": "string",
                                "description": "URL to the paper (DOI, arXiv, or direct link)"
                            },
                            "authors": {
                                "type": "string",
                                "description": "Author names if available"
                            },
                            "year": {
                                "type": "integer",
                                "description": "Publication year if available"
                            },
                            "relevance": {
                                "type": "string",
                                "description": "Why this paper is relevant to SC qubits/circuits"
                            }
                        },
                        "required": ["title", "url"]
                    }
                },
                "has_sc_qubit_content": {
                    "type": "boolean",
                    "description": "Whether the text contains SC qubit related content"
                },
                "summary": {
                    "type": "string",
                    "description": "Brief summary of SC qubit content found"
                }
            },
            "required": ["papers", "has_sc_qubit_content"]
        }
    }
}

SYSTEM_PROMPT = """You are an expert in superconducting quantum computing and circuit QED.
Your task is to extract references to academic papers related to superconducting circuits and qubits.

Focus on papers about:
- Josephson junctions and their fabrication
- Transmon, fluxonium, and other SC qubit types
- Circuit QED (cQED)
- Qubit coherence and materials
- Quantum error correction for SC qubits
- Dilution refrigerators and measurement setups

Extract the paper title, URL, authors (if mentioned), year, and explain why it's relevant.
If no SC qubit related papers are found, set has_sc_qubit_content to false.

IMPORTANT: After your reasoning, you MUST call the extract_papers function with your findings."""


def fetch_blog_post(url: str) -> str:
    """Fetch raw content of a blog post from GitHub."""
    print(f"Fetching: {url}")
    response = httpx.get(url, timeout=30.0)
    response.raise_for_status()
    return response.text


def extract_papers_from_text(text: str, timeout: float = 300.0) -> dict:
    """
    Send text to GLM-4.7 and extract SC qubit papers using tool calling.
    
    Args:
        text: The blog post content
        timeout: Request timeout in seconds (GLM-4.7 can be slow with thinking)
    
    Returns:
        Extracted papers and metadata
    """
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract SC qubit related papers from this blog post:\n\n{text}"}
        ],
        "tools": [EXTRACT_PAPERS_TOOL],
        "tool_choice": {"type": "function", "function": {"name": "extract_papers"}},
        "max_tokens": 8192,  # Large for thinking tokens
        "temperature": 0.1
    }
    
    print(f"Sending request to {API_URL}/chat/completions...")
    print(f"Model: {MODEL_ID}")
    print(f"Timeout: {timeout}s")
    
    start_time = time.time()
    
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            f"{API_URL}/chat/completions",
            headers=headers,
            json=payload
        )
    
    elapsed = time.time() - start_time
    print(f"Response received in {elapsed:.1f}s")
    
    response.raise_for_status()
    result = response.json()
    
    return result


def parse_response(result: dict) -> dict:
    """Parse the API response, handling thinking tokens and tool calls."""
    choice = result.get("choices", [{}])[0]
    message = choice.get("message", {})
    
    # Check for tool calls
    tool_calls = message.get("tool_calls", [])
    if tool_calls:
        for tc in tool_calls:
            if tc.get("function", {}).get("name") == "extract_papers":
                args = tc["function"].get("arguments", "{}")
                return json.loads(args)
    
    # Fallback: try to parse from content (if model didn't use tool call properly)
    content = message.get("content", "")
    
    # Look for </think> to find end of thinking
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()
    
    # Try to find JSON in content
    if "{" in content and "}" in content:
        start = content.find("{")
        end = content.rfind("}") + 1
        try:
            return json.loads(content[start:end])
        except json.JSONDecodeError:
            pass
    
    return {"error": "Could not parse response", "raw_content": content}


def main():
    """Main test function."""
    import sys

    # Default: SC qubits intro post (known to have many SC qubit papers)
    default_url = "https://raw.githubusercontent.com/jwt625/jwt625.github.io/master/_posts/2025-09-28-20250619-sc-qubits-intro.md"

    # Accept URL as command line argument
    test_url = sys.argv[1] if len(sys.argv) > 1 else default_url
    
    print("=" * 60)
    print("SC Qubit Paper Extraction Test")
    print("=" * 60)
    
    # Fetch the blog post
    content = fetch_blog_post(test_url)
    print(f"Fetched {len(content)} characters")
    print()
    
    # Extract papers
    result = extract_papers_from_text(content)
    
    # Save raw response
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "raw_response.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Raw response saved to {output_dir / 'raw_response.json'}")
    
    # Parse and display results
    parsed = parse_response(result)
    
    with open(output_dir / "extracted_papers.json", "w") as f:
        json.dump(parsed, f, indent=2)
    print(f"Parsed results saved to {output_dir / 'extracted_papers.json'}")
    
    print()
    print("=" * 60)
    print("EXTRACTION RESULTS")
    print("=" * 60)
    
    if "error" in parsed:
        print(f"Error: {parsed['error']}")
        if "raw_content" in parsed:
            print(f"Raw content preview: {parsed['raw_content'][:500]}...")
    else:
        has_content = parsed.get("has_sc_qubit_content", False)
        print(f"Has SC qubit content: {has_content}")
        
        if parsed.get("summary"):
            print(f"Summary: {parsed['summary']}")
        
        papers = parsed.get("papers", [])
        print(f"\nExtracted {len(papers)} papers:")
        print("-" * 40)
        
        for i, paper in enumerate(papers, 1):
            print(f"\n{i}. {paper.get('title', 'Unknown')}")
            print(f"   URL: {paper.get('url', 'N/A')}")
            if paper.get("authors"):
                print(f"   Authors: {paper['authors']}")
            if paper.get("year"):
                print(f"   Year: {paper['year']}")
            if paper.get("relevance"):
                print(f"   Relevance: {paper['relevance']}")
    
    # Print usage stats
    usage = result.get("usage", {})
    if usage:
        print()
        print("-" * 40)
        print(f"Token usage: {usage.get('prompt_tokens', 0)} prompt + "
              f"{usage.get('completion_tokens', 0)} completion = "
              f"{usage.get('total_tokens', 0)} total")


if __name__ == "__main__":
    main()

