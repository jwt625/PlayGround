#!/usr/bin/env python3
"""
Comprehensive tool extraction from all Claude Code log files.

Extracts ALL tool definitions including:
- Tool name, description, input_schema
- Tool IDs (from tool_use blocks)
- System prompts associated with each tool configuration
- Variations of the same tool (e.g., Task tool with different system prompts)

Usage:
    python3 extract_all_tools.py           # Simplified output (default)
    python3 extract_all_tools.py --full    # Full detailed output
"""

import json
import hashlib
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Set, Tuple


def compute_hash(text: str, length: int = 200) -> str:
    """Compute SHA256 hash of first N characters."""
    if not text:
        return ""
    normalized = ' '.join(text.split())[:length]
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def extract_system_prompt(log_entry: Dict[str, Any]) -> tuple[str, str]:
    """Extract system prompt text and its hash from log entry."""
    system = log_entry.get('body', {}).get('system', [])
    if not system:
        return "", ""

    # Concatenate all system prompt texts
    texts = []
    for prompt in system:
        if isinstance(prompt, dict):
            text = prompt.get('text', '')
            if text:
                texts.append(text)

    combined = '|||'.join(texts)
    return combined, compute_hash(combined, length=500)


def extract_tool_definitions(log_entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract tool definitions from request body."""
    tools = log_entry.get('body', {}).get('tools', [])
    tool_defs = []
    
    for tool in tools:
        if isinstance(tool, dict):
            tool_defs.append({
                'name': tool.get('name', 'Unknown'),
                'description': tool.get('description', ''),
                'input_schema': tool.get('input_schema', {}),
                'description_hash': compute_hash(tool.get('description', ''), length=300)
            })
    
    return tool_defs


def extract_tool_uses(log_entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract tool_use blocks from response content."""
    tool_uses = []
    response_body = log_entry.get('response', {}).get('body', {})
    content = response_body.get('content', [])
    
    if not isinstance(content, list):
        return tool_uses
    
    for block in content:
        if isinstance(block, dict) and block.get('type') == 'tool_use':
            tool_uses.append({
                'id': block.get('id', ''),
                'name': block.get('name', ''),
                'input': block.get('input', {})
            })
    
    return tool_uses


def parse_all_logs(log_dir: Path) -> Dict[str, Any]:
    """
    Parse all log files and extract comprehensive tool information.
    
    Returns:
        Dictionary with tool definitions, variations, and usage statistics
    """
    # Storage for tool information
    tool_definitions = defaultdict(list)  # tool_name -> list of unique definitions
    tool_variations = defaultdict(set)  # tool_name -> set of (desc_hash, system_hash)
    tool_use_ids = defaultdict(set)  # tool_name -> set of tool_use IDs
    tool_inputs = defaultdict(list)  # tool_name -> list of sample inputs
    system_prompt_contexts = defaultdict(set)  # system_hash -> set of tool names
    system_prompts = {}  # system_hash -> actual prompt text

    # Get all log files
    log_files = sorted(log_dir.glob("*.jsonl"))

    print(f"Found {len(log_files)} log files")

    for log_file in log_files:
        print(f"Processing {log_file.name}...")

        with open(log_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    log_entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Extract system prompt text and hash
                system_text, system_hash = extract_system_prompt(log_entry)
                if system_hash and system_hash not in system_prompts:
                    system_prompts[system_hash] = system_text

                # Extract tool definitions from request
                tool_defs = extract_tool_definitions(log_entry)
                for tool_def in tool_defs:
                    tool_name = tool_def['name']
                    desc_hash = tool_def['description_hash']

                    # Track variation (combination of description + system prompt)
                    variation_key = (desc_hash, system_hash)

                    if variation_key not in tool_variations[tool_name]:
                        tool_variations[tool_name].add(variation_key)
                        tool_definitions[tool_name].append({
                            'description': tool_def['description'],
                            'input_schema': tool_def['input_schema'],
                            'description_hash': desc_hash,
                            'system_prompt_hash': system_hash,
                            'first_seen_file': log_file.name,
                            'first_seen_line': line_num
                        })

                    # Track which tools appear with which system prompts
                    if system_hash:
                        system_prompt_contexts[system_hash].add(tool_name)

                # Extract tool uses from response
                tool_uses = extract_tool_uses(log_entry)
                for tool_use in tool_uses:
                    tool_name = tool_use['name']
                    tool_id = tool_use['id']

                    if tool_id:
                        tool_use_ids[tool_name].add(tool_id)

                    # Store sample inputs (limit to 5 per tool)
                    if len(tool_inputs[tool_name]) < 5:
                        tool_inputs[tool_name].append(tool_use['input'])

    return {
        'tool_definitions': dict(tool_definitions),
        'tool_variations': {k: len(v) for k, v in tool_variations.items()},
        'tool_use_ids': {k: list(v) for k, v in tool_use_ids.items()},
        'tool_inputs': dict(tool_inputs),
        'system_prompt_contexts': {k: list(v) for k, v in system_prompt_contexts.items()},
        'system_prompts': system_prompts
    }


def format_tool_output(data: Dict[str, Any], simplified: bool = True) -> Dict[str, Any]:
    """
    Format extracted tool data.

    Args:
        data: Raw extracted data
        simplified: If True, only include essential fields (default)

    Returns:
        Dictionary with summary stats and tool information
    """
    tools = []

    for tool_name in sorted(data['tool_definitions'].keys()):
        definitions = data['tool_definitions'][tool_name]
        variation_count = data['tool_variations'].get(tool_name, 0)
        use_ids = data['tool_use_ids'].get(tool_name, [])

        # Create base tool entry
        tool_entry = {
            'name': tool_name,
            'variation_count': variation_count,
            'total_use_ids': len(use_ids),
            'variations': []
        }

        # Add use IDs only if simplified
        if simplified:
            tool_entry['use_ids'] = use_ids
        else:
            tool_entry['sample_use_ids'] = use_ids[:10]

        # Add each variation
        for idx, definition in enumerate(definitions, 1):
            if simplified:
                # Get system prompt text
                sys_hash = definition['system_prompt_hash']
                sys_text = data['system_prompts'].get(sys_hash, '')

                variation = {
                    'variation_id': idx,
                    'description': definition['description'],
                    'system_prompt': sys_text,
                }
            else:
                sample_inputs = data['tool_inputs'].get(tool_name, [])
                sys_hash = definition['system_prompt_hash']
                sys_text = data['system_prompts'].get(sys_hash, '')

                variation = {
                    'variation_id': idx,
                    'description': definition['description'],
                    'input_schema': definition['input_schema'],
                    'description_hash': definition['description_hash'],
                    'system_prompt': sys_text,
                    'system_prompt_hash': definition['system_prompt_hash'],
                    'first_seen': {
                        'file': definition['first_seen_file'],
                        'line': definition['first_seen_line']
                    }
                }
                # Add sample input for first variation only
                if idx == 1 and sample_inputs:
                    variation['sample_inputs'] = sample_inputs

            tool_entry['variations'].append(variation)

        tools.append(tool_entry)

    # Create output with summary
    return {
        'summary': {
            'total_tools': len(data['tool_definitions']),
            'total_variations': sum(data['tool_variations'].values()),
            'tools_with_multiple_variations': sum(1 for v in data['tool_variations'].values() if v > 1),
            'total_unique_tool_use_ids': sum(len(ids) for ids in data['tool_use_ids'].values()),
            'system_prompt_contexts': len(data['system_prompt_contexts'])
        },
        'tools': tools
    }


def print_summary(data: Dict[str, Any]):
    """Print a summary of extracted tools."""
    print(f"\n{'='*80}")
    print("TOOL EXTRACTION SUMMARY")
    print(f"{'='*80}\n")

    total_tools = len(data['tool_definitions'])
    total_variations = sum(data['tool_variations'].values())

    print(f"Total unique tools: {total_tools}")
    print(f"Total variations: {total_variations}")
    print(f"\nTools with multiple variations:")

    for tool_name, count in sorted(data['tool_variations'].items(), key=lambda x: -x[1]):
        if count > 1:
            use_count = len(data['tool_use_ids'].get(tool_name, []))
            print(f"  - {tool_name}: {count} variations, {use_count} unique IDs")

    print(f"\nSystem prompt contexts: {len(data['system_prompt_contexts'])}")

    # Show which tools appear together in different contexts
    print(f"\nTool groupings by system prompt:")
    for idx, (sys_hash, tools) in enumerate(data['system_prompt_contexts'].items(), 1):
        if len(tools) > 0:
            print(f"  Context {idx} ({sys_hash[:8]}): {len(tools)} tools")
            if len(tools) <= 5:
                print(f"    Tools: {', '.join(sorted(tools))}")


def main():
    """Main execution function."""
    # Check for --full flag
    simplified = '--full' not in sys.argv

    script_dir = Path(__file__).parent
    log_dir = script_dir.parent / 'proxy' / 'logs'
    output_file = script_dir / 'tools_extracted.json'

    if not log_dir.exists():
        print(f"Error: Log directory not found at {log_dir}")
        return

    mode = "simplified" if simplified else "full"
    print(f"Parsing logs from: {log_dir} (mode: {mode})")

    # Extract all tool information
    data = parse_all_logs(log_dir)

    # Print summary
    print_summary(data)

    # Format for JSON output
    formatted_output = format_tool_output(data, simplified=simplified)

    # Save to JSON file (overwrite existing)
    with open(output_file, 'w') as f:
        json.dump(formatted_output, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Updated {output_file} ({mode} mode)")
    print(f"Total tools: {formatted_output['summary']['total_tools']}")
    print(f"Total variations: {formatted_output['summary']['total_variations']}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()


