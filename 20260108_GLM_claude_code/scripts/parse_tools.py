#!/usr/bin/env python3
"""
Parse tools and their descriptions from the requests_20260108.jsonl log file.
Extracts the tools list from the 5:30:54 PM entry (user msg is "foo").
"""

import json
from pathlib import Path


def parse_tools_from_log(log_file_path: str, target_time: str = "01:30:54"):
    """
    Parse the log file and extract tools with their descriptions.
    
    Args:
        log_file_path: Path to the JSONL log file
        target_time: Time string to search for (format: HH:MM:SS)
    
    Returns:
        List of dictionaries containing tool name and description
    """
    tools_info = []
    
    with open(log_file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
                
            try:
                log_entry = json.loads(line)
                
                # Check if this is the target entry
                timestamp = log_entry.get('timestamp', '')
                if target_time not in timestamp:
                    continue
                
                # Extract tools from the request body
                body = log_entry.get('body', {})
                tools = body.get('tools', [])
                
                if not tools:
                    continue
                
                # Extract name and description for each tool
                for tool in tools:
                    tool_name = tool.get('name', 'Unknown')
                    tool_desc = tool.get('description', 'No description')
                    
                    tools_info.append({
                        'name': tool_name,
                        'description': tool_desc
                    })
                
                # Found the target entry, no need to continue
                if tools_info:
                    break
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON line: {e}")
                continue
    
    return tools_info


def print_tools_summary(tools_info: list):
    """Print a formatted summary of tools and their descriptions."""
    print(f"\n{'='*80}")
    print(f"Found {len(tools_info)} tools in the log entry")
    print(f"{'='*80}\n")
    
    for i, tool in enumerate(tools_info, 1):
        print(f"{i}. Tool: {tool['name']}")
        print(f"   Description (first 200 chars):")
        desc = tool['description']
        if len(desc) > 200:
            print(f"   {desc[:200]}...")
        else:
            print(f"   {desc}")
        print()


def save_tools_to_json(tools_info: list, output_file: str):
    """Save the tools information to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(tools_info, f, indent=2)
    print(f"Saved full tool descriptions to: {output_file}")


def main():
    # Determine the log file path
    script_dir = Path(__file__).parent
    log_file = script_dir.parent / 'proxy' / 'logs' / 'requests_20260108.jsonl'
    
    if not log_file.exists():
        print(f"Error: Log file not found at {log_file}")
        return
    
    print(f"Parsing log file: {log_file}")
    
    # Parse the tools
    tools_info = parse_tools_from_log(str(log_file))
    
    if not tools_info:
        print("No tools found in the log entry!")
        return
    
    # Print summary
    print_tools_summary(tools_info)
    
    # Save to JSON file
    output_file = script_dir / 'tools_extracted.json'
    save_tools_to_json(tools_info, str(output_file))


if __name__ == '__main__':
    main()

