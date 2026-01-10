#!/usr/bin/env python3
"""
Analyze system prompts from Claude Code log files.
Extracts all distinct system prompt combinations and provides statistics.
"""

import json
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any


def extract_system_prompts(log_file_path: str) -> List[Dict[str, Any]]:
    """
    Extract all system prompts from the log file.
    
    Returns:
        List of dictionaries containing system prompt data
    """
    system_prompts = []
    
    with open(log_file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            try:
                log_entry = json.loads(line)
                
                # Extract system prompts from request body
                body = log_entry.get('body', {})
                system = body.get('system', [])
                
                if system:
                    system_prompts.append({
                        'line_num': line_num,
                        'timestamp': log_entry.get('timestamp', ''),
                        'model': body.get('model', 'unknown'),
                        'system': system,
                        'num_prompts': len(system)
                    })
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON at line {line_num}: {e}")
                continue
    
    return system_prompts


def compute_prompt_hash(system_prompts: List[Dict[str, str]]) -> str:
    """
    Compute a hash for a system prompt combination.
    Uses the first 200 characters of each prompt to create a signature.
    """
    signature = []
    for prompt in system_prompts:
        text = prompt.get('text', '')
        # Use first 200 chars as signature
        signature.append(text[:200])
    
    combined = '|||'.join(signature)
    return hashlib.md5(combined.encode()).hexdigest()


def group_by_combination(system_prompts: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group system prompts by their combination (using hash).
    """
    grouped = defaultdict(list)
    
    for entry in system_prompts:
        prompt_hash = compute_prompt_hash(entry['system'])
        grouped[prompt_hash].append(entry)
    
    return grouped


def print_summary(grouped: Dict[str, List[Dict[str, Any]]]):
    """
    Print summary statistics.
    """
    print("=" * 80)
    print("SYSTEM PROMPTS ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"\nTotal log entries with system prompts: {sum(len(v) for v in grouped.values())}")
    print(f"Distinct system prompt combinations: {len(grouped)}")
    print()


def print_prompt_details(grouped: Dict[str, List[Dict[str, Any]]]):
    """
    Print detailed information about each distinct prompt combination.
    """
    for idx, (prompt_hash, entries) in enumerate(sorted(grouped.items(), key=lambda x: -len(x[1])), 1):
        print(f"\n{'=' * 80}")
        print(f"COMBINATION #{idx} (Hash: {prompt_hash[:8]}...)")
        print(f"{'=' * 80}")
        print(f"Occurrences: {len(entries)}")
        print(f"Number of prompts in combination: {entries[0]['num_prompts']}")
        print(f"Models used: {', '.join(set(e['model'] for e in entries))}")
        print()
        
        # Show the actual prompts
        system = entries[0]['system']
        for i, prompt in enumerate(system, 1):
            text = prompt.get('text', '')
            cache_control = prompt.get('cache_control', {})
            
            print(f"\n--- Prompt {i}/{len(system)} ---")
            if cache_control:
                print(f"Cache control: {cache_control}")
            
            # Print first 500 characters
            if len(text) <= 500:
                print(text)
            else:
                print(text[:500])
                print(f"\n... (truncated, total length: {len(text)} characters)")
        
        print(f"\nFirst occurrence: {entries[0]['timestamp']} (line {entries[0]['line_num']})")
        print(f"Last occurrence: {entries[-1]['timestamp']} (line {entries[-1]['line_num']})")


def save_full_prompts(grouped: Dict[str, List[Dict[str, Any]]], output_file: str):
    """
    Save full prompts to a JSON file for detailed analysis.
    """
    output_data = []
    
    for prompt_hash, entries in grouped.items():
        output_data.append({
            'hash': prompt_hash,
            'occurrences': len(entries),
            'num_prompts': entries[0]['num_prompts'],
            'models': list(set(e['model'] for e in entries)),
            'system_prompts': entries[0]['system'],
            'first_occurrence': entries[0]['timestamp'],
            'last_occurrence': entries[-1]['timestamp'],
            'line_numbers': [e['line_num'] for e in entries]
        })
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n\nFull prompts saved to: {output_file}")


def main():
    # Determine the log file path
    script_dir = Path(__file__).parent
    log_file = script_dir.parent / 'proxy' / 'logs' / 'requests_20260109.jsonl'
    
    if not log_file.exists():
        print(f"Error: Log file not found at {log_file}")
        return
    
    print(f"Analyzing log file: {log_file}\n")
    
    # Extract system prompts
    system_prompts = extract_system_prompts(str(log_file))
    
    if not system_prompts:
        print("No system prompts found in the log file!")
        return
    
    # Group by combination
    grouped = group_by_combination(system_prompts)
    
    # Print summary
    print_summary(grouped)
    
    # Print detailed information
    print_prompt_details(grouped)
    
    # Save full prompts to JSON
    output_file = script_dir / 'system_prompts_analysis.json'
    save_full_prompts(grouped, str(output_file))


if __name__ == '__main__':
    main()

