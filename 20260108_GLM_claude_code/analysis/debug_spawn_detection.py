#!/usr/bin/env python3
"""Debug spawn detection for specific agents."""

import json
import hashlib
import re
from pathlib import Path


def compute_hash(text, length=16):
    return hashlib.sha256(text.encode()).hexdigest()[:length]


def normalize_command(command):
    if not command:
        return ''
    # Remove stderr redirects
    normalized = re.sub(r'\s*2>/dev/null\s*', ' ', command)
    normalized = re.sub(r'\s*2>&1\s*', ' ', normalized)
    # Remove pipe commands
    normalized = re.sub(r'\s*\|[^|]*$', '', normalized)
    # Remove all quotes
    normalized = normalized.replace('"', '')
    normalized = normalized.replace("'", '')
    # Normalize whitespace
    normalized = ' '.join(normalized.split())
    return normalized.strip()


def extract_command_from_message(message):
    """Extract command from validation subagent's first message."""
    if not message:
        return None
    # Pattern 1: "Command: <command>" at the start
    match = re.match(r'^Command:\s*(.+?)(?:\n|$)', message, re.MULTILINE)
    if match:
        return match.group(1).strip()
    # Pattern 2: "Command: <command>" anywhere
    match = re.search(r'Command:\s*(.+?)(?:\n|$)', message, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def main():
    with open('proxy/logs/entities_20260110.json', 'r') as f:
        data = json.load(f)

    agents = data.get('entities', {}).get('agent_instances', [])
    tool_uses = data.get('entities', {}).get('tool_uses', [])

    # Find agent_2 (Git history analyzer)
    agent_2 = next((a for a in agents if a['agent_id'] == 'agent_2'), None)
    if agent_2:
        print("=== agent_2 (Git history analyzer) ===")
        print(f"First request ID: {agent_2['first_request_id']}")
        print(f"System prompt hash: {agent_2['system_prompt_hash']}")
        print(f"First user message (first 300 chars):")
        print(agent_2['first_user_message'][:300])
        print()

        # Check if it starts with Command:
        cmd_from_msg = extract_command_from_message(agent_2['first_user_message'])
        print(f"Extracted command: {repr(cmd_from_msg[:100] if cmd_from_msg else None)}")
        print()
    
    # Find agent_72
    agent_72 = next((a for a in agents if a['agent_id'] == 'agent_72'), None)
    if agent_72:
        print("=== agent_72 (File path extractor) ===")
        print(f"First request ID: {agent_72['first_request_id']}")
        print(f"First user message (first 300 chars):")
        print(agent_72['first_user_message'][:300])
        print()
        
        # Extract command from message
        cmd_from_msg = extract_command_from_message(agent_72['first_user_message'])
        print(f"Extracted command from message:")
        print(repr(cmd_from_msg[:200] if cmd_from_msg else None))
        print()
        
        if cmd_from_msg:
            normalized_msg = normalize_command(cmd_from_msg)
            msg_hash = compute_hash(normalized_msg)
            print(f"Normalized command from message:")
            print(repr(normalized_msg[:100]))
            print(f"Hash: {msg_hash}")
            print()
    
    # Find the git commit bash call from agent_4
    print("=== Looking for matching Bash calls ===")
    for tu in tool_uses:
        if tu.get('tool_name') == 'Bash' and tu.get('first_seen_agent') == 'agent_4':
            cmd = tu.get('input', {}).get('command', '')
            if 'git commit' in cmd:
                print(f"Found Bash call: {tu['id']}")
                print(f"First seen request: {tu.get('first_seen_request')}")
                print(f"Command (first 200 chars): {repr(cmd[:200])}")
                
                normalized_bash = normalize_command(cmd)
                bash_hash = compute_hash(normalized_bash)
                print(f"Normalized: {repr(normalized_bash[:100])}")
                print(f"Hash: {bash_hash}")
                print()


if __name__ == '__main__':
    main()

