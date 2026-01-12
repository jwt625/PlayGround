#!/usr/bin/env python3
"""Analyze root agents (agents without parents) in entities.json."""

import json
import sys
from pathlib import Path


def main():
    # Load entities
    entities_path = Path('proxy/logs/entities_20260110.json')
    if not entities_path.exists():
        print(f"Error: {entities_path} not found")
        sys.exit(1)
    
    with open(entities_path, 'r') as f:
        data = json.load(f)
    
    # Get agent instances
    agents = data.get('entities', {}).get('agent_instances', [])
    system_prompts = {p['hash']: p for p in data.get('entities', {}).get('system_prompts', [])}
    
    # Find root agents (no parent)
    root_agents = [a for a in agents if not a.get('parent_agent_id')]
    
    print(f'Total agents: {len(agents)}')
    print(f'Root agents (no parent): {len(root_agents)}')
    print(f'Child agents (with parent): {len(agents) - len(root_agents)}')
    print()
    print('=' * 80)
    print('ROOT AGENTS ANALYSIS')
    print('=' * 80)
    print()
    
    # Group by system prompt hash
    by_prompt = {}
    for agent in root_agents:
        h = agent.get('system_prompt_hash', 'N/A')
        if h not in by_prompt:
            by_prompt[h] = []
        by_prompt[h].append(agent)
    
    print(f'Unique system prompts among root agents: {len(by_prompt)}')
    for h, agents_list in sorted(by_prompt.items(), key=lambda x: -len(x[1])):
        prompt_info = system_prompts.get(h, {})
        prompt_preview = prompt_info.get('text', 'N/A')[:100].replace('\n', ' ')
        print(f'  {h}: {len(agents_list)} agents')
        print(f'    Preview: {prompt_preview}...')
    print()
    
    # Analyze each root agent
    print('=' * 80)
    print('DETAILED ROOT AGENT LIST')
    print('=' * 80)
    print()
    
    for agent in sorted(root_agents, key=lambda x: x.get('first_request_id', 0)):
        agent_id = agent.get('agent_id', 'N/A')
        first_req_id = agent.get('first_request_id', 'N/A')
        system_hash = agent.get('system_prompt_hash', 'N/A')
        first_msg = agent.get('first_user_message', '')[:200].replace('\n', ' ')
        total_reqs = agent.get('total_requests', len(agent.get('requests', [])))
        first_ts = agent.get('first_timestamp', 'N/A')
        child_ids = agent.get('child_agent_ids', [])
        
        # Get system prompt info
        prompt_info = system_prompts.get(system_hash, {})
        prompt_text = prompt_info.get('text', '')[:100].replace('\n', ' ') if prompt_info else 'N/A'
        
        print(f'--- {agent_id} ---')
        print(f'  First request ID (log index): {first_req_id}')
        print(f'  Total requests: {total_reqs}')
        print(f'  First timestamp: {first_ts}')
        print(f'  System prompt hash: {system_hash}')
        print(f'  System prompt preview: {prompt_text}...')
        print(f'  Child agents spawned: {child_ids if child_ids else "None"}')
        print(f'  First user message: {first_msg}...')
        print()


if __name__ == '__main__':
    main()

