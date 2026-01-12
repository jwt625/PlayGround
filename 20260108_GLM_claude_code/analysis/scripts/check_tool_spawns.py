#!/usr/bin/env python3
"""Check tool-spawned subagent detection results."""

import json
import sys
from collections import Counter

def main():
    # Load entities
    with open('proxy/logs/entities_extracted.json', 'r') as f:
        data = json.load(f)
    
    edges = data.get('workflow_dag', {}).get('edges', [])
    agents = data.get('workflow_dag', {}).get('nodes', [])
    
    print(f'Total edges: {len(edges)}')
    print()
    
    # Count by type
    edge_types = Counter(e['type'] for e in edges)
    print('Edge counts by type:')
    for edge_type, count in sorted(edge_types.items()):
        print(f'  {edge_type}: {count}')
    print()
    
    # Show subagent_spawn edges
    spawn_edges = [e for e in edges if e['type'] == 'subagent_spawn']
    print(f'Subagent spawn edges: {len(spawn_edges)}')
    print()
    
    # Group by spawn_method
    spawn_by_method = {}
    for e in spawn_edges:
        method = e.get('spawn_method', 'unknown')
        if method not in spawn_by_method:
            spawn_by_method[method] = []
        spawn_by_method[method].append(e)
    
    for method, edges_list in sorted(spawn_by_method.items()):
        print(f'{method} spawns: {len(edges_list)}')
        for e in edges_list:
            print(f'  {e["source_agent_id"]} -> {e["target_agent_id"]}', end='')
            if 'tool_name' in e:
                print(f' (tool: {e["tool_name"]})', end='')
            if 'command_hash' in e:
                print(f' [cmd_hash: {e["command_hash"]}]', end='')
            print()
        print()
    
    # Show agents with parent info
    print('Agents with parents:')
    child_agents = [a for a in agents if a.get('parent_agent_id')]
    for agent in sorted(child_agents, key=lambda x: x['agent_id']):
        print(f'  {agent["agent_id"]} <- parent: {agent["parent_agent_id"]}')
    print()
    
    print(f'Total child agents: {len(child_agents)}')
    print(f'Total root agents: {len([a for a in agents if not a.get("parent_agent_id")])}')

if __name__ == '__main__':
    main()

