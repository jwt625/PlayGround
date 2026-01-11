#!/usr/bin/env python3
import json

with open('proxy/logs/entities_extracted.json', 'r') as f:
    data = json.load(f)

edges = data.get('workflow_dag', {}).get('edges', [])
agents = data.get('workflow_dag', {}).get('nodes', [])

# Find the specific examples from the devlog
print("=== VERIFICATION OF DEVLOG EXAMPLES ===\n")

print("Example 1: agent_8 -> agent_10 (Bash: git branch -a)")
print("-" * 60)
edge_8_10 = [e for e in edges if e.get('source_agent_id') == 'agent_8' and e.get('target_agent_id') == 'agent_10']
if edge_8_10:
    e = edge_8_10[0]
    print(f"✓ FOUND!")
    print(f"  Type: {e['type']}")
    print(f"  Spawn method: {e.get('spawn_method', 'N/A')}")
    print(f"  Tool name: {e.get('tool_name', 'N/A')}")
    print(f"  Tool use ID: {e.get('spawned_by_tool_use_id', 'N/A')}")
    print(f"  Confidence: {e.get('confidence', 'N/A')}")
else:
    print("✗ NOT FOUND")
print()

print("Example 2: agent_9 -> agent_20 (Bash: find command)")
print("-" * 60)
# Note: Based on the output, agent_20 might not exist or might be a different agent
# Let's check what agent_9 spawns
agent_9_spawns = [e for e in edges if e.get('source_agent_id') == 'agent_9' and e.get('type') == 'subagent_spawn']
print(f"agent_9 spawns {len(agent_9_spawns)} subagents:")
for e in agent_9_spawns[:5]:  # Show first 5
    print(f"  agent_9 -> {e['target_agent_id']} ({e.get('spawn_method', 'N/A')}, tool: {e.get('tool_name', 'N/A')})")
print()

# Check if agent_20 exists and has a parent
agent_20 = [a for a in agents if a['agent_id'] == 'agent_20']
if agent_20:
    a = agent_20[0]
    print(f"agent_20 info:")
    print(f"  Parent: {a.get('parent_agent_id', 'None')}")
    print(f"  Spawned by: {a.get('spawned_by_task_id', 'None')}")
else:
    print("agent_20 not found in agents list")
print()

print("=== SUMMARY ===")
print(f"Total subagent_spawn edges: {len([e for e in edges if e['type'] == 'subagent_spawn'])}")
print(f"  - task spawns: {len([e for e in edges if e.get('spawn_method') == 'task'])}")
print(f"  - tool_call spawns: {len([e for e in edges if e.get('spawn_method') == 'tool_call'])}")
print()
print(f"Total agents: {len(agents)}")
print(f"  - Root agents: {len([a for a in agents if not a.get('parent_agent_id')])}")
print(f"  - Child agents: {len([a for a in agents if a.get('parent_agent_id')])}")

