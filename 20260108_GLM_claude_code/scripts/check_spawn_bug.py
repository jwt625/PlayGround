import json

with open('proxy/logs/entities_20260110.json', 'r') as f:
    data = json.load(f)

spawn_edges = [e for e in data['workflow_dag']['edges'] if e['type'] == 'subagent_spawn']

print('=== SPAWN EDGE FIX VERIFICATION ===')
print(f'Total spawn edges: {len(spawn_edges)}')
print()

sample_edges = spawn_edges[:3]
print('Sample spawn edges:')
for edge in sample_edges:
    print(f'  {edge["source_agent_id"]} -> {edge["target_agent_id"]}')
    print(f'    Has source_request_id: {"source_request_id" in edge}')
    if 'source_request_id' in edge:
        print(f'    source_request_id: {edge["source_request_id"]}')
    print()

# Check agent_20 example
agent_20_edge = [e for e in spawn_edges if e['target_agent_id'] == 'agent_20']
if agent_20_edge:
    edge = agent_20_edge[0]
    print('Agent_20 spawn edge (FIXED):')
    print(f'  Source: {edge["source_agent_id"]}')
    print(f'  Target: {edge["target_agent_id"]}')
    print(f'  source_request_id: {edge.get("source_request_id")}')

    agents = data['entities']['agent_instances']
    parent = [a for a in agents if a['agent_id'] == edge['source_agent_id']][0]
    print(f'  Parent requests: {parent["requests"]}')
    print(f'  Expected: request 45 (4th request)')
    print(f'  Actual: request {edge.get("source_request_id")}')
    print(f'  {"✓ MATCH!" if edge.get("source_request_id") == 45 else "✗ MISMATCH"}')

# Check all edges have source_request_id
edges_with_request_id = [e for e in spawn_edges if e.get('source_request_id') is not None]
print(f'\nEdges with source_request_id: {len(edges_with_request_id)} / {len(spawn_edges)}')
if len(edges_with_request_id) == len(spawn_edges):
    print('✓ ALL spawn edges have source_request_id!')
else:
    print(f'✗ {len(spawn_edges) - len(edges_with_request_id)} edges missing source_request_id')

