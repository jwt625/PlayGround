#!/usr/bin/env python3
"""
Workflow graph construction for Claude Code inference logs.

Builds a directed graph showing:
1. Tool dependencies (tool_use → tool_result)
2. Subagent spawns (parent → child via Task tool)
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime


def build_tool_index(logs: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Create tool_use_id → log_index mapping.
    
    Args:
        logs: List of log entries
        
    Returns:
        Dictionary mapping tool_use_id to log index
    """
    tool_index = {}
    
    for idx, log in enumerate(logs):
        # Extract tool_use blocks from response
        response_body = log.get('response', {}).get('body', {})
        content = response_body.get('content', [])
        
        if not isinstance(content, list):
            continue
            
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'tool_use':
                tool_use_id = block.get('id')
                if tool_use_id:
                    tool_index[tool_use_id] = idx
    
    return tool_index


def match_tool_results(logs: List[Dict[str, Any]], tool_index: Dict[str, int]) -> List[Dict[str, Any]]:
    """
    Find tool result dependencies by matching tool_use_id references.
    
    Args:
        logs: List of log entries
        tool_index: Mapping of tool_use_id to log index
        
    Returns:
        List of edge dictionaries with type='tool_result'
    """
    edges = []
    
    for target_idx, log in enumerate(logs):
        # Extract tool_result blocks from request
        request_body = log.get('body', {})
        messages = request_body.get('messages', [])
        
        if not isinstance(messages, list):
            continue
            
        for message in messages:
            if not isinstance(message, dict):
                continue
                
            content = message.get('content', [])
            if not isinstance(content, list):
                continue
                
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'tool_result':
                    tool_use_id = block.get('tool_use_id')
                    is_error = block.get('is_error', False)
                    
                    if tool_use_id and tool_use_id in tool_index:
                        source_idx = tool_index[tool_use_id]
                        
                        # Extract tool name from source
                        tool_name = None
                        source_log = logs[source_idx]
                        source_content = source_log.get('response', {}).get('body', {}).get('content', [])
                        for src_block in source_content:
                            if isinstance(src_block, dict) and src_block.get('id') == tool_use_id:
                                tool_name = src_block.get('name')
                                break
                        
                        edges.append({
                            'type': 'tool_result',
                            'source': source_idx,
                            'target': target_idx,
                            'metadata': {
                                'tool_use_id': tool_use_id,
                                'tool_name': tool_name,
                                'is_error': is_error
                            },
                            'confidence': 1.0
                        })
    
    return edges


def detect_subagent_spawns(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Identify parent-child relationships via Task tool usage.
    
    Args:
        logs: List of log entries
        
    Returns:
        List of edge dictionaries with type='subagent_spawn'
    """
    edges = []
    
    # Find all Task tool uses
    task_spawns = []
    for idx, log in enumerate(logs):
        response_body = log.get('response', {}).get('body', {})
        content = response_body.get('content', [])
        
        if not isinstance(content, list):
            continue
            
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'tool_use' and block.get('name') == 'Task':
                tool_input = block.get('input', {})
                subagent_type = tool_input.get('subagent_type')
                task_tool_id = block.get('id')
                
                if subagent_type:
                    task_spawns.append({
                        'parent_idx': idx,
                        'subagent_type': subagent_type,
                        'task_tool_id': task_tool_id,
                        'timestamp': log.get('timestamp')
                    })
    
    # Match spawns to subsequent agent instances
    for spawn in task_spawns:
        parent_idx = spawn['parent_idx']
        subagent_type = spawn['subagent_type']
        spawn_time = datetime.fromisoformat(spawn['timestamp'].replace('Z', '+00:00'))

        # Search forward for matching agent type
        best_match = None
        min_time_diff = float('inf')

        for idx in range(parent_idx + 1, len(logs)):
            log = logs[idx]

            # Early termination: stop if we've gone past 1 hour window
            log_time = datetime.fromisoformat(log.get('timestamp').replace('Z', '+00:00'))
            time_diff = (log_time - spawn_time).total_seconds()

            if time_diff > 3600:
                # Logs are sorted by time, so no point continuing
                break

            agent_type = log.get('agent_type', {})

            if agent_type.get('name') == subagent_type:
                # Within 1 hour window
                if 0 <= time_diff and time_diff < min_time_diff:
                    best_match = idx
                    min_time_diff = time_diff

        if best_match is not None:
            edges.append({
                'type': 'subagent_spawn',
                'source': parent_idx,
                'target': best_match,
                'metadata': {
                    'subagent_type': subagent_type,
                    'task_tool_id': spawn['task_tool_id'],
                    'spawn_time': spawn['timestamp'],
                    'time_diff_seconds': min_time_diff
                },
                'confidence': 0.9 if min_time_diff < 60 else 0.85
            })
    
    return edges


def build_workflow_graph(logs: List[Dict[str, Any]], time_window_hours: float = 1.0, max_logs: int = 500) -> Dict[str, Any]:
    """
    Build complete workflow graph with nodes and edges.

    Args:
        logs: List of enriched log entries (can be in any order)
        time_window_hours: Only process logs within this time window (default: 1 hour)
        max_logs: Maximum number of logs to process (default: 500)

    Returns:
        Dictionary with 'nodes' and 'edges' arrays
    """
    if not logs:
        return {'nodes': [], 'edges': [], 'metrics': {}}

    # Sort logs chronologically (oldest first) for graph computation
    sorted_logs = sorted(logs, key=lambda x: x.get('timestamp', ''))

    # Apply time window filter - only process recent logs
    if time_window_hours > 0 and len(sorted_logs) > 0:
        latest_time = datetime.fromisoformat(sorted_logs[-1].get('timestamp', '').replace('Z', '+00:00'))
        cutoff_time = latest_time.timestamp() - (time_window_hours * 3600)

        filtered_logs = []
        for log in sorted_logs:
            log_time = datetime.fromisoformat(log.get('timestamp', '').replace('Z', '+00:00'))
            if log_time.timestamp() >= cutoff_time:
                filtered_logs.append(log)

        sorted_logs = filtered_logs
        print(f"Time window filter: {len(logs)} -> {len(sorted_logs)} logs (last {time_window_hours}h)")

    # Apply hard cap on number of logs
    if len(sorted_logs) > max_logs:
        sorted_logs = sorted_logs[-max_logs:]  # Take most recent
        print(f"Max logs cap: limited to {max_logs} most recent logs")

    if not sorted_logs:
        return {'nodes': [], 'edges': [], 'metrics': {}}

    # Build tool index on sorted logs
    tool_index = build_tool_index(sorted_logs)

    # Find all edges on sorted logs
    tool_edges = match_tool_results(sorted_logs, tool_index)
    spawn_edges = detect_subagent_spawns(sorted_logs)
    all_edges = tool_edges + spawn_edges

    print(f"Graph edges: {len(tool_edges)} tool dependencies, {len(spawn_edges)} subagent spawns")

    # Build nodes from sorted logs
    nodes = []
    for idx, log in enumerate(sorted_logs):
        agent_type = log.get('agent_type', {})
        response = log.get('response', {})
        response_body = response.get('body', {})
        usage = response_body.get('usage', {})

        node = {
            'id': idx,
            'log_index': idx,
            'timestamp': log.get('timestamp'),
            'agent_type': agent_type.get('name', 'unknown'),
            'agent_label': agent_type.get('label', 'Unknown'),
            'agent_color': agent_type.get('color', '#6b7280'),
            'model': log.get('body', {}).get('model', 'unknown'),
            'duration_ms': response.get('duration_ms'),
            'tokens': {
                'input': usage.get('input_tokens', 0),
                'output': usage.get('output_tokens', 0),
                'total': usage.get('input_tokens', 0) + usage.get('output_tokens', 0)
            },
            'stop_reason': log.get('stop_reason'),
            'has_errors': log.get('has_errors', False),
            'tool_count': log.get('tool_info', {}).get('count', 0),
            'subagent_count': log.get('subagent_count', 0)
        }
        nodes.append(node)

    # Compute graph metrics
    metrics = compute_graph_metrics(nodes, all_edges)

    return {
        'nodes': nodes,
        'edges': all_edges,
        'metrics': metrics
    }


def compute_graph_metrics(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate graph statistics.

    Args:
        nodes: List of node dictionaries
        edges: List of edge dictionaries

    Returns:
        Dictionary with graph metrics
    """
    if not nodes:
        return {}

    # Count edge types
    tool_edges = [e for e in edges if e['type'] == 'tool_result']
    spawn_edges = [e for e in edges if e['type'] == 'subagent_spawn']

    # Build adjacency lists
    children = {i: [] for i in range(len(nodes))}
    parents = {i: [] for i in range(len(nodes))}

    for edge in edges:
        source = edge['source']
        target = edge['target']
        children[source].append(target)
        parents[target].append(source)

    # Find root nodes (no parents)
    roots = [i for i in range(len(nodes)) if len(parents[i]) == 0]

    # Calculate max depth
    def get_depth(node_id: int, visited: Set[int]) -> int:
        if node_id in visited:
            return 0
        visited.add(node_id)
        if not children[node_id]:
            return 1
        return 1 + max(get_depth(child, visited.copy()) for child in children[node_id])

    max_depth = max((get_depth(root, set()) for root in roots), default=0)

    # Calculate branching factor
    non_leaf_nodes = [i for i in range(len(nodes)) if children[i]]
    avg_branching = sum(len(children[i]) for i in non_leaf_nodes) / len(non_leaf_nodes) if non_leaf_nodes else 0

    return {
        'total_nodes': len(nodes),
        'total_edges': len(edges),
        'tool_dependency_count': len(tool_edges),
        'subagent_spawn_count': len(spawn_edges),
        'max_depth': max_depth,
        'avg_branching_factor': round(avg_branching, 2),
        'root_count': len(roots)
    }


def enrich_logs_with_workflow_graph(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add workflow graph to enriched logs.

    Args:
        logs: List of enriched log entries

    Returns:
        Same logs list (modified in place) with workflow_graph added
    """
    # This function is called from log_classifier.py after enrichment
    # It returns the graph as a separate structure, not per-log
    # The graph will be added to the API response at the top level
    return logs

