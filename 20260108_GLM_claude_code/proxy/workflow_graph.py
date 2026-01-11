#!/usr/bin/env python3
"""
Workflow graph construction for Claude Code inference logs.

Builds a directed graph showing:
1. Tool dependencies (tool_use → tool_result)
2. Subagent spawns (parent → child via Task tool)
3. Content reuse (agent output → subsequent agent input)
4. Prompt-based subagent matching (Task tool prompt → subagent first message)
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
import hashlib


def detect_sessions(logs: List[Dict[str, Any]], gap_minutes: float = 10.0) -> List[Tuple[int, int]]:
    """
    Detect session boundaries based on time gaps.

    Args:
        logs: List of log entries (must be sorted by timestamp)
        gap_minutes: Minimum gap in minutes to consider a new session (default: 10)

    Returns:
        List of (start_index, end_index) tuples for each session
    """
    if not logs:
        return []

    sessions = []
    current_session_start = 0
    prev_time = None

    for i, log in enumerate(logs):
        timestamp = log.get('timestamp', '')
        if not timestamp:
            continue

        try:
            curr_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            continue

        if prev_time:
            gap_seconds = (curr_time - prev_time).total_seconds()
            if gap_seconds > (gap_minutes * 60):
                # Session boundary detected
                sessions.append((current_session_start, i - 1))
                current_session_start = i

        prev_time = curr_time

    # Add final session
    if current_session_start < len(logs):
        sessions.append((current_session_start, len(logs) - 1))

    return sessions


def build_tool_index(logs: List[Dict[str, Any]], session_id: Optional[int] = None) -> Tuple[Dict[str, int], Dict[str, str]]:
    """
    Create tool_use_id → log_index and tool_use_id → tool_name mappings.

    IMPORTANT: To prevent cross-session edge contamination, tool_use_ids are prefixed
    with session_id if provided. This ensures tool_use_ids are unique across sessions.

    Args:
        logs: List of log entries
        session_id: Optional session ID to prefix tool_use_ids (prevents cross-session matches)

    Returns:
        Tuple of (tool_index, tool_names) dictionaries
    """
    tool_index = {}
    tool_names = {}

    for idx, log in enumerate(logs):
        # Extract tool_use blocks from response
        response_body = log.get('response', {}).get('body', {})
        content = response_body.get('content', [])

        if not isinstance(content, list):
            continue

        for block in content:
            if isinstance(block, dict) and block.get('type') == 'tool_use':
                tool_use_id = block.get('id')
                tool_name = block.get('name')
                if tool_use_id:
                    # Prefix with session_id to prevent cross-session collisions
                    if session_id is not None:
                        tool_use_id = f"session_{session_id}_{tool_use_id}"
                    tool_index[tool_use_id] = idx
                    tool_names[tool_use_id] = tool_name

    return tool_index, tool_names


def match_tool_results(logs: List[Dict[str, Any]], tool_index: Dict[str, int], tool_names: Dict[str, str], session_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Find tool result dependencies by matching tool_use_id references.

    IMPORTANT: Uses session-prefixed tool_use_ids to prevent cross-session edge contamination.

    Args:
        logs: List of log entries
        tool_index: Mapping of tool_use_id to log index
        tool_names: Mapping of tool_use_id to tool name
        session_id: Optional session ID to prefix tool_use_ids (must match build_tool_index)

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

                    if tool_use_id:
                        # Prefix with session_id to match tool_index keys
                        if session_id is not None:
                            prefixed_tool_use_id = f"session_{session_id}_{tool_use_id}"
                        else:
                            prefixed_tool_use_id = tool_use_id

                        if prefixed_tool_use_id in tool_index:
                            source_idx = tool_index[prefixed_tool_use_id]
                            tool_name = tool_names.get(prefixed_tool_use_id)

                            edges.append({
                                'type': 'tool_result',
                                'source': source_idx,
                                'target': target_idx,
                                'metadata': {
                                    'tool_use_id': tool_use_id,  # Store original ID without prefix
                                    'tool_name': tool_name,
                                    'is_error': is_error
                                },
                                'confidence': 1.0
                            })

    return edges


def detect_subagent_spawns(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Identify parent-child relationships via Task tool usage.

    Uses both temporal proximity and prompt matching:
    1. Extract prompt from Task tool input
    2. Search forward for agent with matching first message content
    3. Verify agent type matches subagent_type
    4. Prefer closest temporal match within reasonable window

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
                prompt = tool_input.get('prompt', '')

                if subagent_type:
                    task_spawns.append({
                        'parent_idx': idx,
                        'subagent_type': subagent_type,
                        'task_tool_id': task_tool_id,
                        'prompt': prompt,
                        'prompt_hash': hash_content(prompt, length=300) if prompt else None,
                        'timestamp': log.get('timestamp')
                    })

    # Match spawns to subsequent agent instances
    for spawn in task_spawns:
        parent_idx = spawn['parent_idx']
        subagent_type = spawn['subagent_type']
        prompt_hash = spawn['prompt_hash']
        spawn_time = datetime.fromisoformat(spawn['timestamp'].replace('Z', '+00:00'))

        # Search forward for matching agent by prompt content
        best_match = None
        min_time_diff = float('inf')
        match_method = 'none'

        # Only try to match if we have a prompt hash
        if not prompt_hash:
            continue

        for idx in range(parent_idx + 1, len(logs)):
            log = logs[idx]

            # Early termination: stop if we've gone past 1 hour window
            log_time = datetime.fromisoformat(log.get('timestamp').replace('Z', '+00:00'))
            time_diff = (log_time - spawn_time).total_seconds()

            if time_diff > 3600:
                # Logs are sorted by time, so no point continuing
                break

            # Check first message in request for prompt match
            messages = log.get('body', {}).get('messages', [])
            if isinstance(messages, list) and len(messages) > 0:
                first_msg = messages[0]
                if isinstance(first_msg, dict):
                    msg_content = first_msg.get('content', '')

                    # Handle both string and array content
                    msg_hash = None
                    if isinstance(msg_content, str):
                        msg_hash = hash_content(msg_content, length=300)
                    elif isinstance(msg_content, list):
                        # Extract text from content blocks
                        for block in msg_content:
                            if isinstance(block, dict) and block.get('type') == 'text':
                                text = block.get('text', '')
                                msg_hash = hash_content(text, length=300)
                                break

                    # Check for prompt match
                    if msg_hash and msg_hash == prompt_hash:
                        # Prompt match found - take first match within time window
                        if 0 <= time_diff and time_diff < min_time_diff:
                            best_match = idx
                            min_time_diff = time_diff
                            match_method = 'prompt_hash'
                            # Break early on prompt match since it's very specific
                            break

        if best_match is not None:
            edges.append({
                'type': 'subagent_spawn',
                'source': parent_idx,
                'target': best_match,
                'metadata': {
                    'subagent_type': subagent_type,
                    'task_tool_id': spawn['task_tool_id'],
                    'spawn_time': spawn['timestamp'],
                    'time_diff_seconds': min_time_diff,
                    'match_method': match_method
                },
                'confidence': 0.95
            })

    return edges


def hash_content(text: str, length: int = 200) -> str:
    """
    Hash the first N characters of text content for matching.

    Args:
        text: Text content to hash
        length: Number of characters to use (default: 200)

    Returns:
        SHA256 hash of the first N characters
    """
    if not text:
        return ""

    # Normalize whitespace and take first N characters
    normalized = ' '.join(text.split())[:length]
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def extract_text_content(content_blocks: List[Any]) -> str:
    """
    Extract text content from response content blocks.

    Args:
        content_blocks: List of content blocks from response

    Returns:
        Concatenated text content
    """
    if not isinstance(content_blocks, list):
        return ""

    texts = []
    for block in content_blocks:
        if isinstance(block, dict):
            if block.get('type') == 'text':
                text = block.get('text', '')
                if text:
                    texts.append(text)

    return '\n'.join(texts)


def detect_content_reuse(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Identify content reuse edges where one agent's output is used in another agent's input.

    Uses content hashing (first 200 chars) to match response text to subsequent request messages.

    Args:
        logs: List of log entries

    Returns:
        List of edge dictionaries with type='content_reuse'
    """
    edges = []

    # Build index of response content hashes
    response_hashes = {}  # hash -> list of (log_idx, full_text_preview)

    for idx, log in enumerate(logs):
        response_body = log.get('response', {}).get('body', {})
        content = response_body.get('content', [])

        text_content = extract_text_content(content)
        if text_content:
            content_hash = hash_content(text_content)
            if content_hash:
                if content_hash not in response_hashes:
                    response_hashes[content_hash] = []
                # Store preview (first 100 chars for metadata)
                preview = text_content[:100].replace('\n', ' ')
                response_hashes[content_hash].append((idx, preview))

    # Search for matching content in subsequent requests
    for target_idx, log in enumerate(logs):
        request_body = log.get('body', {})
        messages = request_body.get('messages', [])

        if not isinstance(messages, list):
            continue

        for message in messages:
            if not isinstance(message, dict):
                continue

            content = message.get('content', [])
            if isinstance(content, str):
                # Handle string content
                content_hash = hash_content(content)
                if content_hash in response_hashes:
                    for source_idx, preview in response_hashes[content_hash]:
                        # Only create edge if source comes before target
                        if source_idx < target_idx:
                            edges.append({
                                'type': 'content_reuse',
                                'source': source_idx,
                                'target': target_idx,
                                'metadata': {
                                    'content_preview': preview,
                                    'match_method': 'hash_200char'
                                },
                                'confidence': 0.95
                            })
            elif isinstance(content, list):
                # Handle array content
                for block in content:
                    if isinstance(block, dict) and block.get('type') == 'text':
                        text = block.get('text', '')
                        if text:
                            content_hash = hash_content(text)
                            if content_hash in response_hashes:
                                for source_idx, preview in response_hashes[content_hash]:
                                    if source_idx < target_idx:
                                        edges.append({
                                            'type': 'content_reuse',
                                            'source': source_idx,
                                            'target': target_idx,
                                            'metadata': {
                                                'content_preview': preview,
                                                'match_method': 'hash_200char'
                                            },
                                            'confidence': 0.95
                                        })

    return edges


def build_workflow_graph(logs: List[Dict[str, Any]], max_logs_per_session: int = 1000, session_gap_minutes: float = 10.0) -> Dict[str, Any]:
    """
    Build complete workflow graph with nodes and edges across all sessions.

    Args:
        logs: List of enriched log entries (can be in any order)
        max_logs_per_session: Maximum number of logs to process per session (default: 1000)
        session_gap_minutes: Time gap in minutes to detect session boundaries (default: 10)

    Returns:
        Dictionary with 'nodes', 'edges', 'sessions', and 'metrics' arrays
    """
    if not logs:
        return {'nodes': [], 'edges': [], 'sessions': [], 'metrics': {}}

    # Sort logs chronologically (oldest first) for graph computation
    sorted_logs = sorted(logs, key=lambda x: x.get('timestamp', ''))

    # Detect session boundaries
    sessions = detect_sessions(sorted_logs, gap_minutes=session_gap_minutes)

    print(f"Detected {len(sessions)} sessions from {len(sorted_logs)} logs")
    for idx, (start, end) in enumerate(sessions):
        session_size = end - start + 1
        start_time = sorted_logs[start].get('timestamp', 'unknown')
        end_time = sorted_logs[end].get('timestamp', 'unknown')
        print(f"  Session {idx + 1}: Logs {start}-{end} ({session_size} logs) | {start_time[:19]} to {end_time[:19]}")

    # Process all sessions (apply per-session cap if needed)
    all_nodes = []
    all_edges = []
    session_info = []

    for session_idx, (session_start, session_end) in enumerate(sessions):
        session_logs = sorted_logs[session_start:session_end + 1]

        # Apply per-session cap
        if len(session_logs) > max_logs_per_session:
            print(f"  Session {session_idx + 1}: Capping at {max_logs_per_session} most recent logs")
            session_logs = session_logs[-max_logs_per_session:]

        session_node_start = len(all_nodes)

        # Build tool index for this session with session_id prefix to prevent cross-session contamination
        tool_index, tool_names = build_tool_index(session_logs, session_id=session_idx)

        # Find edges within this session (using session_id prefix)
        tool_edges = match_tool_results(session_logs, tool_index, tool_names, session_id=session_idx)
        spawn_edges = detect_subagent_spawns(session_logs)
        content_edges = detect_content_reuse(session_logs)

        print(f"  Session {session_idx + 1}: {len(tool_edges)} tool edges, {len(spawn_edges)} spawn edges, {len(content_edges)} content reuse edges")

        # Build nodes for this session
        for local_idx, log in enumerate(session_logs):
            agent_type = log.get('agent_type', {})
            response = log.get('response', {})
            response_body = response.get('body', {})
            usage = response_body.get('usage', {})

            global_idx = len(all_nodes)

            node = {
                'id': global_idx,
                'log_index': global_idx,
                'session_id': session_idx,
                'session_local_index': local_idx,
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
            all_nodes.append(node)

        # Adjust edge indices to global indices
        for edge in tool_edges + spawn_edges + content_edges:
            edge['source'] = session_node_start + edge['source']
            edge['target'] = session_node_start + edge['target']
            edge['session_id'] = session_idx

        all_edges.extend(tool_edges + spawn_edges + content_edges)

        # Store session metadata
        session_info.append({
            'session_id': session_idx,
            'node_start': session_node_start,
            'node_end': len(all_nodes) - 1,
            'node_count': len(session_logs),
            'edge_count': len(tool_edges) + len(spawn_edges) + len(content_edges),
            'start_time': session_logs[0].get('timestamp') if session_logs else None,
            'end_time': session_logs[-1].get('timestamp') if session_logs else None
        })

    print(f"\nTotal: {len(all_nodes)} nodes, {len(all_edges)} edges across {len(sessions)} sessions")

    # Skip expensive metrics calculation - not used by frontend
    metrics = {
        'total_nodes': len(all_nodes),
        'total_edges': len(all_edges),
        'session_count': len(sessions)
    }

    return {
        'nodes': all_nodes,
        'edges': all_edges,
        'sessions': session_info,
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

    print(f"Computing metrics for {len(nodes)} nodes and {len(edges)} edges...")

    # Count edge types
    tool_edges = [e for e in edges if e['type'] == 'tool_result']
    spawn_edges = [e for e in edges if e['type'] == 'subagent_spawn']
    print(f"Edge types: {len(tool_edges)} tool, {len(spawn_edges)} spawn")

    # Build adjacency lists
    children = {i: [] for i in range(len(nodes))}
    parents = {i: [] for i in range(len(nodes))}

    for edge in edges:
        source = edge['source']
        target = edge['target']
        children[source].append(target)
        parents[target].append(source)

    print("Built adjacency lists")

    # Find root nodes (no parents)
    roots = [i for i in range(len(nodes)) if len(parents[i]) == 0]
    print(f"Found {len(roots)} root nodes")

    # Calculate max depth using iterative BFS (much faster than recursive)
    max_depth = 0
    if roots:
        from collections import deque
        print("Calculating max depth...")
        for root in roots:
            queue = deque([(root, 1)])
            visited = set()
            while queue:
                node_id, depth = queue.popleft()
                if node_id in visited:
                    continue
                visited.add(node_id)
                max_depth = max(max_depth, depth)
                for child in children[node_id]:
                    if child not in visited:
                        queue.append((child, depth + 1))
        print(f"Max depth: {max_depth}")

    # Calculate branching factor
    print("Calculating branching factor...")
    non_leaf_nodes = [i for i in range(len(nodes)) if children[i]]
    avg_branching = sum(len(children[i]) for i in non_leaf_nodes) / len(non_leaf_nodes) if non_leaf_nodes else 0
    print(f"Avg branching factor: {avg_branching:.2f}")

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

