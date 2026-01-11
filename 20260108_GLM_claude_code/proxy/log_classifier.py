#!/usr/bin/env python3
"""
Log entry classifier for Claude Code inference logs.
Enriches log entries with agent type, tool usage, subagent spawns, and other metadata.
"""

import hashlib
import json
from typing import Dict, List, Any, Optional
from workflow_graph import build_workflow_graph


# Known agent type hashes (from system prompt analysis)
AGENT_TYPE_HASHES = {
    "a7f039fc": {
        "name": "file_path_extractor",
        "label": "File Path Extractor",
        "color": "#10b981",
        "description": "Extract file paths from bash outputs"
    },
    "53e64a1b": {
        "name": "file_search",
        "label": "File Search Specialist",
        "color": "#3b82f6",
        "description": "Read-only codebase exploration"
    },
    "1dbb15f8": {
        "name": "bash_processor",
        "label": "Bash Command Processor",
        "color": "#f59e0b",
        "description": "Process and validate bash commands"
    },
    "0080a1ca": {
        "name": "summarizer",
        "label": "Conversation Summarizer",
        "color": "#8b5cf6",
        "description": "Generate conversation titles"
    },
    "564e15ab": {
        "name": "architect",
        "label": "Software Architect",
        "color": "#ec4899",
        "description": "Design implementation plans"
    },
    "7e56ff8e": {
        "name": "topic_detector",
        "label": "Topic Change Detector",
        "color": "#06b6d4",
        "description": "Detect conversation topic changes"
    },
    "c5e8d165": {
        "name": "main_agent",
        "label": "Main Interactive Agent",
        "color": "#ef4444",
        "description": "Primary CLI assistant"
    }
}


def compute_prompt_hash(system_prompts: List[Dict[str, str]]) -> str:
    """Compute hash for system prompt combination (first 200 chars of each)."""
    signature = []
    for prompt in system_prompts:
        text = prompt.get('text', '')
        signature.append(text[:200])
    
    combined = '|||'.join(signature)
    return hashlib.md5(combined.encode()).hexdigest()


def classify_agent_type(log_entry: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Classify the agent type based on system prompts."""
    system = log_entry.get('body', {}).get('system', [])
    if not system:
        return None
    
    prompt_hash = compute_prompt_hash(system)
    hash_prefix = prompt_hash[:8]
    
    return AGENT_TYPE_HASHES.get(hash_prefix, {
        "name": "unknown",
        "label": "Unknown Agent",
        "color": "#6b7280",
        "description": "Unclassified agent type"
    })


def extract_response_tools(log_entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract tool uses from response content."""
    tools = []
    response_content = log_entry.get('response', {}).get('body', {}).get('content', [])
    
    if isinstance(response_content, list):
        for content_block in response_content:
            if content_block.get('type') == 'tool_use':
                tools.append({
                    'id': content_block.get('id'),
                    'name': content_block.get('name'),
                    'input': content_block.get('input', {})
                })
    
    return tools


def categorize_tool(tool_name: str) -> str:
    """Categorize tool by type."""
    if tool_name in ['Read', 'Glob', 'Grep', 'LSP']:
        return 'read'
    elif tool_name in ['Edit', 'Write']:
        return 'write'
    elif tool_name in ['Bash', 'KillShell']:
        return 'execute'
    elif tool_name in ['Task', 'TodoWrite', 'EnterPlanMode', 'ExitPlanMode']:
        return 'orchestration'
    elif tool_name in ['AskUserQuestion']:
        return 'interaction'
    else:
        return 'other'


def extract_tool_info(log_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Extract comprehensive tool usage information."""
    tools = extract_response_tools(log_entry)
    
    tool_counts = {}
    tool_categories = {}
    
    for tool in tools:
        name = tool['name']
        if name:
            tool_counts[name] = tool_counts.get(name, 0) + 1
            category = categorize_tool(name)
            tool_categories[category] = tool_categories.get(category, 0) + 1
    
    return {
        'count': len(tools),
        'tools': tools,
        'tool_names': [t['name'] for t in tools if t['name']],
        'tool_counts': tool_counts,
        'categories': tool_categories,
        'has_tools': len(tools) > 0
    }


def extract_subagent_spawns(log_entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract subagent spawn information from Task tool usage."""
    spawns = []
    tools = extract_response_tools(log_entry)

    for tool in tools:
        if tool['name'] == 'Task':
            task_input = tool.get('input', {})
            spawns.append({
                'subagent_type': task_input.get('subagent_type'),
                'model': task_input.get('model'),
                'description': task_input.get('description', '')[:100],
                'has_resume': 'resume' in task_input
            })

    return spawns


def extract_tool_errors(log_entry: Dict[str, Any]) -> int:
    """Count tool errors in the request messages."""
    error_count = 0
    messages = log_entry.get('body', {}).get('messages', [])

    for msg in messages:
        if isinstance(msg.get('content'), list):
            for content_block in msg['content']:
                if content_block.get('type') == 'tool_result' and content_block.get('is_error'):
                    error_count += 1

    return error_count


def extract_stop_reason(log_entry: Dict[str, Any]) -> Optional[str]:
    """Extract stop reason from response."""
    return log_entry.get('response', {}).get('body', {}).get('stop_reason')


def extract_model_info(log_entry: Dict[str, Any]) -> Dict[str, str]:
    """Extract model information."""
    request_model = log_entry.get('body', {}).get('model', 'unknown')
    response_model = log_entry.get('response', {}).get('body', {}).get('model')

    # Simplify model names
    model = response_model or request_model
    if 'GLM-4.7' in model:
        model_type = 'GLM-4.7-FP8'
    elif 'GLM-4.6' in model:
        model_type = 'GLM-4.6-FP8'
    else:
        model_type = model

    return {
        'full': model,
        'type': model_type
    }


def enrich_log_entry(log_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich a log entry with all classification metadata."""
    enriched = log_entry.copy()

    # Add agent type classification
    agent_type = classify_agent_type(log_entry)
    if agent_type:
        enriched['agent_type'] = agent_type

    # Add tool information
    tool_info = extract_tool_info(log_entry)
    enriched['tool_info'] = tool_info

    # Add subagent spawn information
    subagent_spawns = extract_subagent_spawns(log_entry)
    if subagent_spawns:
        enriched['subagent_spawns'] = subagent_spawns
        enriched['has_subagent_spawns'] = True
        enriched['subagent_count'] = len(subagent_spawns)
    else:
        enriched['has_subagent_spawns'] = False
        enriched['subagent_count'] = 0

    # Add error information
    tool_errors = extract_tool_errors(log_entry)
    enriched['tool_errors'] = tool_errors
    enriched['has_errors'] = tool_errors > 0

    # Add stop reason
    stop_reason = extract_stop_reason(log_entry)
    if stop_reason:
        enriched['stop_reason'] = stop_reason

    # Add model info
    model_info = extract_model_info(log_entry)
    enriched['model_info'] = model_info

    return enriched


def enrich_logs_only(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enrich all log entries with metadata (agent type, tool info, etc.).
    Does NOT build workflow graph.

    Returns:
        List of enriched log entries with log_index added
    """
    enriched = []
    for idx, log in enumerate(logs):
        enriched_log = enrich_log_entry(log)
        enriched_log['log_index'] = idx
        enriched.append(enriched_log)
    return enriched


def enrich_logs(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Enrich all log entries and build workflow graph.

    Returns:
        Dictionary with 'logs' and 'workflow_graph' keys
    """
    enriched_logs = [enrich_log_entry(log) for log in logs]
    workflow_graph = build_workflow_graph(enriched_logs)

    return {
        'logs': enriched_logs,
        'workflow_graph': workflow_graph
    }

