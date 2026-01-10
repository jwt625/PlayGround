#!/usr/bin/env python3
"""
Analyze Claude Code workflow orchestration from log files.

This script identifies the key objects/subjects in the agentic workflow:
- Main agent requests
- Subagent spawning (via Task tool)
- Tool usage patterns
- Tool results and their relationships
- Request-response chains
- TodoWrite state management
- Plan mode transitions
- Response metadata (stop_reason, model, usage)
- Available tool definitions
"""

import json
import sys
from collections import defaultdict
from typing import Dict, List, Any, Optional
from pathlib import Path


def load_log_file(filepath: str) -> List[Dict[str, Any]]:
    """Load JSONL log file."""
    entries = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def extract_tool_uses(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract tool_use blocks from messages."""
    tool_uses = []
    messages = entry.get('body', {}).get('messages', [])
    for msg in messages:
        if isinstance(msg.get('content'), list):
            for content_block in msg['content']:
                if content_block.get('type') == 'tool_use':
                    tool_uses.append(content_block)
    return tool_uses


def extract_tool_results(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract tool_result blocks from messages."""
    tool_results = []
    messages = entry.get('body', {}).get('messages', [])
    for msg in messages:
        if isinstance(msg.get('content'), list):
            for content_block in msg['content']:
                if content_block.get('type') == 'tool_result':
                    tool_results.append(content_block)
    return tool_results


def extract_response_tool_uses(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract tool_use blocks from response content."""
    tool_uses = []
    response_content = entry.get('response', {}).get('body', {}).get('content', [])
    if isinstance(response_content, list):
        for content_block in response_content:
            if content_block.get('type') == 'tool_use':
                tool_uses.append(content_block)
    return tool_uses


def extract_available_tools(entry: Dict[str, Any]) -> List[str]:
    """Extract tool names from request body tools array."""
    tools = entry.get('body', {}).get('tools', [])
    return [t.get('name') for t in tools if t.get('name')]


def extract_response_metadata(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Extract response metadata (model, stop_reason, usage)."""
    body = entry.get('response', {}).get('body', {})
    return {
        'model': body.get('model'),
        'stop_reason': body.get('stop_reason'),
        'usage': body.get('usage', {})
    }


def extract_todo_states(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract TodoWrite tool invocations and their states."""
    todos = []
    response_content = entry.get('response', {}).get('body', {}).get('content', [])
    if isinstance(response_content, list):
        for content_block in response_content:
            if content_block.get('type') == 'tool_use' and content_block.get('name') == 'TodoWrite':
                todo_input = content_block.get('input', {})
                todos.append({
                    'id': content_block.get('id'),
                    'todos': todo_input.get('todos', [])
                })
    return todos


def analyze_workflow(log_file: str):
    """Analyze workflow orchestration patterns."""
    entries = load_log_file(log_file)
    
    print(f"\n{'='*80}")
    print(f"WORKFLOW ORCHESTRATION ANALYSIS")
    print(f"{'='*80}\n")
    print(f"Total log entries: {len(entries)}")
    print(f"Log file: {log_file}\n")
    
    # Statistics
    stats = {
        'requests_with_tool_uses': 0,
        'requests_with_tool_results': 0,
        'responses_with_tool_uses': 0,
        'tool_use_count': defaultdict(int),
        'tool_result_count': defaultdict(int),
        'response_tool_count': defaultdict(int),
        'subagent_types': defaultdict(int),
        'tool_chains': [],
        'task_spawns': [],
        # New stats
        'available_tools': set(),
        'stop_reasons': defaultdict(int),
        'models': defaultdict(int),
        'total_input_tokens': 0,
        'total_output_tokens': 0,
        'todo_writes': [],
        'todo_states': defaultdict(int),
        'plan_mode_enters': 0,
        'plan_mode_exits': 0,
        'tool_errors': 0,
        'task_model_usage': defaultdict(int),
    }

    for idx, entry in enumerate(entries):
        # Analyze request tool uses (from user/previous responses)
        req_tool_uses = extract_tool_uses(entry)
        if req_tool_uses:
            stats['requests_with_tool_uses'] += 1
            for tool in req_tool_uses:
                stats['tool_use_count'][tool.get('name')] += 1
        
        # Analyze tool results (feedback from previous tool uses)
        tool_results = extract_tool_results(entry)
        if tool_results:
            stats['requests_with_tool_results'] += 1
            for result in tool_results:
                tool_id = result.get('tool_use_id', 'unknown')
                stats['tool_result_count'][tool_id[:20]] += 1
                if result.get('is_error'):
                    stats['tool_errors'] += 1

        # Extract available tools from request
        available_tools = extract_available_tools(entry)
        stats['available_tools'].update(available_tools)

        # Extract response metadata
        resp_meta = extract_response_metadata(entry)
        if resp_meta['stop_reason']:
            stats['stop_reasons'][resp_meta['stop_reason']] += 1
        if resp_meta['model']:
            stats['models'][resp_meta['model']] += 1
        if resp_meta['usage']:
            stats['total_input_tokens'] += resp_meta['usage'].get('input_tokens', 0)
            stats['total_output_tokens'] += resp_meta['usage'].get('output_tokens', 0)

        # Analyze response tool uses (agent's next actions)
        resp_tool_uses = extract_response_tool_uses(entry)
        if resp_tool_uses:
            stats['responses_with_tool_uses'] += 1
            tool_names = []
            for tool in resp_tool_uses:
                tool_name = tool.get('name')
                stats['response_tool_count'][tool_name] += 1
                tool_names.append(tool_name)

                # Check for Task tool (subagent spawning)
                if tool_name == 'Task':
                    task_input = tool.get('input', {})
                    subagent_type = task_input.get('subagent_type')
                    model = task_input.get('model')
                    if subagent_type:
                        stats['subagent_types'][subagent_type] += 1
                        stats['task_spawns'].append({
                            'entry_idx': idx,
                            'timestamp': entry.get('timestamp'),
                            'subagent_type': subagent_type,
                            'model': model,
                            'description': task_input.get('description', '')[:80],
                            'has_resume': 'resume' in task_input
                        })
                    if model:
                        stats['task_model_usage'][model] += 1

                # Check for TodoWrite
                elif tool_name == 'TodoWrite':
                    todo_input = tool.get('input', {})
                    todos = todo_input.get('todos', [])
                    stats['todo_writes'].append({
                        'entry_idx': idx,
                        'timestamp': entry.get('timestamp'),
                        'todo_count': len(todos)
                    })
                    for todo in todos:
                        stats['todo_states'][todo.get('status', 'unknown')] += 1

                # Check for plan mode
                elif tool_name == 'EnterPlanMode':
                    stats['plan_mode_enters'] += 1
                elif tool_name == 'ExitPlanMode':
                    stats['plan_mode_exits'] += 1

            # Record tool chains (multiple tools in one response)
            if len(tool_names) > 1:
                stats['tool_chains'].append({
                    'entry_idx': idx,
                    'timestamp': entry.get('timestamp'),
                    'tools': tool_names
                })
    
    # Print statistics
    print(f"{'='*80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*80}\n")
    print(f"Requests with tool uses (input):     {stats['requests_with_tool_uses']}")
    print(f"Requests with tool results (input):  {stats['requests_with_tool_results']}")
    print(f"Responses with tool uses (output):   {stats['responses_with_tool_uses']}")
    print(f"Tool chains (multi-tool responses):  {len(stats['tool_chains'])}")
    print(f"Subagent spawns (Task tool):         {len(stats['task_spawns'])}")
    print(f"TodoWrite invocations:               {len(stats['todo_writes'])}")
    print(f"Plan mode enters:                    {stats['plan_mode_enters']}")
    print(f"Plan mode exits:                     {stats['plan_mode_exits']}")
    print(f"Tool errors:                         {stats['tool_errors']}")
    print(f"Available tools (unique):            {len(stats['available_tools'])}")
    print(f"Total input tokens:                  {stats['total_input_tokens']:,}")
    print(f"Total output tokens:                 {stats['total_output_tokens']:,}\n")

    return stats, entries


if __name__ == '__main__':
    log_file = sys.argv[1] if len(sys.argv) > 1 else 'proxy/logs/requests_20260109.jsonl'
    stats, entries = analyze_workflow(log_file)

    # Print detailed breakdowns
    print(f"{'='*80}")
    print(f"TOOL USAGE IN REQUESTS (Input to Agent)")
    print(f"{'='*80}\n")
    for tool, count in sorted(stats['tool_use_count'].items(), key=lambda x: -x[1])[:20]:
        print(f"{tool:30s} {count:4d}")

    print(f"\n{'='*80}")
    print(f"TOOL USAGE IN RESPONSES (Agent Output)")
    print(f"{'='*80}\n")
    for tool, count in sorted(stats['response_tool_count'].items(), key=lambda x: -x[1])[:20]:
        tool_name = tool if tool is not None else 'null'
        print(f"{tool_name:30s} {count:4d}")

    print(f"\n{'='*80}")
    print(f"SUBAGENT TYPES (Task Tool)")
    print(f"{'='*80}\n")
    for subagent_type, count in sorted(stats['subagent_types'].items(), key=lambda x: -x[1]):
        print(f"{subagent_type:30s} {count:4d}")

    print(f"\n{'='*80}")
    print(f"SUBAGENT SPAWN EXAMPLES (First 10)")
    print(f"{'='*80}\n")
    for spawn in stats['task_spawns'][:10]:
        model_str = f" | model={spawn.get('model', 'default')}" if spawn.get('model') else ""
        resume_str = " [RESUME]" if spawn.get('has_resume') else ""
        print(f"Entry {spawn['entry_idx']:3d} | {spawn['timestamp']} | {spawn['subagent_type']:15s}{model_str}{resume_str}")
        print(f"  Description: {spawn['description']}")
        print()

    print(f"\n{'='*80}")
    print(f"TOOL CHAINS (Multi-tool Responses, First 10)")
    print(f"{'='*80}\n")
    for chain in stats['tool_chains'][:10]:
        print(f"Entry {chain['entry_idx']:3d} | {chain['timestamp']}")
        tool_names = [t if t is not None else 'null' for t in chain['tools']]
        print(f"  Tools: {' -> '.join(tool_names)}")
        print()

    print(f"\n{'='*80}")
    print(f"STOP REASONS")
    print(f"{'='*80}\n")
    for reason, count in sorted(stats['stop_reasons'].items(), key=lambda x: -x[1]):
        print(f"{reason:30s} {count:4d}")

    print(f"\n{'='*80}")
    print(f"MODELS USED")
    print(f"{'='*80}\n")
    for model, count in sorted(stats['models'].items(), key=lambda x: -x[1]):
        print(f"{model:50s} {count:4d}")

    print(f"\n{'='*80}")
    print(f"TODO STATE DISTRIBUTION")
    print(f"{'='*80}\n")
    for state, count in sorted(stats['todo_states'].items(), key=lambda x: -x[1]):
        print(f"{state:30s} {count:4d}")

    print(f"\n{'='*80}")
    print(f"TASK MODEL SELECTION")
    print(f"{'='*80}\n")
    for model, count in sorted(stats['task_model_usage'].items(), key=lambda x: -x[1]):
        print(f"{model:30s} {count:4d}")

    print(f"\n{'='*80}")
    print(f"AVAILABLE TOOLS ({len(stats['available_tools'])} total)")
    print(f"{'='*80}\n")
    for tool in sorted(stats['available_tools']):
        print(f"  - {tool}")

    # Save detailed analysis
    output_file = log_file.replace('.jsonl', '_workflow_analysis.json')
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_entries': len(entries),
                'requests_with_tool_uses': stats['requests_with_tool_uses'],
                'requests_with_tool_results': stats['requests_with_tool_results'],
                'responses_with_tool_uses': stats['responses_with_tool_uses'],
                'tool_chains_count': len(stats['tool_chains']),
                'subagent_spawns_count': len(stats['task_spawns']),
                'todo_writes_count': len(stats['todo_writes']),
                'plan_mode_enters': stats['plan_mode_enters'],
                'plan_mode_exits': stats['plan_mode_exits'],
                'tool_errors': stats['tool_errors'],
                'total_input_tokens': stats['total_input_tokens'],
                'total_output_tokens': stats['total_output_tokens'],
            },
            'tool_use_count': dict(stats['tool_use_count']),
            'response_tool_count': dict(stats['response_tool_count']),
            'subagent_types': dict(stats['subagent_types']),
            'task_spawns': stats['task_spawns'],
            'tool_chains': stats['tool_chains'],
            'stop_reasons': dict(stats['stop_reasons']),
            'models': dict(stats['models']),
            'todo_states': dict(stats['todo_states']),
            'task_model_usage': dict(stats['task_model_usage']),
            'available_tools': sorted(list(stats['available_tools'])),
            'todo_writes': stats['todo_writes'],
        }, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Detailed analysis saved to: {output_file}")
    print(f"{'='*80}\n")

