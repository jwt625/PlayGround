# Agent Workflow Tracking

Comprehensive documentation for agent instance tracking, entity deduplication, and workflow DAG visualization in Claude Code analysis tools.

## Overview

The workflow tracking system provides:

1. **Agent Instance Tracking**: Identifies and tracks individual agent instances across multiple API requests
2. **Entity Deduplication**: Removes duplicate entities (tool definitions, system prompts, etc.) while preserving references
3. **Conversation Evolution**: Tracks how conversations evolve across requests
4. **Parent-Child Relationships**: Links spawned subagents to their parent agents via Task tool usage
5. **Workflow DAG**: Builds a directed acyclic graph showing agent interactions and tool dependencies

## Key Concepts

### Agent Instance

An **agent instance** represents a single continuous conversation with Claude. It is identified by:

- **Conversation Fingerprint**: A hash of the conversation history (user/assistant message sequence)
- **System Prompt Hash**: The type of agent (e.g., file search specialist, command executor)
- **Message Count**: Number of messages in the conversation
- **First User Message**: The initial user request that started this agent

### Conversation Continuation

When a request contains the same conversation history as a previous request (plus new messages), it's identified as a **continuation** of the same agent instance.

### Agent Spawning

When an agent uses the `Task` tool to spawn a subagent:
1. The task prompt is registered
2. When a new agent starts with a matching first user message, it's linked as a child
3. Parent-child relationships are tracked in the agent hierarchy

## Architecture

### Components

```
analysis/
├── __init__.py               # Package exports
├── agent_tracker.py          # AgentInstanceTracker class
├── entity_deduplicator.py    # EntityDeduplicator class
└── extract_all_entities.py   # Main extraction script

proxy/
├── log_api.py                # REST API server for logs
├── log_classifier.py         # Log entry enrichment (agent type, tool info)
├── workflow_graph.py         # Session-aware workflow DAG construction
└── viewer/
    ├── src/App.jsx           # Main React application
    ├── src/WorkflowPanel.jsx # D3.js force-directed graph
    ├── src/AgentGanttPanel.jsx # Gantt chart with spawn arrows
    └── workflow_tree_viz.html # Standalone HTML viewer
```

### AgentInstanceTracker

**Purpose**: Track agent instances across API requests

**Key Methods**:
- `identify_or_create_agent(request_id, body, timestamp)`: Identifies or creates agent instance
- `register_task_prompt(task_id, prompt, parent_agent_id)`: Registers task for subagent matching
- `build_workflow_dag()`: Builds complete workflow DAG
- `get_agent_hierarchy()`: Returns parent-child relationships
- `get_statistics()`: Returns tracking statistics

**Key Attributes**:
- `instances`: Dict of agent_id → AgentInstance
- `fingerprint_to_agent`: Maps conversation fingerprints to agent IDs
- `task_prompts`: Maps task IDs to prompts for subagent matching
- `request_to_agent`: Maps request IDs to agent IDs
- `workflow_edges`: All edges in the workflow DAG

### EntityDeduplicator

**Purpose**: Deduplicate entities while preserving references

**Key Methods**:
- `deduplicate_entity(entity, entity_type, request_id)`: Deduplicates an entity
- `get_deduplication_stats()`: Returns deduplication statistics

**Deduplication Strategy**:
- **Tool Definitions**: By name
- **System Prompts**: By content hash
- **Tasks**: By prompt content
- **Tool Uses**: By tool name + input hash

## Data Model

### Workflow DAG Structure

```json
{
  "nodes": [
    {
      "id": "agent_4",
      "agent_id": "agent_4",
      "agent_type": "9b285b9c331d663e",
      "parent_agent_id": null,
      "child_agent_ids": ["agent_5", "agent_6"],
      "spawned_by_task_id": null,
      "request_count": 2,
      "tool_use_count": 6,
      "tool_result_count": 0,
      "first_timestamp": "2026-01-10T19:45:00.105423Z",
      "last_timestamp": "2026-01-10T19:45:06.207973Z",
      "is_root": true,
      "is_leaf": false
    }
  ],
  "edges": [
    {
      "type": "subagent_spawn",
      "source_agent_id": "agent_4",
      "target_agent_id": "agent_5",
      "spawned_by_task_id": "chatcmpl-tool-a6399d82b26e7d47",
      "confidence": 0.95
    },
    {
      "type": "tool_result",
      "source_agent_id": "agent_10",
      "target_agent_id": "agent_10",
      "tool_use_id": "chatcmpl-tool-9bd81cfe4057df7b",
      "tool_name": "Glob",
      "source_request_id": 23,
      "target_request_id": 23,
      "is_error": false,
      "confidence": 1.0
    }
  ],
  "metrics": {
    "total_agents": 129,
    "root_agents": 82,
    "leaf_agents": 128,
    "total_edges": 2121,
    "spawn_edges": 47,
    "tool_result_edges": 2074
  },
  "root_agent_ids": ["agent_0", "agent_1"]
}
```

### Edge Types

1. **subagent_spawn**: Parent agent spawns child agent
   - Represents branching in workflow
   - Created when Task tool is used
   - Links parent to child agent

2. **tool_result**: Tool use receives result
   - Represents data flow
   - Created for every tool execution
   - Links tool use to tool result (may be same agent or different)

## Usage

### Extract Entities with Tracking

```bash
python3 -m analysis.extract_all_entities \
    proxy/logs/requests_20260110.jsonl \
    -o proxy/logs/entities_20260110.json
```

### Output Structure

```json
{
  "metadata": {
    "extraction_timestamp": "2026-01-11T...",
    "summary": {
      "counts": { ... },
      "agent_tracking": {
        "total_agents": 129,
        "total_requests": 263,
        "avg_requests_per_agent": 2.04,
        "root_agents": 129,
        "child_agents": 0,
        "unique_agent_types": 8
      },
      "deduplication": {
        "total_unique_entities": 193,
        "total_occurrences": 2412,
        "overall_duplication_ratio": 12.5,
        "duplicates_removed": 2219,
        "by_entity_type": { ... }
      }
    }
  },
  "entities": {
    "agent_instances": [ ... ],
    "api_requests": [ ... ],
    ...
  },
  "relationships": {
    "agent_hierarchy": { ... },
    "request_to_agent": { ... },
    ...
  },
  "workflow_dag": { ... }
}
```

### Visualize Workflow Tree

```bash
open proxy/viewer/workflow_tree_viz.html
# Then load proxy/logs/entities_20260110.json using the file picker
```

Features:
- **Request-level visualization**: Each node represents a single API request
- **Git-style tree layout**: Hierarchical view with spawn and sequential edges
- **Interactive exploration**: Click nodes to view full conversation
- **Zoom and pan**: Navigate large workflows
- **Statistics panel**: Overview of workflow metrics
- **Root selection**: Choose which agent tree to visualize



## Implementation Details

### Conversation Fingerprinting

The conversation fingerprint is computed as:

```python
def compute_conversation_fingerprint(messages):
    """Compute fingerprint from user/assistant message sequence."""
    conversation_sequence = []
    for msg in messages:
        if msg['role'] in ['user', 'assistant']:
            # Use role + content hash
            content_str = json.dumps(msg.get('content', ''), sort_keys=True)
            content_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]
            conversation_sequence.append(f"{msg['role']}:{content_hash}")

    fingerprint_str = '|'.join(conversation_sequence)
    return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]
```

### Agent Matching

When a new request arrives:

1. **Compute fingerprint** of conversation history
2. **Check for exact match**: If fingerprint exists, it's a continuation
3. **Check for prefix match**: If fingerprint is a prefix of existing, it's a continuation
4. **Check for task spawn**: If first user message matches a registered task prompt, link as child
5. **Create new agent**: Otherwise, create a new root agent

### Workflow DAG Construction

The DAG is built in two passes:

1. **First pass**: Track all agent instances and their requests
   - Identify continuations
   - Register task prompts
   - Match spawned agents

2. **Second pass**: Build edges
   - Add spawn edges (parent → child)
   - Add tool result edges (tool_use → tool_result)
   - Compute metrics

## Visualization Features

### Request-Level Nodes

Each node in the tree represents a single API request:

- **Node ID**: `{agent_id}_req_{index}` (e.g., `agent_4_req_0`)
- **Label**: `{agent_id} [{turn}/{total}] - {time}`
- **Color**: Root requests are teal, others are blue
- **Click**: Opens modal with full conversation

### Edge Types

- **Sequential** (gray): Connects consecutive requests in same agent
- **Spawn** (green): Connects parent's last request to child's first request

### Modal View

Clicking a node shows:
- **Request metadata**: Agent type, request ID, turn number, timestamp
- **Full conversation**: All messages with content blocks
- **Tool uses**: Tool name and input parameters
- **Tool results**: Truncated results (first 500 chars)

## Statistics

The workflow DAG includes comprehensive metrics:

```json
{
  "total_agents": 129,
  "root_agents": 82,
  "leaf_agents": 128,
  "total_edges": 2121,
  "spawn_edges": 47,
  "tool_result_edges": 2074,
  "max_depth": 3,
  "avg_children_per_agent": 0.36
}
```

## Future Enhancements

Implemented:
- Timeline view (TimelinePanel.jsx)
- Search and filter with regex support
- Performance metrics (duration display)
- Agent type color coding

Potential improvements:
1. **Tool dependency graph**: Visualize tool usage patterns
2. **Export to formats**: GraphML, DOT, etc.
3. **Color-coded arrows**: By confidence level
4. **Spawn path highlighting**: Trace spawn sequence on hover
5. **Animation**: Animate spawn sequence playback


