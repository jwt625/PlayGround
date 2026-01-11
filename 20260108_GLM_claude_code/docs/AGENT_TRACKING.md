# Agent Instance Tracking & Entity Deduplication

This document describes the agent instance tracking and entity deduplication features added to the Claude Code workflow analysis tools.

## Overview

The enhanced entity extraction system now includes:

1. **Agent Instance Tracking**: Identifies and tracks individual agent instances across multiple API requests
2. **Entity Deduplication**: Removes duplicate entities (tool definitions, system prompts, etc.) while preserving references
3. **Conversation Evolution**: Tracks how conversations evolve across requests
4. **Parent-Child Relationships**: Links spawned subagents to their parent agents via Task tool usage

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
└── extract_all_entities.py   # Main extraction script (enhanced)

proxy/viewer/
└── agent_tracker_viz.html    # Standalone visualization
```

### AgentInstanceTracker

**Purpose**: Track agent instances across API requests

**Key Methods**:
- `identify_or_create_agent(request_id, body)`: Identifies or creates agent instance
- `register_task_prompt(task_id, prompt, parent_agent_id)`: Registers task for subagent matching
- `get_agent_hierarchy()`: Returns parent-child relationships
- `get_statistics()`: Returns tracking statistics

**Key Attributes**:
- `agents`: Dict of agent_id → AgentInstance
- `conversation_fingerprints`: Maps fingerprints to agent IDs
- `task_prompts`: Maps task IDs to prompts for subagent matching
- `request_to_agent`: Maps request IDs to agent IDs

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

### Enhanced EntityExtractor

**New Features**:
- Integrates AgentInstanceTracker and EntityDeduplicator
- Adds agent metadata to API requests
- Tracks conversation evolution
- Exports agent instances and hierarchy

**New Output Fields** (in `api_requests`):
```json
{
  "agent_id": "abc123...",
  "agent_type": "0b142c50...",
  "is_continuation": false,
  "conversation_turn": 1,
  "spawned_by_task": null,
  "parent_agent": null
}
```

## Usage

### Extract Entities with Tracking

```bash
python3 -m analysis.extract_all_entities \
    proxy/logs/requests_20260110.jsonl \
    -o proxy/logs/entities_with_tracking.json
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
  }
}
```

### Visualize Agent Tracking

Open `proxy/viewer/agent_tracker_viz.html` in a browser:

1. **Auto-load**: Automatically loads `../logs/entities_with_tracking.json` if available
2. **Manual load**: Use file picker to load any entities JSON file
3. **Interactive timeline**: Hover over nodes to see detailed tooltips
4. **Sorting**: Sort agents by first request, total requests, or agent type

**Visualization Features**:
- Each row represents one agent instance
- Nodes represent API requests
- Node size indicates message count
- Node color indicates agent type
- Tooltips show comprehensive agent details

## Statistics

### Example Output

```
================================================================================
AGENT TRACKING
================================================================================
  Total Agent Instances: 129
  Total Requests: 263
  Avg Requests/Agent: 2.04
  Root Agents: 129
  Child Agents (spawned): 0
  Unique Agent Types: 8

================================================================================
DEDUPLICATION STATISTICS
================================================================================
  Total Unique Entities: 193
  Total Occurrences: 2412
  Overall Duplication Ratio: 12.50x
  Duplicates Removed: 2219

  By Entity Type:
    tool_use       :  193 unique,  2267 total (11.7x)
```

## Implementation Details

### Conversation Fingerprinting

```python
def compute_conversation_fingerprint(messages):
    """
    Creates a hash of the conversation structure:
    - Role sequence (user/assistant)
    - Message content hashes
    - Ignores tool results (they vary)
    """
    parts = []
    for msg in messages:
        if msg['role'] == 'user':
            # Hash user message content
            content_hash = hash_content(msg['content'])
            parts.append(f"U:{content_hash}")
        elif msg['role'] == 'assistant':
            # Hash assistant message content
            content_hash = hash_content(msg['content'])
            parts.append(f"A:{content_hash}")
    
    return hashlib.sha256('|'.join(parts).encode()).hexdigest()[:16]
```

### Agent Identification Logic

1. Extract messages from request body
2. Compute conversation fingerprint
3. Check if fingerprint exists:
   - **Yes**: Return existing agent, increment turn counter
   - **No**: Create new agent instance
4. Check if first user message matches any task prompt:
   - **Yes**: Link as child agent
5. Update agent state (message count, request list)

### Deduplication Logic

1. Compute entity hash based on type-specific key
2. Check if hash exists:
   - **Yes**: Mark as duplicate, return reference to original
   - **No**: Store as new unique entity
3. Track occurrence count and request IDs
4. Return entity with `is_duplicate` flag and `original_id` if duplicate

## Future Enhancements

1. **Improved Subagent Matching**: Use fuzzy matching for task prompts
2. **Conversation Diff**: Show what changed between continuation requests
3. **Agent Lifecycle**: Track agent creation, evolution, and termination
4. **Performance Metrics**: Track response times, token usage per agent
5. **Interactive Filtering**: Filter visualization by agent type, time range, etc.

## Troubleshooting

### Child Agents Not Detected

**Symptom**: `child_agents: 0` in statistics

**Cause**: Task prompts don't exactly match first user messages of spawned agents

**Solution**: 
- Check task prompt extraction in `extract_content_block`
- Verify first user message extraction in `identify_or_create_agent`
- Consider implementing fuzzy matching

### High Duplication Ratio

**Symptom**: Very high duplication ratios (>20x)

**Cause**: Many requests use the same tools/prompts

**Expected**: This is normal for Claude Code workflows where the same tools are used repeatedly

### Missing Agent Instances

**Symptom**: Fewer agents than expected

**Cause**: Requests without messages or system prompts

**Solution**: Check that all requests have valid `body.messages` and `body.system`

## References

- `analysis/agent_tracker.py`: Agent tracking implementation
- `analysis/entity_deduplicator.py`: Deduplication implementation
- `analysis/extract_all_entities.py`: Main extraction script
- `proxy/viewer/agent_tracker_viz.html`: Visualization tool

