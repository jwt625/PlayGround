# DevLog-004: Workflow Causality Matching and Dependency Graph

**Date**: 2026-01-10
**Status**: Planning (Updated 2026-01-10)
**Author**: System

## Revision Summary (2026-01-10)

**Key Decisions:**
1. Skip sequential edges (low confidence 40-60%) in Phase 1
2. Cache workflow graph in `log_classifier.py` during enrichment
3. Start with Timeline + Arrows visualization (Phase 2), DAG view in Phase 3
4. Add `task_tool_id` to subagent spawn edges for higher confidence
5. Session detection: 10-minute gap OR different user_id

**Performance Optimizations:**
1. Time windowing: Only process last 1 hour of logs (default)
2. Hard cap: Max 500 logs processed
3. Chronological sorting: Logs sorted oldest-first for graph computation
4. Persistent cache: Enriched data saved to disk (`.enriched_cache.pkl`)
5. Memory + disk cache: Fast restarts, no recomputation unless logs change

## Overview

This document outlines the design and implementation strategy for building a workflow causality graph that tracks which agent/inference calls launch other agents and how inference results flow between different inference calls.

## Problem Statement

Currently, the log viewer displays individual inference requests in isolation. To understand the orchestration workflow, we need to:

1. Identify which inference call spawned which subagent
2. Track how tool outputs from one inference feed into subsequent inferences
3. Visualize the dependency graph and data flow between agents
4. Handle long-running subagents that may report back after many minutes

## Data Structure Analysis

### Available Matching Signals

**Tool Use to Tool Result Linkage**:
- Response contains `response.body.content[]` with `tool_use` blocks
- Each tool_use has a unique `id` field (e.g., `chatcmpl-tool-8a71856430b502d7`)
- Subsequent request contains `body.messages[].content[]` with `tool_result` blocks
- Each tool_result references the original via `tool_use_id` field
- This provides exact matching with 100% confidence

**Subagent Spawn Detection**:
- Task tool in response indicates subagent spawn
- `tool_use.name == "Task"` with `input.subagent_type` specifying agent type
- Subsequent inference with matching `agent_type.name` is the spawned child
- Task tool may include `input.resume` for resuming previous subagent instances

**Temporal Information**:
- All logs have ISO 8601 timestamps
- Response duration available in `response.duration_ms`
- Causality constraint: parent timestamp must precede child timestamp

## Matching Strategy

### Level 1: Direct Tool ID Matching (Primary)

**Approach**: Build hash index of all tool_use IDs, then match tool_result references.

**Algorithm**:
1. First pass: Extract all `tool_use` blocks from responses, index by `tool_use.id`
2. Second pass: Extract all `tool_result` blocks from requests, lookup `tool_use_id` in index
3. Create directed edge: source_inference → target_inference

**Confidence**: 100% (exact ID match)

**Search Scope**: All-to-all within session (max 200 requests per session, 800 total)

**Time Window**: Not strictly required for tool ID matching, but validate temporal ordering

### Level 2: Subagent Spawn Matching (Secondary)

**Approach**: Detect Task tool usage and match to subsequent agent instances.

**Algorithm**:
1. Identify inferences with `has_subagent_spawns == true`
2. Extract `subagent_type` from Task tool input
3. Search forward in time for next inference with matching `agent_type.name`
4. Validate temporal ordering and reasonable time gap

**Confidence**: 85-95% (high confidence with temporal validation)

**Search Scope**: Forward search from parent timestamp

**Time Window**: Up to 1 hour forward (to handle long-running subagents)

**Edge Cases**:
- Multiple subagents of same type: Use closest temporal match
- Subagent resume: Track via `input.resume` field if present
- Parallel subagent spawns: Multiple Task tools in single response

### Level 3: Sequential Flow Detection (Tertiary) - DEFERRED

**Status**: Not implemented in Phase 1. Sequential edges have low confidence (40-60%) and may clutter the graph with false relationships.

**Approach**: Identify sequential inference chains within same agent type.

**Algorithm**:
1. Group inferences by `agent_type.name`
2. Within each group, sort by timestamp
3. Link consecutive inferences if no other relationship exists
4. Validate time gap is reasonable (< 5 minutes for interactive agents)

**Confidence**: 40-60% (lower confidence, unreliable for causal relationships)

**Use Case**: Optional "conversation flow" view if explicitly requested by user

**Decision**: Skip in initial implementation. Focus on high-confidence edges (tool dependencies + subagent spawns).

## Data Model

### Workflow Node

Each inference request becomes a node with:

**Identity**:
- Unique node ID (derived from timestamp + log index)
- Original log index reference
- Timestamp

**Agent Information**:
- Agent type (name, label, color)
- Model used
- System prompt hash

**Performance Metrics**:
- Duration (ms)
- Token usage (input, output, total)
- Stop reason

**Relationships**:
- Parent node ID (if spawned by Task tool)
- Child node IDs (subagents spawned by this node)
- Previous node ID (sequential predecessor)
- Next node ID (sequential successor)
- Tool dependency edges (tool_use_id → target_node_id mapping)

**Tool Information**:
- Tool uses in response (with IDs)
- Tool results in request (with source references)
- Matched tool IDs to downstream nodes

### Edge Types

**Parent-Child Edge** (Subagent Spawn):
- Type: `subagent_spawn`
- Direction: Parent → Child
- Metadata: subagent_type, spawn_time, model, task_tool_id (links to Task tool_use ID)
- Confidence: 0.85-0.95 (higher with task_tool_id linkage)

**Tool Dependency Edge** (Data Flow):
- Type: `tool_result`
- Direction: Source → Target
- Metadata: tool_use_id, tool_name, is_error
- Confidence: 1.0

**Sequential Edge** (Conversation Flow) - DEFERRED:
- Type: `sequential`
- Direction: Previous → Next
- Metadata: time_gap_ms, same_agent_type
- Confidence: 0.4-0.6
- Status: Not implemented in Phase 1 due to low confidence

## Implementation Architecture

### Backend Processing (Python)

**New Module**: `workflow_graph.py`

**Functions**:
- `build_tool_index(logs)`: Create tool_use_id → log_index mapping
- `match_tool_results(logs, tool_index)`: Find tool result dependencies
- `detect_subagent_spawns(logs)`: Identify parent-child relationships
- `build_workflow_graph(logs)`: Combine all matching strategies
- `compute_graph_metrics(graph)`: Calculate graph statistics

**Integration**: Extend `log_classifier.py` to compute and cache workflow graph during log enrichment

**Caching Strategy**:
- Compute graph once when logs are loaded
- Cache in memory alongside enriched logs
- Invalidate and recompute when new log entries are added
- Store graph as part of enriched log data structure

**Output**: Add `workflow_graph` field to enriched logs with nodes and edges

### Frontend Visualization (React)

**New Component**: `WorkflowGraphPanel.jsx`

**Features**:
- Graph visualization (using D3.js or similar)
- Interactive node selection (highlight dependencies)
- Filter by agent type, time range
- Zoom and pan controls
- Export graph data

**Integration**: Add as third tab in bottom panel (Timeline | Stats | Workflow)

## Search Optimization

### Scalability Considerations

**Current Scale**:
- 800 total requests across all sessions
- Max 200 requests per session
- Search window: 1 hour (typically 50-100 requests)

**Optimization Strategy**:
- All-to-all search is acceptable at this scale (O(n²) with n=200 is manageable)
- Hash-based tool ID lookup reduces to O(n) for tool matching
- Forward-only search for subagent spawns (O(n×m) where m << n)

**Performance Targets**:
- Graph computation: < 500ms for 200 requests
- Incremental updates: < 100ms for new log entries
- Frontend rendering: < 1s for full graph

### Caching Strategy

**Backend**:
- Compute graph once during log enrichment in `log_classifier.py`
- Cache as part of enriched log data structure
- Invalidate and recompute when new log entries are added
- Store in memory (no persistence needed)
- Graph computation integrated into `enrich_logs()` function

**Frontend**:
- Receive pre-computed graph from API
- Cache graph data in component state
- Use React.useMemo for rendering computations only
- No client-side graph computation needed

## Visualization Options

### Option 1: Hierarchical Tree View
- Root: Main agent
- Branches: Subagent spawns
- Leaves: Terminal inferences
- Pros: Clear parent-child relationships
- Cons: Doesn't show tool dependencies well

### Option 2: Directed Acyclic Graph (DAG)
- Nodes: Inferences (colored by agent type)
- Edges: Tool dependencies and spawns
- Layout: Topological sort (left-to-right or top-to-bottom)
- Pros: Shows all relationships
- Cons: Can be complex with many nodes

### Option 3: Sankey Diagram
- Flows: Tool outputs → Tool inputs
- Width: Proportional to data size or token count
- Pros: Emphasizes data flow
- Cons: Less clear for spawn relationships

### Option 4: Timeline with Dependency Arrows (SELECTED FOR PHASE 1)
- Base: Existing timeline visualization (already implemented)
- Overlay: Arrows showing dependencies
- Arrow types:
  - Solid arrows: Tool dependencies (tool_use → tool_result)
  - Dashed arrows: Subagent spawns (parent → child)
- Pros: Combines temporal and causal views, leverages existing UI
- Cons: Can be cluttered with many dependencies
- Mitigation: Add filtering to show/hide specific edge types

**Implementation Decision**:
- **Phase 1**: Option 4 (Timeline + Arrows) - fastest to implement, leverages existing TimelinePanel
- **Phase 2**: Option 2 (DAG) - comprehensive standalone view
- **Phase 3**: Option 1 (Tree) - optional hierarchical view for subagent spawns only

## Metrics and Statistics

### Graph-Level Metrics
- Total nodes (inferences)
- Total edges (dependencies)
- Graph depth (max chain length)
- Branching factor (avg children per parent)
- Subagent spawn count
- Tool dependency count

### Node-Level Metrics
- Degree (in/out edges)
- Descendants count (total subagents spawned)
- Critical path (longest dependency chain)
- Execution time (including all descendants)

### Agent-Level Aggregations
- Spawn frequency by agent type
- Average subagent lifetime
- Tool usage patterns per agent type
- Error rates in dependency chains

## Implementation Phases

### Phase 1: Backend Graph Construction
- Implement tool ID indexing
- Implement tool result matching
- Implement subagent spawn detection
- Add workflow_graph to API response
- Write unit tests for matching logic

### Phase 2: Frontend Timeline Arrow Overlay
- Extend TimelinePanel component to render dependency arrows
- Add arrow rendering for tool dependencies (solid lines)
- Add arrow rendering for subagent spawns (dashed lines)
- Add toggle controls to show/hide arrow types
- Add hover highlighting for dependency chains
- Implement arrow click to highlight related nodes

### Phase 3: Standalone DAG Visualization
- Create WorkflowGraphPanel component as third bottom panel tab
- Implement DAG layout algorithm (D3.js force-directed or hierarchical)
- Add node/edge rendering with agent type colors
- Add basic interactivity (hover, click, zoom, pan)
- Integrate as "Workflow" tab alongside Timeline and Stats

### Phase 4: Advanced Features
- Add graph filtering (by agent, time, tool)
- Add graph metrics display (depth, branching factor, critical path)
- Add export functionality (JSON, SVG, GraphML)
- Add search/highlight specific execution paths
- Add animation for temporal flow
- Add hierarchical tree view (Option 1) for subagent spawns

### Phase 5: Optimization and Polish
- Performance profiling and optimization
- Add loading states and error handling
- Add user preferences (layout, colors, arrow visibility)
- Documentation and examples
- Unit tests for graph matching algorithms

## Open Questions

1. **Session Detection**: Use time gaps > 10 minutes OR different user_id in metadata
2. **Graph Simplification**: No - keep all nodes for Phase 1, consider collapsing in Phase 4
3. **Error Propagation**: Use red dashed arrows for tool dependencies with errors
4. **Real-time Updates**: Graph computed during log enrichment, updates when new logs added
5. **Export Format**: JSON for Phase 1, add SVG/GraphML in Phase 4

## Success Criteria

- Correctly identify 95%+ of tool dependencies via ID matching
- Correctly identify 85%+ of subagent spawns
- Graph computation completes in < 500ms for typical session
- Visualization renders smoothly for graphs with 100+ nodes
- User can trace complete execution path from root to any leaf

