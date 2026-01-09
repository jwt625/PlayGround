# DevLog-001: Workflow Visualization Planning

## Date
2026-01-08

## Context
The proxy viewer currently displays flat chronological logs of all HTTP requests/responses between Claude Code and inference endpoints. This makes it difficult to understand how Claude Code orchestrates complex workflows involving planning, tool calls, subagent spawning, and result aggregation.

## Goal
Unflatten the logs into a hierarchical orchestration workflow visualization with a middle panel showing compact workflow trees between the timeline (left) and detailed logs (right).

## Current Log Structure

### Data Available
- JSONL format with request-response pairs
- Each entry contains:
  - `timestamp`: ISO 8601 timestamp
  - `method`, `path`, `url`: HTTP metadata
  - `body.model`: Model used (e.g., "zai-org/GLM-4.6-FP8")
  - `body.messages`: Conversation history array
  - `body.tools`: Available tools for this call
  - `response.body.content`: Array of content blocks (text, tool_use)
  - `response.body.stop_reason`: "end_turn", "tool_use", etc.

### Tool Use Patterns
From analysis of logs:
- 16 text responses
- 20 tool_use responses
- Tool types include: Task (subagent), Bash, Read, Edit, Glob, Grep, Write, etc.
- `stop_reason: "tool_use"` indicates Claude is calling tools
- Tool name "Task" indicates subagent spawn

## Proposed Architecture

### Phase 1: Data Model Design
Create hierarchical data structure representing orchestration tree:
- **Node structure**:
  - Unique ID (timestamp + index)
  - Type: planning, tool_use, subagent, response
  - Metadata: timestamp, model, tokens, duration
  - Relationships: parent_id, children[], sequence_index
  - Content: tool names, stop_reason, preview text

- **Relationships**:
  - Parent-child: subagent spawns
  - Sequential: tool call chains (call -> result -> next inference)
  - Parallel: concurrent tool calls in single message

### Phase 2: Log Parser
Parse JSONL and build workflow tree:
1. Extract key fields from each log entry
2. Identify conversation sessions (group related calls)
3. Detect tool use patterns:
   - Extract tool_use content blocks
   - Match tool results in subsequent messages
4. Build parent-child relationships:
   - Task tool use creates child node
   - Tool results link back to parent inference
5. Detect parallel vs sequential execution

### Phase 3: Workflow Reconstruction
Group and classify inference calls:
- **Planning phase**: Initial user request, no tool use
- **Execution phase**: Tool use with stop_reason "tool_use"
- **Subagent workflow**: Task tool spawns nested workflow
- **Tool chains**: Sequential tool call -> result -> next call
- **Parallel execution**: Multiple tool uses in single response

### Phase 4: Middle Panel Visualization
Display compact workflow tree:
- **Visual elements**:
  - Collapsible tree nodes
  - Indentation for hierarchy
  - Icons for node types (planning, tool, subagent)
  - Color coding: planning (blue), tool use (green), subagent (purple), error (red)
  
- **Node display**:
  - Timestamp (relative or absolute)
  - Model name (abbreviated)
  - Tool names (comma-separated if multiple)
  - Token count or duration
  - Status indicator (success, error, in-progress)

- **Interactions**:
  - Click node to show details in main panel
  - Expand/collapse subtrees
  - Highlight current node in timeline
  - Filter by type, model, or time range

## Implementation Strategy

### Option A: Bottom-up (Data-first)
1. Implement data model and parser
2. Build workflow reconstruction logic
3. Create visualization components
4. Integrate with existing viewer

### Option B: Top-down (UI-first)
1. Create middle panel with basic grouping
2. Iterate on visualization design
3. Enhance parser as needed
4. Refine data model based on UI needs

### Option C: Prototype-first
1. Quick minimal working version
2. Validate concept with user
3. Iterate based on feedback
4. Full implementation

## Technical Considerations

### Parser Challenges
- Matching tool results to tool calls (no explicit IDs in current format)
- Detecting session boundaries (when does a new workflow start?)
- Handling malformed or incomplete logs
- Performance with large log files

### Visualization Challenges
- Rendering large trees efficiently
- Synchronizing three panels (timeline, workflow, details)
- Handling deep nesting (subagents spawning subagents)
- Responsive layout for different screen sizes

### Data Model Challenges
- Representing parallel tool calls
- Tracking token usage across workflow
- Handling errors and retries
- Supporting incremental updates (live logging)

## Next Steps
Decision needed on implementation approach:
1. Start with data model and parser (Phase 1-2)
2. Start with UI prototype (Phase 4)
3. Build minimal end-to-end prototype first

## References
- Proxy logs: `proxy/logs/requests_20260108.jsonl`
- Current viewer: `proxy/viewer/src/App.jsx`
- Proxy server: `proxy/proxy_server.py`

