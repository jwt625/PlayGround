# DevLog-003: Agent Type and Workflow Visualization Enhancements

**Date:** 2026-01-10
**Status:** Planning
**Related:** DevLog-001, DevLog-002

## Context

Analysis of the Claude Code logging proxy reveals a sophisticated multi-agent architecture with 7 distinct agent types, subagent spawning mechanisms, and complex tool orchestration patterns. The current viewer lacks visualization of these architectural elements, making it difficult to understand the workflow orchestration.

## Analysis Summary

### System Prompt Analysis

**Script:** `scripts/analyze_system_prompts.py`

Identified 7 distinct agent types based on system prompt combinations:

1. **File Path Extractor** (44.2% of requests)
   - Model: GLM-4.6-FP8
   - Purpose: Extract file paths from bash command outputs
   - Post-execution analysis agent

2. **File Search Specialist** (38.0% of requests)
   - Model: GLM-4.6-FP8
   - Purpose: Read-only codebase exploration
   - Tools: Glob, Grep, Read
   - Strict read-only mode enforcement

3. **Bash Command Processor** (8.1% of requests)
   - Model: GLM-4.6-FP8
   - Purpose: Pre-execution command validation
   - Policy-based security checks

4. **Conversation Summarizer** (7.3% of requests)
   - Model: GLM-4.6-FP8
   - Purpose: Generate conversation titles (<50 chars)
   - Metadata generation

5. **Software Architect/Planner** (1.3% of requests)
   - Model: GLM-4.7-FP8 (different model)
   - Purpose: Design implementation plans
   - Read-only architecture analysis
   - Longer prompts (3,747 characters)

6. **Topic Change Detector** (0.5% of requests)
   - Model: GLM-4.6-FP8
   - Purpose: Detect conversation topic changes
   - JSON output format

7. **Main Interactive Agent** (0.5% of requests)
   - Model: GLM-4.7-FP8 (different model)
   - Purpose: Primary CLI assistant
   - Longest prompt (14,260 characters)
   - Full tool access (not read-only)

### Workflow Orchestration Analysis

**Script:** `scripts/analyze_workflow_orchestration.py`

Key metrics from current logs (requests_20260110.jsonl):
- Total entries: 227
- Subagent spawns (Task tool): 5
- Tool chains (multi-tool responses): 66
- Tool errors: 70
- TodoWrite invocations: 3
- Available tools: 22 unique tools

**Subagent Types Detected:**
- Explore (5 occurrences)

**Tool Usage Patterns:**
- Input tools: Read (658), Glob (616), Bash (426), Grep (82)
- Output tools: null (79), Read (78), Bash (46), Glob (42)

**Stop Reasons:**
- end_turn: 106
- tool_use: 87

## Proposed Visualization Enhancements

### 1. System Prompt Type Classification & Color Coding

**Implementation:**
- Compute hash of system prompts to identify agent type
- Color-code metadata bar by agent type
- Add agent type badge/label
- Display legend/key for agent types

**Color Palette:**
```javascript
const AGENT_COLORS = {
  'file_path_extractor': '#10b981',    // green
  'file_search': '#3b82f6',            // blue
  'bash_processor': '#f59e0b',         // amber
  'summarizer': '#8b5cf6',             // purple
  'architect': '#ec4899',              // pink
  'topic_detector': '#06b6d4',         // cyan
  'main_agent': '#ef4444',             // red
  'unknown': '#6b7280'                 // gray
}
```

### 2. Subagent Creation Highlighting

**Detection Criteria:**
- Response contains Task tool usage
- Extract subagent_type from tool input
- Track subagent descriptions

**Visual Elements:**
- Prominent badge/icon for subagent-spawning entries
- Count of subagents spawned
- Expandable section showing subagent types and descriptions
- Special border or background highlight
- Link/reference to spawned subagent entries

### 3. Tool Usage Visualization

**Metrics:**
- Tool count badge (number of tools in response)
- Tool type indicators by category:
  - Read operations: Read, Glob, Grep
  - Write operations: Edit, Write
  - Execution: Bash
  - Orchestration: Task, TodoWrite
  - Planning: EnterPlanMode, ExitPlanMode

**Visual Elements:**
- Tool usage summary in metadata bar
- Color-coded tool badges by category
- Tool chain visualization for multi-tool responses
- Hover tooltip with tool details and parameters

### 4. Additional Metadata Enhancements

**Stop Reason Indicator:**
- Badge showing end_turn vs tool_use vs max_tokens
- Different styling for each stop reason

**Error Highlighting:**
- Flag entries with tool errors (is_error in tool_result)
- Error count badge
- Red/warning styling

**Token Usage Visualization:**
- Input/output token breakdown
- Visual bar for token consumption
- Highlight high-token entries (>10k tokens)

**Model Indicator:**
- Distinguish GLM-4.6-FP8 vs GLM-4.7-FP8
- Different styling for different models

### 5. Workflow State Indicators

**TodoWrite State:**
- Show when TodoWrite is invoked
- Display todo count and states (pending, in_progress, completed)
- Progress indicator

**Plan Mode:**
- Highlight EnterPlanMode/ExitPlanMode usage
- Show planning session boundaries

### 6. Timeline Enhancements

**Agent Type Lanes:**
- Separate timeline rows by agent type
- Show parallel execution
- Visualize agent orchestration flow

**Subagent Spawn Connections:**
- Draw connections from parent to spawned subagents
- Show hierarchical relationships

## Implementation Plan

### Phase 1 (High Priority)
1. System prompt type detection and color coding
2. Tool usage count and badges
3. Subagent creation highlighting

### Phase 2 (Medium Priority)
4. Tool chain visualization
5. Error highlighting
6. Stop reason indicators

### Phase 3 (Enhancement)
7. Timeline agent lanes
8. TodoWrite state tracking
9. Subagent connection visualization

## Technical Requirements

### New Data Extraction Functions

```javascript
extractSystemPromptType(log)      // Hash system prompts to identify agent type
extractToolUsage(log)             // Count and categorize tools from response
extractSubagentSpawns(log)        // Detect Task tool usage
extractToolErrors(log)            // Find tool_result with is_error
extractStopReason(log)            // Get stop_reason from response
extractToolChain(log)             // Extract sequence of tools used
extractTodoState(log)             // Extract TodoWrite state information
```

### System Prompt Hash Mapping

Need to create mapping from system prompt hash to agent type name. Reference hashes from `scripts/system_prompts_analysis.json`:
- `53e64a1b...` -> file_search
- `a7f039fc...` -> file_path_extractor
- `1dbb15f8...` -> bash_processor
- `0080a1ca...` -> summarizer
- `564e15ab...` -> architect
- `7e56ff8e...` -> topic_detector
- `c5e8d165...` -> main_agent

## Files to Modify

- `proxy/viewer/src/App.jsx` - Add extraction functions and metadata display
- `proxy/viewer/src/App.css` - Add styling for agent types, badges, highlights
- `proxy/viewer/src/TimelinePanel.jsx` - Add agent type lanes and connections
- `proxy/log_api.py` - Optionally add server-side analysis endpoint

## Next Steps

1. Implement Phase 1 features
2. Test with current log data
3. Iterate based on visual clarity
4. Proceed to Phase 2 and 3

