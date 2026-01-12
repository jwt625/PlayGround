# DevLog-005-03: Dependency Tracking Analysis and Enhancement

**Date**: 2026-01-11
**Status**: IN PROGRESS
**Parent**: DevLog-005 (Entity Extraction and Agent Architecture Analysis)
**Related Files**:
- `analysis/agent_tracker.py`
- `analysis/entity_deduplicator.py`
- `analysis/extract_all_entities.py`
- `proxy/workflow_graph.py`

## Overview

Analysis of current dependency tracking implementation and enhancement to include content reuse and request sequence dependencies in entities.json for improved Gantt chart visualization.

## Current Implementation

### Deduplication Methods

**1. Content-Based Hashing**
- **Tool Uses**: By `tool_use_id` (unique identifier)
- **Tasks**: By `task_id` (tool_use_id for Task tools)
- **System Prompts**: SHA-256 hash (16 chars) of combined text
- **Tool Definitions**: By `tool_name`

**Results**: 57x reduction (570 task occurrences → 10 unique)

**2. Conversation Fingerprinting**
- **Method**: SHA-256 hash of message sequence based on content, not IDs
- **Components**: role + text_hash + tool_name:input_hash + result_hash
- **Ignores**: Unique IDs, timestamps, formatting variations

**Agent Identification Strategy** (in order):
1. Exact fingerprint match → Same agent (replay)
2. Parent conversation match (backtrack 1-5 messages) → Continuation
3. Task spawn detection (first message matches task prompt) → Child agent
4. Otherwise → New root agent

### Current Dependency Types in entities.json

**Type 1: TOOL_RESULT** (2,074 edges in sample data)
```json
{
  "type": "tool_result",
  "source_agent_id": "agent_6",
  "target_agent_id": "agent_6",
  "tool_use_id": "chatcmpl-tool-9bd81cfe4057df7b",
  "tool_name": "Glob",
  "source_request_id": 23,
  "target_request_id": 23,
  "is_error": false,
  "confidence": 1.0
}
```
- Tracks: Agent uses tool → Agent receives result
- Usually self-loops (same agent)
- Source: `agent_tracker.py` lines 434-488

**Type 2: SUBAGENT_SPAWN** (5 edges in sample data)
```json
{
  "type": "subagent_spawn",
  "source_agent_id": "agent_4",
  "target_agent_id": "agent_5",
  "spawned_by_task_id": "chatcmpl-tool-a6399d82b26e7d47",
  "confidence": 0.95
}
```
- Tracks: Parent spawns child via Task tool
- Method: Hash matching of task prompt to first user message
- Source: `agent_tracker.py` lines 519-528

### Missing from entities.json

**Type 3: CONTENT_REUSE** (implemented in workflow_graph.py but NOT in entities.json)
- Tracks: Agent A output → Agent B input
- Method: Hash first 200 chars of text content
- Status: Available via `/api/workflow` but not in entities.json

**Type 4: REQUEST_SEQUENCE** (not implemented)
- Tracks: Sequential requests within same agent
- Example: request_1 → request_2 → request_3 for agent_0

## Proposed Enhancement

### Add to entities.json workflow_dag.edges

**Type 3: CONTENT_REUSE**
```json
{
  "type": "content_reuse",
  "source_agent_id": "agent_4",
  "target_agent_id": "agent_5",
  "source_request_id": 42,
  "target_request_id": 58,
  "content_hash": "a3f5e8c2",
  "confidence": 0.85
}
```

**Type 4: REQUEST_SEQUENCE**
```json
{
  "type": "request_sequence",
  "source_agent_id": "agent_1",
  "target_agent_id": "agent_1",
  "source_request_id": 2,
  "target_request_id": 3,
  "time_gap_ms": 1523,
  "confidence": 1.0
}
```

### Implementation Plan

**Phase 1: Add Content Reuse Tracking to agent_tracker.py**
- Build response content hash index during extraction
- Match against subsequent request messages
- Create content_reuse edges in workflow_dag

**Phase 2: Add Request Sequence Tracking**
- For each agent, iterate through requests in chronological order
- Create edges between consecutive requests
- Include time gap for temporal analysis

**Phase 3: Update export_to_json**
- Include new edge types in workflow_dag.edges
- Maintain backward compatibility

## Implementation Results

**Status**: COMPLETE

### Changes Made

**1. agent_tracker.py** (lines 151-735)
- Added `response_content_index` for tracking response content hashes
- Added `track_response_content()` method to index response text
- Added `track_request_content()` method to detect content reuse
- Added `build_request_sequence_edges()` method for temporal flow
- Updated `build_workflow_dag()` to include new edge types

**2. extract_all_entities.py** (lines 83-136)
- Call `track_request_content()` after extracting messages
- Call `track_response_content()` after extracting response

### Test Results (requests_20260110.jsonl - 263 requests)

**Edge Counts**:
- subagent_spawn: 5 edges
- tool_result: 2,074 edges
- request_sequence: 190 edges (NEW)
- content_reuse: 0 edges (NEW - none detected in this dataset)

**Agent Statistics**:
- Total agents: 73
- Agents with multiple requests: 68
- Root agents: 68
- Child agents: 5

**Performance**:
- Extraction time: < 5 seconds (no noticeable degradation)
- File size increase: Minimal (190 new edges vs 2,079 existing)

### Edge Type Details

**Request Sequence Edges**:
- Created between consecutive requests within same agent
- Includes time gap in milliseconds
- Example: agent_1 has 6 requests → 5 sequence edges
- Total: 190 edges across 68 multi-request agents

**Content Reuse Edges**:
- Detects when agent A's output appears in agent B's input
- Uses 200-char hash matching
- Only creates edges between different agents
- Result: 0 edges (no cross-agent content sharing in this dataset)

## Success Criteria

1. DONE - entities.json includes content_reuse edges (0 found, but tracking works)
2. DONE - entities.json includes request_sequence edges (190 edges)
3. TODO - Gantt chart can visualize all 4 dependency types
4. DONE - No performance degradation (< 10% increase in extraction time)
5. DONE - Backward compatible with existing visualization code

## Files Modified

1. `analysis/agent_tracker.py`: Added content reuse and request sequence tracking
2. `analysis/extract_all_entities.py`: Integrated new tracking methods
3. `proxy/viewer/src/AgentGanttPanel.jsx`: Render new edge types (COMPLETE)

## Tool-Spawned Subagent Discovery

**Date**: 2026-01-11
**Status**: COMPLETE ✓

### Problem Identified

Current subagent spawn detection only captures Task tool spawns via prompt hash matching. This misses validation subagents spawned by Bash tool calls.

### Critical Bug Found and Fixed

**Bug**: Command normalization was converting `"` to `'` but not removing quotes entirely. This caused hash mismatches when:
- Parent command: `find ... -name "*.md"` → normalized to `find ... -name '*.md'` → hash: `30be1d9a23ea26ef`
- Child command: `find ... -name *.md` → normalized to `find ... -name *.md` → hash: `10076fe56755b07f`

The same command with/without quotes produced different hashes, preventing spawn detection.

**Fix**: Modified `normalize_command()` to remove ALL quotes (both single and double) instead of converting between them:
```python
# Before:
normalized = normalized.replace('"', "'")

# After:
normalized = normalized.replace('"', '')
normalized = normalized.replace("'", '')
```

This ensures consistent matching regardless of quoting style in the original command.

### Verified Examples

**Example 1: agent_8 → agent_10**
- Parent: agent_8, request 2 (request_id=19)
  - Tool call: Bash with command `git branch -a`
  - tool_use_id: `chatcmpl-tool-a5599c4c35f87d0c`
- Child: agent_10, request 1 (request_id=27)
  - First user message: `Command: git branch -a\nOutput: ...`
  - System prompt: "You are Claude Code, Anthropic's official CLI for Claude."
- **Status: ✓ DETECTED** - agent_10.parent_agent_id = agent_8

**Example 2: agent_9 → agent_20**
- Parent: agent_9, request 4 (request_id=45)
  - Tool call: Bash with command `find /Users/.../proxy -maxdepth 1 -name "*.md" -o -name "*.txt" -o -name "*.json" -o -name "*.py" 2>/dev/null | head -20`
  - tool_use_id: `chatcmpl-tool-bc38cc57532bce46`
- Child: agent_20, request 1 (request_id=58)
  - First user message: `Command: find /Users/.../proxy -maxdepth 1 -name *.md -o -name *.txt -o -name *.json -o -name *.py`
  - Includes policy_spec for command validation
- **Status: ✓ DETECTED** - agent_20.parent_agent_id = agent_9 (after quote normalization fix)

### Pattern Analysis

**Spawn Mechanism**:
1. Parent agent calls Bash tool with `{command, description}`
2. System spawns validation subagent (agent_10, agent_20)
3. Child receives user message: `Command: <command>\n[Output: <output>]`
4. Child validates/processes the command execution

**Why Current Detection Fails**:
- Current method only detects Task tool spawns via prompt hash matching
- Bash tool spawns don't use Task tool
- No task prompt to hash-match (command is in tool arguments, not task prompt)
- Command appears in child's first user message, not in a task prompt

### Proposed Solution

**Method: Tool Argument Matching**

Track Bash tool calls and match their `command` argument to child agent's first user message pattern `Command: <command>`.

**Implementation Strategy**:

1. Index Tool Call Commands (in `agent_tracker.py`):
   - Build index: `tool_command_index[command_normalized] = {tool_use_id, agent_id, request_id, tool_name, command}`
   - Normalize command: remove quotes, stderr redirects (`2>/dev/null`), pipes (`| head -20`)

2. Match Child Agent First Message:
   - Extract command from pattern `Command: <command>\n`
   - Normalize and lookup in `tool_command_index`
   - If match found → link as parent-child

3. Enhanced Edge Structure:
```json
{
  "type": "subagent_spawn",
  "spawn_method": "tool_call",
  "source_agent_id": "agent_9",
  "target_agent_id": "agent_20",
  "spawned_by_tool_use_id": "chatcmpl-tool-bc38cc57532bce46",
  "spawned_by_task_id": null,
  "tool_name": "Bash",
  "command_hash": "a3f5e8c2",
  "confidence": 0.90
}
```

**New Fields**:
- `spawn_method`: "task" or "tool_call"
- `spawned_by_tool_use_id`: ID of the tool call that spawned the child
- `tool_name`: Name of the tool that triggered spawn (e.g., "Bash")
- `command_hash`: Hash of normalized command for debugging

### Final Results

**Metrics**:
- Root agents: 25 → 21 (reduced by 4)
- Spawn edges: 5 → 52 (5 Task + 47 Bash)
- Child agents: 48 → 52

**Impact**:
- Successfully detected **47 additional Bash-spawned subagents**
- Captured full workflow hierarchy including all validation subagents
- Quote normalization bug fix was critical for matching commands with different quoting styles
- Both simple and policy-spec message patterns now handled correctly

## Spawn Edge source_request_id Bug

**Date**: 2026-01-11
**Status**: FIXED

### Problem

Spawn arrows in the Gantt visualization were incorrectly pointing from the **last** request of the parent agent instead of the specific request where the tool/task was called.

Example: agent_9 spawns agent_20 at request 45 (4th request), but the arrow was drawn from request 216 (18th/last request).

### Root Cause

The `tool_use_index`, `task_prompts`, and `tool_command_index` were being **overwritten** each time a tool_use_id was encountered. Since conversation history accumulates (each new request includes all previous assistant messages), the same tool_use_id appears in messages across multiple requests. The last request to process the tool_use_id would overwrite the original `request_id`.

### Fix

Modified `agent_tracker.py` to only store the **first** occurrence of each tool_use_id:

```python
# Before: Always overwrite
self.tool_use_index[tool_use_id] = {...}

# After: Only store first occurrence
if tool_use_id not in self.tool_use_index:
    self.tool_use_index[tool_use_id] = {...}
```

Applied same fix to:
- `self.tool_use_index` (line 518)
- `self.task_prompts` (line 553)
- `self.tool_command_index` (line 571)

Added `source_request_id` to spawn edges in `build_workflow_dag()` (line 690).

Updated `AgentGanttPanel.jsx` to use `edge.source_request_id` directly instead of looking up through taskIndex/toolUseIndex.

### Verification

```
Agent_20 spawn edge (FIXED):
  Source: agent_9
  source_request_id: 45
  Parent requests: [17, 22, 29, 45, 81, 93, ...]
  Expected: request 45 (4th request)
  Actual: request 45
  MATCH!

Edges with source_request_id: 52 / 52
ALL spawn edges have source_request_id!
```

## Additional Normalization Bugs

**Date**: 2026-01-12
**Status**: FIXED

### Bug 1: Multi-line Heredoc Command Extraction

**Problem**: Commands using heredoc syntax (e.g., `git commit -m "$(cat <<'EOF'\n...\nEOF\n)"`) were not being extracted correctly. The regex only captured the first line, causing hash mismatches.

**Fix**: Modified `extract_command_from_message()` to detect heredoc patterns and extract the full multi-line command until the closing marker:
```python
# Detect heredoc pattern
heredoc_match = re.search(r"<<\s*'?(\w+)'?", first_line)
if heredoc_match:
    marker = heredoc_match.group(1)
    # Find closing marker and include full content
```

### Bug 2: Stderr Redirect with Space

**Problem**: Command normalization did not handle `2> /dev/null` (with space after `>`). Only `2>/dev/null` was matched.

**Examples**:
- Parent: `find ... -name "*.md" 2>/dev/null | head -20`
- Child: `find ... -name *.md 2> /dev/null`
- After quote removal, the space difference in redirect still caused hash mismatch.

**Fix**: Updated regex to handle optional whitespace after redirect operator:
```python
# Before:
normalized = re.sub(r'\s*2>/dev/null\s*', ' ', command)

# After:
normalized = re.sub(r'\s*2>+\s*/dev/null\s*', ' ', command)
normalized = re.sub(r'\s*>+\s*/dev/null\s*', ' ', normalized)
```

### Results After Fixes

**Metrics**:
- Root agents: 21 -> 18 (3 additional parents detected)
- Child agents: 52 -> 55

### Final Root Agent Count: 11

**Legitimate Root Agents**:
- agent_0-4: Startup agents (warmup, git history, message analyzer, main CLI)
- agent_65, agent_66, agent_68: User-initiated conversations

**File Content Agents**:
- agent_41, agent_49, agent_64: Receive raw file content (lockfiles, React code), no command pattern to match

## Command Normalization Overhaul

**Date**: 2026-01-12
**Status**: FIXED

### Problem

The previous normalization approach used multiple regex patterns to remove specific variations (quotes, redirects, pipes) but still failed to match commands due to:

1. **Quote style variations**: `"*.md"` vs `'*.md'` vs `*.md`
2. **Spacing differences**: `\( -name` vs `\(-name`
3. **Redirect variations**: `2>/dev/null` vs `2> /dev/null`
4. **Compound commands**: Child receives `echo "error"` from parent's `curl ... || echo "error"`
5. **Heredoc truncation**: Child receives first line only, parent has full multi-line content

### Solution

Replaced complex regex patterns with simple alphanumeric-only normalization:

```python
def normalize_command(command: str) -> str:
    if not command:
        return ""
    # Keep only alphanumeric characters
    normalized = re.sub(r'[^a-zA-Z0-9]', '', command)
    return normalized.lower()
```

Added prefix/suffix matching in `detect_tool_spawn()`:

```python
# Prefix match: child is truncated version of parent
if parent_normalized.startswith(child_normalized):
    return cmd_info
# Suffix match: child is latter part of compound command (|| or &&)
if parent_normalized.endswith(child_normalized):
    return cmd_info
```

### Results

| Metric | Before | After |
|--------|--------|-------|
| Root agents | 18 | 11 |
| Child agents | 55 | 62 |
| Spawn edges | 55 | 62 |

All 7 previously unmatched policy spec validators now correctly linked to parents:
- agent_29, agent_31 -> agent_9 (curl/echo compound command)
- agent_48, agent_50 -> agent_7 (find with grep pipe)
- agent_53, agent_58 -> agent_6 (find with spacing variation)
- agent_70 -> agent_4 (git commit heredoc)

## Content Reuse Detection Bug

**Date**: 2026-01-12
**Status**: FIXED

### Problem

Content reuse detection was not finding any edges (0 detected) despite clear evidence of Task tool results flowing from child agents back to parent agents.

**Example**: agent_5's response at request 218 appears in agent_4's request 219 as a `tool_result` block, but no content_reuse edge was created.

### Root Cause

The `track_request_content()` function only examined:
1. String content in user messages
2. `text` type blocks in array content

It did NOT examine `tool_result` blocks, which is the primary mechanism for Task/Bash tool responses to flow back to parent agents.

**Structure of tool_result in user message**:
```json
{
  "role": "user",
  "content": [
    {
      "type": "tool_result",
      "tool_use_id": "chatcmpl-tool-a6399d82b26e7d47",
      "content": [
        {"type": "text", "text": "Now I have a comprehensive understanding..."}
      ]
    }
  ]
}
```

### Fix

Modified `agent_tracker.py`:

1. **Added `_check_content_reuse()` helper**: Refactored content matching with deduplication to avoid duplicate edges from accumulated conversation history.

2. **Added `_extract_text_from_tool_result()` helper**: Handles tool_result content which can be string or array of text blocks.

3. **Updated `track_request_content()`**: Now also examines `tool_result` blocks inside user messages.

### Results

| Metric | Before | After |
|--------|--------|-------|
| content_reuse edges | 0 | 5 |

**Detected Edges**:
| Source | Source Req | Target | Target Req | Tool Use ID |
|--------|------------|--------|------------|-------------|
| agent_5 | 218 | agent_4 | 219 | chatcmpl-tool-a6399d82b26e7d47 |
| agent_6 | 217 | agent_4 | 219 | chatcmpl-tool-b90e643bb8855247 |
| agent_7 | 215 | agent_4 | 219 | chatcmpl-tool-a2360b3e412511c7 |
| agent_8 | 183 | agent_4 | 219 | chatcmpl-tool-bdac5fe90ee5f8aa |
| agent_9 | 216 | agent_4 | 219 | chatcmpl-tool-8bd93476eed2e430 |

All 5 edges correctly point from the child agent's final response request to the parent agent's request where the tool_result was received.

### Visualization Update

Updated `AgentGanttPanel.jsx`:
- Added `sourceRequestId`, `targetRequestId`, `tool_use_id` to edge data
- Updated tooltip to show precise request-level identification
- Adjusted arrow styling: solid line, smaller arrowhead, brighter purple (#c084fc)
