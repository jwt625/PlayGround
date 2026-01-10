# Claude Code Agentic Workflow Analysis

This document describes the orchestration patterns and components identified from analyzing Claude Code API request logs.

## Overview

Claude Code implements a multi-agent architecture with specialized subagents, planning modes, task management, and tool orchestration. The workflow supports concurrent agent execution, state management, and IDE integration.

## Tool Categories

### Core File and Shell Tools

| Tool | Description |
|------|-------------|
| `Bash` | Execute shell commands |
| `BashOutput` | Read shell command output |
| `Read` | Read file contents |
| `Write` | Write new files |
| `Edit` | Edit existing files |
| `Glob` | Find files by pattern |
| `Grep` | Search file contents |
| `KillShell` | Terminate shell processes |

### Planning and Mode Tools

| Tool | Description |
|------|-------------|
| `EnterPlanMode` | Transition to planning mode for complex tasks requiring architectural decisions |
| `ExitPlanMode` | Exit planning mode after writing plan to file for user approval |

### Task Management

| Tool | Description |
|------|-------------|
| `TodoWrite` | Create and manage structured task lists with state tracking |

**TodoWrite State Machine:**
- `pending` - Task not yet started
- `in_progress` - Currently working on (limit to one at a time)
- `completed` - Task finished successfully

**TodoWrite Fields:**
- `content` - Imperative form (e.g., "Run tests")
- `activeForm` - Present continuous form (e.g., "Running tests")
- `status` - Current state

### Subagent Orchestration

| Tool | Description |
|------|-------------|
| `Task` | Launch specialized subagents for complex, multi-step tasks |

**Task Tool Parameters:**
- `subagent_type` (required) - Type of specialized agent
- `description` (required) - Short 3-5 word task description
- `prompt` (required) - Detailed task instructions
- `model` (optional) - Model selection: `sonnet`, `opus`, `haiku`
- `resume` (optional) - Agent ID to resume previous execution

## Subagent Types

### 1. general-purpose
- **Purpose:** General-purpose agent for researching complex questions, searching code, executing multi-step tasks
- **Tools:** All tools (`*`)
- **Use Case:** When not confident about finding matches in first few tries

### 2. Explore
- **Purpose:** Fast agent specialized for exploring codebases
- **Tools:** All tools
- **Thoroughness Levels:**
  - `quick` - Basic searches
  - `medium` - Moderate exploration
  - `very thorough` - Comprehensive analysis across multiple locations
- **Use Cases:** Find files by patterns, search code for keywords, answer codebase questions

### 3. Plan
- **Purpose:** Fast agent for exploration and planning (similar to Explore)
- **Tools:** All tools
- **Thoroughness Levels:** Same as Explore
- **Use Cases:** Planning-focused exploration tasks

### 4. statusline-setup
- **Purpose:** Configure user's Claude Code status line setting
- **Tools:** `Read`, `Edit`
- **Use Case:** Status line configuration only

### 5. claude-code-guide
- **Purpose:** Answer questions about Claude Code or Claude Agent SDK
- **Tools:** `Glob`, `Grep`, `Read`, `WebFetch`, `WebSearch`
- **Features:** Supports `resume` parameter to reuse existing agent context
- **Use Cases:** Feature questions, usage guidance, SDK architecture

## Skills and Extensions

| Tool | Description |
|------|-------------|
| `Skill` | Execute specialized skills (e.g., `pdf`, `xlsx`) within conversation |
| `SlashCommand` | Execute custom slash commands from `.claude/commands/` |

## IDE Integration (MCP)

| Tool | Description |
|------|-------------|
| `mcp__ide__getDiagnostics` | Get language diagnostics from VS Code |
| `mcp__ide__executeCode` | Execute Python code in Jupyter kernel |

## Web and Notebook Tools

| Tool | Description |
|------|-------------|
| `WebFetch` | Fetch webpage content |
| `WebSearch` | Search the web |
| `NotebookEdit` | Edit Jupyter notebook cells (replace/insert/delete) |
| `AskUserQuestion` | Prompt user for input |

## Response Metadata

**Stop Reasons:**
- `end_turn` - Normal completion
- `tool_use` - Response contains tool calls
- `null` - Incomplete or error

**Cache Control:**
- System prompts support `cache_control: {"type": "ephemeral"}` for caching optimization

## Orchestration Patterns

### 1. Parallel Agent Execution
Multiple Task tool calls can be sent in a single message for concurrent execution.

### 2. Tool Chaining
Multiple tools can be called in sequence within a single response.

### 3. Agent Resume
Subagents can be resumed using the `resume` parameter with a previous agent ID.

### 4. Plan Mode Workflow
1. `EnterPlanMode` - Enter planning state
2. Explore codebase with Glob, Grep, Read
3. Write plan to designated file
4. `ExitPlanMode` - Submit for user approval
5. Implement after approval

### 5. Task State Management
TodoWrite maintains real-time task state with immediate status updates.

## Tool Use Linking

- Tool uses have unique `id` fields (e.g., `chatcmpl-tool-8a71856430b502d7`)
- Tool results reference via `tool_use_id` field
- `is_error` field indicates execution status

## Observed Statistics (from requests_20260109.jsonl)

| Metric | Count |
|--------|-------|
| Total log entries | 541 |
| Requests with tool uses (input) | 210 |
| Requests with tool results (input) | 210 |
| Responses with tool uses (output) | 154 |
| Tool chains (multi-tool responses) | 121 |
| Subagent spawns (Task tool) | 5 |
| TodoWrite invocations | 11 |
| Plan mode enters | 0 |
| Plan mode exits | 0 |
| Tool errors | 294 |
| Unique tools available | 20 |
| Total input tokens | 8,657,489 |
| Total output tokens | 188,693 |

### Stop Reason Distribution

| Stop Reason | Count |
|-------------|-------|
| end_turn | 285 |
| tool_use | 174 |

### Model Usage

| Model | Count |
|-------|-------|
| hosted_vllm/zai-org/GLM-4.7-FP8 | 263 |
| zai-org/GLM-4.7-FP8 | 196 |

### Top Tool Usage (Response)

| Tool | Count |
|------|-------|
| null | 123 |
| Bash | 110 |
| Read | 79 |
| Glob | 38 |
| Grep | 23 |
| TodoWrite | 11 |
| Edit | 10 |
| Task | 5 |
| Write | 2 |

### Todo State Distribution

| State | Count |
|-------|-------|
| completed | 33 |
| pending | 16 |
| in_progress | 10 |

### Subagent Types Used

| Type | Count |
|------|-------|
| Explore | 5 |

