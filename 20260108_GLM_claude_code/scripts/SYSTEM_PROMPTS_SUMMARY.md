# System Prompts Analysis Summary

**Analysis Date:** 2026-01-09  
**Log File:** `proxy/logs/requests_20260109.jsonl`  
**Total Entries with System Prompts:** 371  
**Distinct Combinations:** 7

---

## Overview

Claude Code uses a **two-part system prompt structure**:
1. **Base Identity Prompt** - Always present: "You are Claude Code, Anthropic's official CLI for Claude."
2. **Role-Specific Prompt** - Varies based on the agent's task

All prompts use `cache_control: {"type": "ephemeral"}` for prompt caching optimization.

---

## Distinct System Prompt Combinations

### 1. File Path Extractor (164 occurrences - 44.2%)
**Hash:** `a7f039fc...`  
**Model:** GLM-4.6-FP8  
**Purpose:** Extract file paths from bash command outputs

**Role Description:**
> Extract any file paths that this command reads or modifies. For commands like "git diff" and "cat", include the paths of files being shown.

**Key Characteristics:**
- Used after bash commands execute
- Parses command output to identify file paths
- Distinguishes between commands that display file contents vs. list files
- Returns paths verbatim without resolution

---

### 2. File Search Specialist (141 occurrences - 38.0%)
**Hash:** `53e64a1b...`  
**Model:** GLM-4.6-FP8  
**Purpose:** Read-only codebase exploration and search

**Role Description:**
> You are a file search specialist for Claude Code. You excel at thoroughly navigating and exploring codebases.

**Key Characteristics:**
- **READ-ONLY MODE** - Strictly prohibited from file modifications
- Tools: Glob (pattern matching), Grep (regex search), Read (file contents)
- Bash limited to read-only operations (ls, git status, cat, etc.)
- Must use absolute paths
- No emoji usage
- Includes environment context (working directory, git status, platform info)

**Environment Variables Injected:**
- Working directory
- Git repo status
- Platform (darwin)
- OS version
- Current date
- Model being used
- Recent git commits

---

### 3. Bash Command Processor (30 occurrences - 8.1%)
**Hash:** `1dbb15f8...`  
**Model:** GLM-4.6-FP8  
**Purpose:** Process and validate bash commands before execution

**Role Description:**
> Your task is to process Bash commands that an AI coding agent wants to run.

**Key Characteristics:**
- Policy-based command validation
- Determines command prefix/safety
- Pre-execution security checks

---

### 4. Conversation Summarizer (27 occurrences - 7.3%)
**Hash:** `0080a1ca...`  
**Model:** GLM-4.6-FP8  
**Purpose:** Generate conversation titles

**Role Description:**
> Summarize this coding conversation in under 50 characters. Capture the main task, key files, problems addressed, and current status.

**Key Characteristics:**
- Ultra-concise summaries (< 50 chars)
- Used for conversation metadata
- Captures task, files, problems, status

---

### 5. Software Architect/Planner (5 occurrences - 1.3%)
**Hash:** `564e15ab...`  
**Model:** GLM-4.7-FP8 (Note: Different model!)  
**Purpose:** Design implementation plans

**Role Description:**
> You are a software architect and planning specialist for Claude Code. Your role is to explore the codebase and design implementation plans.

**Key Characteristics:**
- **READ-ONLY MODE** - Similar restrictions to File Search Specialist
- Focus on architecture and design
- Creates implementation strategies
- Must output "Critical Files for Implementation" section
- Longer, more detailed prompts (3,747 characters)

---

### 6. Topic Change Detector (2 occurrences - 0.5%)
**Hash:** `7e56ff8e...`  
**Model:** GLM-4.6-FP8  
**Purpose:** Detect conversation topic changes

**Role Description:**
> Analyze if this message indicates a new conversation topic. If it does, extract a 2-3 word title.

**Key Characteristics:**
- JSON output format: `{"isNewTopic": boolean, "title": string|null}`
- 2-3 word titles
- No markdown, pure JSON

---

### 7. Main Interactive Agent (2 occurrences - 0.5%)
**Hash:** `c5e8d165...`  
**Model:** GLM-4.7-FP8 (Note: Different model!)  
**Purpose:** Primary interactive CLI assistant

**Role Description:**
> You are an interactive CLI tool that helps users with software engineering tasks.

**Key Characteristics:**
- **Longest prompt** (14,260 characters!)
- Comprehensive security guidelines
- Authorized for: security testing, defensive security, CTF challenges, education
- Refuses: destructive techniques, DoS, mass targeting, supply chain compromise
- Full tool access (not read-only)
- Detailed instructions for various tasks

---

## Key Patterns Observed

### Model Usage
- **GLM-4.6-FP8:** Used for most specialized agents (361/371 = 97.3%)
- **GLM-4.7-FP8:** Used for architect/planner and main interactive agent (10/371 = 2.7%)

### Prompt Structure
All prompts follow this pattern:
```json
{
  "system": [
    {
      "type": "text",
      "text": "You are Claude Code, Anthropic's official CLI for Claude.",
      "cache_control": {"type": "ephemeral"}
    },
    {
      "type": "text", 
      "text": "<role-specific instructions>",
      "cache_control": {"type": "ephemeral"}
    }
  ]
}
```

### Cache Control
- All prompts use ephemeral caching
- Optimizes for repeated similar requests
- Reduces latency and cost

### Read-Only Enforcement
Multiple agents have explicit READ-ONLY restrictions:
- File Search Specialist
- Software Architect/Planner

This suggests a security-conscious design preventing accidental modifications.

---

## Usage Distribution

1. **File Path Extractor** - 44.2% (post-bash execution)
2. **File Search Specialist** - 38.0% (codebase exploration)
3. **Bash Command Processor** - 8.1% (pre-bash validation)
4. **Conversation Summarizer** - 7.3% (metadata generation)
5. **Software Architect** - 1.3% (planning tasks)
6. **Topic Detector** - 0.5% (conversation management)
7. **Main Interactive Agent** - 0.5% (general assistance)

---

## Files Generated

- `analyze_system_prompts.py` - Analysis script
- `system_prompts_analysis.json` - Full JSON data with all prompts
- `SYSTEM_PROMPTS_SUMMARY.md` - This summary document

