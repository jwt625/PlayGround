# DevLog-002: Digital Twin Components

**Date**: 2026-01-13  
**Status**: Planning  
**Parent**: DevLog-001-Pattern-Mining-Strategy.md  
**Goal**: Define comprehensive components of a digital twin/persona for autonomous decision-making

## Overview

The digital twin serves as a retrievable knowledge base enabling SOTA LLMs to make decisions aligned with user preferences without explicit guidance. It must capture not just explicit preferences but decision-making patterns, context-dependent behaviors, and implicit values.

## Core Components

### 1. Technical Preferences

**Tool Selection Matrix**
- Language-specific tooling (Python: uv, JavaScript: pnpm, etc.)
- Package managers per ecosystem
- Testing frameworks and approaches
- Build systems and CI/CD preferences
- Version control practices
- IDE and editor configurations

**Technology Stack Preferences**
- Framework choices per use case (React vs Vue, FastAPI vs Flask)
- Database preferences (SQL vs NoSQL, specific engines)
- Infrastructure choices (Docker, Kubernetes, serverless)
- Cloud provider preferences
- Authentication/authorization approaches

**Architecture Patterns**
- Preferred design patterns per context
- Code organization principles
- Module/component structure preferences
- API design patterns (REST, GraphQL, gRPC)
- State management approaches
- Error handling strategies

### 2. Code Style and Quality Standards

**Formatting and Aesthetics**
- Indentation, spacing, line length
- Naming conventions (camelCase vs snake_case contexts)
- Comment style and density
- Documentation format preferences
- UI/UX aesthetic preferences (no rounded corners, etc.)

**Code Quality Thresholds**
- When to refactor vs accept technical debt
- Test coverage expectations
- Performance optimization triggers
- Security considerations priority
- Acceptable complexity levels

**Review Standards**
- What triggers request for changes
- Acceptable shortcuts vs must-fix issues
- Code review focus areas
- Documentation requirements

### 3. Workflow and Process Patterns

**Project Initialization**
- Standard setup sequences
- Initial file structure
- Configuration file templates
- Dependency installation order
- Environment setup steps

**Development Cycle**
- Research depth before implementation
- Planning vs diving in threshold
- Iteration approach (big bang vs incremental)
- Testing cadence (TDD, test-after, test-when-done)
- Commit frequency and message style

**Problem-Solving Approach**
- Debugging methodology
- When to read docs vs experiment
- Acceptable trial-and-error scope
- When to ask for help vs persist
- Rollback vs fix-forward preference

**Decision-Making Process**
- Information gathering depth
- Speed vs thoroughness trade-offs
- Risk tolerance levels
- When to choose proven vs cutting-edge
- Build vs buy thresholds

### 4. Communication and Collaboration Style

**Instruction Giving**
- Explicitness level expected
- Context provision patterns
- Assumption tolerance
- Preferred response format
- Acceptable verbosity range

**Feedback Patterns**
- Correction style and tone
- Approval expression patterns
- Frustration indicators
- Satisfaction signals
- Clarification request patterns

**Autonomy Boundaries**
- When LLM should ask permission vs proceed
- Acceptable initiative scope
- Destructive action thresholds
- Creativity vs conservatism balance

### 5. Domain Knowledge and Context

**Project Types and Contexts**
- Web applications (frontend, backend, fullstack)
- Data processing and analysis
- Infrastructure and DevOps
- Machine learning and AI
- Tooling and automation

**Business and Product Thinking**
- Feature prioritization criteria
- User experience priorities
- Performance vs feature trade-offs
- Scalability considerations
- Maintenance burden tolerance

**Domain-Specific Expertise**
- Known strong areas
- Learning areas
- Delegation preferences
- Research depth per domain

### 6. Meta-Preferences and Values

**Development Philosophy**
- Pragmatism vs perfectionism balance
- Innovation vs stability preference
- Simplicity vs feature-richness
- Explicit vs implicit code
- DRY vs YAGNI application

**Time and Resource Management**
- Speed vs quality trade-offs
- When to optimize vs ship
- Technical debt tolerance
- Learning investment willingness
- Automation threshold (when to automate repetitive tasks)

**Risk and Error Tolerance**
- Acceptable failure modes
- Backup and safety requirements
- Experimentation boundaries
- Production vs development risk profiles

### 7. Contextual Decision Rules

**Conditional Preferences**
- "If project type X, then use Y"
- "If timeline is tight, then Z"
- "If working with team, then A, if solo then B"
- "If production, then strict rules, if prototype, then flexible"

**Priority Hierarchies**
- Security vs convenience
- Performance vs maintainability
- User experience vs development speed
- Consistency vs pragmatism

**Evolution and Learning**
- How preferences change over time
- Adoption curve for new technologies
- Deprecated preferences
- Experimental preferences (trying out new approaches)

### 8. Anti-Patterns and Constraints

**Hard Constraints**
- Never do X (e.g., no emojis in code/docs)
- Always do Y (e.g., always use package managers)
- Forbidden approaches
- Required safety checks

**Soft Constraints**
- Avoid X unless Z
- Prefer Y but accept alternatives
- Discouraged but not forbidden patterns

**Common Mistakes to Avoid**
- LLM behaviors that consistently trigger corrections
- Misunderstandings that recur
- Assumption errors
- Scope creep patterns

## Data Organization Architecture

### Format Strategy

**Key Decision**: Use LLM-native formats instead of JSON
- Markdown for primary content (what LLMs train on and understand best)
- YAML for structured rules (more readable than JSON)
- Natural language with examples embedded
- Human-readable and version-controllable

### Tiered Storage Architecture

The architecture uses 4 storage tiers, ordered by **retrieval frequency** (not by layer number):

```
┌─────────────────────────────────────────────────────────┐
│                     ALWAYS LOADED                        │
│  ┌─────────────────────────────────────────────────┐    │
│  │ Index                                           │    │
│  │ - hard_constraints.yaml                         │    │
│  │ - lookup_tables/                                │    │
│  │ - quick_reference.md                            │    │
│  │ Size: ~3-5KB                                    │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                          │
              retrieval needed?
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    ON-DEMAND (Tier 1)                    │
│  ┌─────────────────────────────────────────────────┐    │
│  │ Knowledge Base                                  │    │
│  │ - Preference documents (markdown)               │    │
│  │ - Workflow guides                               │    │
│  │ - Full reasoning and context                    │    │
│  │ Size: ~50-200KB total, retrieve ~5KB at a time │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                          │
              still uncertain?
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    ON-DEMAND (Tier 2)                    │
│  ┌─────────────────────────────────────────────────┐    │
│  │ Evidence                                        │    │
│  │ - Actual conversation excerpts                  │    │
│  │ - Correction examples                           │    │
│  │ - Approval examples                             │    │
│  │ Size: ~500KB-1MB, retrieve specific examples   │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                          │
              never accessed at runtime
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                       COLD STORAGE                       │
│  ┌─────────────────────────────────────────────────┐    │
│  │ Archive                                         │    │
│  │ - Raw conversation JSON                         │    │
│  │ - Only for re-processing pipeline              │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

**Index** (Always in System Prompt)
- YAML lookup tables for common decisions
- Hard constraints list (~20 rules that apply 80% of the time)
- Quick reference before any action
- Size budget: ~3-5KB to fit in every context

**Knowledge Base** (Retrieved On-Demand)
- Markdown documents organized hierarchically
- Primary source for preferences, workflows, reasoning
- YAML sidecars for machine-readable rules
- Cross-references as markdown links
- Retrieved via semantic search or task-triggered lookup

**Evidence** (Retrieved When Uncertain)
- Curated exchanges supporting Knowledge Base claims
- Markdown format with conversation context
- Only accessed when LLM needs proof or analogous examples
- Much smaller than full history (20-30% of exchanges)
- Filtered to: corrections, explicit preferences, decision moments

**Archive** (Cold Storage - Never at Runtime)
- Original JSON from extraction
- Only accessed for re-processing if extraction logic improves
- Not consumed by LLM during normal operation

### Directory Structure

```
digital_twin/
├── index/                       # Always loaded in system prompt
│   ├── hard_constraints.yaml   # Non-negotiable rules
│   ├── lookup_tables/          # Tool/framework quick lookups
│   │   ├── python.yaml
│   │   └── javascript.yaml
│   └── quick_reference.md      # Condensed preference summary
│
├── knowledge_base/              # Retrieved on-demand (Tier 1)
│   ├── _master_index.md        # Navigation hub
│   ├── technical_preferences/
│   │   ├── README.md
│   │   ├── python/
│   │   │   ├── README.md
│   │   │   ├── package_management.md
│   │   │   ├── testing.md
│   │   │   └── _rules.yaml
│   │   └── javascript/
│   │       ├── README.md
│   │       ├── package_management.md
│   │       └── _rules.yaml
│   ├── workflows/
│   │   ├── README.md
│   │   ├── project_initialization/
│   │   └── development/
│   ├── style_guidelines/
│   ├── communication/
│   ├── meta_preferences/
│   └── dependencies/
│       ├── overrides.md
│       └── _graph.yaml
│
├── evidence/                    # Retrieved when uncertain (Tier 2)
│   ├── by_preference/
│   │   ├── python_pkg_mgr/
│   │   │   ├── correction_examples.md
│   │   │   └── approval_examples.md
│   │   └── no_emoji/
│   └── by_domain/
│
└── archive/                     # Cold storage - never at runtime
    └── raw_conversations/
```

### Preference Document Format

Each preference is a markdown document with YAML frontmatter:

```markdown
---
id: pref_python_pkg_mgr
domain: python
category: tooling
type: hard_constraint
confidence: 0.95
priority: 100
last_updated: 2026-01-13
evidence_count: 15
---

# Python Package Management

## Summary
Use `uv` for all Python package management operations.

## Priority
**100** (Strong preference)

## The Preference

Always use `uv` for Python package management instead of `pip`, `poetry`, or other tools.

### Commands

**Installing packages:**
```bash
uv pip install <package>
```

**Creating virtual environments:**
```bash
uv venv
```

## Context and Exceptions

### When This Applies
- All new Python projects
- Projects where you have control over tooling

### Exceptions
- Legacy projects explicitly marked as using pip/poetry
- When working in a team with established different tooling

## Reasoning

Based on conversation history, the user consistently corrects usage of pip to uv.
This appears to be a strong, non-negotiable preference.

## Related Preferences

- [Virtual Environment Management](virtual_environments.md)
- [Dependency Files](dependency_files.md)
- [Project Initialization](../../workflows/project_initialization/python_web_app.md)

## Override Rules

This preference can be overridden by:
- Explicit user instruction: "use pip for this project"
- Project context flag: `legacy_tooling: true`

Priority of override must be >= 100 to take effect.

## Evidence

Extracted from 15 conversations:
- 2024-11-03: [conv_123](../../evidence/conv_123.md) - Correction from pip to uv
- 2024-11-15: [conv_145](../../evidence/conv_145.md) - Approval of uv usage
- [See full evidence list](../../evidence/pref_python_pkg_mgr.md)
```

### Cross-Dependency System

**Dependency Types**
- Hierarchical Override: Child overrides parent in specific context
- Conditional Dependency: "If A, then B applies"
- Mutual Exclusion: "A and B cannot both be true"
- Prerequisite: "B requires A to be set first"
- Modifier: "A changes the value/priority of B"

**Example: Workflow depends on Technical Stack**
```yaml
dependency_id: dep_001
type: conditional
source: tech.language.python
target: workflow.initialization.sequence
condition: "project.language == 'python'"
effect:
  type: replace
  value: ["create_venv_with_uv", "install_deps_with_uv", "setup_pytest"]
priority: 80
```

**Priority Levels** (0-1000)
- 1000: Hard constraints (never override)
- 900-999: Critical preferences (override only with explicit user instruction)
- 700-899: Strong preferences (override with strong contextual evidence)
- 400-699: Moderate preferences (override with contextual rules)
- 100-399: Weak preferences (easily overridden by context)
- 0-99: Suggestions (lowest priority)

### Retrieval Strategy

**Tiered Retrieval Flow**

```
┌─────────────────────────────────────────────────────────┐
│                   User Message                           │
│  "Set up a new FastAPI project"                         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│         Step 1: Index Check (No retrieval needed)        │
│                                                          │
│  Index is already in system prompt. Agent checks:        │
│  - hard_constraints.yaml: "Always use uv for Python"    │
│  - quick_reference.md: Python project conventions       │
│                                                          │
│  → Hard constraint found? Follow it.                    │
│  → No match or need more detail? Continue to Step 2.    │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│         Step 2: Knowledge Base Retrieval                 │
│                                                          │
│  Agent calls: retrieve_context("Python project setup")  │
│                                                          │
│  Returns:                                                │
│  - workflows/project_initialization/python_web_app.md   │
│  - technical_preferences/python/package_management.md   │
│                                                          │
│  → High-confidence match? Follow it.                    │
│  → Ambiguous or conflicting? Continue to Step 3.        │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│         Step 3: Evidence Retrieval (Rare)                │
│                                                          │
│  Agent calls: retrieve_evidence("Python setup")         │
│                                                          │
│  Returns:                                                │
│  - evidence/python_pkg_mgr/correction_examples.md       │
│  - Actual conversation excerpts showing preference      │
│                                                          │
│  → Found analogous case? Follow precedent.              │
│  → Still uncertain? Ask user.                           │
└─────────────────────────────────────────────────────────┘
```

**Retrieval Tool Interface**

```python
retrieve_context(
  query: str,
  max_files: int = 5,
  tier: str = "knowledge_base"  # or "evidence"
) -> str  # Combined markdown context
```

**Escalation Triggers**

| Scenario | Start At | Escalate When |
|----------|----------|---------------|
| Common task (Python setup) | Index | Never - constraint sufficient |
| Tool choice question | Index → KB | No explicit rule in Index |
| Style/formatting decision | KB | Conflicting rules found |
| Novel situation | KB → Evidence | No direct match, need examples |
| User correction received | Evidence | Check if known pattern |
| Destructive action | Evidence | Always verify before proceeding |

**Context Assembly**

The retrieval tool combines multiple files into a single markdown context:

```markdown
# Digital Twin Context

Retrieved for query: "Python web application setup"
Tier: Knowledge Base | Files: 2 | Confidence: High

---

## 1. workflows/project_initialization/python_web_app.md

[full content]

---

## 2. technical_preferences/python/package_management.md

[full content]

---

## Related Files (available on request)
- technical_preferences/python/virtual_environments.md
- workflows/development/iteration_patterns.md
```

**Context Budget Management**
- Index: Always loaded (~3-5KB)
- Knowledge Base: Retrieve ~5KB per query, top 2-3 files
- Evidence: Retrieve only specific examples when needed
- Deduplication: Same file returned by multiple queries included only once

## Success Metrics

**Reduction in Corrections**
- Measure frequency of user corrections over time
- Track types of corrections (should decrease for covered areas)

**Decision Alignment**
- LLM decisions match user preferences without prompting
- Reduced need for explicit guidance

**Autonomy Increase**
- LLM can proceed confidently in more scenarios
- Fewer permission requests for routine decisions

**Consistency**
- Similar contexts produce similar decisions
- Predictable behavior aligned with user expectations

## Implementation Considerations

### Retrieval Tool Design

**Option 1: Tool Approach** (Recommended for initial implementation)
- Main agent calls retrieval function
- Agent decides when to call and with what query
- Can call multiple times with different queries
- Builds up context incrementally

**Option 2: Subagent Approach** (Future evolution)
- Separate agent specialized in knowledge base navigation
- Understands the structure deeply
- Can make multiple lookups autonomously
- Assembles coherent context
- Returns to main agent

### Context Budget Strategies

**Progressive Summarization**
```markdown
## Full Content (High Relevance)

### 1. technical_preferences/python/package_management.md
[complete file]

### 2. workflows/project_initialization/python_web_app.md
[complete file]

## Summaries (Medium Relevance)

### 3. technical_preferences/python/testing.md
**Summary**: Use pytest for all Python testing. Prefer fixtures over setup/teardown...
[Available for full retrieval if needed]
```

**Caching Strategy**
- If using Claude: cache the knowledge base structure
- Cache master index and category READMEs
- Reduce token costs for repeated queries

### Evolution Tracking

Track how preferences change over time:

```json
{
  "preference_id": "tech.python.package_manager",
  "current_value": "uv",
  "history": [
    {"value": "pip", "period": "2020-2023", "confidence": 0.9},
    {"value": "poetry", "period": "2023-2024", "confidence": 0.7},
    {"value": "uv", "period": "2024-present", "confidence": 0.95}
  ],
  "trend": "evolving",
  "stability": 0.6
}
```

## Open Questions

- How to handle conflicting preferences from different time periods?
- How to represent uncertainty and learning preferences?
- How to balance specificity vs generalization?
- How to update the twin as user preferences evolve?
- How to handle context-dependent contradictions?
- What confidence threshold for autonomous decision-making?
- How to represent "it depends" scenarios?
- How to capture implicit knowledge vs explicit rules?
- What threshold for filtering exchanges into Evidence tier?
- How to detect when preferences have evolved vs are context-dependent?

## Decisions Made

1. **Format**: Markdown + YAML instead of JSON for LLM-native consumption
2. **Tiered Architecture**: 4 tiers ordered by retrieval frequency:
   - Index (always loaded, ~3-5KB)
   - Knowledge Base (on-demand, semantic search)
   - Evidence (when uncertain, specific examples)
   - Archive (cold storage, never at runtime)
3. **Primary Interface**: Knowledge Base in markdown with prose and examples
4. **Retrieval**: Tiered escalation (Index → KB → Evidence → Ask User)
5. **Evidence Filtering**: Keep only 20-30% of exchanges (corrections, preferences, decisions)
6. **Priority System**: 0-1000 scale with hard constraints at 1000
7. **Cross-Dependencies**: Encoded in YAML with conditional logic

## Next Steps

To be determined based on discussion and implementation planning.

