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

### 4-Layer Architecture

**Layer 1: Archive** (Raw Conversations)
- Original JSON from extraction
- Rarely accessed, mainly for re-processing if extraction logic improves
- Not directly consumed by LLM

**Layer 2: Knowledge Base** (Primary LLM Interface)
- Markdown documents organized hierarchically
- LLM reads this 95% of the time
- Includes preferences, workflows, examples, reasoning in prose
- YAML sidecars for machine-readable rules
- Cross-references as markdown links

**Layer 3: Evidence & Provenance** (Supporting Material)
- Curated exchanges supporting Layer 2 claims
- Markdown format with context
- Only accessed when LLM needs deeper context or examples
- Much smaller than full conversation history (20-30% of exchanges)
- Filtered to keep only: corrections, explicit preferences, decision moments, problem-solving patterns

**Layer 4: Decision Index** (Fast Lookup)
- YAML lookup tables for common decisions
- Decision trees in markdown with Mermaid diagrams
- Hard constraints list
- Quick reference before deeper retrieval

### Directory Structure

```
digital_twin/
├── knowledge_base/              # Layer 2 - Primary interface
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
├── evidence/                    # Layer 3 - Supporting material
│   ├── by_preference/
│   │   ├── python_pkg_mgr/
│   │   │   ├── correction_examples.md
│   │   │   └── approval_examples.md
│   │   └── no_emoji/
│   └── by_domain/
│
├── decision_index/              # Layer 4 - Fast lookup
│   ├── hard_constraints.yaml
│   ├── lookup_tables/
│   ├── decision_trees/
│   └── quick_reference.md
│
└── archive/                     # Layer 1 - Raw data
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

**Multi-File Context Retrieval**

LLM has access to a context retrieval tool that can be called multiple times:

```python
retrieve_context(
  query: str,
  max_files: int = 5,
  include_evidence: bool = False,
  domains: list[str] = None
) -> str  # Combined markdown context
```

**Retrieval Flow**

```
LLM Query: "What package manager for Python?"
  ↓
1. Quick lookup in decision_index/ (Layer 4)
   - If high confidence match: return + supporting files
  ↓
2. If no match or low confidence:
   - Semantic search in knowledge_base/ (Layer 2)
   - Return top 3-5 most relevant files
  ↓
3. If still uncertain:
   - Check evidence/ (Layer 3) for similar examples
   - Return analogous situations
```

**Multi-File Context Assembly**

The retrieval tool combines multiple files into a single markdown context:

```markdown
# Digital Twin Context

Retrieved for query: "Python web application setup"
Files: 4 | Confidence: High

---

## 1. workflows/project_initialization/python_web_app.md

[full content]

---

## 2. technical_preferences/python/package_management.md

[full content]

---

## 3. technical_preferences/python/testing.md

[full content]

---

## Related Files (not included, available on request)
- technical_preferences/python/virtual_environments.md
- workflows/development/iteration_patterns.md
```

**Context Budget Management**
- Full content for top 2-3 most relevant files
- Summaries for medium relevance files
- Progressive disclosure: start with index, let agent request more detail
- Deduplication: if multiple retrievals return same file, include only once

**Query Patterns**
- "What are preferences for Python package management?"
- "How should I handle user authentication in a React app?"
- "What's the testing approach for this type of project?"
- "Should I ask permission for this action?"

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
- What threshold for filtering exchanges into Layer 3 (evidence)?
- How to detect when preferences have evolved vs are context-dependent?

## Decisions Made

1. **Format**: Markdown + YAML instead of JSON for LLM-native consumption
2. **Layers**: 4-layer architecture (Archive, Knowledge Base, Evidence, Decision Index)
3. **Primary Interface**: Layer 2 (Knowledge Base) in markdown with prose and examples
4. **Retrieval**: Multi-file context assembly via retrieval tool
5. **Evidence Filtering**: Keep only 20-30% of exchanges (corrections, preferences, decisions)
6. **Priority System**: 0-1000 scale with hard constraints at 1000
7. **Cross-Dependencies**: Encoded in YAML with conditional logic

## Next Steps

To be determined based on discussion and implementation planning.

