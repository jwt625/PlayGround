# DevLog-001: Pattern Mining Strategy

**Date**: 2026-01-13 (Updated: 2026-01-17)
**Status**: Phase 3 In Progress - Type Consolidation Complete
**Goal**: Extract patterns from Augment conversations to build consolidated instruction documentation

## Project Objective

Mine LLM conversation histories to extract user preferences, practices, and workflow patterns. Consolidate findings into global instruction documentation to reduce repeated corrections and guidance, ultimately building a digital twin of development practices.

## Strategic Approaches

### 1. Pattern Mining Categories

**Preference Extraction**
- Explicit corrections: Direct statements of preference
- Repeated patterns: Consistent choices across conversations
- Negative feedback: Constraints and prohibitions
- Positive reinforcement: Approved patterns and approaches

**Workflow Pattern Recognition**
- Project initialization sequences
- Development phase transitions
- Decision-making patterns
- Quality gates and triggers

### 2. Categorization Dimensions

**By Domain**
- Language/Framework (Python, React, Docker, etc.)
- Activity Type (setup, debugging, feature development, refactoring)
- Tool Category (package managers, testing, deployment)

**By Preference Type**
- Hard constraints: "NEVER do X"
- Soft preferences: "Prefer X over Y"
- Contextual rules: "When doing X, use Y"
- Style guidelines: Aesthetic and formatting preferences

**By Interaction Pattern**
- Corrections: User fixing LLM mistakes
- Guidance: User steering direction
- Approvals: User confirming approaches
- Rejections: User vetoing suggestions

**By Project Phase**
- Initialization/Setup
- Active Development
- Testing/Validation
- Deployment/Production
- Maintenance

### 3. Processing Pipeline

```
Raw Conversations
    ↓
Preprocessing & Segmentation
    ↓
Exchange Classification
    ↓
Pattern Extraction
    ↓
Preference Consolidation
    ↓
Conflict Resolution
    ↓
Instruction Generation
```

**Pipeline Stages**

1. **Preprocessing**: Split into exchanges, extract metadata, identify code/commands
2. **Classification**: Apply multi-label tags (domain, intent, interaction type)
3. **Pattern Extraction**: Mine explicit preferences, implicit patterns, workflow sequences, error-correction pairs
4. **Consolidation**: Group similar preferences, calculate confidence scores, identify universal vs contextual rules
5. **Conflict Resolution**: Detect contradictions, prioritize recent over old, flag ambiguities
6. **Instruction Generation**: Convert to LLM instruction format, organize by priority, generate examples

### 4. Analysis Questions

**Preference Mining**
- Tool choices per language/framework
- Documentation standards
- Test triggers and requirements
- Error handling preferences
- LLM initiative vs permission-seeking tolerance

**Workflow Mining**
- Project setup sequences
- Debugging approaches
- Exploration vs direct implementation preferences
- Review and iteration patterns

**Communication Mining**
- Frustration indicators (LLM mistakes)
- Approval indicators (good patterns)
- Instruction explicitness
- Context provision vs inference expectations

### 5. Implementation Phases

**Phase 1: Quick Wins (Manual + Simple Scripts)**
- Keyword search for explicit preferences
- Frequency analysis of approved tools/commands
- Extract correction messages

**Phase 2: LLM-Assisted Analysis**
- Use LLM to analyze conversations for patterns
- Aggregate results across conversations
- Identify high-confidence patterns

**Phase 3: Structured Extraction**
- Build preference schema (domain, type, confidence, examples)
- Use LLM with structured output (JSON mode)
- Build pattern database

**Phase 4: Validation & Refinement**
- Generate draft instructions
- Test on new conversations
- Iterate based on remaining corrections

### 6. Output Formats

**Immediate Use**
- Enhanced system prompts
- Custom instructions
- Project-specific templates

**Long-term Use**
- Personal development playbook
- Automated project scaffolding
- Custom LLM fine-tuning data

### 7. Recommended Approach

**Hybrid Strategy**
1. Manual first pass for pattern recognition
2. LLM-assisted systematic analysis
3. Incremental instruction building

**Priority by ROI**
1. Tool preferences (concrete, easy to extract)
2. Style guidelines (reduces friction)
3. Workflow patterns (harder to codify, valuable long-term)

## Open Questions

- Schema design for extracted patterns
- Confidence scoring methodology
- Conflict resolution rules
- Validation metrics
- Instruction format and organization
- Integration with existing LLM systems

## Implementation Results

### Completed Work

**Data Mining Script** (`analysis/mine_patterns.py`)
- Consolidated pattern extraction across all categories
- Processes 52,391 exchanges from 22 workspaces
- Extracts: statistics, explicit preferences, tool mentions, corrections, commands, communication style
- Generates single comprehensive report

**Sentiment Analysis** (`analysis/sentiment_analysis.py`)
- Transformer-based analysis using DistilBERT
- Analyzed 9,271 user messages in 53 seconds (batched, MPS)
- Full metadata traceability: workspace, conversation ID, exchange index, source file
- Results saved to `sentiment_results.json` (7.9MB)

### Key Findings

**Data Quality**
- Total exchange records: 52,391 (includes tool calls, streaming chunks, and actual user-assistant exchanges)
- User messages: 9,271 (the actual data for preference mining)
- Note: Augment's LevelDB stores each tool call and streaming response chunk as a separate exchange record, inflating the total count. The relevant data for pattern mining is the 9,271 user messages, not the 52k total records.

**Tool Preferences** (High Confidence)
- Python package manager: uv (81.2%)
- Python testing: pytest (100%)
- Python web framework: FastAPI (73.9%)
- JavaScript package manager: npm (60.0%)

**Hard Constraints Extracted**
- NEVER rules: 5 unique (e.g., "NEVER CREATE NEW DOCS UNLESS EXPLICITLY ASKED")
- ALWAYS rules: 3 unique (e.g., "ALWAYS USE THE STREAMING VERSION")

**Anti-Patterns Identified**
- Unauthorized changes: 54 instances
- Prohibited actions: 55 instances
- Unsolicited creation: 5 instances
- Ignored instructions: 1 instance

**Explicit Preferences**
- ALWAYS statements: 141
- NEVER statements: 70
- MUST statements: 172
- DON'T statements: 428
- SHOULD statements: 1,549

**Sentiment Distribution**
- Positive: 1,198 (12.9%)
- Negative: 8,073 (87.1%)
- High-confidence negative (score > 0.95): 7,598 (82.0%)

Note: High negative rate includes neutral instructions/questions misclassified by sentiment model. True frustration rate requires additional filtering with frustration markers.

**Top Workspaces by Volume**
1. lambda-daily-slackbot: 14,351 exchanges
2. ClusterScope: 11,548 exchanges
3. lambda-chat-backend: 9,886 exchanges
4. lambda-chat-frontend-new: 6,342 exchanges

### Deliverables

**Analysis Folder** (`analysis/`)
- `mine_patterns.py`: Pattern mining script
- `pattern_analysis.md`: Consolidated pattern report (11KB)
- `sentiment_analysis.py`: Sentiment analysis script
- `sentiment_results.json`: Full sentiment results with metadata (7.9MB)
- `sentiment_results_summary.md`: Sentiment statistics summary

### Limitations

**Data Coverage**
- Single machine extraction only
- Requires consolidation from other machines for complete dataset

**Sentiment Analysis**
- Model trained on product reviews, not technical conversations
- Classifies many neutral instructions as negative
- Requires frustration marker filtering for accurate frustration detection

**Pattern Extraction**
- No distinction between universal and project-specific rules
- Confidence scoring based on frequency only
- No temporal analysis (preference evolution over time)

## Phase 2: LLM-Based Pattern Mining

### Design Decisions

**Message Type Classification**
- Multi-label classification (a message can be multiple types simultaneously)
- Types: bug_report, feature_request, preference_statement, decision, correction, clarification, approval, conversational
- Each type has optional confidence score
- Example: "this is broken, let's use X instead" -> bug_report + decision

Example types:
- bug_report - reporting specific broken behavior
- feature_request - asking for specific functionality
- preference_statement - expressing how things should be done
- decision - making a choice between alternatives
- correction - fixing LLM mistake
- clarification - providing more context
- approval - confirming LLM's approach
- conversational - filler/acknowledgment

**Generalizability Scoring**
- 0.0-0.3: Task-specific (applies only to current context)
- 0.4-0.6: Contextual (applies to similar projects/domains)
- 0.7-1.0: Universal (applies to all projects)

**Insight Extraction Schema**
```python
{
    "insight_type": "tool_preference|workflow|style|constraint|decision|value",
    "domain": "python|javascript|docker|testing|general|...",
    "statement": str,  # Concise free-form
    "applies_to": "all_projects|web_apps|this_project_only|...",
    "constraint_strength": "hard|soft"
}
```

**Context Strategy for Message Analysis**
- Project summary and metadata (extracted once per workspace)
- Previous user message (up to 500 chars)
- Previous final LLM response (up to 1000 chars, otherwise summarized) - the synthesized answer, not tool calls
- LLM response (up to 1000 chars, otherwise summarized)
- Optional: Next user message (up to 300 chars)

### Workspace-Level Context Extraction (WIP)

**Approach**: Extract project summaries from LLM responses (not user messages).

When the LLM is asked to "inspect", "analyze", or "research" a codebase, the final response often contains comprehensive project summaries with architecture, components, and design decisions.

**Extraction Script**: `analysis/extract_project_summaries.py`
- Criteria: response_text > 1500 chars, contains markdown headers
- Finds trigger request that prompted the summary
- Outputs per-workspace JSON files to `analysis/project_summaries/`

**Initial Extraction Results** (2026-01-13)
- 2,408 summaries extracted from 22 workspaces
- Top workspaces: lambda-daily-slackbot (632), lambda-chat-frontend-new (497), lambda-chat-backend (469)
- Longest summaries: up to 32KB per response

**Issues Identified**
- Many extracted "summaries" are not actual project-level context (e.g., bug fix explanations, implementation details)
- Over-extraction: some workspaces have 400+ summaries when only 3-5 high-quality ones are needed
- Duplicate project names from different workspace IDs require disambiguation
- Need stricter filtering criteria or LLM-based validation

### Planned Pipeline

**Step 1: Project Context Consolidation** (COMPLETE)
- Heuristic-based filtering reduced 2,408 candidates to 155 high-quality summaries (93.6% reduction)
- LLM-based validation using Llama-4-Maverick to classify each extracted summary as VALID/INVALID
- For workspaces with multiple valid summaries: consolidation via GLM-4.6-FP8
- For workspaces with zero valid summaries: generation from sampled responses, file types, and tech stack
- Final output: 21 unique project summaries covering all workspaces
- Results by source: 12 generated, 7 consolidated, 2 single-valid

**Implementation**: `analysis/generate_missing_summaries.py`
- Workflow: validate extracted summaries -> consolidate if multiple valid -> generate if none valid
- Models: GLM-4.6-FP8 for generation/consolidation, Llama-4-Maverick-17B for validation
- Output: `analysis/project_summaries/_consolidated_summaries.json`

**Step 2: Message Classification** (IN PROGRESS)

Two-stage hybrid pipeline implemented in `analysis/classify_messages.py`:

*Stage 1: Fast Filter (Llama-4-Maverick-17B)*
- Binary classification: HIGH_VALUE vs LOW_VALUE
- HIGH_VALUE: preferences, corrections, decisions, constraints, frustration, feature requests
- LOW_VALUE: acknowledgments, questions without opinions, conversational filler
- Batch size: 20 messages, incremental saves after each batch
- Purpose: reduce expensive GLM calls by filtering out ~50% of messages

*Stage 2: Deep Classification (GLM-4.6-FP8)*
- Multi-label classification: preference_statement, decision, correction, constraint, frustration, clarification, feature_request, approval, bug_report, conversational
- Generalizability score (0.0-1.0): task-specific to universal applicability
- Insight extraction with type, content, and confidence
- Batch size: 10 messages, incremental saves after each batch

*Context Assembly*
- Project summary from consolidated summaries (Step 1 output)
- Previous assistant response (extracted from response_nodes, type=0 text content)
- Next assistant response (same extraction method)
- Handles tool-only exchanges by searching backwards for actual user messages

*Data Processing*
- Raw data: 52,391 exchange records (includes tool calls and streaming chunks)
- Deduplicated user messages: 6,591 unique messages
- Deduplication via SHA-256 hash of message content

*Interrupt Safety*
- Both stages save results incrementally to JSON after each batch
- Resume capability: loads existing results, tracks processed message hashes, skips already-processed
- Separate execution: `--run-stage1` and `--run-stage2` flags allow independent runs
- Stage 2 reads Stage 1 output file once at startup

*Logging* (2026-01-14)
- Python logging module with dual output: console (INFO+) and file (DEBUG+)
- Timestamped format: `YYYY-MM-DD HH:MM:SS | LEVEL | message`
- Log files saved to `analysis/logs/` with unique timestamp per run

*Concurrency Optimization* (2026-01-14)
- Stage 2 updated to use async/concurrent LLM calls via `httpx.AsyncClient`
- 10 concurrent requests per batch (configurable)
- Performance: ~33 sec per 10 messages vs ~2.7 min sequential (4.9x speedup)
- Estimated total runtime: ~2.7 hours vs ~14 hours sequential

*Final Results* (2026-01-14)
- Two separate datasets from different machines (see Combined Dataset Summary below)
- Stage 1 and Stage 2 complete for both datasets
- Ready for pattern aggregation

**Step 3: Pattern Aggregation** (IN PROGRESS)

*Type Consolidation* (2026-01-17) - COMPLETE
- Reduced 352 insight type variants to 8 canonical types
- Consolidation script: `analysis/consolidate_insight_types.sh`
- Backup created: `backup_classification_and_leveldb_*.zip`
- Output files: `stage2_consolidated.json` (preserves original `type`, adds `canonical_type`)

Canonical type mapping:
| Canonical | Count | Maps From |
|-----------|-------|-----------|
| workflow | 4,175 | workflow_pattern, debugging_approach, testing_approach, planning_preference, etc. |
| constraint | 2,278 | constraint, requirement, functional_requirement, technical_requirement, etc. |
| quality | 2,126 | quality_standard, code_style, naming_convention, documentation_practice, etc. |
| communication | 1,200 | communication_preference, tone_preference, expectation, frustration, etc. |
| misc | 930 | preference, decision, feature_request, bug_report, single-count types |
| tool | 681 | tool_preference, framework, configuration, deployment_preference, etc. |
| ui_ux | 228 | ui_preference, design_pattern, visual_preference, layout_preference, etc. |
| architecture | 184 | architecture_preference, data_model_preference, state_management, etc. |

*Remaining Steps*
- Semantic deduplication within canonical types
- Domain extraction (Python/JS/Docker/etc.)
- Priority calculation from frequency + confidence + generalizability
- Conflict detection and resolution
- Generate Index tier (hard_constraints.yaml) and Knowledge Base docs

### Technical Notes

- Internal inference endpoint at `internal-inference.bugnest.net`
- GLM-4.6 includes thinking tokens; parsed via `</think>` delimiter
- Validation check must distinguish VALID from INVALID (substring match is insufficient)

## Combined Dataset Summary (2026-01-14)

### Dataset Sources

Two independent datasets extracted from separate machines with minimal overlap (2 shared messages, 1 shared workspace):

| Dataset | Path | Workspaces | Conversations | Exchanges | User Messages | High-Value | Stage 2 |
|---------|------|------------|---------------|-----------|---------------|------------|---------|
| Current (wentao) | `analysis/classification_results/` | 22 | 871 | 52,391 | 6,591 | 3,684 | 3,684 (100%) |
| Archive (wentaojiang) | `augment_export_archive/analysis/classification_results/` | 36 | 417 | 40,675 | 3,476 | 2,250 | 2,250 (100%) |
| **Combined** | - | 58 | 1,288 | 93,066 | 10,067 | 5,934 | 5,934 |

Note: Exchange count includes tool calls and streaming chunks stored by Augment's LevelDB. User Messages is the deduplicated count of actual user messages relevant for pattern mining.

### Combined Statistics

| Metric | Value |
|--------|-------|
| Total Workspaces | 58 |
| Total Conversations | 1,288 |
| Total Exchanges (raw) | 93,066 |
| Unique User Messages | 10,067 |
| Messages Classified (Stage 2) | 5,934 |
| Total Insights Extracted | 11,802 |
| Unique Insight Types | 352 raw -> 8 canonical |
| High-Value Insights (gen >= 0.8, conf >= 0.85) | 1,567 |

### Label Distribution

| Label | Count | % of messages |
|-------|-------|---------------|
| clarification | 3,797 | 64% |
| correction | 2,291 | 39% |
| preference_statement | 1,991 | 34% |
| constraint | 1,977 | 33% |
| decision | 1,022 | 17% |
| bug_report | 984 | 17% |
| frustration | 979 | 17% |
| feature_request | 970 | 16% |
| approval | 488 | 8% |
| conversational | 242 | 4% |

### Generalizability Distribution

| Range | Count | % |
|-------|-------|---|
| 0.0-0.2 (task-specific) | 1,481 | 25% |
| 0.3-0.5 (contextual) | 2,419 | 41% |
| 0.6-0.8 (broadly applicable) | 1,529 | 26% |
| 0.9-1.0 (universal) | 499 | 8% |

### Insight Types (after consolidation)

| Canonical Type | Count |
|----------------|-------|
| workflow | 4,175 |
| constraint | 2,278 |
| quality | 2,126 |
| communication | 1,200 |
| misc | 930 |
| tool | 681 |
| ui_ux | 228 |
| architecture | 184 |

### Sample High-Value Insights

**Tool Preferences** (gen >= 0.8, conf >= 0.85)
- Python: uv for packages, mypy for types, ruff for lint, pytest for testing
- JavaScript/TypeScript: pnpm for packages
- Database: psql over Python scripts for simple queries
- API testing: curl over complex scripts
- Media: ffmpeg over Python libraries

**Constraints**
- No file creation without explicit permission
- No emojis in responses or documentation
- No sycophantic phrases ("You're absolutely right")
- Always save to disk, never just display
- Zero tolerance for lint/type errors

**Communication**
- Concise, direct responses
- Use own judgment instead of validating user statements
- No hedging language when code is available to verify
- No premature apologies; acknowledge errors directly

**Quality Standards**
- No slop, no redundant code
- Minimal abstractions, prefer modifying existing code
- Code must be lint and type clean
- Avoid creating unnecessary files

## Next Steps

1. ~~Normalize insight types (352 variants to ~6-8 canonical types)~~ DONE
2. Deduplicate insights by semantic similarity within canonical types
3. Extract domain tags (Python/JS/Docker/etc.) from insight content
4. Calculate priority scores and detect conflicts
5. Generate Index tier (hard_constraints.yaml, quick_reference.md)
6. Generate Knowledge Base documents per canonical type

