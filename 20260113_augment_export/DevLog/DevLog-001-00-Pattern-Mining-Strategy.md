# DevLog-001: Pattern Mining Strategy

**Date**: 2026-01-13 (Updated: 2026-01-14)
**Status**: Phase 2 In Progress - Message Classification Running
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

*Initial Results* (partial run, 2026-01-14)
- Stage 1: 497 high-value out of 940 processed (52.9% acceptance rate)
- Stage 2: running in parallel on available high-value messages

**Step 3: Pattern Aggregation** (TODO)
- Group insights by type and domain
- Calculate confidence based on frequency and generalizability
- Resolve conflicts (recent > old, explicit > implicit)
- Generate consolidated instruction documents

### Technical Notes

- Internal inference endpoint at `internal-inference.bugnest.net`
- GLM-4.6 includes thinking tokens; parsed via `</think>` delimiter
- Validation check must distinguish VALID from INVALID (substring match is insufficient)

## Next Steps

1. Complete message classification (Stage 1 + Stage 2) on all 6,591 messages
2. Analyze classification results: distribution of labels, generalizability scores, insight types
3. Build pattern aggregation and conflict resolution pipeline
4. Generate actionable instruction documents for LLM integration

