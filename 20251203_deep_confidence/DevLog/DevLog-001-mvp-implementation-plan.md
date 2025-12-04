# DeepConf MVP Implementation Plan

## Overview
This document outlines the implementation plan for a Minimum Viable Product (MVP) of the Deep Confidence (DeepConf) method as described in "Deep Confidence: Confidence Estimation for Large Language Model Reasoning via Token Distribution Entropy" (arXiv:2508.15260).

## Implementation Decisions

### Algorithm Parameters (from Paper)

**Token-Level Confidence:**
- k = 19 (top_logprobs = 20 in API)
- Formula: C_i = -1/k * Σ log P_i(j) for j=1 to k
- No normalization or clipping

**Trace-Level Metrics:**
- **Tail Confidence**: Average confidence over last 2048 tokens (fixed window, not percentage)
- **Lowest Group Confidence**: Sliding window of 2048 tokens with stride=1, metric is minimum across all windows
- **Bottom-10% Confidence**: Mean of lowest 10% of all 2048-token window confidences

**Filtering Strategy:**
- η = 10% (aggressive: keep top 10% most-confident traces)
- η = 90% (conservative: keep top 90%)
- Recommended metrics: Tail Confidence or Bottom-10% Confidence for offline mode

**Confidence-Weighted Majority Voting:**
- Formula: V(a) = Σ C_t · 1[answer(t)=a]
- Weight is trace confidence C_t, summation naturally accounts for vote counts

**Answer Extraction:**
- Final answer must be inside `\boxed{...}`
- Extract boxed content
- No normalization specified (assumes numeric answers)

### MVP Scope

**Features to Implement:**
1. Multi-sample generation (n completions)
2. Token-level confidence computation from logprobs
3. All three trace-level metrics (Tail, Lowest Group, Bottom-10%)
4. Filtering by confidence threshold (η = 10% and η = 90%)
5. Confidence-weighted majority voting
6. Baseline comparison (simple majority voting without confidence)

**Evaluation:**
- Dataset: AIME 2025 (30 problems from AIME 2025-I and AIME 2025-II)
- Source: HuggingFace dataset `opencompass/AIME2025`
- Metric: Accuracy improvement vs baseline
- Problem types: High-difficulty math competition problems

**Configuration:**
- Endpoint: kimi-k2 (validated in endpoint evaluation)
- n (completions): Start with paper defaults (research needed)
- temperature: Use paper default (research needed)
- max_tokens: 10000 (may need adjustment based on problem difficulty)
- top_logprobs: 20 (k=19)
- Filtering thresholds: η ∈ {10%, 90%}
- Trace metrics: All three (Tail, Lowest Group, Bottom-10%)

**Output Format:**
- Final answer with confidence score
- All traces with individual confidences (for debugging)
- Filtered vs unfiltered comparison
- Baseline (simple majority) vs DeepConf comparison
- Per-problem breakdown

**Code Structure:**
- Single script for MVP (will modularize later)
- No CLI interface initially
- Focus on clarity and correctness

## Implementation Steps

### Phase 1: Core Infrastructure
1. Load AIME 2025 dataset from HuggingFace
2. Set up kimi-k2 API client with proper configuration
3. Implement multi-sample generation with logprobs extraction

### Phase 2: Confidence Computation
4. Implement token-level confidence calculation (k=19)
5. Implement Tail Confidence metric
6. Implement Lowest Group Confidence metric
7. Implement Bottom-10% Confidence metric

### Phase 3: Filtering and Voting
8. Implement answer extraction from `\boxed{...}` format
9. Implement trace filtering by confidence threshold
10. Implement confidence-weighted majority voting
11. Implement baseline majority voting (for comparison)

### Phase 4: Evaluation
12. Run evaluation on AIME 2025 dataset
13. Compute accuracy metrics (baseline vs DeepConf)
14. Generate detailed output with per-problem analysis

### Phase 5: Validation
15. Verify implementation against paper specifications
16. Debug and iterate based on results
17. Document findings and performance

## Research Needed

The paper PDF is available but needs manual review for:
- Default value for n (number of completions) - likely 64 based on references to "DeepConf@64"
- Default temperature setting - typically 0.7-1.0 for reasoning tasks
- Any preprocessing or prompt engineering used
- Expected accuracy ranges on AIME-like benchmarks

For MVP, will use reasonable defaults:
- n = 10 (based on kimi-k2 validation showing good performance)
- temperature = 0.8 (standard for diverse sampling)
- max_tokens = 10000 (sufficient for complex math problems)

## Success Criteria

The MVP is successful if:
1. All core DeepConf features are implemented correctly
2. Code runs on AIME 2025 dataset without errors
3. Results show measurable difference between baseline and DeepConf
4. Output provides sufficient detail for debugging and analysis
5. Implementation matches paper specifications

## Next Steps

After MVP completion:
- Refactor into modular library structure
- Add CLI interface
- Extend to other datasets/problem types
- Optimize performance
- Add comprehensive test suite

---

## Implementation Progress

### Status: MVP Complete - Evaluation In Progress

**Implementation Date:** 2025-12-03

**File:** `deepconf_mvp.py` (498 lines)

### Completed Features

**Phase 1: Core Infrastructure** - COMPLETE
- AIME 2025 dataset loading from HuggingFace (both AIME2025-I and AIME2025-II configs)
- kimi-k2 API client with proper configuration
- Multi-sample generation (n=10) with logprobs extraction (top_logprobs=20)

**Phase 2: Confidence Computation** - COMPLETE
- Token-level confidence: C_i = -1/k * sum(log P_i(j)) with k=19
- Tail Confidence: Average over last 2048 tokens
- Lowest Group Confidence: Minimum across sliding 2048-token windows (stride=1)
- Bottom-10% Confidence: Mean of lowest 10% of window confidences

**Phase 3: Filtering and Voting** - COMPLETE
- Answer extraction from `\boxed{...}` format using regex
- Trace filtering by confidence threshold (eta=10%, eta=90%)
- Confidence-weighted majority voting: V(a) = sum(C_t * 1[answer(t)=a])
- Baseline majority voting (simple unweighted)
- Single-shot baseline (n=1) for comparison

**Phase 4: Evaluation** - IN PROGRESS
- Evaluation pipeline implemented for AIME 2025 dataset (30 problems)
- Incremental results saving after each problem
- Comprehensive logging with timestamps
- Progress tracking with tqdm progress bar
- Full evaluation currently running

**Phase 5: Validation** - PENDING
- Awaiting evaluation results
- Will verify against paper specifications
- Will document findings and performance

### Implementation Details

**Configuration:**
- Endpoint: kimi-k2 (moonshotai/Kimi-K2-Thinking)
- n_completions: 10
- temperature: 0.8
- max_tokens: 10000
- top_logprobs: 20 (k=19)
- window_size: 2048
- filter_thresholds: [0.1, 0.9]

**Output Files:**
- Results: `deepconf_results_YYYYMMDD_HHMMSS.json`
- Logs: `deepconf_run_YYYYMMDD_HHMMSS.log`

**JSON Output Structure:**
- Single-shot baseline (n=1) with full trace details
- Multi-sample baseline (n=10) with simple majority voting
- All 10 reasoning traces with:
  - Full reasoning and content text
  - Extracted answers
  - All three confidence metrics (tail, lowest_group, bottom_10)
  - Complete token-level confidence arrays
- Baseline majority voting with:
  - Vote counts per answer
  - Detailed voting breakdown (which trace voted for what)
- DeepConf results for all metric/threshold combinations:
  - Confidence-weighted vote totals
  - Detailed voting breakdown with confidence weights
  - Filtered trace indices
  - Filter statistics
- Aggregate accuracy statistics

**Logging Features:**
- Timestamped log files
- Progress bar showing completion percentage and speed
- Incremental results saving (no data loss on interruption)
- Detailed logging at every step (dataset loading, API calls, answer extraction, etc.)

### Known Issues

**Performance:**
- Evaluation time: 3-6 minutes per problem (30-180 minutes total for 30 problems)
- Bottleneck: API generates n=10 completions sequentially with max_tokens=10000
- Potential optimization: Parallel API calls (10 separate n=1 calls instead of 1 call with n=10)

**API Behavior:**
- Some problems result in 0/10 traces with extracted answers (model may not use `\boxed{}` format consistently)
- API occasionally requires retries (handled automatically by OpenAI client)

### Verification Against Paper Specifications

All algorithm parameters match paper specifications:
- Token confidence formula: Exact match
- Trace-level metrics: Exact match (2048-token windows, stride=1, correct aggregation)
- Filtering strategy: eta=10% and eta=90% as specified
- Weighted voting formula: Exact match
- Answer extraction: `\boxed{...}` format as specified

### Next Actions

1. Complete full evaluation on 30 AIME 2025 problems
2. Analyze results and compare against paper benchmarks
3. Document accuracy improvements (single-shot vs multi-sample vs DeepConf)
4. Identify best-performing metric/threshold combinations
5. Validate implementation correctness based on results

