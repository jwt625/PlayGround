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

Before implementation, research the following from the paper:
- Default value for n (number of completions)
- Default temperature setting
- Any preprocessing or prompt engineering used
- Expected accuracy ranges on AIME-like benchmarks

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

