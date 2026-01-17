# DevLog-001-02: Insight Deduplication Approaches

**Date**: 2026-01-17  
**Status**: In Progress  
**Parent**: DevLog-001-01-Digital-Twin-Components.md  
**Goal**: Consolidate 6,935 raw insights into a deduplicated, organized knowledge base

## Problem Statement

The LLM-based classification pipeline extracted 6,935 insights from 5,934 user messages across 20 projects. These insights are highly redundant:
- Only 167 exact string duplicates (6,768 unique strings)
- Massive semantic redundancy: 100+ variants of "NEVER create files", 558 variants of "concise", 72 variants of "prefers minimal"
- Target: Reduce to ~64 canonical statements as the foundation for the Digital Twin index tier

### Data Distribution by Canonical Type

| Type | Count | Percentage |
|------|-------|------------|
| workflow | 2,545 | 36.7% |
| constraint | 1,324 | 19.1% |
| quality | 1,215 | 17.5% |
| communication | 645 | 9.3% |
| misc | 552 | 8.0% |
| tool | 449 | 6.5% |
| architecture | 126 | 1.8% |
| ui_ux | 79 | 1.1% |

## Approach 1: Embedding-Based Clustering

### Method
Used sentence-transformers (all-MiniLM-L6-v2, 384-dim embeddings) with Agglomerative Clustering (n_clusters=64, cosine metric, average linkage).

### Results
Highly imbalanced clusters:
- Cluster 0: 2,287 insights (33% of all data)
- Cluster 1: 677 insights
- Top 10 clusters: ~5,000 insights (72%)
- Bottom 20 clusters: <20 insights each

### Analysis
Embeddings group by grammatical structure rather than semantic meaning. Insights starting with "prefers", "expects", "wants" clustered together regardless of actual content.

## Approach 2: Prefix-Stripped HDBSCAN

### Method
Strip common prefixes ("prefers", "expects", "wants", etc.) before embedding to focus on semantic content. Used HDBSCAN with varying min_cluster_size values.

### Results

| min_cluster_size | Clusters | In Clusters | % of Total | Max Size |
|------------------|----------|-------------|------------|----------|
| 5 | 124 | 1,596 | 23.0% | 100 |
| 8 | 67 | 1,408 | 20.3% | 100 |
| 10 | 52 | 1,337 | 19.3% | 100 |
| 15 | 36 | 1,221 | 17.6% | 100 |

### Key Finding
Only 23% of insights (1,596) form tight semantic clusters. The remaining 77% (~5,300) are unique or near-unique. The insight space is fundamentally sparse.

## Approach 3: LLM-Based Incremental Clustering (Current)

### Method
Use Llama-4-Maverick-17B to incrementally classify insights into subclusters with memory:
1. Start with parent clusters from Approach 1 (64 embedding-based clusters)
2. Process insights in batches of 10
3. For each batch, LLM decides: assign to existing subcluster OR create new one
4. Each subcluster maintains: summary, representative samples (3-5), all member contents, count
5. LLM is instructed to be conservative about creating new subclusters

### Test Run Results (3 Clusters)

| Parent Cluster | Insights | Subclusters | Compression |
|----------------|----------|-------------|-------------|
| 0 (mega-cluster) | 2,287 | 36 | 63.5:1 |
| 1 | 677 | 101 | 6.7:1 |
| 3 | 433 | 40 | 10.8:1 |
| **Total** | **3,397** | **177** | **19.2:1** |

### Observations

1. **Cluster 0 was genuinely cohesive**: 2,287 insights compressed to 36 subclusters validates that these insights share common themes despite embedding limitations.

2. **Cluster 1 was heterogeneous**: 677 insights became 101 subclusters. The embedding model incorrectly grouped disparate topics (Kubernetes config, CI/CD, visualization, IAM). LLM correctly identified the semantic differences.

3. **Cluster 3 moderate**: 433 to 40 subclusters (10.8:1), mostly visualization and UI preferences.

### Example Subclusters from Cluster 0

**Subcluster 0** (364 members): "Specific rules or restrictions on coding practices"
- "NEVER create files unless explicitly instructed to do so"
- "maintains a DevLog to track project progress"
- "strictly avoid type ignore comments; views them as shortcuts"

**Subcluster 1** (74 members): "Specific functionality requirements"
- "expects database to mirror source file data exactly"
- "prioritizes data validation logic during refactoring"
- "validation errors (HTTP 422) must return array format"

**Subcluster 2**: "Approach to problem-solving and task handling"
- "demands concrete code examples rather than theoretical explanations"
- "expects deep root cause analysis rather than superficial fixes"
- "prefers understanding existing codebase before proposing changes"

## Current Status

- Test run completed successfully on 3 parent clusters (3,397 insights -> 177 subclusters)
- Results saved to `analysis/classification_results/llm_subclusters.json`
- Remaining work: run on all 64 parent clusters, then potentially merge similar subclusters across parents

## Open Questions

1. Should we run LLM clustering on all 64 embedding clusters, or skip embedding pre-clustering entirely?
2. How to handle the 77% of insights that don't naturally cluster?
3. Need a second LLM pass to improve summary quality (some are too generic)
4. Should we add multi-dimensional labels (behavior, strength, scope) during or after consolidation?

## Files

| File | Description |
|------|-------------|
| `analysis/classification_results/stage2_consolidated.json` | Source: 6,935 insights |
| `analysis/classification_results/insight_clusters.json` | Approach 1: 64 embedding clusters |
| `analysis/classification_results/insight_clusters_v2.json` | Approach 2: HDBSCAN results |
| `analysis/classification_results/llm_subclusters.json` | Approach 3: LLM subcluster results (3 clusters) |
| `analysis/llm_cluster_insights.py` | LLM incremental clustering script |

