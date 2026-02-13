# DevLog-000-04: Reference Extraction and Citation Graph

## Objective

Extract all references from 260 converted markdown files, build a citation graph, compute statistics, and identify the top 100 most-cited papers for continued download.

---

## Current State

| Metric | Count |
|--------|-------|
| PDF documents collected | 263 |
| Markdown files converted (Marker) | 260 |
| Theses | 112 |
| Papers | 151 |
| Total size | 3.6 GB |

**Markdown location:** `semiconductor_processing_dataset/processed_documents/text_extracted/marker/`

---

## Reference Format Variations Observed

### Format 1: Numbered List (Nature/Science style)
```markdown
## References and Notes
- 1. A. Montanaro, *npj Quantum Information* 2, 15023 (2016).
- 2. A. Kandala, *et al.*, *Nature* 549, 242 (2017).
```

### Format 2: Bracketed (Thesis style)
```markdown
# Bibliography
- [1] D. Jaksch, J. I. Cirac, P. Zoller. Fast quantum gates. Phys. Rev. Lett., 85:2208–2211, 2000.
- [2] M. Saffman, T. G. Walker. Quantum information with Rydberg atoms. Rev. Mod. Phys., 82:2313–2363, 2010.
```

### Format 3: Superscript Inline (Review paper style)
```markdown
<sup>1</sup>R. Feynman, "Simulating physics with computers," Int. J Theor. Phys **21**, 467–488 (1982).
<sup>2</sup>S. Lloyd, "Universal quantum simulators," Science **273**, 1073–1078 (1996).
```

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  260 MD Files   │────▶│  Reference       │────▶│  Citation Graph │
│                 │     │  Extractor       │     │  + Stats        │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                        ┌──────┴──────┐
                        │  Inference  │
                        │  Service    │
                        │ GLM-4.7-FP8 │
                        └─────────────┘
```

---

## Implementation Plan

### Phase 1: Reference Section Extraction (Regex)

**Goal:** Fast pre-processing to extract reference sections from all markdown files.

**Script:** `tests/inference_test/extract_reference_sections.py`

**Output:** `tests/inference_test/output/reference_sections.jsonl`
```json
{"document_id": "place_2021_ncomms_tantalum_qubits", "ref_section_start": 151, "ref_count_estimate": 52, "references_text": "..."}
```

### Phase 2: LLM-Based Structured Extraction

**Goal:** Parse each reference into structured data using function calling.

**Script:** `tests/inference_test/extract_references_llm.py`

**Model:** `zai-org/GLM-4.7-FP8` (proven for extraction tasks)

**Output Schema:**
```json
{
  "document_id": "place_2021_ncomms_tantalum_qubits",
  "references": [
    {
      "ref_num": 1,
      "title": "npj Quantum Information article",
      "authors": ["A. Montanaro"],
      "year": 2016,
      "journal": "npj Quantum Information",
      "volume": "2",
      "pages": "15023",
      "doi": null,
      "arxiv_id": null
    }
  ]
}
```

### Phase 3: Deduplication and Graph Building

**Goal:** Match references across documents, build citation adjacency list.

**Script:** `tests/inference_test/build_citation_graph.py`

**Matching strategy:**
1. Exact match by DOI or arXiv ID
2. Fuzzy match by normalized title + year (Levenshtein distance < 0.1)
3. Author overlap as tiebreaker

**Output:** `tests/inference_test/output/citation_graph.json`
```json
{
  "nodes": {
    "place_2021_ncomms_tantalum_qubits": {"title": "...", "year": 2021, "in_dataset": true},
    "koch_2007_transmon": {"title": "...", "year": 2007, "in_dataset": true},
    "external_ref_001": {"title": "...", "year": 2016, "in_dataset": false}
  },
  "edges": [
    {"from": "place_2021_ncomms_tantalum_qubits", "to": "koch_2007_transmon"}
  ]
}
```

### Phase 4: Statistics and Top 100

**Goal:** Rank papers by citation count, identify download candidates.

**Output:** `tests/inference_test/output/citation_stats.json`
```json
{
  "total_unique_references": 5000,
  "references_in_dataset": 150,
  "references_external": 4850,
  "top_100_external": [
    {"title": "...", "authors": [...], "year": 2007, "citation_count": 45, "doi": "...", "arxiv_id": "..."}
  ]
}
```

---

## Progress Log

### 2026-02-11: Planning

- Created this DevLog
- Analyzed reference format variations in markdown files
- Designed 4-phase extraction pipeline

### 2026-02-11: Phase 1 Complete (Final)

**Script:** `tests/inference_test/extract_reference_sections.py`

**Results:**
| Metric | Count |
|--------|-------|
| Success (references found) | 260 (100%) |
| No references detected | 0 |
| Errors | 0 |
| Total references estimated | 13,063 |

**Detection Methods Breakdown:**
| Method | Files Detected |
|--------|----------------|
| Sequential number detection | 131 |
| Header-based (regex) | 125 |
| Manual | 4 |

**Detection Strategies:**
1. **Header-based**: Look for `# References`, `## Bibliography`, etc. including span-embedded headers
2. **Sequential number detection**: Find 4+ consecutive lines with incrementing reference numbers (`[1]`, `[2]`, `1.`, `2.`, `<sup>1</sup>`, etc.). Allows gaps of up to 5 lines between references.
3. **Manual**: For 4 files with non-standard formats, reference sections were manually identified and added

**Manually Added Files:**
| File | Lines | Ref Count | Format |
|------|-------|-----------|--------|
| paper_ma_2019_nature_mott_insulator | 133-187 | 53 | Numbered list with span IDs |
| paper_saxberg_2022_nature_correlated_fluids | 133-184 | 50 | Numbered list with span IDs |
| paper_schreier_2008_prb_transmon | 78-126 | 25 | Superscript format |
| paper_schuster_2007_nature_photon_number | 67-96 | 30 | Mixed bullet/numbered |

### 2026-02-11: Phase 1 Quality Validation

**Methodology:** Manual verification of 24 documents (9.2% of 260) by directly viewing original markdown files and comparing with extracted `references_text` content.

**Validation Results:**
| Metric | Result |
|--------|--------|
| Documents verified | 24 |
| Correct extractions | 24 (100%) |
| Missing references | 0 |
| Included non-reference content | 0 |

**Documents Verified:**

| Document | Ref Lines | Total Lines | Boundary Handling |
|----------|-----------|-------------|-------------------|
| `altoe_2022_localization_reduction_circuit_losses` | 88-125 | ~125 | Stops before Acknowledgements |
| `chen_2014_aluminum_airbridges` | 70-97 | 190 | Stops before Supplementary Material |
| `reagor_matthew_yale_2015` | 3173-3370 | 3814 | 176 refs, stops before APPENDIX A |
| `chow_jerry_yale_2010` | 3649-3837 | 5476 | 172 refs, stops before APPENDIX A |
| `bosonic_cqed_review_2023` | 113-272 | 272 | 142 refs, to EOF |
| `barends_2013_prl_xmon` | 94-265 | 265 | Main + Supplementary refs, to EOF |
| `gold_2021_npjqi_entanglement_separate_dies` | 262-292 | 292 | 47 refs, to EOF |
| `paper_earnest_2018_prl_fluxonium` | 91-131 | 131 | 39 refs, to EOF |
| `groh_2025_tes_multiplexer` | 136-228 | 228 | 48 refs, to EOF |
| `paper_gyenis_2021_prxq_0pi_qubit` | 60-112 | 452 | 48 refs, stops before Acknowledgments |
| `niedzielski_2022_qubit_compatible_tsv_substrates` | 239-294 | 294 | 48 refs, to EOF |
| `rosenberg_2019_arxiv_3d_integration_packaging` | 143-208 | 208 | 55 refs, to EOF |
| `rigetti_chad_yale_2009` | 3334-3496 | 3496 | 146 refs, to EOF |
| `thesis_owens_clai_uchicago_2019` | 1219-1299 | 1299 | 70 refs, to EOF |
| `paper_zhou_2024_natphys_electron_charge` | 165-196 | 196 | 48 refs, to EOF |
| `paper_mckay_2015_prl_multimode_cqed` | 81-119 | 119 | 37 refs, to EOF |
| `paper_chakram_2022_natphys_photon_blockade` | 88-132 | 374 | 39 refs, stops before Supplementary Info |
| `mohseni_2024_quantum_supercomputer_scaling` | 1227-1554 | 1931 | 304 refs, stops before Appendices |
| `bal_2024_npjq_nb_surface_encapsulation` | - | - | to EOF |
| `dhundhwal_2025_tantalum_resonators` | - | - | to EOF |
| `mcdermott_2014_sfq_qubit_control` | 114-130 | 130 | to EOF |
| `chen_zijun_ucsb_2018` | - | - | thesis, to EOF |
| `bialczak_radoslaw_ucsb_2011` | - | - | thesis, to EOF |
| `petrenko_andrei_yale_2016` | - | - | thesis, to EOF |

**Key Findings:**
1. **Complete reference capture**: All verified documents have complete reference sections (first to last reference)
2. **Proper section boundaries**: Extractions correctly stop before appendices, supplementary materials, and acknowledgments
3. **Various document types handled well**: Short papers (100-300 lines), long theses (3000-5000+ lines), papers with supplementary material, papers with appendices

**Note on `ref_count_estimate` field:** This field uses a simple regex heuristic and significantly underestimates actual reference counts in many documents. This is acceptable because Phase 2 LLM will parse and count actual references. The field is only used for progress tracking, not for extraction quality.

**Conclusion:** Phase 1 extraction quality is **excellent**. Ready for Phase 2.

### 2026-02-11: Phase 2 Implementation and Batch Run

**Scripts:**
- `tests/inference_test/test_extract_references_llm.py` - Single document test script
- `tests/inference_test/batch_extract_refs_phase2.py` - Batch processing script

**Configuration:**
| Parameter | Value |
|-----------|-------|
| Model | `zai-org/GLM-4.7-FP8` |
| Max concurrent API calls | 5 |
| References per chunk | 10 (line-based splitting) |
| Max tokens | 8192 |
| Timeout | 600s |
| Max retries on JSON error | 3 |

**Output Files:**
- `output/phase2_extracted_refs.jsonl` - Extracted references (one JSON per document)
- `output/phase2_progress.json` - Resume checkpoint
- `output/phase2_run_YYYYMMDD_HHMMSS.log` - Timestamped run log

**Key Implementation Notes:**
1. **Chunking strategy:** Simple line-based splitting (each line = one reference). Avoids relying on unreliable `ref_count_estimate` from Phase 1.
2. **Tool calling enforced:** Uses `tool_choice: {"type": "function", "function": {"name": "extract_references"}}` for structured output.
3. **Resume-safe:** Progress saved after each document. Script skips already-completed documents on restart.
4. **Retry logic:** JSON parsing errors trigger up to 3 retries with 1s delay.

### 2026-02-11: Phase 2 Complete

**Final Results:**
| Metric | Count |
|--------|-------|
| Documents processed | 260/260 (100%) |
| Total references extracted | 25,075 |
| Documents with 0 refs | 2 |
| Chunk warnings (retried) | 46 |
| Chunk errors (failed after 3 retries) | 3 |

**Documents with 0 references extracted:**
- `bu_2024_npjqi_tantalum_airbridges` - Needs investigation
- `smith_clarke_yale_2019` - Needs investigation

**Error Analysis:**
The retry logic significantly reduced permanent failures. Out of 49 total JSON parsing issues:
- 46 were warnings (recovered on retry)
- 3 failed after 3 retries (lost ~30 refs total, negligible impact)

**Root cause of JSON errors:** Large author lists in review papers and collaboration papers (e.g., Google Quantum AI with 50+ authors) cause the LLM to generate extremely large JSON responses (250KB+), which occasionally get truncated or malformed. Future improvement: instruct LLM to truncate author lists to first 3 + last 2 for papers with >5 authors.

**Runtime:** Approximately 12 hours for 260 documents (started 02:03, completed 20:55).

**Average processing time:** ~2.8 minutes per document.

### 2026-02-13: Phase 3 Complete - Deduplication and Citation Graph

**Script:** `tests/inference_test/build_citation_graph.py`

**Deduplication Strategy:**

Multi-pass approach to handle various identifier formats:

1. **Pass 1: Primary ID grouping** - Group references by DOI > arXiv > composite key
2. **Pass 2: Union-find merge** - Merge groups that share DOI or arXiv across different primary IDs
3. **Pass 3: Fuzzy title matching** - Merge by normalized title + year + author (first 50 chars)
4. **Pass 4: Source paper integration** - Merge source papers into the pool, matching references to source papers

**Composite Key Format:** `{first_author_last_name}_{year}_{first_significant_title_word}`

**Source Paper Handling:**
- Loaded metadata from `semiconductor_processing_dataset/processed_documents/metadata/` (194 files)
- Generated canonical IDs for source papers using same logic as references
- Source papers without metadata (66 of 260) added as fallback nodes using document_id

**Final Results:**

| Metric | Count |
|--------|-------|
| Total raw references | 25,075 |
| Unique references | 12,479 |
| Dedup ratio | 2.01x |
| Documents with metadata | 194 |
| Total in-dataset nodes | 264 |
| Total graph nodes | 12,550 |
| Total graph edges | 23,367 |
| Inter-dataset citations | 746 |

**Match Method Breakdown:**

| Method | Count |
|--------|-------|
| DOI | 4,435 |
| Composite key | 6,738 |
| arXiv | 885 |
| Hash (fallback) | 421 |

**Top 10 Most-Cited In-Dataset Papers:**

| Rank | Citations | Paper |
|------|-----------|-------|
| 1 | 132 | Koch 2007 - Transmon design (PRA) |
| 2 | 63 | Paik 2011 - High coherence JJ qubits (PRL) |
| 3 | 50 | Manucharyan 2009 - Fluxonium (Science) |
| 4 | 47 | Devoret & Schoelkopf 2013 - Science review |
| 5 | 40 | Barends 2013 - Xmon (PRL) |
| 6 | 35 | Houck 2008 - Controlling spontaneous emission (PRL) |
| 7 | 27 | Kjaergaard 2020 - State of Play review |
| 8 | 20 | Krinner 2019 - Engineering cryogenic setups |
| 9 | 16 | Chow 2010 - PhD thesis |
| 10 | 14 | Foxen 2018 - Qubit interconnects |

**Output Files:**
- `output/unique_references.jsonl` - 12,479 deduplicated references with citation counts
- `output/citation_graph.json` - Graph with 12,550 nodes and 23,367 edges
- `output/dedup_stats.json` - Summary statistics

### 2026-02-13: Visualization Complete

**File:** `tests/inference_test/output/citation_graph_viewer.html`

Interactive D3.js force-directed graph visualization with:
- Node size proportional to log(citation_count) or log(reference_count)
- Color coding: teal for in-dataset papers, red for external references
- Hover tooltip showing title, authors, year, journal, DOI, arXiv, citation count, reference count
- "Size by" dropdown: Auto / Citations received / References made
- Search functionality
- Min citations filter slider
- Zoom and pan controls

### Next Steps

- [x] Implement Phase 1: `extract_reference_sections.py`
- [x] Run Phase 1 on all 260 markdown files - **100% success rate**
- [x] Validate Phase 1 quality - **100% correct on 24-document sample**
- [x] Implement Phase 2: `test_extract_references_llm.py` and `batch_extract_refs_phase2.py`
- [x] Run Phase 2 batch extraction - **260/260 complete, 25,075 refs extracted**
- [x] Implement Phase 3: `build_citation_graph.py` - **2.01x dedup, 746 inter-dataset citations**
- [x] Implement visualization: `citation_graph_viewer.html`
- [ ] Investigate 2 documents with 0 refs
- [ ] Generate download list for top 100 cited external papers

---

## File Locations

```
tests/inference_test/
├── extract_reference_sections.py      # Phase 1: regex extraction
├── test_extract_references_llm.py     # Phase 2: single doc test script
├── batch_extract_refs_phase2.py       # Phase 2: batch processing
├── build_citation_graph.py            # Phase 3: dedup + graph construction
├── output/
│   ├── reference_sections.jsonl       # Phase 1 output
│   ├── phase2_extracted_refs.jsonl    # Phase 2 output (25,075 refs)
│   ├── phase2_progress.json           # Phase 2 resume checkpoint
│   ├── phase2_run_*.log               # Phase 2 timestamped logs
│   ├── unique_references.jsonl        # Phase 3 output (12,479 unique refs)
│   ├── citation_graph.json            # Phase 3 output (12,550 nodes, 23,367 edges)
│   ├── dedup_stats.json               # Phase 3 statistics
│   └── citation_graph_viewer.html     # Interactive visualization
```

