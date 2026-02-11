# DevLog-000-03: PDF to Markdown Conversion Tool Evaluation

## Objective

Evaluate PDF to Markdown conversion tools for processing the SC qubit dataset (263 documents, 3.6 GB). The goal is to select the best tool for:
1. Converting PDFs to clean Markdown for downstream LLM processing
2. Preserving document structure (sections, tables, figures, equations)
3. Extracting references for consolidation and deduplication

---

## Tool Candidates

### Top 3 Tools for Evaluation

| Tool | Description | License | GPU Support |
|------|-------------|---------|-------------|
| **Marker** | Fast PDF to Markdown, good table/figure handling | MIT | Yes (optional) |
| **Docling** (IBM) | Scientific document understanding, structure preservation | MIT | Yes |
| **GROBID** | Gold standard for scientific PDF parsing, reference extraction | Apache 2.0 | No |

### Why These Three

1. **Marker** - Speed-focused, community-proven, good balance of quality and performance
2. **Docling** - IBM's scientific document understanding, explicit structure modeling
3. **GROBID** - 15+ years of scientific PDF parsing, excellent reference extraction

---

## Test Documents Selection

### Criteria

- Range of document lengths (short papers to long theses)
- Range of file sizes (1 MB to 300+ MB)
- Different layouts (arXiv preprints, journal papers, theses)
- Different time periods (1999 to 2025)
- Mix of content types (experimental, theoretical, review)

### Selected Documents (7 total)

| # | Document | Type | Size | Year | Layout Notes |
|---|----------|------|------|------|--------------|
| 1 | `nakamura_1999_charge_qubit.pdf` | Paper | 113 KB | 1999 | Nature letter, historical, scanned |
| 2 | `krantz_2019_apr_quantum_engineers_guide.pdf` | Review | 5.0 MB | 2019 | Long review, many equations |
| 3 | `muschinske_2023_dolan_manhattan_jj_uniformity.pdf` | Paper | 6.6 MB | 2023 | Fabrication, tables, figures |
| 4 | `place_2021_ncomms_tantalum_qubits.pdf` | Paper | 9.4 MB | 2021 | Nature Comms, supplementary |
| 5 | `putterman_2025_bosonic_qec.pdf` | Paper | 39 MB | 2025 | Recent, complex figures |
| 6 | `spietz_lafe_yale_2006.pdf` | Thesis | 76 MB | 2006 | Yale thesis, older format |
| 7 | `eichinger_michaela_copenhagen_2023.pdf` | Thesis | 328 MB | 2023 | Large thesis, many chapters |

---

## Evaluation Metrics

### Quality Metrics

| Metric | Description | Weight |
|--------|-------------|--------|
| **Text Accuracy** | Correct text extraction, no garbled characters | High |
| **Structure Preservation** | Headers, sections, lists maintained | High |
| **Table Handling** | Tables converted to Markdown tables | Medium |
| **Figure Handling** | Figure captions preserved, references intact | Medium |
| **Equation Rendering** | LaTeX equations preserved or converted | Medium |
| **Reference Extraction** | Bibliography parsed correctly | High |

### Performance Metrics

| Metric | Description |
|--------|-------------|
| **Processing Time** | Seconds per page, total time |
| **Memory Usage** | Peak RAM during processing |
| **Output Size** | Markdown file size vs input |
| **Failure Rate** | Documents that fail to process |

---

## Implementation Plan

### Phase 1: Tool Setup

1. Install Marker: `pip install marker-pdf`
2. Install Docling: `pip install docling`
3. Setup GROBID: Docker or Python client

### Phase 2: Benchmark Execution

For each tool, process all 7 test documents and record:
- Processing time
- Output quality (manual inspection)
- Reference extraction count
- Error/warning messages

### Phase 3: Evaluation

Create comparison matrix with scores for each metric.

---

## Progress Log

### 2026-02-10: Planning

- Created this DevLog
- Selected 7 representative test documents
- Defined evaluation metrics

### Next Steps

- [ ] Install Marker
- [ ] Install Docling  
- [ ] Setup GROBID
- [ ] Run conversion on 7 test documents
- [ ] Evaluate results
- [ ] Document findings and recommendation

---

## Output Directory Structure

```
semiconductor_processing_dataset/
  processed_documents/
    markdown_evaluation/
      marker/
        <document_id>.md
      docling/
        <document_id>.md
      grobid/
        <document_id>.tei.xml
        <document_id>.md
    evaluation_results.json
```

