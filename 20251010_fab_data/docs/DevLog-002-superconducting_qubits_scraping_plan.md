# Superconducting Qubits Research Database & Scraping Plan

## Overview
This document defines a structured plan to collect, organize, and manage materials (theses, papers, publications, etc.) from the world’s leading superconducting qubit research groups, including the labs of Rob Schoelkopf, Michel Devoret, John Martinis, and their academic descendants.

---

## 1. Objectives
- Build a structured, queryable database of superconducting qubit research materials.
- Include theses, papers, and datasets from top labs and their students.
- Capture advisor–student relationships to map academic lineages.
- Prepare for automated scraping, enrichment, and deduplication.

---

## 2. Data Model Schema

### `works` table
| Field | Type | Description |
|-------|------|--------------|
| work_id | string | Hash of title+year or DOI/arXiv |
| type | enum(thesis, paper, presentation, dataset) | Type of work |
| title | string | Work title |
| authors | list[string] | Authors |
| advisors | list[string] | Advisors (for theses) |
| institution | string | University or lab |
| year | integer | Publication year |
| lab | string | Associated lab name |
| url_pdf | string | Direct PDF link |
| url_page | string | Webpage link |
| doi | string | DOI if available |
| arxiv_id | string | arXiv ID if available |
| keywords | list[string] | Key concepts (e.g., transmon, fluxonium) |
| descendants | list[string] | Linked labs |
| source_host | string | Source host domain |
| notes | string | Additional notes |
| access | enum(open, restricted) | Access control flag |

### `labs` table
| Field | Type | Description |
|-------|------|--------------|
| lab_id | string | Short unique name (e.g., yale_rsl) |
| pi | string | Principal investigator |
| institution | string | University |
| region | string | Country/region |
| home_url | string | Homepage |
| publications_url | string | Publication list URL |
| theses_url | string | Theses list URL |
| people_url | string | People/alumni list URL |
| lineage | list[string] | Advisor or descendant labs |

### `people` table
| Field | Type | Description |
|-------|------|--------------|
| name | string | Full name |
| role | enum(student, postdoc, PI) | Position |
| lab_id | string | Associated lab |
| advisors | list[string] | Advisor(s) |
| alumni_of | string | Institution of degree |
| homepage | string | Personal/lab page |
| works | list[string] | List of work IDs |

---

## 3. Source List (YAML format)

```yaml
sources:
  yale_schoelkopf:
    home: "https://rsl.yale.edu"
    theses: "https://rsl.yale.edu/theses"
    publications: "https://rsl.yale.edu/publications"
    people: "https://rsl.yale.edu/people"
  yale_devoret:
    home: "https://qulab.eng.yale.edu"
    theses: "https://qulab.eng.yale.edu/theses"
    publications: "https://qulab.eng.yale.edu/publications"
  ucsb_martinis:
    home: "https://physics.ucsb.edu/~martinisgroup"
    theses: "https://physics.ucsb.edu/~martinisgroup/theses.shtml"
    publications: "https://physics.ucsb.edu/~martinisgroup/publications.shtml"
  princeton_houck:
    home: "https://houcklab.princeton.edu"
    theses: "https://houcklab.princeton.edu/publications"
  tu_delft_dicarlo:
    home: "https://dicarlo.tudelft.nl"
    theses: "https://repository.tudelft.nl"
    publications: "https://dicarlo.tudelft.nl/publications"
  eth_wallraff:
    home: "https://qudev.phys.ethz.ch"
    theses: "https://qudev.phys.ethz.ch/teaching/theses.html"
    publications: "https://qudev.phys.ethz.ch/publications.html"
  berkeley_siddiqi:
    home: "https://qnl.berkeley.edu"
    publications: "https://qnl.berkeley.edu/publications"
  nist_lehnert:
    home: "https://jila.colorado.edu/lehnert"
    theses: "https://jila.colorado.edu/lehnert/theses"
```

---

## 4. Phases & Execution Plan

### Phase 1 — Discovery
- Crawl each lab’s `/theses`, `/publications`, `/people` pages.
- Parse metadata: title, author, year, advisor, URLs.
- Store in raw JSON with minimal normalization.

### Phase 2 — Normalization
- Normalize names, institutions, and years.
- Assign unique IDs (`work_id`, `lab_id`, `person_id`).
- Map people to labs and advisors.

### Phase 3 — Enrichment
- Query arXiv and CrossRef APIs to fill in DOIs, abstracts, keywords.
- Tag keywords automatically (transmon, fluxonium, cQED, bosonic, JPA, etc.).

### Phase 4 — Deduplication
- Use DOI/arXiv as primary key when available.
- Hash PDFs (SHA256) for duplicate detection.
- Merge duplicate entries across sources.

### Phase 5 — Lineage Mapping
- Build advisor–student relationships.
- Generate lab graph (e.g., Yale → Delft → Princeton → Google AI).

### Phase 6 — Export
- Export database to CSV/Parquet/SQLite.
- Generate visual graph of lab lineages.

---

## 5. TODO Checklist

- [ ] Implement base crawler for static lab pages.
- [ ] Integrate PDF downloader with checksum.
- [ ] Add parser for thesis front matter (extract advisor).
- [ ] Build keyword tagging pipeline.
- [ ] Write enrichment script (arXiv, CrossRef).
- [ ] Create relational database schema.
- [ ] Generate advisor-lab graph (NetworkX/Graphviz).
- [ ] Verify access and licensing constraints.
- [ ] Final export to data formats for analysis.

---

## 6. Suggested Keywords
transmon, fluxonium, flux qubit, cQED, JPA, JTWPA, JPC, JRM, bosonic code, cat qubit, GKP, error correction, surface code, tantalum, niobium, TiN, coherence, quasiparticle, TLS, materials, readout, quantum amplifier.

---

## 7. License & Access Notes
- Respect institutional copyright restrictions.
- Prefer open-access university repositories.
- For restricted theses (e.g., Yale network access), store metadata only.

---

© 2025 Superconducting Qubit Knowledge Map Initiative
