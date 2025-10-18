# Superconducting Qubits Research Database & Scraping Plan (REFINED)

## Overview
This document defines a structured plan to collect, organize, and manage materials (theses, papers, publications, etc.) from the world's leading superconducting qubit research groups, including the labs of Rob Schoelkopf, Michel Devoret, John Martinis, and their academic descendants.

---

## 1. Objectives
- Build a structured, queryable knowledge graph of superconducting qubit research.
- Collect all available PDFs (theses, papers, presentations) from top labs.
- Track all name variations for researchers (deduplication deferred to Phase 6).
- Capture temporal relationships: advisor-student (with start/end years), lab affiliations (with duration).
- Support multiple lab affiliations per researcher with role and temporal data.
- Map 2 generations of academic lineage through lab descendants.
- Extract structured data from CVs/bios using AI agents.
- Quality over quantity: focus on accuracy and completeness.

---

## 2. Data Model Schema

### Core Entities (Nodes/Tables)

#### `works` table
| Field | Type | Description |
|-------|------|--------------|
| work_id | string | Hash of URL or DOI/arXiv |
| type | enum | thesis, paper, presentation |
| title | string | Work title |
| year | integer | Publication/completion year |
| doi | string | DOI if available |
| arxiv_id | string | arXiv ID if available |
| url_pdf | string | Direct PDF link |
| url_page | string | Webpage link |
| pdf_local_path | string | Local storage path |
| pdf_file_size | integer | File size in bytes |
| pdf_hash | string | SHA256 hash for deduplication |
| scrape_timestamp | string | When PDF was downloaded |
| source_host | string | Source host domain |
| source_url | string | Source page URL |
| keywords | list | Key concepts (transmon, fluxonium, etc.) |
| access | enum | open, restricted |
| notes | string | Additional notes |

#### `people` table
| Field | Type | Description |
|-------|------|--------------|
| person_id | string | Unique identifier |
| name_canonical | string | Canonical name (for deduplication) |
| name_variations | list | All known name variations |
| homepage | string | Personal/lab page |
| bio_text | string | Raw biography text |
| bio_structured | object | Structured data extracted from bio |

#### `labs` table
| Field | Type | Description |
|-------|------|--------------|
| lab_id | string | Short unique name (e.g., yale_rsl) |
| name | string | Full lab name |
| pi_id | string | FK to people (principal investigator) |
| institution | string | University name |
| region | string | Country/region |
| founded_year | integer | Year lab was founded |
| home_url | string | Homepage |
| publications_url | string | Publication list URL |
| theses_url | string | Theses list URL |
| people_url | string | People/alumni list URL |

#### `institutions` table
| Field | Type | Description |
|-------|------|--------------|
| institution_id | string | Unique identifier |
| name | string | University name |
| country | string | Country |
| homepage | string | Homepage |

### Relationship Tables (Edges/Link Tables)

#### `authorship` table (Links people to works)
| Field | Type | Description |
|-------|------|--------------|
| authorship_id | string | Unique identifier |
| person_id | string | FK to people |
| work_id | string | FK to works |
| author_order | integer | Position in author list |
| role | enum | author, advisor |

#### `lab_affiliations` table (Links people to labs with temporal data)
| Field | Type | Description |
|-------|------|--------------|
| affiliation_id | string | Unique identifier |
| person_id | string | FK to people |
| lab_id | string | FK to labs |
| role | string | student, postdoc, PI, researcher, alumni |
| start_year | integer | Year affiliation started |
| end_year | integer | Year affiliation ended (NULL if ongoing) |
| source | string | Where info came from |
| confidence | float | Confidence score (0-1) |

#### `advisor_relationships` table (Links advisors to students)
| Field | Type | Description |
|-------|------|--------------|
| relationship_id | string | Unique identifier |
| advisor_id | string | FK to people (advisor) |
| student_id | string | FK to people (student) |
| start_year | integer | Year relationship started |
| end_year | integer | Year relationship ended |
| degree_type | enum | phd, masters, postdoc |
| institution_id | string | FK to institutions (where degree earned) |
| source | string | Where info came from |
| confidence | float | Confidence score (0-1) |

#### `citations` table (Links papers that cite each other)
| Field | Type | Description |
|-------|------|--------------|
| citation_id | string | Unique identifier |
| citing_work_id | string | FK to works (paper doing citing) |
| cited_work_id | string | FK to works (paper being cited) |
| source | enum | google_scholar, crossref, manual |

#### `lab_lineage` table (Links labs through PI lineage)
| Field | Type | Description |
|-------|------|--------------|
| lineage_id | string | Unique identifier |
| ancestor_lab_id | string | FK to labs (ancestor) |
| descendant_lab_id | string | FK to labs (descendant) |
| generation | integer | Generation depth (1 = direct descendant) |
| connecting_person_id | string | FK to people (PI who moved labs) |

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

### Phase 1: Source Inspection & Strategy
- Inspect each lab website to understand structure.
- Create YAML configuration with CSS selectors for PDF links.
- Document whether Playwright is needed (dynamic JS vs static HTML).
- Check robots.txt compliance and rate limiting requirements.
- Output: `sources_inspection.yaml` with parsing rules for each lab.

### Phase 2: PDF Scraping & Storage
- Use Playwright for respectful, slow scraping with random intervals.
- Download all available PDFs from configured sources.
- Track metadata: file size, SHA256 hash, scrape timestamp, source URL.
- Store PDFs in organized directory structure.
- Store metadata in SQLite database.
- Run overnight with random intervals between labs (30 min to 2 hours).
- Output: PDFs + `metadata.db` with full metadata.

### Phase 3: Metadata Extraction
- Extract text from PDFs using docling or marker.
- Parse metadata from PDFs and HTML: title, authors, year, advisor.
- Use AI agents to process CV/bio information into structured data.
- Populate `works` and `people` tables.
- Track all name variations (no deduplication yet).
- Output: Populated works, people, authorship tables.

### Phase 4: Relationship Mapping
- Extract advisor-student relationships from thesis metadata.
- Map researcher affiliations with temporal data (start/end years).
- Build lab lineage (2 generations through PI movements).
- Assign confidence scores based on source reliability.
- Output: advisor_relationships, lab_affiliations, lab_lineage tables.

### Phase 5: Enrichment
- Query arXiv API for DOI/arXiv ID and metadata.
- Query CrossRef API for citation metadata.
- Fetch citation data from Google Scholar (via Playwright, slow and respectful).
- Tag keywords automatically (transmon, fluxonium, cQED, etc.).
- Output: Updated works table, citations table.

### Phase 6: Deduplication
- Exact match on DOI/arXiv ID.
- Fuzzy match on title + authors (>95% similarity).
- PDF hash matching for identical files.
- Manual review queue for ambiguous cases.
- Merge duplicates while preserving all name variations.
- Output: Cleaned database.

### Phase 7: Knowledge Graph Analysis
- Build NetworkX graph from database.
- Generate lineage visualizations.
- Analyze collaboration networks and co-authorship patterns.
- Export for further analysis and visualization.
- Output: Graphs, visualizations, analysis reports.

---

## 5. Implementation Principles

1. **Quality over Quantity** - Focus on accuracy and completeness
2. **Preserve All Variations** - Track all name variations for later deduplication
3. **Temporal Tracking** - Capture start/end years for all relationships
4. **Multiple Affiliations** - Support researchers with multiple lab affiliations
5. **Slow & Respectful** - Use Playwright with random intervals, run overnight
6. **PDF Priority** - Download PDFs first, convert later with docling/marker
7. **Confidence Scores** - Track confidence for uncertain relationships
8. **Educational Focus** - Non-commercial, volunteer-driven, open access

---

## 6. TODO Checklist

- [ ] Phase 1: Inspect all lab websites
- [ ] Phase 1: Create sources_inspection.yaml with CSS selectors
- [ ] Phase 2: Implement Playwright-based PDF scraper
- [ ] Phase 2: Set up SQLite metadata database
- [ ] Phase 2: Test on 1-2 labs
- [ ] Phase 2: Run overnight scraping
- [ ] Phase 3: Extract text from PDFs (docling/marker)
- [ ] Phase 3: Parse metadata from PDFs and HTML
- [ ] Phase 3: Use AI agents for CV/bio processing
- [ ] Phase 4: Extract advisor-student relationships
- [ ] Phase 4: Map lab affiliations with temporal data
- [ ] Phase 4: Build lab lineage (2 generations)
- [ ] Phase 5: Query arXiv and CrossRef APIs
- [ ] Phase 5: Fetch citation data
- [ ] Phase 6: Implement deduplication logic
- [ ] Phase 6: Manual review for ambiguous cases
- [ ] Phase 7: Build NetworkX graph
- [ ] Phase 7: Generate visualizations

---

## 7. Suggested Keywords

transmon, fluxonium, flux qubit, cQED, JPA, JTWPA, JPC, JRM, bosonic code, cat qubit, GKP, error correction, surface code, tantalum, niobium, TiN, coherence, quasiparticle, TLS, materials, readout, quantum amplifier.

---

## 8. License & Access Notes

- Respect institutional copyright restrictions.
- Prefer open-access university repositories.
- For restricted theses (e.g., Yale network access), store metadata only.

---

© 2025 Superconducting Qubit Knowledge Map Initiative
