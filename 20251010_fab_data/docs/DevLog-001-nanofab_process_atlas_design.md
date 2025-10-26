# Nanofabrication Process Atlas — Design Document (v0.1)
**Date:** 2025-10-11


## 1) Executive Summary
We will build an open, structured, and queryable “process atlas” for nanofabrication, starting with **superconducting qubits** (transmons, fluxonium, resonators) and then expanding to **nanophotonics** and **MEMS**. The system stores machine-readable recipes, supports **visual analytics** (e.g., T₁ distributions, yield trends), and provides **RAG-based** natural-language search grounded in cited sources. The project will launch **local-first** for rapid iteration, with a low-cost path to a public demo (e.g., Hugging Face Space) using a managed Postgres later.

**Primary outcomes:**
- A normalized **Postgres** database with **pgvector** for semantic search and FTS for keyword search.
- A **YAML-based recipe** format validated by JSON Schema and exploded into SQL tables.
- A curation workflow covering **literature** (papers, methods) and **materials** (resists, developers, solvents).
- A retrieval layer enabling **hybrid search** (vector + FTS + filters) and **chat** with citations and figure crops.
- A visualization layer (Metabase/Superset) for trend analysis.

---

## 2) Scope & Goals
**Initial scope:** superconducting qubit fabrication processes (Al/AlOx/Al transmons, Nb/NbN resonators, fluxonium), spanning substrate preparation, lithography, deposition, oxidation, etch/lift-off, and measurement summaries.

**Goals:**
- Make processes **discoverable** by materials, steps, equipment, and outcomes.
- Enable **queries** such as:
  - “All Al/AlOx/Al transmon processes on sapphire with T₁ > 50 µs and base pressure < 1e-7 Torr.”
  - “Resist stacks associated with higher yield for NbN CPW resonators.”
  - “Two-angle Al shadow-evaporation recipes and their oxidation conditions.”
- Generate **readable process sheets** directly from structured recipes.
- Provide **RAG** answers with explicit **provenance** (page/figure references) and **YAML step IDs**.

**Out-of-scope for Phase 0:** automated device simulation, waferscale yield maps, proprietary tool telemetry.

---

## 3) Data Model Overview (system of record = Postgres)
We use Postgres (16+) with JSONB for flexible fields and **pgvector** for embeddings. Core entities provide structure; JSONB fields capture variable details and maintain forward compatibility.

### 3.1 Core Entities
- **papers**: DOI, title, year, journal, arXiv, open-access link, authors/affiliations, bibliography.
- **processes**: links to `papers`; domain tag (qubit | nanophotonics | mems); schema version; **YAML recipe** (authoritative); provenance (who/when/how extracted).
- **steps**: exploded from YAML; ordered; `kind` (clean, lithography, deposition, oxidation, etch, anneal, lift_off, measure, etc.); detail (JSONB) for parameters.
- **materials**: canonical registry (name, formula, synonyms, vendor/product, CAS/pubchem IDs, datasheet/SDS links).
- **process_materials**: links processes to materials with roles (substrate, film, resist, mask, etchant, gas, developer, stripper).
- **metrics**: outcome metrics (e.g., T₁, T₂, Qᵢ, yield, Ic) with normalized units and optional context (frequency, power regime).
- **chunks**: RAG content units (section text or single step) with metadata (device/materials/methods tags) and an embedding.
- **files**: PDFs, figure crops, chart overlays; storage URLs (object storage), mime type, page, bbox, OCR text; optional CLIP-like embedding.
- **material_curves**: digitized curves (e.g., spin speed vs. thickness) with conditions, points, and a fit model (power law), linked to `files` for provenance.

### 3.2 Indexing & Search
- **FTS** (tsvector) on text content for keywords.
- **pgvector** HNSW index on `chunks.embedding` for semantic search.
- **GIN** on JSONB metadata for fast faceting (materials, device type, methods).

### 3.3 Controlled Vocabulary & Units
- Controlled fields: device type, substrate, deposition method, lithography type, etch type, resist names, gases, equipment (make/model).
- Units normalized to SI; conversion on ingest (keep original in provenance).
- Material and equipment **synonym** maps ensure consistent analytics.

---

## 4) Recipe Representation (YAML, validated by schema)
**Intent:** Human-readable, machine-validated source of truth. YAML contains:
- `schema_version`, `paper` (DOI/arXiv), `device` (type, substrate, geometry summary).
- `process`: name and ordered `steps` with `kind` and `detail` parameters (e.g., resist stack, bake, developer, spin speeds, deposition rates/pressures, oxidation).
- `outcomes`: key metrics (e.g., T₁, yield) with units.
- `provenance`: extraction tool/version, curator, date, confidence, evidence pointers (page/figure/bbox).

**Operational use:**
- YAML is stored in `processes`; a loader explodes it into `steps`, `materials`, `metrics`.
- A renderer generates printable “process sheets” (HTML/Markdown/PDF) from YAML.
- All surfaced answers show **step IDs** and **citations** from provenance.

---

## 5) Curation Strategy (Phase 0–1)
We start with two parallel streams and a tight loop (seed → validate → refine schema → scale).

### 5.1 Literature Stream (processes)
- Seed 25–40 open-access, high-signal papers across major labs/companies; diversity in materials (Al/Nb/NbN), substrates (sapphire/Si), lithography stacks (MMA/PMMA, PMMA-only, HSQ).
- Capture MVP fields first: substrate; resist stack + bake; patterning method; metal + deposition (base pressure, rate, thickness); oxidation (mode, pressure, time); etch/lift-off chemistry; core outcomes (T₁, Qᵢ, yield); and **evidence locations**.
- Require page/figure references for each fact (empowers grounded RAG).

**Prioritization rubric:**
1. Methods completeness & supplementary detail.
2. Open access availability.
3. Diversity coverage (materials/methods).
4. Impact/citations and community uptake.
5. Recency.

### 5.2 Materials Stream (datasheets)
- Curate 15–25 common materials first: PMMA 950 A4, MMA EL11, ZEP520A, HSQ/FOx; developers (MIBK:IPA 1:3, TMAH-based); strippers (NMP/alternatives); solvents (anisole, etc.).
- Normalize: solids wt%, viscosity, solvent system, recommended spin curves (rpm→nm), bake schedules, developer/time, stripper, hazard (GHS), storage. Maintain aliases/vendor strings.

---

## 6) Ingest & Extraction Workflow
1. **Discover**: Crossref/DOI and arXiv; manual uploads when needed; store only OA PDFs; for gated papers store derived facts + DOIs and short, fair-use quotes.
2. **Parse PDFs**: text + table extraction; figure detection; OCR for image-only sections.
3. **Process detection**: section tagging (“Methods”, “Fabrication”, “Deposition”, units patterns like nm, Torr, sccm).
4. **Structured extraction**: LLM-assisted mapping to the YAML schema; deterministic rules for units and common patterns.
5. **Normalize**: units (SI), materials & equipment synonyms, step kinds.
6. **Human-in-the-loop**: curation UI shows side-by-side PDF and proposed YAML; curator accepts/edits with provenance retained.
7. **Load**: write `papers`/`processes`/`steps`/`materials`/`metrics`; build `chunks` with embeddings and rich metadata.
8. **Index**: FTS, pgvector, and JSONB indexes.
9. **QA**: schema validation, unit checks, completeness thresholds, and outlier detection.

**Provenance is mandatory** at each step.

---

## 7) Datasheet & Curve Extraction (spin curves, etc.)
**Tables-first (robust):**
- Prefer parsing numeric tables directly from PDFs; if tables are images, OCR + table-structure recovery.
- Store tidy tables with conditions (time, temperature, substrate).

**Plots/curves (spin speed vs. thickness):**
- **MVP**: semi-automatic digitization for axis calibration (log/linear) and point capture; store points + calibration + curve crop for provenance.
- **Model**: fit a **power law** (thickness = A · rpm^(−b)) at fixed time/viscosity; store fit parameters and RMSE; keep raw points.
- **QA**: render an overlay of the fitted curve on the crop; flag low-confidence extractions for review.

**What to capture beyond curves:**
- Spin/bake protocols (dispense, time, acceleration; soft/post-exposure bakes).
- Chemistry (solvent system, solids %, viscosity, developer ratios/times, stripper).
- Versioning (datasheet/SDS version date; product lineage).
- Conditions (ambient temperature, RH, substrate), recorded as machine-readable fields.

---

## 8) Storage of Images & Large Artifacts
- **Do not** store large images inline in Postgres; store them in **object storage** (local MinIO in dev; S3/GCS or a public dataset host in prod/demo).
- Keep **thumbnails** inline (optional) and store **metadata** (page, bbox, caption, OCR text, hashes, mime).
- Link derived facts (e.g., digitized curves) back to the **source file** and region (page/bbox).

---

## 9) Retrieval & RAG
- **Chunking**: per section and per step (each YAML step becomes a chunk). Metadata includes domain, device type, materials list, substrate, and methods.
- **Embeddings**: start with a strong open model; store in `chunks.embedding`.
- **Hybrid search**: combine pgvector kNN, FTS, and structured filters (JSONB/GiN). Use a reranker on the top-N to boost precision.
- **Grounded answers**: return matched step IDs, DOI + page/figure/bbox, and YAML excerpts used in the answer.

---

## 10) Visualizations & Analytics
- **BI tooling**: Metabase or Superset on top of Postgres (no ETL).
- **Initial views**: 
  - T₁/T₂ distributions by substrate, device type, and oxidation presence.
  - Yield vs. resist stack, base pressure, or oxidation time.
  - Heatmaps (thickness vs. oxidation time → outcome metric).
  - Yearly trends by material/method adoption.
- Maintain materialized views for common slices to improve dashboard responsiveness.

---

## 11) Platform & Deployment Strategy
### 11.1 Local-First (Phase 0)
- Run locally with Docker: Postgres + pgvector, FastAPI (or Gradio), Adminer (DB UI), MinIO (object storage), and Metabase.
- Benefits: zero external dependencies; fast iteration; simple debugging.
- Keep secrets in a local `.env` file; mount volumes for DB persistence.

### 11.2 Public Demo (Phase 1)
- **Hugging Face Space** for the **web app** (Gradio/Streamlit or FastAPI via Docker).
- **Managed Postgres** (Neon or Supabase) with pgvector for the database.
- **Artifacts**: host YAML, images, and Parquet in a versioned public dataset repo (CDN-served).
- **Batch ingest**: GitHub Actions on a schedule (crawl arXiv/DOIs, write to DB, push artifacts).
- Notes: free Spaces can **sleep**; acceptable for a demo. Do not run Postgres inside a Space.

### 11.3 Scale-Out (Later)
- Keep Postgres as the system of record; add an external ANN vector service only if needed.
- Move artifacts from MinIO to durable cloud storage (S3/GCS) or keep using a public dataset host for artifacts.

---

## 12) API & UI (high-level, no code)
**Core endpoints (illustrative):**
- `POST /ingest_pdf` — upload a PDF; returns ingestion job status and provenance summary.
- `GET /papers/{{id}}` — paper metadata and citations.
- `GET /processes/{{id}}` — full YAML + provenance; links to steps, materials, metrics.
- `GET /search` — hybrid search over text + embeddings + filters.
- `POST /answer` — RAG answer; returns passages, step IDs, and citations.
- `GET /materials/{{id}}` — canonical properties, synonyms, links to datasheets/SDS.
- `GET /curves/{{material_id}}` — digitized points, fit, conditions; provides interpolation at requested rpm.
- `GET /files/{{id}}` — metadata and signed URL for the asset.

**Curation UI:**
- PDF viewer with region highlights; YAML editor with schema validation; diff view; provenance panel.
- Materials registry editor (aliases, vendor/product mapping, hazard/storage).
- Process sheet renderer (one-click export to HTML/PDF).

**User-facing search UI:**
- Facets for device type, substrate, materials, methods; free-text query; sort by recency/impact.
- Chat interface with “copy recipe steps” and “show citations” actions.

---

## 13) Governance, Licensing, and Quality
- **Schema versioning**: track `schema_version` in recipes; use DB migrations for changes.
- **Provenance & confidence**: every extracted field carries a source and confidence; allow filtering by threshold.
- **Licensing**: store only OA PDFs; for paywalled papers, store **facts with citations** and very short quotes (fair use).
- **Community contributions**: PR-style edits to YAML with review; optional ORCID/GitHub login for attribution.
- **Evaluation**: maintain a small, labeled QA set; track precision@k for retrieval and extraction accuracy.

---

## 14) Risks & Mitigations
- **Publisher restrictions**: mitigate by storing derived facts + citations; link out to DOIs.
- **OCR/parse quality**: keep human-in-the-loop, especially for tables/figures; log confidence; require approval for low-confidence extracts.
- **Unit/notation variability**: enforce conversion on ingest; keep originals in provenance; maintain synonym maps.
- **Hallucination risk** in LLM extraction: constrain with schema, explicit quoting, and curator approval.
- **Scaling embeddings**: start with pgvector; add external ANN service only if needed.

---

## 15) Phased Plan & Deliverables
**Phase 0 (Local-first):**
- Finalize YAML schema (MVP) and Postgres tables/indexes.
- Curate ~30 literature entries and ~20 materials; validate ingestion and rendering.
- Enable FTS + basic pgvector retrieval; build 2–3 Metabase dashboards.

**Phase 1 (Public Demo):**
- Deploy app to a public Space; move DB to managed Postgres; publish artifacts to a versioned dataset repo.
- Add RAG chat with citations and a curation UI for maintainers.
- Nightly ingest of new arXiv entries in target categories.

**Phase 2 (Expand Domains + Automation):**
- Add nanophotonics and MEMS tags and vocabularies.
- Semi-automatic curve digitization pipeline with QA overlays.
- Community contribution workflow and moderation.

---

## 16) Open Questions & Decisions
- **Equipment ontology** depth (how much model-specific detail to capture).
- **Embedding model** choice and dimension (balance quality, license, and cost).
- **Contribution policy** (who can edit/publish; review thresholds).
- **Hosting for artifacts** long-term (public dataset host vs. object storage).

---

## 17) Appendices (Field Lists)
**Step kinds (initial):** clean, lithography, deposition, oxidation, etch, anneal, lift_off, measure.  
**Common parameters:** spin rpm/time/accel; bake temp/time; developer & time; base pressure; deposition rate; thickness; oxidation mode/pressure/time; etch chemistry/gases/power/time; tool make/model.  
**Outcome metrics:** T₁, T₂, f₀ (GHz), Ic, Qᵢ, yield (die or device-level), device count, temperature/power context.  
**Materials roles:** substrate, film, resist, mask, etchant, gas, developer, stripper.

---

**Summary**: This design emphasizes a structured, provenance-rich foundation with hybrid retrieval and practical curation. Start local, prove the schema and workflows on a small curated set, then publish a low-cost public demo backed by managed Postgres and a versioned artifact store. The same schema and APIs support later domain expansion to nanophotonics and MEMS.
