# Semiconductor Processing Dataset

This directory contains the complete dataset for extracting semiconductor fabrication processes from academic literature using LangExtract.

## Directory Structure

### raw_documents/
Original PDF files organized by domain and document type.

**papers/** - Journal articles and conference papers
- superconducting_qubits/ - SC qubit fabrication papers
  - transmon/ - Transmon qubit papers
  - fluxonium/ - Fluxonium qubit papers
  - flux_qubits/ - Flux qubit papers
  - materials_studies/ - Materials and coherence studies
  - reviews/ - Review papers
- 2d_materials/ - 2D materials fabrication
  - mos2_fabrication/ - MoS2 and TMD fabrication
  - graphene_transfer/ - Graphene transfer and CVD
  - heterostructures/ - Van der Waals heterostructures
- photonics/ - Photonic device fabrication
  - silicon_photonics/ - Silicon photonic circuits
  - tfln/ - Thin-film lithium niobate
  - optomechanics/ - Optomechanical devices
- mems/ - MEMS fabrication

**supplementary/** - Supplementary information files (same structure as papers/)

**theses/** - PhD theses and dissertations
- yale/ - Yale University theses
- ucsb/ - UC Santa Barbara theses
- eth_zurich/ - ETH Zurich dissertations
- mit/ - MIT theses
- berkeley/ - UC Berkeley theses
- delft/ - TU Delft theses
- others/ - Other institutions

**patents/** - Patent documents with detailed process specifications

### processed_documents/
Extracted text and metadata from raw documents.

**text_extracted/** - Text extraction outputs
- docling/ - Docling extraction results
- marker/ - Marker extraction results
- grobid/ - GROBID extraction results (academic papers)
- manual_corrections/ - Manually corrected extractions

**metadata/** - JSON metadata files for each document
- Document ID, authors, year, DOI, etc.
- Extraction status and quality flags
- Topic tags and keywords

### annotations/
LangExtract output and structured extractions.

- process_flows/ - Extracted process sequences
- chemicals/ - Chemical inventory and formulations
- equipment/ - Tool and equipment database
- parameters/ - Process parameters (temperature, pressure, time, etc.)

### knowledge_base/
Normalized databases and ontologies.

- chemical_database.json - Canonical chemical names and variants
- equipment_database.json - Tool specifications and models
- abbreviations.json - Domain-specific abbreviations
- process_ontology.json - Process step taxonomy

### extraction_configs/
LangExtract configuration files.

**prompts/** - Extraction prompt definitions
- chemical_extraction.py
- process_flow_extraction.py
- parameter_extraction.py
- equipment_extraction.py

**examples/** - Few-shot examples for LangExtract
- few_shot_examples/ - High-quality example extractions

## Workflow

1. **Collection**: Download PDFs to raw_documents/
2. **Metadata**: Create metadata JSON for each document
3. **Extraction**: Run text extraction tools (docling, marker, grobid)
4. **Comparison**: Compare extraction quality, select best
5. **LangExtract**: Run structured extraction with domain-specific prompts
6. **Annotation**: Store results in annotations/
7. **Normalization**: Build knowledge base from extractions
8. **Validation**: Expert review and quality control

## Canonical Naming

- Document filename format: `lastname_firstname_institution_year.pdf`
- Document ID format: `lastname_firstname_institution_year`
- Metadata path: `semiconductor_processing_dataset/processed_documents/metadata/<document_id>.json`
- Source document path convention:
  - papers: `semiconductor_processing_dataset/raw_documents/papers/<domain>/<subdomain>/`
  - supplementary: `semiconductor_processing_dataset/raw_documents/supplementary/<domain>/<subdomain>/`
  - theses: `semiconductor_processing_dataset/raw_documents/theses/<institution>/`
  - patents: `semiconductor_processing_dataset/raw_documents/patents/`

## Metadata Minimum Fields

Each metadata JSON must include:
- `document_id`
- `source_type`
- `title`
- `year`
- `url`
- `download_date`
- `quality_assessment`
- `extraction_status`
- `quality_flags`

Collection tracking files:
- `semiconductor_processing_dataset/processed_documents/metadata/manifest_documents.jsonl`
  - One latest-state record per `document_id`.
- `semiconductor_processing_dataset/processed_documents/metadata/collection_attempts.jsonl`
  - Append-only log of each collection attempt (attempted, failed, succeeded, skipped).

## Storage Strategy

- Local storage during active processing
- Move raw PDFs to NAS after successful extraction and ingestion
- Keep processed documents and annotations local for analysis

## Data Quality

Each document should have:
- Metadata JSON file
- At least one text extraction
- Quality assessment flag
- Extraction status tracking

High-value documents:
- Detailed fabrication sections
- Process flow diagrams
- Chemical lists with specifications
- Equipment models and parameters
- Supplementary information

## Notes

- Focus initially on SC qubits (Phase 1)
- Expand to other domains in later phases
- Maintain consistent naming conventions
- Track document provenance and download dates
