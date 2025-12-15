# DevLog-001: Thesis PDF Processing and Graph Construction

**Date**: 2025-12-06  
**Status**: Complete  
**Project**: GraphRAG Implementation - Thesis Analysis

## Objective

Extract text from academic thesis PDF, build knowledge graph using GraphRAG, and implement visualization and querying capabilities.

## PDF Text Extraction

### Initial Approach: Docling
- Attempted IBM Docling for academic PDF parsing
- Issue: Extremely slow due to ML model downloads (torch, torchvision, layout detection models)
- Conclusion: Overkill for simple text extraction needed for GraphRAG indexing

### Final Approach: pypdf
- Installed pypdf (lightweight, no ML dependencies)
- Created `parse_thesis.py` script for fast text extraction
- Successfully extracted 255 pages in seconds

### Extraction Results
- Input: `pdfs/Schuster_thesis.pdf` (Circuit Quantum Electrodynamics thesis)
- Output: `thesis_output/thesis.txt`
- Statistics:
  - Pages: 255
  - Lines: 10,301
  - Words: 91,823
  - Characters: 516,109
  - File size: 515.2 KB

### Text Quality Assessment
- Main narrative text: Readable and suitable for GraphRAG
- Mathematical notation: Degraded (broken subscripts, Greek letters)
- Equations: Formatting artifacts present
- Conclusion: Acceptable for entity/relationship extraction despite math formatting issues

## GraphRAG Project Setup

### Project Initialization
```bash
graphrag init --root ./thesis
```

### Configuration
- Copied `.env` from existing Christmas project
- Input directory: `./thesis/input/`
- Output directory: `./thesis/output/`
- Settings: `./thesis/settings.yaml` (default configuration)

### Input Preparation
- Moved extracted text to `./thesis/input/schuster_thesis.txt`

## Indexing Pipeline Execution

### Command
```bash
graphrag index --root ./thesis
```

### Processing Statistics
- Text chunks created: 138
- Entities extracted: 825
- Relationships identified: 829
- Communities detected: 106
- Embeddings generated: Successfully completed

### Entity Type Distribution
- ORGANIZATION: 108
- EVENT: 73
- PERSON: 71
- GEO: 12
- Other: 17

### Top Entities by Degree
1. COOPER PAIR BOX (EVENT): degree=78
2. QUBIT (ORGANIZATION): degree=62
3. CAVITY QED (EVENT): degree=54
4. CPB (EVENT): degree=29
5. CAVITY (ORGANIZATION): degree=28

## Visualization Enhancement

### Script Updates
Modified `visualize_graph.py` to accept command-line arguments:

**New Parameters:**
- `--input-dir`: Path to GraphRAG output directory
- `--output-dir`: Path for visualization outputs
- `--title`: Graph title for display

**Usage Examples:**
```bash
# Thesis visualization
uv run python visualize_graph.py \
  --input-dir ./thesis/output \
  --output-dir ./thesis/plot \
  --title "Circuit QED Thesis"

# Christmas Carol visualization (default)
uv run python visualize_graph.py --title "A Christmas Carol"
```

### Visualization Outputs
- Static PNG: High-resolution network graph
- Interactive HTML: Pyvis-based interactive exploration with hover tooltips
- Node information: Entity type, degree, description
- Edge information: Relationship weight, description

## Query Testing

### Local Search Example
**Query:** "What is a Cooper Pair Box and how does it work?"

**Response Summary:**
- Definition: Quantum bit component in superconducting circuits
- Structure: Josephson tunnel junction with gate capacitor
- Working principle: Quantum state manipulation via gate voltage and Josephson energy
- Applications: Cavity QED and Circuit QED experiments
- Evolution: Advanced to transmon design for reduced charge noise

### Global Search Example
**Query:** "What are the main contributions and findings of this thesis?"

**Response Summary:**
- Focus: Cavity Quantum Electrodynamics research
- Key concepts: Jaynes-Cummings Hamiltonian, Cooper Pair Box, Circuit QED
- Applications: Quantum computing and quantum information processing
- Challenges: Noise mitigation (critical current noise, flux noise)
- Contributors: Andreas Wallraff, David Schuster, Houck

## Technical Implementation

### PDF Parsing Script
```python
# parse_thesis.py - Fast text extraction using pypdf
reader = pypdf.PdfReader(pdf_path)
text_content = [page.extract_text() for page in reader.pages]
full_text = "\n\n".join(text_content)
```

### Query Commands
```bash
# Local search (specific questions)
graphrag query --root ./thesis --method local --query "<question>"

# Global search (broad questions)
graphrag query --root ./thesis --method global --query "<question>"
```

## Lessons Learned

1. **PDF Extraction Tool Selection**: Simple pypdf sufficient for GraphRAG; avoid ML-heavy tools unless layout preservation critical
2. **Text Quality Requirements**: GraphRAG robust to formatting artifacts; entity extraction works despite broken mathematical notation
3. **Visualization Flexibility**: Parameterized scripts enable reuse across multiple GraphRAG projects
4. **Query Method Selection**: Local search for specific entities, global search for thematic overview
5. **Graph Statistics**: Academic thesis produces denser entity networks than narrative fiction

## Next Steps

- Compare graph structures between narrative (Christmas Carol) and technical (thesis) documents
- Experiment with custom prompts for academic entity extraction
- Evaluate query performance on domain-specific technical questions
- Consider chunking parameter optimization for academic papers

## References

- pypdf: https://pypdf.readthedocs.io/
- GraphRAG Query Documentation: https://microsoft.github.io/graphrag/query/
- Thesis: "Circuit Quantum Electrodynamics" by David Isaac Schuster (2007)

