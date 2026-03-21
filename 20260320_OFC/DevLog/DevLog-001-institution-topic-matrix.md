# Institution-Topic Analysis Interactive Matrix

## Summary

This log documents the development of an interactive web-based visualization analyzing OFC 2026 conference data across two dimensions: institutions (222) and technical topics (45). The result is a comprehensive heatmap deployed to GitHub Pages with advanced sorting, filtering, and exploration features.

## Objectives

- Extract and normalize institution names from author affiliations
- Identify technical topics from presentation abstracts and titles
- Build a co-occurrence matrix showing which institutions work on which topics
- Create an interactive HTML visualization for exploration
- Deploy to GitHub Pages for public access

## Data Processing

### Institution Extraction

- Source: Author affiliation fields from presentation metadata
- Normalization: Consolidated duplicate institution names (NTT, Huawei, Nokia, etc.)
- Coverage: 222 unique institutions after consolidation
- Top institution: NTT (233 author mentions)

### Topic Extraction

- Source: Presentation titles and abstracts
- Method: Regex pattern matching for technical terms
- Coverage: 45 technical keywords
- Categories: Modulation formats, amplification, photonic integration, networking, quantum, AI/ML

### Matrix Construction

- Dimensions: 222 institutions × 45 topics
- Values: Co-occurrence counts (institution authors on papers mentioning topic)
- Multi-author handling: Papers with multiple institutions credit all co-author affiliations
- Total cells: 9,990

## Interactive Features Implemented

### Core Visualization

- Color-coded heatmap with 6-level intensity scale
- Sticky row and column headers for navigation
- Zoom controls (50%-300%) with state preservation
- Responsive tooltip system with example presentations

### Sorting and Filtering

- Click column headers to sort institutions by activity in that topic
- Click row headers to sort topics by activity for that institution
- Toggle sort on/off to reset to original order
- Institution type filter: All / Universities & Labs / Companies
- Academic detection via regex: university, lab, college, institute, school, academy

### Tooltip System

- Hover to preview, click to pin
- Shows top 5 example presentations per cell
- Expandable abstracts (click to expand/collapse)
- Displays: presentation code, title, presenter, affiliation, abstract
- Sticky positioning with viewport-aware placement

## Technical Implementation

### State Management

- Deep cloning of original DOM state for reliable reset
- Separate tracking of row and column sort states
- Zoom level preservation across all UI interactions
- Event listener re-attachment after DOM manipulation

### Key Bug Fixes

1. **Zoom reset on sort toggle**: Replaced page reload with in-memory DOM reordering
2. **Lost event listeners after reset**: Implemented reusable listener attachment functions
3. **Row sort reset failure**: Changed from DOM references to deep clones of original state
4. **Filter rendering issues**: Switched from `display: none` to `visibility: collapse` for table rows
5. **Tooltip positioning offset**: Changed from `pageX/Y` to `clientX/Y` for fixed positioning

### Institution Consolidation

Major organizations consolidated from multiple variants:

- NTT: 6+ variants → 233 papers
- Huawei Technologies: 7+ variants → 132 papers
- Nokia: 5+ variants → 131 papers (separate from Nokia Bell Labs: 166 papers)
- Beijing University of Posts and Telecommunications: 3 variants → 175 papers
- Pengcheng Laboratory: 3 variants → 147 papers
- Samsung, ZTE, Microsoft, NVIDIA, Cisco, and 10+ others

## Deployment

- Deployment target: GitHub Pages
- Format: Single standalone HTML file (1.7MB)
- Additional: Full metadata JSON included (4.8MB)

## Key Statistics

- Total presentations analyzed: 830
- Unique institutions: 222
- Technical topics tracked: 45
- Matrix cells with data: ~2,000 non-zero entries
- Top institution-topic pair: NTT × coherent (multiple high-activity combinations)

## Design Decisions

### Multi-Author Attribution

Papers with multiple institutional affiliations credit all institutions. This design choice:

- Captures collaboration patterns
- Shows multi-institutional research efforts
- May display presenter from different institution than row header (intentional)
- Provides complete picture of institutional involvement

### Academic vs Industry Classification

Regex-based classification using institution name patterns:

- Academic: university, lab, college, institute, school, academy
- Industry: everything else
- Limitations: Some research labs may be misclassified
- Trade-off: Simple, transparent, mostly accurate

### Sticky Headers

- Column headers: `position: sticky; top: 0; z-index: 10`
- Row headers: `position: sticky; left: 0; z-index: 9`
- Corner cell: `position: sticky; left: 0; top: 0; z-index: 20`
- Shadows added for visual depth

## Known Limitations

- Institution classification is heuristic-based
- Some institution name variants may remain undetected
- Topic extraction relies on keyword matching (may miss semantic variations)
- Multi-author papers counted for all institutions (intentional but may inflate counts)
- Large matrix size may impact performance on older devices

## Files Modified

- `20260320_OFC/scripts/analyze_institutions_vs_topics.py`: Core analysis and HTML generation
- `ofc-2026/index.html`: Deployed visualization
- `ofc-2026/ofc_full_metadata.json`: Source metadata

## Commit History

1. Initial commit: Basic matrix visualization
2. Add metadata and expandable abstracts
3. Fix zoom reset bug when toggling sort
4. Fix header click listeners after reset
5. Add institution type filter
6. Fix filter rendering bug
7. Consolidate duplicate institutions - major cleanup
8. Fix row sort reset bug - deep clone original state
9. Enhance sticky headers with better z-index and shadows
10. Fix tooltip positioning - use clientX/Y for fixed positioning
