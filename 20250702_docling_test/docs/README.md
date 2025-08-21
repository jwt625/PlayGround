# Documentation

This folder contains technical documentation and evaluation reports for document processing tools and libraries.

## Contents

### RFD-001: Docling Evaluation Report
**File**: `RFD-001-docling-evaluation.md`

A comprehensive evaluation of the Docling document processing library, including:

- **Executive Summary**: Key findings and recommendations
- **Installation Guide**: Step-by-step setup instructions
- **Performance Analysis**: Detailed test results and metrics
- **Feature Evaluation**: Formula extraction, OCR, image processing
- **Usage Examples**: Code samples for common use cases
- **Integration Patterns**: Framework integrations (LangChain, LlamaIndex)
- **Troubleshooting**: Common issues and solutions
- **Marker Comparison**: Head-to-head analysis with Marker

#### Key Findings
- ⭐⭐⭐⭐⭐ **Overall Rating**: Excellent for academic document processing
- 🧮 **Formula Extraction**: Outstanding LaTeX conversion capabilities
- 🖼️ **Image Processing**: Good AI-powered classification and description
- 📝 **Text Quality**: Excellent structure preservation
- 🚀 **Performance**: Good processing speed (2.4 min for 4-page academic paper)

#### Test Results Summary
- **Document Tested**: iVehicles academic paper (4 pages)
- **Processing Time**: 143 seconds with full enrichment
- **Formulas Extracted**: 5/5 successfully converted to LaTeX
- **Images Processed**: 7 with AI descriptions
- **Output Formats**: Markdown, HTML, JSON

### RFD-002: Marker Evaluation Report
**File**: `RFD-002-marker-evaluation.md`

A detailed evaluation of the Marker document processing library, focusing on:

- **Advanced Structure Analysis**: Cell-level table detection and hierarchical analysis
- **Multi-Stage Processing**: Specialized models for different document elements
- **Debug Capabilities**: Comprehensive visualization and metadata generation
- **Technical Architecture**: Deep dive into processing pipeline
- **Comparison Analysis**: Detailed comparison with Docling
- **Research Applications**: Ideal use cases for detailed document analysis

#### Key Findings
- ⭐⭐⭐⭐⭐ **Overall Rating**: Outstanding for research and detailed analysis
- 🔬 **Structure Analysis**: Superior detail (71 table cells vs 2 tables)
- 🧮 **Formula Extraction**: Excellent (13 mathematical expressions)
- 🖼️ **Image Processing**: High-quality extraction with proper naming
- 🔧 **Debug Features**: Unmatched visualization capabilities
- ⏱️ **Performance**: Slower but more thorough (8 min vs 2.4 min)

#### Test Results Summary
- **Document Tested**: iVehicles academic paper (4 pages)
- **Processing Time**: 31.5 seconds (0.5 minutes) with full configuration
- **Formulas Extracted**: 5 display + 8 inline mathematical expressions
- **Images Processed**: 6 high-quality images with debug visualizations
- **Output Formats**: Markdown with extensive debug data

## Quick Start

### Docling Setup
```bash
# Setup environment
uv venv && source .venv/bin/activate
uv pip install docling

# Run basic test
python -c "
from docling.document_converter import DocumentConverter
converter = DocumentConverter()
result = converter.convert('your_document.pdf')
print(result.document.export_to_markdown())
"
```

### Marker Setup
```bash
# Install Marker with full dependencies
uv pip install marker-pdf[full]

# Run with maximum configuration
marker_single \
  --highres_image_dpi 300 \
  --extract_images true \
  --debug \
  --output_format markdown \
  your_document.pdf
```

## Tool Comparison Summary

| Feature | Docling | Marker | Best For |
|---------|---------|--------|----------|
| **Setup Time** | Fast | Slow (model downloads) | Docling |
| **Processing Speed** | 2.4 min | 0.5 min | Marker |
| **Formula Extraction** | 5 formulas | 13 expressions | Marker |
| **Structure Analysis** | Good | Outstanding | Marker |
| **Debug Features** | Moderate | Comprehensive | Marker |
| **Ease of Use** | Simple | Complex | Docling |
| **Production Ready** | Yes | Research-focused | Docling |
| **Configuration** | Moderate | Extensive | Marker |

### Recommendations
- **Choose Docling** for: Production systems, fast processing, simple integration
- **Choose Marker** for: Research, detailed analysis, debugging, comprehensive structure understanding

## Document Format

This documentation follows the RFD (Request for Discussion) format:
- **RFD-XXX**: Sequential numbering
- **Status**: Draft, Review, Complete
- **Structured Sections**: Clear organization for technical evaluation
- **Code Examples**: Practical implementation guidance
- **Appendices**: Detailed technical information

## Contributing

When adding new documentation:
1. Follow the RFD numbering scheme
2. Include practical examples and code samples
3. Provide clear setup and usage instructions
4. Document test environments and results
5. Include troubleshooting sections

---

**Last Updated**: 2025-01-21  
**Maintainer**: Technical Evaluation Team
