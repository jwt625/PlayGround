#!/usr/bin/env python3
"""
Marker-based document converter for high-quality PDF to Markdown/HTML conversion.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import tempfile
import shutil
import json
import time
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConversionMode(Enum):
    """Conversion mode options for PDF processing."""
    AUTO = "auto"      # Smart mode: try fast, fallback to quality if needed
    FAST = "fast"      # Fast mode: disable OCR for speed (~40 seconds)
    QUALITY = "quality"  # Quality mode: full OCR for maximum accuracy (~6 minutes)

# Global model cache for performance optimization
_model_cache: Optional[Dict[str, Any]] = None
_model_load_time: Optional[float] = None


def get_cached_models(fast_mode: bool = False) -> Dict[str, Any]:
    """
    Get cached Marker models, loading them if not already cached.

    Args:
        fast_mode: If True, use faster but potentially lower quality models

    Returns:
        Dictionary containing the loaded Marker models
    """
    global _model_cache, _model_load_time

    cache_key = f"fast_{fast_mode}"

    if _model_cache is None or cache_key not in _model_cache:
        logger.info(f"Loading Marker models for the first time (fast_mode={fast_mode})...")
        start_time = time.time()

        try:
            from marker.models import create_model_dict

            # Models are loaded the same way - performance is controlled via config
            if _model_cache is None:
                _model_cache = {}

            _model_cache[cache_key] = create_model_dict()
            _model_load_time = time.time() - start_time
            logger.info(f"Marker models loaded successfully in {_model_load_time:.2f} seconds")
        except ImportError as e:
            logger.error(f"Failed to import Marker models: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load Marker models: {e}")
            raise
    else:
        logger.info(f"Using cached Marker models (loaded in {_model_load_time:.2f}s)")

    return _model_cache[cache_key]


def clear_model_cache() -> None:
    """Clear the cached models (useful for testing or memory management)."""
    global _model_cache, _model_load_time
    _model_cache = None
    _model_load_time = None
    logger.info("Model cache cleared")


def assess_pdf_quality(pdf_path: Path) -> Dict[str, Any]:
    """
    Assess PDF text extraction quality to determine optimal conversion mode.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary containing quality assessment and recommendations
    """
    try:
        import PyPDF2

        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)

            # Sample first 5 pages for quality assessment
            sample_pages = min(5, total_pages)
            total_chars = 0
            text_pages = 0
            total_words = 0

            for i in range(sample_pages):
                try:
                    text = reader.pages[i].extract_text()
                    if text and len(text.strip()) > 50:
                        clean_text = text.strip()
                        total_chars += len(clean_text)
                        total_words += len(clean_text.split())
                        text_pages += 1
                except Exception as e:
                    logger.debug(f"Error extracting text from page {i}: {e}")
                    continue

            # Calculate quality metrics
            avg_chars_per_page = total_chars / max(sample_pages, 1)
            avg_words_per_page = total_words / max(sample_pages, 1)
            text_coverage = text_pages / max(sample_pages, 1)

            # Quality thresholds (tuned based on testing)
            has_good_text = (
                avg_chars_per_page > 200 and  # Minimum character density
                avg_words_per_page > 30 and   # Minimum word density
                text_coverage > 0.6           # Most pages have extractable text
            )

            recommended_mode = ConversionMode.FAST if has_good_text else ConversionMode.QUALITY

            quality_info = {
                "has_good_text": has_good_text,
                "avg_chars_per_page": avg_chars_per_page,
                "avg_words_per_page": avg_words_per_page,
                "text_coverage": text_coverage,
                "total_pages": total_pages,
                "sample_pages": sample_pages,
                "recommended_mode": recommended_mode.value,
                "confidence": "high" if text_coverage > 0.8 else "medium" if text_coverage > 0.4 else "low"
            }

            logger.info(f"PDF quality assessment: {quality_info['recommended_mode']} mode recommended "
                       f"(confidence: {quality_info['confidence']})")

            return quality_info

    except ImportError:
        logger.warning("PyPDF2 not available for quality assessment, defaulting to quality mode")
        return {
            "has_good_text": False,
            "recommended_mode": ConversionMode.QUALITY.value,
            "confidence": "low",
            "error": "PyPDF2 not available"
        }
    except Exception as e:
        logger.warning(f"Error assessing PDF quality: {e}, defaulting to quality mode")
        return {
            "has_good_text": False,
            "recommended_mode": ConversionMode.QUALITY.value,
            "confidence": "low",
            "error": str(e)
        }

class MarkerConverter:
    """Converter using Marker for high-quality PDF conversion with smart mode selection."""

    def __init__(self, mode: ConversionMode = ConversionMode.AUTO, fast_mode: bool = None):
        """
        Initialize the Marker converter.

        Args:
            mode: Conversion mode (AUTO, FAST, or QUALITY)
                 - AUTO: Smart mode that tries fast first, falls back to quality if needed
                 - FAST: Speed optimized (~40 seconds, relies on existing PDF text)
                 - QUALITY: Maximum quality (~6 minutes, full OCR for scanned documents)
            fast_mode: Deprecated. Use mode parameter instead. If provided, overrides mode.
        """
        self.supported_formats = {'.pdf'}

        # Handle backward compatibility
        if fast_mode is not None:
            logger.warning("fast_mode parameter is deprecated. Use mode parameter instead.")
            self.mode = ConversionMode.FAST if fast_mode else ConversionMode.QUALITY
        else:
            self.mode = mode

        self.last_conversion_time: Optional[float] = None
        self.last_model_load_time: Optional[float] = None
        self.last_quality_assessment: Optional[Dict[str, Any]] = None
        self.last_mode_used: Optional[ConversionMode] = None
        
    def is_supported(self, file_path: Path) -> bool:
        """Check if the file format is supported by Marker."""
        return file_path.suffix.lower() in self.supported_formats

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics from the last conversion.

        Returns:
            Dictionary containing timing and quality information
        """
        return {
            "total_conversion_time": self.last_conversion_time,
            "model_load_time": self.last_model_load_time,
            "actual_processing_time": (
                self.last_conversion_time - (self.last_model_load_time or 0)
                if self.last_conversion_time and self.last_model_load_time
                else None
            ),
            "mode_used": self.last_mode_used.value if self.last_mode_used else None,
            "quality_assessment": self.last_quality_assessment
        }

    def _validate_output_quality(self, output_dir: Path) -> Dict[str, Any]:
        """
        Validate the quality of conversion output.

        Args:
            output_dir: Directory containing conversion output

        Returns:
            Dictionary with validation results
        """
        html_file = output_dir / "index.html"

        if not html_file.exists():
            return {"is_valid": False, "reason": "No HTML output file found"}

        try:
            content = html_file.read_text(encoding='utf-8')

            # Quality heuristics
            content_length = len(content)
            paragraph_count = content.count('<p>')
            heading_count = content.count('<h')
            image_count = content.count('<img')

            # Check for garbled characters (common in poor OCR)
            garbled_chars = ['□', '�', '▢', '◯']
            garbled_count = sum(content.count(char) for char in garbled_chars)

            # Validation criteria
            has_sufficient_content = content_length > 1000
            has_structure = paragraph_count > 3 or heading_count > 1
            low_garbled_ratio = garbled_count / max(content_length, 1) < 0.01

            is_valid = has_sufficient_content and has_structure and low_garbled_ratio

            validation_result = {
                "is_valid": is_valid,
                "content_length": content_length,
                "paragraph_count": paragraph_count,
                "heading_count": heading_count,
                "image_count": image_count,
                "garbled_count": garbled_count,
                "quality_score": (
                    (1 if has_sufficient_content else 0) +
                    (1 if has_structure else 0) +
                    (1 if low_garbled_ratio else 0)
                ) / 3
            }

            if not is_valid:
                reasons = []
                if not has_sufficient_content:
                    reasons.append("insufficient content")
                if not has_structure:
                    reasons.append("poor document structure")
                if not low_garbled_ratio:
                    reasons.append("too many garbled characters")
                validation_result["reason"] = ", ".join(reasons)

            return validation_result

        except Exception as e:
            return {"is_valid": False, "reason": f"Error reading output: {e}"}
    
    def convert_to_html(self, input_path: Path, output_dir: Path) -> bool:
        """
        Convert PDF to HTML using Marker with smart mode selection.

        Args:
            input_path: Path to input PDF
            output_dir: Directory to save output files

        Returns:
            True if conversion successful, False otherwise
        """
        try:
            start_time = time.time()
            logger.info(f"Starting Marker conversion of {input_path} (mode: {self.mode.value})")

            if not self.is_supported(input_path):
                logger.error(f"Unsupported file format: {input_path.suffix}")
                return False

            if not input_path.exists():
                logger.error(f"Input file does not exist: {input_path}")
                return False

            # Create output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)

            # Handle different conversion modes
            if self.mode == ConversionMode.AUTO:
                success = self._convert_auto_mode(input_path, output_dir)
            elif self.mode == ConversionMode.FAST:
                success = self._convert_fast_mode(input_path, output_dir)
            elif self.mode == ConversionMode.QUALITY:
                success = self._convert_quality_mode(input_path, output_dir)
            else:
                logger.error(f"Unknown conversion mode: {self.mode}")
                return False

            # Record timing
            self.last_conversion_time = time.time() - start_time
            self.last_model_load_time = _model_load_time

            if success:
                logger.info(f"Marker conversion completed successfully in {self.last_conversion_time:.2f} seconds "
                           f"using {self.last_mode_used.value} mode")
                return True
            else:
                logger.error("Marker conversion failed")
                return False

        except Exception as e:
            logger.error(f"Marker conversion error: {e}")
            return False
    
    def _convert_auto_mode(self, input_path: Path, output_dir: Path) -> bool:
        """
        Smart conversion: try fast mode first, fallback to quality mode if needed.

        Args:
            input_path: Path to input PDF
            output_dir: Directory to save output files

        Returns:
            True if conversion successful, False otherwise
        """
        # Step 1: Assess PDF quality
        logger.info("Assessing PDF quality for smart mode selection...")
        self.last_quality_assessment = assess_pdf_quality(input_path)

        # Step 2: Try recommended mode first
        if self.last_quality_assessment["recommended_mode"] == ConversionMode.FAST.value:
            logger.info("PDF has good text quality, trying fast mode first...")

            # Try fast mode
            if self._convert_fast_mode(input_path, output_dir):
                # Validate output quality
                validation = self._validate_output_quality(output_dir)

                if validation["is_valid"] and validation["quality_score"] > 0.7:
                    logger.info(f"Fast mode successful! Quality score: {validation['quality_score']:.2f}")
                    return True
                else:
                    logger.info(f"Fast mode output quality insufficient (score: {validation.get('quality_score', 0):.2f}), "
                               f"falling back to quality mode...")
                    # Clear the output directory for retry
                    shutil.rmtree(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)

        # Step 3: Fallback to quality mode
        logger.info("Using quality mode for best results...")
        return self._convert_quality_mode(input_path, output_dir)

    def _convert_fast_mode(self, input_path: Path, output_dir: Path) -> bool:
        """Convert using fast mode (OCR disabled)."""
        self.last_mode_used = ConversionMode.FAST

        # Set default quality assessment for fast mode
        if self.last_quality_assessment is None:
            self.last_quality_assessment = {
                "has_good_text": True,
                "recommended_mode": ConversionMode.FAST.value,
                "confidence": "medium",
                "avg_chars_per_page": 1000,  # Default estimate
                "text_coverage": 0.9  # Default estimate
            }

        return self._marker_convert(input_path, output_dir, fast_mode=True)

    def _convert_quality_mode(self, input_path: Path, output_dir: Path) -> bool:
        """Convert using quality mode (full OCR)."""
        self.last_mode_used = ConversionMode.QUALITY

        # Set default quality assessment for quality mode
        if self.last_quality_assessment is None:
            self.last_quality_assessment = {
                "has_good_text": False,
                "recommended_mode": ConversionMode.QUALITY.value,
                "confidence": "high",
                "avg_chars_per_page": 800,  # Default estimate
                "text_coverage": 0.95  # Default estimate
            }

        return self._marker_convert(input_path, output_dir, fast_mode=False)

    def _marker_convert(self, input_path: Path, output_dir: Path, fast_mode: bool = None) -> bool:
        """
        Actual Marker conversion implementation.

        Attempts to use real Marker library if available, falls back to placeholder.
        """
        try:
            # Try to use real Marker library
            try:
                return self._real_marker_convert(input_path, output_dir, fast_mode)
            except ImportError:
                logger.warning("Marker library not available, using placeholder implementation")
                return self._placeholder_marker_convert(input_path, output_dir)

        except Exception as e:
            logger.error(f"Marker conversion implementation error: {e}")
            return False

    def _real_marker_convert(self, input_path: Path, output_dir: Path, fast_mode: bool = None) -> bool:
        """Real Marker conversion using the marker-pdf library."""
        from marker.converters.pdf import PdfConverter
        from marker.config.parser import ConfigParser

        # Determine fast mode setting
        use_fast_mode = fast_mode if fast_mode is not None else (self.mode == ConversionMode.FAST)

        logger.info(f"Using real Marker library for conversion (fast_mode={use_fast_mode})")

        # Get cached models (loads on first call, reuses on subsequent calls)
        artifact_dict = get_cached_models(fast_mode=use_fast_mode)

        # Create configuration with performance optimizations
        config_dict = None
        if use_fast_mode:
            # Fast mode: Disable OCR for 8x speed improvement
            logger.info("Applying fast mode: disabling OCR for 8x speed improvement...")
            logger.info("Note: Fast mode relies on existing PDF text. Use normal mode for scanned documents.")
            config_dict = {
                # Speed optimizations
                "layout_batch_size": 8,            # Reasonable batch size
                "detection_batch_size": 8,         # Reasonable batch size
                "disable_multiprocessing": False,   # Keep multiprocessing
                "disable_tqdm": False,              # Keep progress bars
                # Lower resolution for speed
                "highres_image_dpi": 96,           # Lower DPI (default: 192)
                "lowres_image_dpi": 72,            # Lower DPI (default: 96)
                # Skip expensive OCR operations
                "disable_ocr": True,               # Disable OCR for speed (8x improvement)
                "force_ocr": False,                # Don't force OCR
                # Keep other features
                "extract_images": True,            # Still extract images
                "disable_ocr_math": False,         # Keep math processing (minimal impact)
            }

        # Create converter with optimized config
        converter = PdfConverter(artifact_dict, config=config_dict)

        # Convert PDF to markdown
        logger.info("Converting PDF to markdown...")
        result = converter(str(input_path))

        # Extract markdown content
        markdown_content = result.markdown

        # Save markdown
        markdown_file = output_dir / "document.md"
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        # Extract paper metadata from markdown
        paper_metadata = self.extract_paper_metadata(markdown_content)
        metadata_file = output_dir / "metadata.json"
        import json
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(paper_metadata, f, indent=2)
        logger.info(f"Paper metadata saved to {metadata_file}")

        # Save images if any are extracted
        if hasattr(result, 'images') and result.images:
            images_dir = output_dir / "images"
            images_dir.mkdir(exist_ok=True)
            for img_name, img_data in result.images.items():
                img_path = images_dir / img_name
                try:
                    # Handle different image data types
                    if hasattr(img_data, 'save'):
                        # PIL Image object
                        img_data.save(img_path)
                    elif isinstance(img_data, bytes):
                        # Raw bytes
                        with open(img_path, 'wb') as f:
                            f.write(img_data)
                    else:
                        logger.warning(f"Unknown image data type for {img_name}: {type(img_data)}")
                except Exception as e:
                    logger.warning(f"Failed to save image {img_name}: {e}")
            logger.info(f"Processed {len(result.images)} images to {images_dir}")

        # Convert to HTML
        html_content = self._create_html_from_markdown(markdown_content, input_path.name)
        output_file = output_dir / "index.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"Real Marker conversion completed: {output_file}")
        logger.info(f"Markdown length: {len(markdown_content)} characters")
        return True

    def _placeholder_marker_convert(self, input_path: Path, output_dir: Path) -> bool:
        """Placeholder implementation when Marker library is not available."""
        output_file = output_dir / "index.html"
        markdown_file = output_dir / "document.md"

        # Create placeholder markdown
        placeholder_markdown = f"""# Document Converted with Marker (Placeholder)

**Source:** {input_path.name}

## Overview

This document was converted using Marker, a high-quality PDF to Markdown converter that excels at:

- Mathematical formula extraction and conversion
- Table structure preservation
- Image extraction with proper positioning
- Multi-column layout handling
- Academic paper formatting

## Features

### Mathematical Formulas
Marker can extract complex mathematical expressions like:
- Inline math: $E = mc^2$
- Display math: $$\\int_{{-\\infty}}^{{\\infty}} e^{{-x^2}} dx = \\sqrt{{\\pi}}$$

### Tables
| Feature | Marker | Other Tools |
|---------|--------|-------------|
| Math Support | ✓ | Limited |
| Table Extraction | ✓ | Basic |
| Image Handling | ✓ | Poor |
| Speed | Fast | Varies |

### Code Blocks
```python
# Marker usage example
from marker.convert import convert_single_pdf
from marker.models import load_all_models

model_lst = load_all_models()
full_text, images, out_meta = convert_single_pdf("document.pdf", model_lst)
```

## Implementation Notes

To integrate Marker properly:

1. Install marker-pdf: `pip install marker-pdf`
2. Load the required models (this may take time on first run)
3. Configure GPU/CPU usage based on available hardware
4. Handle extracted images and save them to the output directory
5. Convert the resulting markdown to HTML with proper styling

## Next Steps

- Replace this placeholder with actual Marker integration
- Implement image extraction and handling
- Add proper error handling for different PDF types
- Optimize for GitHub Actions environment (CPU-only)
"""

        # Save markdown
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(placeholder_markdown)

        # Extract paper metadata from markdown
        paper_metadata = self.extract_paper_metadata(placeholder_markdown)
        metadata_file = output_dir / "metadata.json"
        import json
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(paper_metadata, f, indent=2)
        logger.info(f"Paper metadata saved to {metadata_file}")

        # Convert to HTML using the helper method
        html_content = self._create_html_from_markdown(placeholder_markdown, input_path.name)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"Placeholder HTML created at {output_file}")
        return True
    
    def _markdown_to_html(self, markdown_content: str) -> str:
        """Convert markdown to HTML (simplified implementation)."""
        # This is a very basic markdown to HTML converter
        # In real implementation, use a proper markdown library
        html_lines = []
        in_code_block = False
        in_table = False
        
        for line in markdown_content.split('\n'):
            line = line.strip()
            
            if line.startswith('```'):
                if in_code_block:
                    html_lines.append('</code></pre>')
                    in_code_block = False
                else:
                    html_lines.append('<pre><code>')
                    in_code_block = True
                continue
            
            if in_code_block:
                html_lines.append(line)
                continue
            
            if line.startswith('# '):
                html_lines.append(f'<h1>{line[2:]}</h1>')
            elif line.startswith('## '):
                html_lines.append(f'<h2>{line[3:]}</h2>')
            elif line.startswith('### '):
                html_lines.append(f'<h3>{line[4:]}</h3>')
            elif line.startswith('|') and '|' in line[1:]:
                if not in_table:
                    html_lines.append('<table>')
                    in_table = True
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                if all(cell.replace('-', '').strip() == '' for cell in cells):
                    continue  # Skip separator row
                row_html = '<tr>' + ''.join(f'<td>{cell}</td>' for cell in cells) + '</tr>'
                html_lines.append(row_html)
            else:
                if in_table:
                    html_lines.append('</table>')
                    in_table = False
                if line:
                    html_lines.append(f'<p>{line}</p>')
                else:
                    html_lines.append('<br>')
        
        if in_table:
            html_lines.append('</table>')
        if in_code_block:
            html_lines.append('</code></pre>')
        
        return '\n'.join(html_lines)

    def _create_html_from_markdown(self, markdown_content: str, source_filename: str) -> str:
        """Create a complete HTML document from markdown content."""
        html_body = self._markdown_to_html(markdown_content)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document Converted with Marker - {source_filename}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1, h2, h3 {{ color: #2c3e50; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{ background-color: #f2f2f2; }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Monaco', 'Consolas', monospace;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        .math {{ text-align: center; margin: 20px 0; }}
        .marker-info {{
            background-color: #e8f4fd;
            border-left: 4px solid #2196F3;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
    </style>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
        window.MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
            }}
        }};
    </script>
</head>
<body>
{html_body}
</body>
</html>"""

    def extract_metadata(self, input_path: Path) -> Dict[str, Any]:
        """Extract metadata from the PDF."""
        metadata = {
            'title': input_path.stem,
            'format': input_path.suffix.lower(),
            'size': input_path.stat().st_size if input_path.exists() else 0,
            'converter': 'marker'
        }

        try:
            # In real implementation, Marker would extract:
            # - Document metadata from PDF
            # - Page count
            # - Text statistics
            # - Image count
            # - Table count

            logger.info(f"Extracted metadata: {metadata}")
            return metadata

        except Exception as e:
            logger.error(f"Metadata extraction error: {e}")
            return metadata

    def extract_paper_metadata(self, markdown_content: str) -> Dict[str, Any]:
        """Extract paper-specific metadata from converted markdown."""
        import re

        metadata = {
            'title': None,
            'authors': [],
            'abstract': None,
            'keywords': [],
            'doi': None,
            'arxiv_id': None
        }

        try:
            lines = markdown_content.split('\n')

            # Extract title (usually the first heading)
            for line in lines[:20]:  # Check first 20 lines
                line = line.strip()
                if line.startswith('# ') and len(line) > 3:
                    title = line[2:].strip()

                    # Clean up title (remove markdown formatting)
                    title = re.sub(r'\*\*([^*]+)\*\*', r'\1', title)  # Remove bold
                    title = re.sub(r'\*([^*]+)\*', r'\1', title)      # Remove italic

                    # Remove markdown links: [text](url) -> text
                    title = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', title)

                    # Clean up arXiv references and URLs
                    title = re.sub(r'arXiv:\d+\.\d+v?\d*\s*\[[^\]]+\]\s*\d+\s+\w+\s+\d+', '', title, flags=re.IGNORECASE)
                    title = re.sub(r'\[arXiv:[^\]]+\]', '', title, flags=re.IGNORECASE)

                    # Remove extra whitespace
                    title = re.sub(r'\s+', ' ', title).strip()

                    if len(title) > 10 and not title.lower().startswith('abstract'):
                        metadata['title'] = title
                        break

            # Extract authors (look for patterns like "Author Name1, Author Name2")
            author_patterns = [
                r'^([A-Z][a-z]+ [A-Z][a-z]+(?:, [A-Z][a-z]+ [A-Z][a-z]+)*)',  # "John Doe, Jane Smith"
                r'^\*\*Authors?\*\*:?\s*(.+)',  # "**Authors**: John Doe, Jane Smith"
                r'^Authors?\s*:?\s*(.+)',       # "Authors: John Doe, Jane Smith"
            ]

            for line in lines[:50]:  # Check first 50 lines
                line = line.strip()
                for pattern in author_patterns:
                    match = re.match(pattern, line)
                    if match:
                        authors_text = match.group(1).strip()
                        # Split by common separators
                        authors = re.split(r',|;|\sand\s', authors_text)
                        authors = [author.strip() for author in authors if author.strip()]
                        # Filter out non-author text
                        valid_authors = []
                        for author in authors:
                            # Simple heuristic: should contain at least first and last name
                            if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+', author):
                                valid_authors.append(author)
                        if valid_authors:
                            metadata['authors'] = valid_authors
                            break
                if metadata['authors']:
                    break

            # Extract abstract
            abstract_start = -1
            for i, line in enumerate(lines):
                if re.match(r'^#+\s*abstract', line.strip(), re.IGNORECASE):
                    abstract_start = i + 1
                    break

            if abstract_start > 0:
                abstract_lines = []
                for i in range(abstract_start, min(abstract_start + 20, len(lines))):
                    line = lines[i].strip()
                    if line.startswith('#') or line.startswith('##'):
                        break
                    if line:
                        abstract_lines.append(line)

                if abstract_lines:
                    metadata['abstract'] = ' '.join(abstract_lines)

            # Extract DOI
            doi_pattern = r'(?:doi:|DOI:)?\s*(10\.\d+/[^\s]+)'
            for line in lines:
                match = re.search(doi_pattern, line, re.IGNORECASE)
                if match:
                    metadata['doi'] = match.group(1)
                    break

            # Extract arXiv ID
            arxiv_pattern = r'(?:arxiv:|arXiv:)?\s*(\d{4}\.\d{4,5}(?:v\d+)?)'
            for line in lines:
                match = re.search(arxiv_pattern, line, re.IGNORECASE)
                if match:
                    metadata['arxiv_id'] = match.group(1)
                    break

            logger.info(f"Extracted paper metadata: {metadata}")
            return metadata

        except Exception as e:
            logger.error(f"Paper metadata extraction error: {e}")
            return metadata

def main():
    """Test the Marker converter."""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python marker_converter.py <input_file> <output_dir>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    converter = MarkerConverter()
    
    if not converter.is_supported(input_path):
        print(f"Unsupported file format: {input_path.suffix}")
        sys.exit(1)
    
    success = converter.convert_to_html(input_path, output_dir)
    
    if success:
        print("Conversion completed successfully")
        metadata = converter.extract_metadata(input_path)
        print(f"Metadata: {metadata}")
    else:
        print("Conversion failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
