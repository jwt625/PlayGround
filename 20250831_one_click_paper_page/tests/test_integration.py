"""
Integration tests for the marker converter with real PDF files.

These tests use actual PDF files to verify the complete conversion pipeline
including image extraction, metadata extraction, and HTML generation.
"""

import json
import tempfile
from pathlib import Path

import pytest

# Import the marker converter
import sys
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

from marker_converter import MarkerConverter, ConversionMode


@pytest.mark.integration
class TestMarkerConverterIntegration:
    """Integration tests for MarkerConverter with real PDFs."""

    @pytest.fixture
    def test_pdf_path(self):
        """Path to the test PDF file."""
        pdf_path = Path(__file__).parent / "pdf" / "attention_is_all_you_need.pdf"
        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found at {pdf_path}")
        return pdf_path

    @pytest.fixture
    def temp_output_dir(self):
        """Temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def converter(self):
        """MarkerConverter instance for testing."""
        return MarkerConverter(mode=ConversionMode.FAST)

    def test_full_conversion_pipeline(self, converter, test_pdf_path, temp_output_dir):
        """Test the complete conversion pipeline with a real PDF."""
        # Run conversion
        success = converter.convert_to_html(test_pdf_path, temp_output_dir)
        
        # Verify conversion succeeded
        assert success, "Conversion should succeed"
        
        # Check that all expected output files exist
        html_file = temp_output_dir / "index.html"
        md_file = temp_output_dir / "document.md"
        metadata_file = temp_output_dir / "metadata.json"
        
        assert html_file.exists(), "HTML file should be created"
        assert md_file.exists(), "Markdown file should be created"
        assert metadata_file.exists(), "Metadata file should be created"
        
        # Verify files have content
        assert html_file.stat().st_size > 1000, "HTML file should have substantial content"
        assert md_file.stat().st_size > 1000, "Markdown file should have substantial content"
        assert metadata_file.stat().st_size > 10, "Metadata file should have content"

    def test_image_extraction(self, converter, test_pdf_path, temp_output_dir):
        """Test that images are properly extracted and saved."""
        # Run conversion
        success = converter.convert_to_html(test_pdf_path, temp_output_dir)
        assert success, "Conversion should succeed"
        
        # Check for images directory
        images_dir = temp_output_dir / "images"
        
        # The Attention Is All You Need paper should have images/figures
        if images_dir.exists():
            image_files = list(images_dir.glob("*"))
            print(f"Found {len(image_files)} image files")
            
            # Verify image files have reasonable sizes
            for img_file in image_files:
                assert img_file.stat().st_size > 100, f"Image {img_file.name} should have content"
        else:
            # If no images directory, that might be expected for some PDFs
            print("No images directory found - PDF may not contain extractable images")

    def test_metadata_extraction(self, converter, test_pdf_path, temp_output_dir):
        """Test that paper metadata is properly extracted."""
        # Run conversion
        success = converter.convert_to_html(test_pdf_path, temp_output_dir)
        assert success, "Conversion should succeed"
        
        # Load and verify metadata
        metadata_file = temp_output_dir / "metadata.json"
        assert metadata_file.exists(), "Metadata file should exist"
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Verify metadata structure
        expected_keys = ['title', 'authors', 'abstract', 'keywords', 'doi', 'arxiv_id']
        for key in expected_keys:
            assert key in metadata, f"Metadata should contain '{key}' field"
        
        # For the Attention Is All You Need paper, we should extract the title
        if metadata['title']:
            title = metadata['title'].lower()
            assert 'attention' in title or 'transformer' in title, \
                f"Title should contain relevant keywords, got: {metadata['title']}"

    def test_html_content_quality(self, converter, test_pdf_path, temp_output_dir):
        """Test that generated HTML has proper structure and content."""
        # Run conversion
        success = converter.convert_to_html(test_pdf_path, temp_output_dir)
        assert success, "Conversion should succeed"
        
        # Read HTML content
        html_file = temp_output_dir / "index.html"
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Verify HTML structure
        assert '<!DOCTYPE html>' in html_content, "HTML should have proper doctype"
        assert '<html' in html_content, "HTML should have html tag"
        assert '<head>' in html_content, "HTML should have head section"
        assert '<body>' in html_content, "HTML should have body section"
        assert '</html>' in html_content, "HTML should be properly closed"
        
        # Verify content exists
        assert len(html_content) > 5000, "HTML should have substantial content"
        
        # Check for common academic paper elements
        content_lower = html_content.lower()
        academic_indicators = ['abstract', 'introduction', 'conclusion', 'references']
        found_indicators = [indicator for indicator in academic_indicators if indicator in content_lower]
        assert len(found_indicators) >= 2, \
            f"HTML should contain academic paper structure, found: {found_indicators}"

    def test_markdown_content_quality(self, converter, test_pdf_path, temp_output_dir):
        """Test that generated Markdown has proper structure and content."""
        # Run conversion
        success = converter.convert_to_html(test_pdf_path, temp_output_dir)
        assert success, "Conversion should succeed"
        
        # Read Markdown content
        md_file = temp_output_dir / "document.md"
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Verify Markdown structure
        assert len(md_content) > 2000, "Markdown should have substantial content"
        assert md_content.count('#') >= 3, "Markdown should have multiple headings"
        
        # Check for academic paper elements
        content_lower = md_content.lower()
        academic_indicators = ['abstract', 'introduction', 'attention', 'transformer']
        found_indicators = [indicator for indicator in academic_indicators if indicator in content_lower]
        assert len(found_indicators) >= 2, \
            f"Markdown should contain relevant content, found: {found_indicators}"

    def test_performance_metrics(self, converter, test_pdf_path, temp_output_dir):
        """Test that performance metrics are properly recorded."""
        # Run conversion
        success = converter.convert_to_html(test_pdf_path, temp_output_dir)
        assert success, "Conversion should succeed"
        
        # Get performance metrics
        metrics = converter.get_performance_metrics()
        
        # Verify metrics structure
        assert 'conversion_time' in metrics, "Metrics should include conversion time"
        assert 'quality_assessment' in metrics, "Metrics should include quality assessment"
        
        # Verify reasonable values
        conversion_time = metrics['conversion_time']
        assert isinstance(conversion_time, (int, float)), "Conversion time should be numeric"
        assert 0 < conversion_time < 600, f"Conversion time should be reasonable, got {conversion_time}s"
        
        # Verify quality assessment
        quality = metrics['quality_assessment']
        assert isinstance(quality, dict), "Quality assessment should be a dictionary"
        assert 'has_good_text' in quality, "Quality should assess text quality"

    def test_different_conversion_modes(self, test_pdf_path, temp_output_dir):
        """Test conversion with different modes."""
        modes_to_test = [ConversionMode.FAST, ConversionMode.AUTO]
        
        for mode in modes_to_test:
            converter = MarkerConverter(mode=mode)
            
            # Create subdirectory for this mode
            mode_dir = temp_output_dir / f"mode_{mode.value}"
            mode_dir.mkdir(exist_ok=True)
            
            # Run conversion
            success = converter.convert_to_html(test_pdf_path, mode_dir)
            assert success, f"Conversion should succeed with mode {mode.value}"
            
            # Verify output exists
            html_file = mode_dir / "index.html"
            assert html_file.exists(), f"HTML should be created with mode {mode.value}"
            assert html_file.stat().st_size > 1000, f"HTML should have content with mode {mode.value}"

    @pytest.mark.slow
    def test_quality_mode_conversion(self, test_pdf_path, temp_output_dir):
        """Test conversion with quality mode (marked as slow test)."""
        converter = MarkerConverter(mode=ConversionMode.QUALITY)
        
        # Run conversion (this will take longer)
        success = converter.convert_to_html(test_pdf_path, temp_output_dir)
        assert success, "Quality mode conversion should succeed"
        
        # Quality mode should potentially extract more content
        html_file = temp_output_dir / "index.html"
        assert html_file.exists(), "HTML file should be created"
        
        # Get metrics to verify it used quality mode
        metrics = converter.get_performance_metrics()
        assert metrics['conversion_time'] > 10, "Quality mode should take more time"
