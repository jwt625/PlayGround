#!/usr/bin/env python3
"""
Unit tests for the Marker converter.
Tests both the placeholder implementation and prepares for real Marker integration.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from marker_converter import MarkerConverter


class TestMarkerConverter:
    """Test suite for MarkerConverter class."""
    
    @pytest.fixture
    def converter(self):
        """Create a MarkerConverter instance for testing."""
        return MarkerConverter()
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_pdf_path(self):
        """Path to the sample PDF for testing."""
        return Path(__file__).parent / "pdf" / "2508.19977v1.pdf"
    
    @pytest.fixture
    def fake_pdf_path(self, temp_dir):
        """Create a fake PDF file for testing."""
        fake_pdf = temp_dir / "fake.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4\n%fake pdf content")
        return fake_pdf

    def test_initialization(self, converter):
        """Test MarkerConverter initialization."""
        assert converter.supported_formats == {'.pdf'}
        assert isinstance(converter, MarkerConverter)
    
    def test_is_supported_pdf(self, converter):
        """Test that PDF files are supported."""
        pdf_path = Path("test.pdf")
        assert converter.is_supported(pdf_path) is True
    
    def test_is_supported_case_insensitive(self, converter):
        """Test that file extension checking is case insensitive."""
        pdf_path = Path("test.PDF")
        assert converter.is_supported(pdf_path) is True
    
    def test_is_not_supported_docx(self, converter):
        """Test that non-PDF files are not supported."""
        docx_path = Path("test.docx")
        assert converter.is_supported(docx_path) is False
    
    def test_is_not_supported_txt(self, converter):
        """Test that text files are not supported."""
        txt_path = Path("test.txt")
        assert converter.is_supported(txt_path) is False

    def test_convert_nonexistent_file(self, converter, temp_dir):
        """Test conversion with non-existent input file."""
        nonexistent = Path("nonexistent.pdf")
        result = converter.convert_to_html(nonexistent, temp_dir)
        assert result is False
    
    def test_convert_unsupported_format(self, converter, temp_dir):
        """Test conversion with unsupported file format."""
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("test content")
        result = converter.convert_to_html(txt_file, temp_dir)
        assert result is False
    
    def test_convert_creates_output_directory(self, converter, fake_pdf_path):
        """Test that conversion creates output directory if it doesn't exist."""
        output_dir = fake_pdf_path.parent / "new_output_dir"
        assert not output_dir.exists()
        
        result = converter.convert_to_html(fake_pdf_path, output_dir)
        
        assert result is True
        assert output_dir.exists()
        assert output_dir.is_dir()
    
    def test_convert_placeholder_success(self, converter, fake_pdf_path, temp_dir):
        """Test successful conversion with placeholder implementation."""
        result = converter.convert_to_html(fake_pdf_path, temp_dir)
        
        assert result is True
        
        # Check that output files are created
        html_file = temp_dir / "index.html"
        markdown_file = temp_dir / "document.md"
        
        assert html_file.exists()
        assert markdown_file.exists()
        
        # Check file contents
        html_content = html_file.read_text()
        markdown_content = markdown_file.read_text()
        
        assert "Document Converted with Marker" in html_content
        assert "Document Converted with Marker" in markdown_content
        assert fake_pdf_path.name in markdown_content
    
    def test_convert_real_pdf_placeholder(self, converter, sample_pdf_path, temp_dir):
        """Test conversion with real PDF file (placeholder implementation)."""
        if not sample_pdf_path.exists():
            pytest.skip(f"Sample PDF not found at {sample_pdf_path}")
        
        result = converter.convert_to_html(sample_pdf_path, temp_dir)
        
        assert result is True
        
        # Check output files
        html_file = temp_dir / "index.html"
        markdown_file = temp_dir / "document.md"
        
        assert html_file.exists()
        assert markdown_file.exists()
        
        # Verify content includes PDF name
        markdown_content = markdown_file.read_text()
        assert sample_pdf_path.name in markdown_content
    
    def test_extract_metadata_basic(self, converter, fake_pdf_path):
        """Test basic metadata extraction."""
        metadata = converter.extract_metadata(fake_pdf_path)
        
        assert metadata['title'] == fake_pdf_path.stem
        assert metadata['format'] == '.pdf'
        assert metadata['converter'] == 'marker'
        assert metadata['size'] > 0
    
    def test_extract_metadata_nonexistent_file(self, converter):
        """Test metadata extraction with non-existent file."""
        nonexistent = Path("nonexistent.pdf")
        metadata = converter.extract_metadata(nonexistent)
        
        assert metadata['title'] == 'nonexistent'
        assert metadata['format'] == '.pdf'
        assert metadata['converter'] == 'marker'
        assert metadata['size'] == 0
    
    def test_markdown_to_html_headers(self, converter):
        """Test markdown to HTML conversion for headers."""
        markdown = "# Header 1\n## Header 2\n### Header 3"
        html = converter._markdown_to_html(markdown)
        
        assert '<h1>Header 1</h1>' in html
        assert '<h2>Header 2</h2>' in html
        assert '<h3>Header 3</h3>' in html
    
    def test_markdown_to_html_table(self, converter):
        """Test markdown to HTML conversion for tables."""
        markdown = """| Col1 | Col2 |
|------|------|
| A    | B    |
| C    | D    |"""
        html = converter._markdown_to_html(markdown)
        
        assert '<table>' in html
        assert '</table>' in html
        assert '<tr>' in html
        assert '<td>A</td>' in html
        assert '<td>B</td>' in html
    
    def test_markdown_to_html_code_block(self, converter):
        """Test markdown to HTML conversion for code blocks."""
        markdown = """```python
print("hello")
```"""
        html = converter._markdown_to_html(markdown)
        
        assert '<pre><code>' in html
        assert '</code></pre>' in html
        assert 'print("hello")' in html
    
    def test_html_output_structure(self, converter, fake_pdf_path, temp_dir):
        """Test that HTML output has proper structure."""
        converter.convert_to_html(fake_pdf_path, temp_dir)
        
        html_file = temp_dir / "index.html"
        html_content = html_file.read_text()
        
        # Check HTML structure
        assert '<!DOCTYPE html>' in html_content
        assert '<html lang="en">' in html_content
        assert '<head>' in html_content
        assert '<body>' in html_content
        assert '</html>' in html_content
        
        # Check MathJax integration
        assert 'MathJax' in html_content
        assert 'tex-mml-chtml.js' in html_content
    
    def test_html_output_styling(self, converter, fake_pdf_path, temp_dir):
        """Test that HTML output includes proper styling."""
        converter.convert_to_html(fake_pdf_path, temp_dir)
        
        html_file = temp_dir / "index.html"
        html_content = html_file.read_text()
        
        # Check CSS styling
        assert '<style>' in html_content
        assert 'font-family:' in html_content
        assert 'max-width: 800px' in html_content
        assert 'border-collapse: collapse' in html_content


class TestMarkerConverterIntegration:
    """Integration tests for MarkerConverter with real Marker library."""

    @pytest.fixture
    def converter(self):
        return MarkerConverter()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_pdf_path(self):
        return Path(__file__).parent / "pdf" / "2508.19977v1.pdf"
    
    @pytest.mark.integration
    @pytest.mark.skipif(not Path(__file__).parent.joinpath("pdf", "2508.19977v1.pdf").exists(),
                       reason="Sample PDF not available")
    def test_real_marker_conversion(self, converter, sample_pdf_path, temp_dir):
        """Test conversion with real Marker library (when available)."""
        # This test will be skipped unless real Marker is installed
        # and the sample PDF is available
        
        try:
            import marker
            # If marker is available, test real conversion
            # This would replace the placeholder implementation
            pytest.skip("Real Marker integration not yet implemented")
        except ImportError:
            pytest.skip("Marker library not installed")


class TestMarkerConverterErrorHandling:
    """Test error handling and edge cases."""

    @pytest.fixture
    def converter(self):
        return MarkerConverter()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_convert_with_exception_in_marker_convert(self, converter, temp_dir):
        """Test handling of exceptions in _marker_convert method."""
        fake_pdf = temp_dir / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4\n")
        
        # Mock _marker_convert to raise an exception
        with patch.object(converter, '_marker_convert', side_effect=Exception("Test error")):
            result = converter.convert_to_html(fake_pdf, temp_dir)
            assert result is False
    
    def test_metadata_extraction_with_exception(self, converter, temp_dir):
        """Test metadata extraction when an exception occurs."""
        fake_pdf = temp_dir / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4\n")
        
        # The current implementation should handle exceptions gracefully
        metadata = converter.extract_metadata(fake_pdf)
        
        # Should return basic metadata even if extraction fails
        assert 'title' in metadata
        assert 'converter' in metadata
        assert metadata['converter'] == 'marker'


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
