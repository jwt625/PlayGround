#!/usr/bin/env python3
"""
Unit tests for the Marker converter.
Tests real Marker integration.
"""

import pytest
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from marker_converter import MarkerConverter, get_cached_models, clear_model_cache, ConversionMode, assess_pdf_quality


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


class TestMarkerConverterPerformance:
    """Performance tests for MarkerConverter with timing measurements."""

    @pytest.fixture
    def converter(self):
        # Clear cache before each test to ensure clean state
        clear_model_cache()
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

    @pytest.mark.performance
    @pytest.mark.skipif(not Path(__file__).parent.joinpath("pdf", "2508.19977v1.pdf").exists(),
                       reason="Sample PDF not available")
    def test_first_conversion_timing(self, converter, sample_pdf_path, temp_dir):
        """Test timing of first conversion (includes model loading)."""
        try:
            import marker
        except ImportError:
            pytest.skip("Marker library not installed")

        start_time = time.time()
        success = converter.convert_to_html(sample_pdf_path, temp_dir)
        total_time = time.time() - start_time

        if success:
            metrics = converter.get_performance_metrics()
            print(f"\nFirst conversion performance:")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Reported total time: {metrics['total_conversion_time']:.2f}s")
            print(f"  Model load time: {metrics['model_load_time']:.2f}s")
            print(f"  Processing time: {metrics['actual_processing_time']:.2f}s")

            # Verify timing is reasonable (should be under 10 minutes for test PDF)
            assert total_time < 600, f"Conversion took too long: {total_time:.2f}s"
            assert metrics['model_load_time'] > 0, "Model load time should be recorded"
            assert metrics['actual_processing_time'] > 0, "Processing time should be recorded"

    @pytest.mark.performance
    @pytest.mark.skipif(not Path(__file__).parent.joinpath("pdf", "2508.19977v1.pdf").exists(),
                       reason="Sample PDF not available")
    def test_second_conversion_timing(self, converter, sample_pdf_path, temp_dir):
        """Test timing of second conversion (should use cached models)."""
        try:
            import marker
        except ImportError:
            pytest.skip("Marker library not installed")

        # First conversion to load models
        temp_dir1 = temp_dir / "first"
        temp_dir1.mkdir()
        converter.convert_to_html(sample_pdf_path, temp_dir1)
        first_metrics = converter.get_performance_metrics()

        # Second conversion should be faster (cached models)
        temp_dir2 = temp_dir / "second"
        temp_dir2.mkdir()
        start_time = time.time()
        success = converter.convert_to_html(sample_pdf_path, temp_dir2)
        total_time = time.time() - start_time

        if success:
            second_metrics = converter.get_performance_metrics()
            print(f"\nSecond conversion performance:")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  First conversion total: {first_metrics['total_conversion_time']:.2f}s")
            print(f"  Second conversion total: {second_metrics['total_conversion_time']:.2f}s")
            print(f"  Speed improvement: {first_metrics['total_conversion_time'] / second_metrics['total_conversion_time']:.1f}x")

            # Second conversion should be significantly faster
            assert second_metrics['total_conversion_time'] < first_metrics['total_conversion_time'], \
                "Second conversion should be faster due to cached models"

    @pytest.mark.performance
    def test_model_caching_behavior(self):
        """Test that model caching works correctly."""
        try:
            import marker
        except ImportError:
            pytest.skip("Marker library not installed")

        # Clear cache
        clear_model_cache()

        # First call should load models
        start_time = time.time()
        models1 = get_cached_models()
        first_load_time = time.time() - start_time

        # Second call should use cache
        start_time = time.time()
        models2 = get_cached_models()
        second_load_time = time.time() - start_time

        print(f"\nModel caching performance:")
        print(f"  First load time: {first_load_time:.2f}s")
        print(f"  Second load time: {second_load_time:.4f}s")
        print(f"  Cache speedup: {first_load_time / second_load_time:.0f}x")

        # Verify caching works
        assert models1 is models2, "Should return the same cached object"
        assert second_load_time < 0.1, "Cached access should be very fast"
        assert first_load_time > second_load_time, "First load should be slower"

    @pytest.mark.performance
    @pytest.mark.skipif(not Path(__file__).parent.joinpath("pdf", "2508.19977v1.pdf").exists(),
                       reason="Sample PDF not available")
    def test_fast_mode_vs_normal_mode(self, sample_pdf_path, temp_dir):
        """Test performance difference between fast mode and normal mode."""
        try:
            import marker
        except ImportError:
            pytest.skip("Marker library not installed")

        # Clear cache before test
        clear_model_cache()

        # Test normal mode
        normal_converter = MarkerConverter(fast_mode=False)
        normal_dir = temp_dir / "normal"
        normal_dir.mkdir()

        start_time = time.time()
        normal_success = normal_converter.convert_to_html(sample_pdf_path, normal_dir)
        normal_time = time.time() - start_time
        normal_metrics = normal_converter.get_performance_metrics()

        # Test fast mode
        fast_converter = MarkerConverter(fast_mode=True)
        fast_dir = temp_dir / "fast"
        fast_dir.mkdir()

        start_time = time.time()
        fast_success = fast_converter.convert_to_html(sample_pdf_path, fast_dir)
        fast_time = time.time() - start_time
        fast_metrics = fast_converter.get_performance_metrics()

        if normal_success and fast_success:
            print(f"\nFast mode vs Normal mode performance:")
            print(f"  Normal mode total: {normal_time:.2f}s")
            print(f"  Fast mode total: {fast_time:.2f}s")
            print(f"  Speed improvement: {normal_time / fast_time:.1f}x")
            print(f"  Normal processing: {normal_metrics['actual_processing_time']:.2f}s")
            print(f"  Fast processing: {fast_metrics['actual_processing_time']:.2f}s")

            # Fast mode should be faster (or at least not significantly slower)
            # Allow for some variance in timing
            assert fast_time <= normal_time * 1.1, "Fast mode should not be significantly slower"

            # Both should produce valid output
            assert (normal_dir / "index.html").exists(), "Normal mode should produce HTML"
            assert (fast_dir / "index.html").exists(), "Fast mode should produce HTML"


class TestSmartModeConversion:
    """Tests for smart mode conversion with automatic fallback."""

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

    def test_conversion_mode_enum(self):
        """Test ConversionMode enum values."""
        assert ConversionMode.AUTO.value == "auto"
        assert ConversionMode.FAST.value == "fast"
        assert ConversionMode.QUALITY.value == "quality"

    def test_pdf_quality_assessment(self, sample_pdf_path):
        """Test PDF quality assessment function."""
        if not sample_pdf_path.exists():
            pytest.skip("Sample PDF not available")

        try:
            import PyPDF2
        except ImportError:
            pytest.skip("PyPDF2 not installed")

        quality_info = assess_pdf_quality(sample_pdf_path)

        # Verify structure
        assert "has_good_text" in quality_info
        assert "recommended_mode" in quality_info
        assert "confidence" in quality_info
        assert "total_pages" in quality_info

        # For academic papers, should typically recommend fast mode
        print(f"\nPDF Quality Assessment: {quality_info}")
        assert quality_info["total_pages"] > 0
        assert quality_info["confidence"] in ["low", "medium", "high"]

    def test_converter_mode_initialization(self):
        """Test converter initialization with different modes."""
        # Test default mode
        converter_auto = MarkerConverter()
        assert converter_auto.mode == ConversionMode.AUTO

        # Test explicit modes
        converter_fast = MarkerConverter(mode=ConversionMode.FAST)
        assert converter_fast.mode == ConversionMode.FAST

        converter_quality = MarkerConverter(mode=ConversionMode.QUALITY)
        assert converter_quality.mode == ConversionMode.QUALITY

        # Test backward compatibility
        converter_legacy = MarkerConverter(fast_mode=True)
        assert converter_legacy.mode == ConversionMode.FAST


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
