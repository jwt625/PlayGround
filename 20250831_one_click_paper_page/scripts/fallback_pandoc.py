#!/usr/bin/env python3
"""
Pandoc-based fallback converter for various document formats.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
import tempfile
import shutil
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PandocConverter:
    """Fallback converter using Pandoc for various document formats."""
    
    def __init__(self):
        """Initialize the Pandoc converter."""
        self.supported_formats = {'.tex', '.latex', '.docx', '.doc', '.md', '.markdown', '.rst'}
        
    def is_supported(self, file_path: Path) -> bool:
        """Check if the file format is supported by Pandoc."""
        return file_path.suffix.lower() in self.supported_formats
    
    def is_pandoc_available(self) -> bool:
        """Check if Pandoc is available on the system."""
        try:
            result = subprocess.run(['pandoc', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def convert_to_html(self, input_path: Path, output_dir: Path, 
                       input_format: Optional[str] = None) -> bool:
        """
        Convert document to HTML using Pandoc.
        
        Args:
            input_path: Path to input document
            output_dir: Directory to save output files
            input_format: Optional input format specification
            
        Returns:
            True if conversion successful, False otherwise
        """
        try:
            logger.info(f"Starting Pandoc conversion of {input_path}")
            
            if not self.is_supported(input_path):
                logger.error(f"Unsupported file format: {input_path.suffix}")
                return False
            
            if not input_path.exists():
                logger.error(f"Input file does not exist: {input_path}")
                return False
            
            if not self.is_pandoc_available():
                logger.error("Pandoc is not available on the system")
                return False
            
            # Create output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Perform conversion
            success = self._pandoc_convert(input_path, output_dir, input_format)
            
            if success:
                logger.info("Pandoc conversion completed successfully")
                return True
            else:
                logger.error("Pandoc conversion failed")
                return False
                
        except Exception as e:
            logger.error(f"Pandoc conversion error: {e}")
            return False
    
    def _pandoc_convert(self, input_path: Path, output_dir: Path, 
                       input_format: Optional[str] = None) -> bool:
        """
        Actual Pandoc conversion implementation.
        """
        try:
            output_file = output_dir / "index.html"
            
            # Build Pandoc command
            cmd = ['pandoc']
            
            # Add input format if specified
            if input_format:
                cmd.extend(['-f', input_format])
            elif input_path.suffix.lower() in ['.tex', '.latex']:
                cmd.extend(['-f', 'latex'])
            elif input_path.suffix.lower() in ['.md', '.markdown']:
                cmd.extend(['-f', 'markdown'])
            elif input_path.suffix.lower() in ['.rst']:
                cmd.extend(['-f', 'rst'])
            
            # Add input file
            cmd.append(str(input_path))
            
            # Add output format and file
            cmd.extend(['-t', 'html5'])
            cmd.extend(['-o', str(output_file)])
            
            # Add common options
            cmd.extend([
                '--standalone',           # Create complete HTML document
                '--mathjax',             # Enable MathJax for math rendering
                '--toc',                 # Generate table of contents
                '--toc-depth=3',         # TOC depth
                '--number-sections',     # Number sections
                '--highlight-style=github',  # Code highlighting
                '--css=style.css',       # Reference to CSS file
                '--metadata', 'title=Academic Paper'
            ])
            
            # Add LaTeX-specific options
            if input_path.suffix.lower() in ['.tex', '.latex']:
                cmd.extend([
                    '--bibliography-style=ieee',  # Bibliography style
                    '--citeproc'                   # Process citations
                ])
            
            logger.info(f"Running Pandoc command: {' '.join(cmd)}")
            
            # Run Pandoc
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("Pandoc conversion successful")
                
                # Create a basic CSS file
                self._create_css_file(output_dir)
                
                # Post-process the HTML if needed
                self._post_process_html(output_file)
                
                return True
            else:
                logger.error(f"Pandoc conversion failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Pandoc conversion timed out")
            return False
        except Exception as e:
            logger.error(f"Pandoc conversion implementation error: {e}")
            return False
    
    def _create_css_file(self, output_dir: Path) -> None:
        """Create a basic CSS file for styling."""
        css_content = """
/* Academic Paper Styling */
body {
    font-family: 'Times New Roman', serif;
    line-height: 1.6;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    color: #333;
    background-color: #fff;
}

h1, h2, h3, h4, h5, h6 {
    color: #2c3e50;
    margin-top: 2em;
    margin-bottom: 1em;
}

h1 {
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
    text-align: center;
}

h2 {
    border-bottom: 1px solid #bdc3c7;
    padding-bottom: 5px;
}

/* Table of Contents */
#TOC {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 5px;
    padding: 20px;
    margin: 20px 0;
}

#TOC ul {
    list-style-type: none;
    padding-left: 0;
}

#TOC li {
    margin: 5px 0;
}

#TOC a {
    text-decoration: none;
    color: #007bff;
}

#TOC a:hover {
    text-decoration: underline;
}

/* Tables */
table {
    border-collapse: collapse;
    width: 100%;
    margin: 20px 0;
}

th, td {
    border: 1px solid #ddd;
    padding: 12px;
    text-align: left;
}

th {
    background-color: #f2f2f2;
    font-weight: bold;
}

/* Code blocks */
pre {
    background-color: #f4f4f4;
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 15px;
    overflow-x: auto;
    font-family: 'Monaco', 'Consolas', 'Courier New', monospace;
}

code {
    background-color: #f4f4f4;
    padding: 2px 4px;
    border-radius: 3px;
    font-family: 'Monaco', 'Consolas', 'Courier New', monospace;
}

/* Math */
.math {
    text-align: center;
    margin: 20px 0;
}

/* Figures */
figure {
    text-align: center;
    margin: 20px 0;
}

figcaption {
    font-style: italic;
    margin-top: 10px;
    color: #666;
}

/* Citations and references */
.references {
    margin-top: 40px;
    border-top: 2px solid #ddd;
    padding-top: 20px;
}

.references h1 {
    border-bottom: none;
    text-align: left;
}

/* Blockquotes */
blockquote {
    border-left: 4px solid #3498db;
    margin: 20px 0;
    padding: 10px 20px;
    background-color: #f8f9fa;
    font-style: italic;
}

/* Links */
a {
    color: #007bff;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

/* Print styles */
@media print {
    body {
        font-size: 12pt;
        line-height: 1.4;
    }
    
    h1, h2, h3 {
        page-break-after: avoid;
    }
    
    pre, blockquote {
        page-break-inside: avoid;
    }
}
"""
        
        css_file = output_dir / "style.css"
        with open(css_file, 'w', encoding='utf-8') as f:
            f.write(css_content)
        
        logger.info(f"CSS file created at {css_file}")
    
    def _post_process_html(self, html_file: Path) -> None:
        """Post-process the generated HTML file."""
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add viewport meta tag if not present
            if 'viewport' not in content:
                content = content.replace(
                    '<head>',
                    '<head>\n    <meta name="viewport" content="width=device-width, initial-scale=1.0">'
                )
            
            # Ensure MathJax is properly configured
            if 'MathJax' in content and 'window.MathJax' not in content:
                mathjax_config = """
    <script>
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
            }
        };
    </script>"""
                content = content.replace('</head>', f'{mathjax_config}\n</head>')
            
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info("HTML post-processing completed")
            
        except Exception as e:
            logger.warning(f"HTML post-processing failed: {e}")
    
    def extract_metadata(self, input_path: Path) -> Dict[str, Any]:
        """Extract metadata from the document."""
        metadata = {
            'title': input_path.stem,
            'format': input_path.suffix.lower(),
            'size': input_path.stat().st_size if input_path.exists() else 0,
            'converter': 'pandoc'
        }
        
        try:
            # Try to extract metadata using Pandoc
            if self.is_pandoc_available():
                cmd = ['pandoc', '--template=', str(input_path)]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    # Parse any metadata from the output
                    pass  # Implementation would depend on document format
            
            logger.info(f"Extracted metadata: {metadata}")
            return metadata
            
        except Exception as e:
            logger.error(f"Metadata extraction error: {e}")
            return metadata

def main():
    """Test the Pandoc converter."""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python fallback_pandoc.py <input_file> <output_dir> [input_format]")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    input_format = sys.argv[3] if len(sys.argv) > 3 else None
    
    converter = PandocConverter()
    
    if not converter.is_supported(input_path):
        print(f"Unsupported file format: {input_path.suffix}")
        sys.exit(1)
    
    success = converter.convert_to_html(input_path, output_dir, input_format)
    
    if success:
        print("Conversion completed successfully")
        metadata = converter.extract_metadata(input_path)
        print(f"Metadata: {metadata}")
    else:
        print("Conversion failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
