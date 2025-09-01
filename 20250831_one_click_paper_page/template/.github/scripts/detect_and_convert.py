#!/usr/bin/env python3
"""
Detect input file type and convert to HTML using appropriate converter.
"""

import os
import sys
import glob
from pathlib import Path
from typing import Optional, Tuple
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_input_type() -> Tuple[str, Optional[Path]]:
    """
    Detect the type of input file in the repository.
    Returns: (file_type, file_path)
    """
    # Check for different file types in order of preference
    
    # Check for LaTeX files
    tex_files = list(Path('.').glob('*.tex'))
    if tex_files:
        main_tex = None
        for tex_file in tex_files:
            # Look for main.tex or files with \documentclass
            if tex_file.name == 'main.tex':
                main_tex = tex_file
                break
            try:
                with open(tex_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if '\\documentclass' in content:
                        main_tex = tex_file
                        break
            except Exception:
                continue
        
        if main_tex:
            logger.info(f"Detected LaTeX file: {main_tex}")
            return "latex", main_tex
    
    # Check for ZIP files (could be LaTeX or other)
    zip_files = list(Path('.').glob('*.zip'))
    if zip_files:
        logger.info(f"Detected ZIP file: {zip_files[0]}")
        return "zip", zip_files[0]
    
    # Check for DOCX files
    docx_files = list(Path('.').glob('*.docx'))
    if docx_files:
        logger.info(f"Detected DOCX file: {docx_files[0]}")
        return "docx", docx_files[0]
    
    # Check for PDF files
    pdf_files = list(Path('.').glob('*.pdf'))
    if pdf_files:
        logger.info(f"Detected PDF file: {pdf_files[0]}")
        return "pdf", pdf_files[0]
    
    # Check for Overleaf Git repository (look for .git and .tex files)
    if Path('.git').exists() and tex_files:
        logger.info("Detected Overleaf Git repository")
        return "overleaf", tex_files[0] if tex_files else None
    
    logger.error("No supported input files found")
    return "unknown", None

def convert_with_docling(input_path: Path, output_dir: Path) -> bool:
    """Convert using Docling."""
    try:
        logger.info(f"Converting {input_path} with Docling...")
        # This is a placeholder - actual Docling integration would go here
        # For now, we'll use a simple conversion approach
        return False  # Fallback to other methods
    except Exception as e:
        logger.error(f"Docling conversion failed: {e}")
        return False

def convert_with_marker(input_path: Path, output_dir: Path) -> bool:
    """Convert using Marker with optimized settings."""
    try:
        logger.info(f"Converting {input_path} with Marker...")

        # Import marker components
        from marker.models import create_model_dict
        from marker.converters.pdf import PdfConverter
        from marker.settings import settings
        import time

        # Get cached models for performance
        logger.info("Loading Marker models...")
        start_time = time.time()
        model_dict = create_model_dict()
        load_time = time.time() - start_time
        logger.info(f"Marker models loaded in {load_time:.2f} seconds")

        # Create converter with smart mode (fast by default, quality fallback)
        converter = PdfConverter(
            artifact_dict=model_dict,
            processor_config={
                "ocr": False,  # Start with fast mode
                "extract_images": True,
            }
        )

        # Convert PDF
        logger.info("Starting PDF conversion...")
        conversion_start = time.time()
        result = converter(str(input_path))
        conversion_time = time.time() - conversion_start
        logger.info(f"Conversion completed in {conversion_time:.2f} seconds")

        # Save markdown content
        markdown_file = output_dir / "document.md"
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(result.text_content)

        # Save images
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)

        for img_name, img_data in result.images.items():
            img_path = images_dir / img_name
            try:
                if hasattr(img_data, 'save'):
                    # PIL Image object
                    img_data.save(img_path)
                elif isinstance(img_data, bytes):
                    # Raw bytes
                    with open(img_path, 'wb') as f:
                        f.write(img_data)
                logger.info(f"Saved image: {img_name}")
            except Exception as e:
                logger.warning(f"Failed to save image {img_name}: {e}")

        # Generate HTML from markdown
        html_content = generate_html_from_markdown(result.text_content, images_dir)
        html_file = output_dir / "index.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"Marker conversion successful: {len(result.text_content)} characters, {len(result.images)} images")
        return True

    except ImportError:
        logger.warning("Marker library not available, falling back to other methods")
        return False
    except Exception as e:
        logger.error(f"Marker conversion failed: {e}")
        return False

def convert_with_pandoc(input_path: Path, output_dir: Path, file_type: str) -> bool:
    """Convert using Pandoc as fallback."""
    try:
        logger.info(f"Converting {input_path} with Pandoc...")
        output_file = output_dir / "index.html"
        
        if file_type == "latex":
            cmd = ["pandoc", str(input_path), "-o", str(output_file), "--standalone", "--mathjax"]
        elif file_type == "docx":
            cmd = ["pandoc", str(input_path), "-o", str(output_file), "--standalone", "--mathjax"]
        else:
            logger.error(f"Pandoc doesn't support {file_type} directly")
            return False
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("Pandoc conversion successful")
            return True
        else:
            logger.error(f"Pandoc failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Pandoc conversion failed: {e}")
        return False

def generate_html_from_markdown(markdown_content: str, images_dir: Path) -> str:
    """Generate HTML from markdown content with academic styling."""
    # Basic markdown to HTML conversion with MathJax support
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Academic Paper</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']]
            }
        };
    </script>
    <style>
        body {
            font-family: 'Times New Roman', serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
            margin-top: 1.5em;
        }
        h1 {
            text-align: center;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        pre {
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
        blockquote {
            border-left: 4px solid #3498db;
            margin: 0;
            padding-left: 20px;
            font-style: italic;
        }
    </style>
</head>
<body>
{content}
</body>
</html>"""

    # Simple markdown to HTML conversion
    html_content = markdown_content

    # Convert headers
    html_content = html_content.replace('# ', '<h1>').replace('\n', '</h1>\n', 1)
    html_content = html_content.replace('## ', '<h2>').replace('\n', '</h2>\n')
    html_content = html_content.replace('### ', '<h3>').replace('\n', '</h3>\n')

    # Convert paragraphs
    paragraphs = html_content.split('\n\n')
    html_paragraphs = []
    for para in paragraphs:
        if para.strip() and not para.startswith('<h'):
            html_paragraphs.append(f'<p>{para.strip()}</p>')
        else:
            html_paragraphs.append(para)

    html_content = '\n\n'.join(html_paragraphs)

    return html_template.format(content=html_content)

def main():
    """Main conversion logic."""
    # Create output directory
    output_dir = Path("dist")
    output_dir.mkdir(exist_ok=True)
    
    # Detect input type
    file_type, input_path = detect_input_type()
    
    if file_type == "unknown" or input_path is None:
        logger.error("No supported input files found")
        sys.exit(1)
    
    # Try conversion methods in order - prioritize Marker for PDFs
    success = False

    if file_type == "pdf":
        # For PDFs, try Marker first (best quality and performance)
        logger.info("PDF detected - using optimized Marker converter")
        success = convert_with_marker(input_path, output_dir)

        if not success:
            logger.info("Marker failed, trying Docling fallback")
            success = convert_with_docling(input_path, output_dir)

    elif file_type == "docx":
        # For DOCX, try Docling first, then Marker
        success = convert_with_docling(input_path, output_dir)
        if not success:
            success = convert_with_marker(input_path, output_dir)

    elif file_type in ["latex", "tex"]:
        # For LaTeX, use Pandoc
        success = convert_with_pandoc(input_path, output_dir, file_type)

    # Final fallback to Pandoc for supported formats
    if not success and file_type in ["latex", "docx"]:
        logger.info("Trying Pandoc as final fallback")
        success = convert_with_pandoc(input_path, output_dir, file_type)

    if not success:
        logger.error("All conversion methods failed")
        sys.exit(1)

    logger.info("Conversion completed successfully")

    # Log conversion results
    output_files = list(output_dir.glob("*"))
    logger.info(f"Generated {len(output_files)} output files:")
    for file_path in output_files:
        logger.info(f"  - {file_path.name}")

    # Set GitHub Actions output for next steps
    if (output_dir / "index.html").exists():
        print("::set-output name=conversion_success::true")
        print(f"::set-output name=output_dir::{output_dir}")
    else:
        print("::set-output name=conversion_success::false")

if __name__ == "__main__":
    main()
