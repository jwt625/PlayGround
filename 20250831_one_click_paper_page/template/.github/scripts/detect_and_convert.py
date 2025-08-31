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
    """Convert using Marker."""
    try:
        logger.info(f"Converting {input_path} with Marker...")
        # This is a placeholder - actual Marker integration would go here
        return False  # Fallback to other methods
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
    
    # Try conversion methods in order
    success = False
    
    if file_type in ["pdf", "docx"]:
        # Try Docling first for PDF/DOCX
        success = convert_with_docling(input_path, output_dir)
        if not success:
            success = convert_with_marker(input_path, output_dir)
    
    if not success and file_type in ["latex", "docx"]:
        # Try Pandoc as fallback
        success = convert_with_pandoc(input_path, output_dir, file_type)
    
    if not success:
        logger.error("All conversion methods failed")
        sys.exit(1)
    
    logger.info("Conversion completed successfully")

if __name__ == "__main__":
    main()
