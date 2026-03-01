#!/usr/bin/env python3
"""
PDF to Markdown Evaluation Benchmark

Evaluates Marker and Docling for converting PDFs to Markdown.
Records: processing time, output size, success/failure, page count.
"""

import json
import os
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# Test documents from DevLog-000-03
TEST_DOCUMENTS = [
    ("nakamura_1999_charge_qubit", "papers/superconducting_qubits/ofs_batch"),
    ("krantz_2019_apr_quantum_engineers_guide", "papers/superconducting_qubits/reviews"),
    ("muschinske_2023_dolan_manhattan_jj_uniformity", "papers/superconducting_qubits/fabrication_processes"),
    ("place_2021_ncomms_tantalum_qubits", "papers/superconducting_qubits/materials_studies"),
    ("putterman_2025_bosonic_qec", "papers/superconducting_qubits/ofs_batch"),
    ("spietz_lafe_yale_2006", "theses/yale"),
    ("eichinger_michaela_copenhagen_2023", "theses/copenhagen"),
]

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RAW_DOCS_BASE = PROJECT_ROOT / "semiconductor_processing_dataset" / "raw_documents"
OUTPUT_BASE = PROJECT_ROOT / "semiconductor_processing_dataset" / "processed_documents" / "text_extracted"


@dataclass
class ConversionResult:
    document_id: str
    tool: str
    success: bool
    processing_time_seconds: float
    input_size_bytes: int
    output_size_bytes: Optional[int]
    page_count: Optional[int]
    error_message: Optional[str]
    timestamp: str


def get_pdf_path(doc_id: str, subdir: str) -> Path:
    """Get full path to PDF file."""
    return RAW_DOCS_BASE / subdir / f"{doc_id}.pdf"


def convert_with_marker(pdf_path: Path, output_dir: Path) -> tuple[str, Optional[int]]:
    """Convert PDF using Marker. Returns (markdown_text, page_count)."""
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    
    # Create models (CPU mode)
    model_dict = create_model_dict()
    converter = PdfConverter(artifact_dict=model_dict)
    
    # Convert
    result = converter(str(pdf_path))
    markdown_text = result.markdown
    page_count = len(result.pages) if hasattr(result, 'pages') else None
    
    return markdown_text, page_count


def convert_with_docling(pdf_path: Path, output_dir: Path) -> tuple[str, Optional[int]]:
    """Convert PDF using Docling. Returns (markdown_text, page_count)."""
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))

    markdown_text = result.document.export_to_markdown()
    # num_pages can be a method or property depending on version
    page_count = None
    if hasattr(result.document, 'num_pages'):
        np = result.document.num_pages
        page_count = np() if callable(np) else np

    return markdown_text, page_count


def run_conversion(doc_id: str, subdir: str, tool: str) -> ConversionResult:
    """Run a single conversion and record metrics."""
    pdf_path = get_pdf_path(doc_id, subdir)
    output_dir = OUTPUT_BASE / tool
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{doc_id}.md"
    
    input_size = pdf_path.stat().st_size
    
    start_time = time.time()
    try:
        if tool == "marker":
            markdown_text, page_count = convert_with_marker(pdf_path, output_dir)
        elif tool == "docling":
            markdown_text, page_count = convert_with_docling(pdf_path, output_dir)
        else:
            raise ValueError(f"Unknown tool: {tool}")
        
        # Write output
        output_file.write_text(markdown_text, encoding="utf-8")
        output_size = output_file.stat().st_size
        
        elapsed = time.time() - start_time
        return ConversionResult(
            document_id=doc_id, tool=tool, success=True,
            processing_time_seconds=round(elapsed, 2),
            input_size_bytes=input_size, output_size_bytes=output_size,
            page_count=page_count, error_message=None,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        elapsed = time.time() - start_time
        return ConversionResult(
            document_id=doc_id, tool=tool, success=False,
            processing_time_seconds=round(elapsed, 2),
            input_size_bytes=input_size, output_size_bytes=None,
            page_count=None, error_message=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            timestamp=datetime.now().isoformat()
        )


def run_benchmark(tools: list[str] = None, docs: list[tuple[str, str]] = None):
    """Run full benchmark and save results."""
    tools = tools or ["marker", "docling"]
    docs = docs or TEST_DOCUMENTS
    
    results = []
    results_file = OUTPUT_BASE / "evaluation_results.jsonl"
    
    print(f"Running benchmark: {len(docs)} documents × {len(tools)} tools")
    print(f"Results will be saved to: {results_file}")
    print("-" * 60)
    
    for doc_id, subdir in docs:
        for tool in tools:
            print(f"\n[{tool}] Processing: {doc_id}")
            result = run_conversion(doc_id, subdir, tool)
            results.append(result)
            
            if result.success:
                print(f"  ✓ Success: {result.processing_time_seconds}s, "
                      f"{result.output_size_bytes/1024:.1f} KB output")
            else:
                print(f"  ✗ Failed: {result.error_message[:100]}...")
            
            # Append to results file incrementally
            with open(results_file, "a") as f:
                f.write(json.dumps(asdict(result)) + "\n")
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PDF to Markdown Evaluation Benchmark")
    parser.add_argument("--tool", choices=["marker", "docling"], help="Run only one tool")
    parser.add_argument("--doc", help="Run only one document (by ID)")
    parser.add_argument("--list", action="store_true", help="List test documents")
    args = parser.parse_args()
    
    if args.list:
        print("Test documents:")
        for doc_id, subdir in TEST_DOCUMENTS:
            pdf_path = get_pdf_path(doc_id, subdir)
            size_mb = pdf_path.stat().st_size / 1024 / 1024
            print(f"  {doc_id}: {size_mb:.1f} MB")
    else:
        tools = [args.tool] if args.tool else None
        docs = [(d, s) for d, s in TEST_DOCUMENTS if d == args.doc] if args.doc else None
        run_benchmark(tools=tools, docs=docs)
