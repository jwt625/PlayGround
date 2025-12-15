#!/usr/bin/env python3
"""
Parse thesis PDF using pypdf and extract text for GraphRAG indexing.
Simple and fast - just extracts text without fancy layout analysis.
"""

from pathlib import Path
import pypdf
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract text from PDF for GraphRAG indexing')
    parser.add_argument('--input', '-i', type=str, default='pdfs/Schuster_thesis.pdf',
                        help='Input PDF file path (default: pdfs/Schuster_thesis.pdf)')
    parser.add_argument('--output', '-o', type=str, default='thesis_output/thesis.txt',
                        help='Output text file path (default: thesis_output/thesis.txt)')
    args = parser.parse_args()

    # Configuration
    pdf_path = Path(args.input)
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PARSING PDF WITH PYPDF (FAST TEXT EXTRACTION)")
    print("=" * 80)
    print(f"Input PDF: {pdf_path}")
    print(f"Output file: {output_path}")
    print()

    # Check if PDF exists
    if not pdf_path.exists():
        print(f"ERROR: PDF file not found at {pdf_path}")
        return

    # Extract text from PDF
    print("Extracting text from PDF...")
    reader = pypdf.PdfReader(str(pdf_path))

    print(f"Found {len(reader.pages)} pages")

    # Extract text from all pages
    text_content = []
    for i, page in enumerate(reader.pages, 1):
        if i % 10 == 0:
            print(f"  Processing page {i}/{len(reader.pages)}...")
        text = page.extract_text()
        text_content.append(text)

    # Combine all text
    full_text = "\n\n".join(text_content)

    # Save to plain text file
    print("\nSaving extracted text...")
    output_path.write_text(full_text, encoding='utf-8')
    print(f"Saved plain text to: {output_path}")

    # Print statistics
    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Text file: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")

    # Count lines and words
    lines = full_text.split('\n')
    words = full_text.split()
    print(f"\nStatistics:")
    print(f"  Pages: {len(reader.pages):,}")
    print(f"  Lines: {len(lines):,}")
    print(f"  Words: {len(words):,}")
    print(f"  Characters: {len(full_text):,}")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print(f"1. Review the extracted text in {output_path}")
    print("2. Copy the text file to a GraphRAG project input directory")
    print("3. Initialize GraphRAG: graphrag init --root ./project_name")
    print("4. Configure settings.yaml with your API keys")
    print("5. Run indexing: graphrag index --root ./project_name")
    print("=" * 80)

if __name__ == "__main__":
    main()

