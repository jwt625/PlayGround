#!/usr/bin/env python3
"""
Parse thesis PDF using pypdf and extract text for GraphRAG indexing.
Simple and fast - just extracts text without fancy layout analysis.
"""

from pathlib import Path
import pypdf

def main():
    # Configuration
    pdf_path = Path("pdfs/Schuster_thesis.pdf")
    output_dir = Path("thesis_output")
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("PARSING THESIS PDF WITH PYPDF (FAST TEXT EXTRACTION)")
    print("=" * 80)
    print(f"Input PDF: {pdf_path}")
    print(f"Output directory: {output_dir}")
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
    text_path = output_dir / "thesis.txt"
    text_path.write_text(full_text, encoding='utf-8')
    print(f"Saved plain text to: {text_path}")

    # Print statistics
    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Text file: {text_path} ({text_path.stat().st_size / 1024:.1f} KB)")

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
    print("1. Review the extracted text in thesis_output/thesis.txt")
    print("2. Copy the text file to a new GraphRAG project directory")
    print("3. Initialize GraphRAG: graphrag init --root ./thesis")
    print("4. Configure settings.yaml with your API keys")
    print("5. Run indexing: graphrag index --root ./thesis")
    print("=" * 80)

if __name__ == "__main__":
    main()

