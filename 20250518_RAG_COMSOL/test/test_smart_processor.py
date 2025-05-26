import os
import fitz  # PyMuPDF
import json
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import hashlib

class DocumentChunk:
    """Represents a chunk of text with full metadata for citation."""
    def __init__(self, text: str, source_file: str, page_start: int, page_end: int, 
                 chunk_id: str, chunk_index: int, total_chunks: int):
        self.text = text
        self.source_file = source_file
        self.page_start = page_start
        self.page_end = page_end
        self.chunk_id = chunk_id
        self.chunk_index = chunk_index
        self.total_chunks = total_chunks
        self.char_count = len(text)
        
    def to_dict(self):
        return {
            'text': self.text,
            'source_file': self.source_file,
            'page_start': self.page_start,
            'page_end': self.page_end,
            'chunk_id': self.chunk_id,
            'chunk_index': self.chunk_index,
            'total_chunks': self.total_chunks,
            'char_count': self.char_count
        }
    
    def get_citation(self):
        """Generate a proper citation for this chunk."""
        if self.page_start == self.page_end:
            return f"{self.source_file}, page {self.page_start}"
        else:
            return f"{self.source_file}, pages {self.page_start}-{self.page_end}"

class SmartPDFProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200, min_chunk_size=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.processed_chunks = []
        self.metadata = {}
        
    def extract_text_by_page(self, pdf_path: str) -> List[Tuple[int, str]]:
        """Extract text from PDF, returning (page_number, text) tuples."""
        doc = fitz.open(pdf_path)
        pages = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            pages.append((page_num + 1, text))  # 1-indexed page numbers
        
        doc.close()
        return pages
    
    def smart_chunk_text(self, pages: List[Tuple[int, str]], source_file: str) -> List[DocumentChunk]:
        """Intelligently chunk text while preserving page boundaries and metadata."""
        chunks = []
        current_chunk = ""
        current_pages = []
        chunk_index = 0
        
        # First pass: combine all text to estimate total chunks
        all_text = " ".join([text for _, text in pages])
        estimated_total_chunks = max(1, len(all_text) // self.chunk_size)
        
        for page_num, page_text in pages:
            # Clean the page text
            page_text = page_text.strip()
            if not page_text:
                continue
                
            # If adding this page would exceed chunk size, finalize current chunk
            if (len(current_chunk) + len(page_text) > self.chunk_size and 
                len(current_chunk) > self.min_chunk_size):
                
                if current_chunk and current_pages:
                    chunk = self._create_chunk(
                        current_chunk, source_file, current_pages, 
                        chunk_index, estimated_total_chunks
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + " " + page_text
                else:
                    current_chunk = page_text
                current_pages = [page_num]
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += " " + page_text
                else:
                    current_chunk = page_text
                current_pages.append(page_num)
        
        # Don't forget the last chunk
        if current_chunk and current_pages:
            chunk = self._create_chunk(
                current_chunk, source_file, current_pages, 
                chunk_index, estimated_total_chunks
            )
            chunks.append(chunk)
        
        # Update total chunks count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
            
        return chunks
    
    def _create_chunk(self, text: str, source_file: str, pages: List[int], 
                     chunk_index: int, total_chunks: int) -> DocumentChunk:
        """Create a DocumentChunk with proper metadata."""
        page_start = min(pages)
        page_end = max(pages)
        
        # Create unique chunk ID
        chunk_content = f"{source_file}_{page_start}_{page_end}_{chunk_index}"
        chunk_id = hashlib.md5(chunk_content.encode()).hexdigest()[:12]
        
        return DocumentChunk(
            text=text,
            source_file=source_file,
            page_start=page_start,
            page_end=page_end,
            chunk_id=chunk_id,
            chunk_index=chunk_index,
            total_chunks=total_chunks
        )

def main():
    # Configuration
    processor = SmartPDFProcessor(
        chunk_size=1000,    # Reasonable size for embeddings
        chunk_overlap=200,  # Good overlap for context
        min_chunk_size=100  # Avoid tiny chunks
    )
    
    # Process only first 3 PDFs for testing
    pdf_directory = "pdf"
    if not os.path.exists(pdf_directory):
        print(f"Error: Directory '{pdf_directory}' not found!")
        return
    
    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
    pdf_files.sort()  # Consistent ordering
    
    # Limit to first 3 files for testing
    test_files = pdf_files[:3]
    print(f"🧪 Testing with first {len(test_files)} PDF files:")
    for i, f in enumerate(test_files, 1):
        print(f"  {i}. {f}")
    
    all_chunks = []
    
    for i, pdf_file in enumerate(test_files, 1):
        pdf_path = os.path.join(pdf_directory, pdf_file)
        print(f"\n[{i}/{len(test_files)}] Processing: {pdf_file}")
        
        try:
            # Extract text by page
            pages = processor.extract_text_by_page(pdf_path)
            print(f"  📄 Extracted {len(pages)} pages")
            
            # Show sample of first page
            if pages:
                first_page_preview = pages[0][1][:200] + "..." if len(pages[0][1]) > 200 else pages[0][1]
                print(f"  📖 First page preview: {first_page_preview}")
            
            # Create chunks
            chunks = processor.smart_chunk_text(pages, pdf_file)
            all_chunks.extend(chunks)
            
            print(f"  ✅ Created {len(chunks)} chunks")
            
            # Show sample chunk info
            if chunks:
                sample_chunk = chunks[0]
                print(f"  📝 Sample chunk: {sample_chunk.get_citation()}")
                print(f"     Length: {sample_chunk.char_count} chars")
                print(f"     ID: {sample_chunk.chunk_id}")
                chunk_preview = sample_chunk.text[:150] + "..." if len(sample_chunk.text) > 150 else sample_chunk.text
                print(f"     Preview: {chunk_preview}")
            
        except Exception as e:
            print(f"  ❌ Error processing {pdf_file}: {e}")
            continue
    
    print(f"\n🎉 Test processing complete!")
    print(f"Total chunks created: {len(all_chunks)}")
    
    if all_chunks:
        # Save test results
        print("\n💾 Saving test results...")
        
        # Save metadata as JSON
        metadata = [chunk.to_dict() for chunk in all_chunks]
        with open("test_chunks_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        # Save just the text for embedding
        texts = [chunk.text for chunk in all_chunks]
        np.save("test_chunk_texts.npy", np.array(texts, dtype=object))
        
        print(f"✅ Test files saved:")
        print(f"  📊 test_chunks_metadata.json ({len(metadata)} chunks)")
        print(f"  🔢 test_chunk_texts.npy")
        
        # Show statistics
        total_chars = sum(chunk.char_count for chunk in all_chunks)
        avg_chunk_size = total_chars / len(all_chunks)
        
        print(f"\n📈 Test Statistics:")
        print(f"  Total chunks: {len(all_chunks)}")
        print(f"  Total characters: {total_chars:,}")
        print(f"  Average chunk size: {avg_chunk_size:.0f} chars")
        
        # Group by source
        by_source = {}
        for chunk in all_chunks:
            if chunk.source_file not in by_source:
                by_source[chunk.source_file] = []
            by_source[chunk.source_file].append(chunk)
        
        print(f"\n📚 Chunks by source:")
        for source, source_chunks in by_source.items():
            print(f"  {source}: {len(source_chunks)} chunks")

if __name__ == "__main__":
    main() 