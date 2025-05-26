import os
import fitz  # PyMuPDF
import json
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

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
    def __init__(self, chunk_size=1000, chunk_overlap=200, min_chunk_size=100, use_parallel=True, max_workers=4):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.processed_chunks = []
        self.metadata = {}
        self.use_parallel = use_parallel
        self.max_workers = min(max_workers, mp.cpu_count())
        
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

    def _process_single_pdf(self, pdf_info: Tuple[str, str, int, int]) -> Tuple[List[DocumentChunk], str, bool]:
        """Process a single PDF file. Used for parallel processing."""
        pdf_file, pdf_path, file_idx, total_files = pdf_info
        
        try:
            # Extract text by page
            pages = self.extract_text_by_page(pdf_path)
            
            # Create chunks
            chunks = self.smart_chunk_text(pages, pdf_file)
            
            return chunks, f"[{file_idx}/{total_files}] ✅ {pdf_file}: {len(chunks)} chunks from {len(pages)} pages", True
            
        except Exception as e:
            return [], f"[{file_idx}/{total_files}] ❌ {pdf_file}: Error - {e}", False
    
    def process_pdf_directory(self, pdf_dir: str) -> List[DocumentChunk]:
        """Process all PDFs in a directory with optimized parallel processing."""
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
        pdf_files.sort()  # Consistent ordering
        
        print(f"🚀 Processing {len(pdf_files)} PDF files with {self.max_workers} workers...")
        
        # Prepare work items
        work_items = []
        for i, pdf_file in enumerate(pdf_files, 1):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            work_items.append((pdf_file, pdf_path, i, len(pdf_files)))
        
        all_chunks = []
        
        if self.use_parallel and len(pdf_files) > 1:
            # Use ThreadPoolExecutor for I/O bound PDF processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(self._process_single_pdf, work_items))
            
            # Process results
            for chunks, message, success in results:
                print(message)
                if success:
                    all_chunks.extend(chunks)
        else:
            # Sequential processing for single files or when parallel is disabled
            for work_item in work_items:
                chunks, message, success = self._process_single_pdf(work_item)
                print(message)
                if success:
                    all_chunks.extend(chunks)
        
        self.processed_chunks = all_chunks
        print(f"\n🎉 Processing complete! Created {len(all_chunks)} total chunks")
        return all_chunks
    
    def save_processed_data(self, chunks: List[DocumentChunk], 
                           output_dir: str = ".") -> Dict[str, str]:
        """Save processed chunks and metadata."""
        
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Save chunks as text for embedding
        chunks_file = os.path.join(output_dir, "processed_chunks.txt")
        with open(chunks_file, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                f.write(f"=== CHUNK {i+1} ===\n")
                f.write(f"SOURCE: {chunk.source_file}\n")
                f.write(f"PAGES: {chunk.page_start}-{chunk.page_end}\n")
                f.write(f"CHUNK_ID: {chunk.chunk_id}\n")
                f.write("=" * 50 + "\n")
                f.write(chunk.text)
                f.write("\n\n")
        
        # Save metadata as JSON
        metadata_file = os.path.join(output_dir, "chunks_metadata.json")
        metadata = [chunk.to_dict() for chunk in chunks]
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        # Save just the text for embedding (numpy array)
        texts_file = os.path.join(output_dir, "chunk_texts.npy")
        texts = [chunk.text for chunk in chunks]
        np.save(texts_file, np.array(texts, dtype=object))
        
        # Save summary statistics
        stats_file = os.path.join(output_dir, "processing_stats.json")
        stats = self._generate_stats(chunks)
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nProcessed data saved:")
        print(f"  📄 Chunks text: {chunks_file}")
        print(f"  📊 Metadata: {metadata_file}")
        print(f"  🔢 Texts array: {texts_file}")
        print(f"  📈 Statistics: {stats_file}")
        
        return {
            'chunks_file': chunks_file,
            'metadata_file': metadata_file,
            'texts_file': texts_file,
            'stats_file': stats_file
        }
    
    def _generate_stats(self, chunks: List[DocumentChunk]) -> Dict:
        """Generate processing statistics."""
        if not chunks:
            return {}
        
        # Group by source file
        by_source = {}
        for chunk in chunks:
            if chunk.source_file not in by_source:
                by_source[chunk.source_file] = []
            by_source[chunk.source_file].append(chunk)
        
        # Calculate statistics
        total_chars = sum(chunk.char_count for chunk in chunks)
        avg_chunk_size = total_chars / len(chunks) if chunks else 0
        
        source_stats = {}
        for source, source_chunks in by_source.items():
            source_stats[source] = {
                'chunk_count': len(source_chunks),
                'total_chars': sum(c.char_count for c in source_chunks),
                'avg_chunk_size': sum(c.char_count for c in source_chunks) / len(source_chunks),
                'page_range': f"{min(c.page_start for c in source_chunks)}-{max(c.page_end for c in source_chunks)}"
            }
        
        return {
            'total_chunks': len(chunks),
            'total_sources': len(by_source),
            'total_characters': total_chars,
            'avg_chunk_size': avg_chunk_size,
            'chunk_size_config': self.chunk_size,
            'chunk_overlap_config': self.chunk_overlap,
            'parallel_workers': self.max_workers,
            'source_statistics': source_stats
        }

def main():
    # Optimized configuration for your H100 setup
    processor = SmartPDFProcessor(
        chunk_size=1000,    # Reasonable size for embeddings
        chunk_overlap=200,  # Good overlap for context
        min_chunk_size=100, # Avoid tiny chunks
        use_parallel=True,  # Enable parallel processing
        max_workers=8       # Use more workers for your powerful setup
    )
    
    # Process PDFs
    pdf_directory = "pdf"
    if not os.path.exists(pdf_directory):
        print(f"Error: Directory '{pdf_directory}' not found!")
        return
    
    chunks = processor.process_pdf_directory(pdf_directory)
    
    if not chunks:
        print("No chunks were created!")
        return
    
    # Save processed data
    output_files = processor.save_processed_data(chunks)
    
    print(f"\n🎉 Processing complete!")
    print(f"Created {len(chunks)} chunks from PDF files")
    print(f"Ready for embedding generation!")

if __name__ == "__main__":
    main() 