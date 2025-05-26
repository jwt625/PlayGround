import json
import numpy as np
from typing import List, Dict

def load_test_data():
    """Load the test chunks and metadata."""
    with open("test_chunks_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    texts = np.load("test_chunk_texts.npy", allow_pickle=True).tolist()
    
    return metadata, texts

def demonstrate_citation_system():
    """Demonstrate how the citation system works."""
    print("🔍 Citation System Demonstration")
    print("=" * 50)
    
    metadata, texts = load_test_data()
    
    print(f"Loaded {len(metadata)} chunks with perfect metadata alignment")
    
    # Show examples from different sources
    sources_shown = set()
    examples_shown = 0
    
    for i, chunk_meta in enumerate(metadata):
        source = chunk_meta['source_file']
        
        # Show one example from each source
        if source not in sources_shown and examples_shown < 5:
            sources_shown.add(source)
            examples_shown += 1
            
            # Generate citation
            page_start = chunk_meta['page_start']
            page_end = chunk_meta['page_end']
            
            if page_start == page_end:
                citation = f"{source}, page {page_start}"
            else:
                citation = f"{source}, pages {page_start}-{page_end}"
            
            print(f"\n📄 Example {examples_shown}:")
            print(f"   Citation: {citation}")
            print(f"   Chunk ID: {chunk_meta['chunk_id']}")
            print(f"   Size: {chunk_meta['char_count']} characters")
            print(f"   Index: {i} (embedding index {i})")
            
            # Show text preview
            text_preview = texts[i][:200] + "..." if len(texts[i]) > 200 else texts[i]
            print(f"   Preview: {text_preview}")
    
    # Show statistics by source
    print(f"\n📊 Statistics by Source:")
    print("-" * 30)
    
    by_source = {}
    for chunk_meta in metadata:
        source = chunk_meta['source_file']
        if source not in by_source:
            by_source[source] = {
                'chunks': 0,
                'total_chars': 0,
                'page_ranges': []
            }
        
        by_source[source]['chunks'] += 1
        by_source[source]['total_chars'] += chunk_meta['char_count']
        by_source[source]['page_ranges'].append((chunk_meta['page_start'], chunk_meta['page_end']))
    
    for source, stats in by_source.items():
        min_page = min(r[0] for r in stats['page_ranges'])
        max_page = max(r[1] for r in stats['page_ranges'])
        avg_chars = stats['total_chars'] / stats['chunks']
        
        print(f"  {source}:")
        print(f"    - {stats['chunks']} chunks")
        print(f"    - Pages {min_page}-{max_page}")
        print(f"    - Avg chunk size: {avg_chars:.0f} chars")
    
    # Demonstrate search simulation
    print(f"\n🔍 Simulated Search Results:")
    print("-" * 30)
    
    # Simulate finding chunks about "electromagnetic"
    electromagnetic_chunks = []
    for i, text in enumerate(texts):
        if "electromagnetic" in text.lower():
            electromagnetic_chunks.append((i, metadata[i]))
            if len(electromagnetic_chunks) >= 3:  # Limit to 3 examples
                break
    
    if electromagnetic_chunks:
        print(f"Found {len(electromagnetic_chunks)} chunks about 'electromagnetic':")
        
        for rank, (idx, chunk_meta) in enumerate(electromagnetic_chunks, 1):
            source = chunk_meta['source_file']
            page_start = chunk_meta['page_start']
            page_end = chunk_meta['page_end']
            
            if page_start == page_end:
                citation = f"{source}, page {page_start}"
            else:
                citation = f"{source}, pages {page_start}-{page_end}"
            
            print(f"\n  Result {rank}:")
            print(f"    Citation: {citation}")
            print(f"    Chunk ID: {chunk_meta['chunk_id']}")
            
            # Find the word in context
            text = texts[idx]
            em_pos = text.lower().find("electromagnetic")
            if em_pos >= 0:
                start = max(0, em_pos - 50)
                end = min(len(text), em_pos + 100)
                context = text[start:end]
                print(f"    Context: ...{context}...")

def main():
    try:
        demonstrate_citation_system()
    except FileNotFoundError:
        print("❌ Test data not found!")
        print("Please run test_smart_processor.py first to generate test data.")

if __name__ == "__main__":
    main() 