import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import faiss
import json
from typing import List, Dict, Tuple
from tqdm import tqdm
import os
import textwrap

class SmartRAGTester:
    def __init__(self, model_name="Alibaba-NLP/gte-Qwen2-7B-instruct", device="cuda:0"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.embed_dim = 3584
        self.index = None
        self.metadata = None
        
    def load_model(self):
        """Load the embedding model."""
        print(f"🤖 Loading embedding model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16
        ).half().eval()
        self.model = self.model.to(self.device)
        print("✅ Model loaded successfully!")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.cpu().numpy().astype("float32")
    
    def generate_embeddings_if_needed(self):
        """Generate embeddings from processed chunks if not already done."""
        
        # Check if embeddings already exist
        if os.path.exists("smart_embeddings.npy") and os.path.exists("smart_faiss_index.index"):
            print("📁 Found existing embeddings, loading...")
            self.load_existing_system()
            return
        
        print("🔄 Generating new embeddings from processed chunks...")
        
        # Load processed data
        print("📖 Loading processed chunks...")
        with open("chunks_metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        
        texts = np.load("chunk_texts.npy", allow_pickle=True).tolist()
        
        if len(self.metadata) != len(texts):
            raise ValueError(f"Metadata count ({len(self.metadata)}) doesn't match texts count ({len(texts)})")
        
        print(f"✅ Loaded {len(self.metadata)} chunks")
        
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Generate embeddings in batches
        print("🔢 Generating embeddings...")
        batch_size = 16  # Conservative for GPU memory
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = []
            
            for text in batch_texts:
                embedding = self.embed_text(text)
                batch_embeddings.append(embedding[0])  # Remove batch dimension
            
            all_embeddings.extend(batch_embeddings)
        
        embeddings = np.array(all_embeddings, dtype="float32")
        
        # Create FAISS index
        print("🔍 Creating FAISS index...")
        index = faiss.IndexFlatIP(self.embed_dim)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        self.index = index
        
        # Save embeddings and index
        print("💾 Saving embeddings and index...")
        np.save("smart_embeddings.npy", embeddings)
        faiss.write_index(index, "smart_faiss_index.index")
        
        print(f"✅ Generated embeddings for {len(texts)} chunks")
    
    def load_existing_system(self):
        """Load existing embedding system."""
        # Load metadata
        with open("chunks_metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        
        # Load FAISS index
        self.index = faiss.read_index("smart_faiss_index.index")
        
        print(f"✅ Loaded existing system with {len(self.metadata)} chunks")
    
    def search_with_citations(self, query: str, k: int = 10) -> List[Dict]:
        """Search and return results with proper citations."""
        if self.model is None:
            self.load_model()
        
        # Generate query embedding
        query_embedding = self.embed_text(query)
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            chunk_meta = self.metadata[idx]
            
            result = {
                'text': chunk_meta['text'],
                'source_file': chunk_meta['source_file'],
                'page_start': chunk_meta['page_start'],
                'page_end': chunk_meta['page_end'],
                'chunk_id': chunk_meta['chunk_id'],
                'relevance_score': float(distance),
                'citation': self.get_citation(chunk_meta)
            }
            results.append(result)
        
        return results
    
    def get_citation(self, chunk_meta: Dict) -> str:
        """Generate proper citation from chunk metadata."""
        source = chunk_meta['source_file']
        page_start = chunk_meta['page_start']
        page_end = chunk_meta['page_end']
        
        if page_start == page_end:
            return f"{source}, page {page_start}"
        else:
            return f"{source}, pages {page_start}-{page_end}"
    
    def format_results(self, results: List[Dict], query: str):
        """Format and display search results with citations."""
        print("\n" + "="*80)
        print(f"🔍 SEARCH RESULTS FOR: {query}")
        print("="*80)
        
        # Group by source
        sources = {}
        for result in results:
            source = result['source_file']
            if source not in sources:
                sources[source] = []
            sources[source].append(result)
        
        print(f"\n📚 SOURCES FOUND: {len(sources)} different PDFs")
        for source, source_results in sources.items():
            print(f"  📄 {source}: {len(source_results)} excerpts")
        
        print(f"\n📝 TOP {len(results)} RESULTS WITH CITATIONS:")
        print("-" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. 📖 {result['citation']}")
            print(f"   🎯 Relevance: {result['relevance_score']:.4f}")
            print(f"   📄 Chunk ID: {result['chunk_id']}")
            print("   " + "-" * 70)
            
            # Show text preview
            text = result['text']
            preview = text[:400] + "..." if len(text) > 400 else text
            wrapped_text = textwrap.fill(preview, width=75, initial_indent="   ", subsequent_indent="   ")
            print(wrapped_text)
            print("   " + "-" * 70)
    
    def test_queries(self):
        """Test the system with sample queries."""
        test_queries = [
            "periodic boundary condition for ewfd simulation",
            "heat transfer in COMSOL",
            "mesh generation techniques",
            "boundary conditions in structural mechanics",
            "electromagnetic field simulation"
        ]
        
        print("\n🧪 TESTING RAG SYSTEM WITH SAMPLE QUERIES")
        print("="*80)
        
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            results = self.search_with_citations(query, k=5)
            self.format_results(results, query)
            
            # Wait for user input to continue
            input("\nPress Enter to continue to next query...")
    
    def interactive_search(self):
        """Interactive search mode."""
        print("\n🎯 INTERACTIVE SEARCH MODE")
        print("="*80)
        print("Enter your questions (or 'quit' to exit):")
        
        while True:
            query = input("\n> ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            print("🔍 Searching...")
            results = self.search_with_citations(query, k=20)
            self.format_results(results, query)

def main():
    print("🚀 SMART RAG SYSTEM TESTER")
    print("="*80)
    
    # Initialize tester
    tester = SmartRAGTester()
    
    # Generate embeddings if needed
    tester.generate_embeddings_if_needed()
    
    # Show system info
    print(f"\n📊 SYSTEM READY!")
    print(f"  📚 Total chunks: {len(tester.metadata)}")
    print(f"  🔍 Index size: {tester.index.ntotal}")
    print(f"  🎯 Embedding dimension: {tester.embed_dim}")
    
    # Ask user what to do
    print(f"\n🎮 CHOOSE TEST MODE:")
    print("1. Run sample queries")
    print("2. Interactive search")
    print("3. Both")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        tester.test_queries()
    
    if choice in ['2', '3']:
        tester.interactive_search()

    print(f"\n🎉 Testing complete!")

if __name__ == "__main__":
    main() 