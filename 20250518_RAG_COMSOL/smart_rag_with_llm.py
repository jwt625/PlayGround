import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import faiss
import json
from typing import List, Dict, Tuple
from tqdm import tqdm
import os
import textwrap
import requests
import time

class SmartRAGWithLLM:
    def __init__(self, model_name="Alibaba-NLP/gte-Qwen2-7B-instruct", device="cuda:4", llm_url="http://localhost:8000"):
        self.model_name = model_name
        self.device = device  # Use GPU 4-7 since LLM uses GPU 0-3
        self.llm_url = llm_url
        self.tokenizer = None
        self.model = None
        self.embed_dim = 3584
        self.index = None
        self.metadata = None
        
    def check_gpu_memory(self):
        """Check GPU memory availability."""
        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return
        
        print("🔍 GPU Memory Status:")
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_device(i)
                total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                free = total_memory - reserved
                
                status = "🔴 BUSY" if allocated > 1.0 else "🟢 FREE"
                print(f"  GPU {i}: {status} - {allocated:.1f}GB/{total_memory:.1f}GB used, {free:.1f}GB free")
            except Exception as e:
                print(f"  GPU {i}: ❌ Error checking - {e}")
    
    def load_model(self):
        """Load the embedding model."""
        print(f"🤖 Loading embedding model: {self.model_name} on {self.device}")
        print(f"💡 Note: LLM uses GPUs 0-3, embedding model uses {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model with memory optimization
        if self.device == "cpu":
            print("🖥️  Using CPU for embeddings...")
            self.model = AutoModel.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float32  # Use float32 for CPU
            ).eval()
        else:
            print(f"🚀 Using {self.device} for embeddings...")
            # Clear cache first
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.model = AutoModel.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16
            ).half().eval()
        
        self.model = self.model.to(self.device)
        print("✅ Model loaded successfully!")
        
        # Show GPU memory usage if using CUDA
        if self.device.startswith("cuda") and torch.cuda.is_available():
            gpu_id = int(self.device.split(":")[-1])
            memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
            print(f"📊 GPU {gpu_id} Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
    
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
        # Try to use pre-computed embeddings if available
        if os.path.exists("smart_embeddings.npy") and self.model is None:
            print("🔍 Using pre-computed embeddings for faster search...")
            # Load embedding model with GPU fallback strategy
            try:
                self.load_model()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️  {self.device} out of memory. Trying alternative GPUs...")
                    # Try GPUs 4-7 (since LLM uses 0-3)
                    for gpu_id in [5, 6, 7, 4]:  # Try 5,6,7 first, then back to 4
                        if gpu_id != int(self.device.split(":")[-1]):  # Skip current device
                            try:
                                print(f"🔄 Trying GPU {gpu_id}...")
                                self.device = f"cuda:{gpu_id}"
                                self.load_model()
                                break
                            except RuntimeError as gpu_e:
                                if "out of memory" in str(gpu_e):
                                    print(f"❌ GPU {gpu_id} also out of memory")
                                    continue
                                else:
                                    raise gpu_e
                    else:
                        # All GPUs failed, fall back to CPU
                        print("🖥️  All GPUs failed, falling back to CPU...")
                        self.device = "cpu"
                        self.load_model()
                else:
                    raise e
        elif self.model is None:
            try:
                self.load_model()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️  {self.device} out of memory. Trying CPU...")
                    self.device = "cpu"
                    self.load_model()
                else:
                    raise e
        
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
    
    def check_llm_availability(self) -> bool:
        """Check if local LLM is available."""
        try:
            response = requests.get(f"{self.llm_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def query_llm(self, prompt: str, max_tokens: int = 2000) -> str:
        """Query the local LLM with a prompt."""
        try:
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
                "stop": ["Human:", "Assistant:"]
            }
            
            response = requests.post(
                f"{self.llm_url}/v1/completions",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("choices", [{}])[0].get("text", "").strip()
            else:
                return f"Error: LLM returned status {response.status_code}"
                
        except Exception as e:
            return f"Error querying LLM: {str(e)}"
    
    def create_context_with_citations(self, results: List[Dict], max_context_length: int = 4000) -> Tuple[str, List[str]]:
        """Create context string with numbered citations."""
        context_parts = []
        citations = []
        current_length = 0
        
        for i, result in enumerate(results, 1):
            text = result['text']
            citation = result['citation']
            
            # Add citation number to the text
            numbered_text = f"[{i}] {text}"
            
            # Check if adding this would exceed limit
            if current_length + len(numbered_text) + 2 > max_context_length:  # +2 for newlines
                break
            
            context_parts.append(numbered_text)
            citations.append(f"[{i}] {citation}")
            current_length += len(numbered_text) + 2  # +2 for newlines
        
        context = "\n\n".join(context_parts)
        return context, citations
    
    def create_summary_prompt(self, query: str, context: str, citations: List[str]) -> str:
        """Create a prompt for LLM summarization with citations."""
        prompt = f"""You are a helpful research assistant. Based on the provided context from technical documents, answer the user's question with a comprehensive summary. Always include proper citations using the reference numbers provided.

User Question: {query}

Context from Documents:
{context}

Available Citations:
{chr(10).join(citations)}

Instructions:
1. Provide a comprehensive answer based on the context
2. Use citation numbers [1], [2], etc. to reference specific information
3. If information comes from multiple sources, cite all relevant ones
4. Be specific and technical when appropriate
5. If the context doesn't fully answer the question, mention what information is available

Answer:"""
        return prompt
    
    def rag_with_llm_summary(self, query: str, k: int = 15, max_context_length: int = 4000) -> Dict:
        """Perform RAG search and generate LLM summary with citations."""
        print(f"🔍 Searching for: {query}")
        
        # Get search results
        results = self.search_with_citations(query, k=k)
        
        if not results:
            return {
                'query': query,
                'summary': "No relevant documents found for your query.",
                'citations': [],
                'search_results': [],
                'llm_available': False
            }
        
        # Create context with citations
        context, citations = self.create_context_with_citations(results, max_context_length)
        
        # Check LLM availability
        llm_available = self.check_llm_availability()
        
        if llm_available:
            print("🤖 Generating LLM summary...")
            prompt = self.create_summary_prompt(query, context, citations)
            summary = self.query_llm(prompt)
        else:
            print("⚠️  LLM not available, providing fallback summary...")
            summary = self.create_fallback_summary(query, results[:5])
        
        return {
            'query': query,
            'summary': summary,
            'citations': citations,
            'search_results': results,
            'llm_available': llm_available,
            'context_used': len(context),
            'sources_count': len(set(r['source_file'] for r in results))
        }
    
    def create_fallback_summary(self, query: str, results: List[Dict]) -> str:
        """Create a fallback summary when LLM is not available."""
        if not results:
            return "No relevant information found."
        
        # Group by source
        sources = {}
        for result in results:
            source = result['source_file']
            if source not in sources:
                sources[source] = []
            sources[source].append(result)
        
        summary_parts = [
            f"Based on the search for '{query}', I found relevant information from {len(sources)} source(s):",
            ""
        ]
        
        for i, (source, source_results) in enumerate(sources.items(), 1):
            summary_parts.append(f"{i}. From {source}:")
            for j, result in enumerate(source_results[:2], 1):  # Show top 2 per source
                preview = result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
                summary_parts.append(f"   - {result['citation']}: {preview}")
            summary_parts.append("")
        
        return "\n".join(summary_parts)
    
    def format_rag_response(self, response: Dict, show_raw_results: bool = False):
        """Format and display the RAG response with LLM summary."""
        print("\n" + "="*80)
        print(f"🎯 RAG RESPONSE FOR: {response['query']}")
        print("="*80)
        
        # Show summary
        print(f"\n📝 SUMMARY:")
        print("-" * 40)
        summary_wrapped = textwrap.fill(response['summary'], width=75)
        print(summary_wrapped)
        
        # Show citations
        if response['citations']:
            print(f"\n📚 CITATIONS:")
            print("-" * 40)
            for citation in response['citations']:
                print(f"  {citation}")
        
        # Show metadata
        print(f"\n📊 METADATA:")
        print("-" * 40)
        print(f"  🤖 LLM Available: {'Yes' if response['llm_available'] else 'No'}")
        print(f"  📄 Sources Used: {response['sources_count']}")
        print(f"  📝 Context Length: {response['context_used']} characters")
        print(f"  🔍 Search Results: {len(response['search_results'])}")
        
        # Show top search results (brief)
        print(f"\n🔍 TOP SEARCH RESULTS (BRIEF):")
        print("-" * 40)
        for i, result in enumerate(response['search_results'][:5], 1):
            print(f"{i}. {result['citation']} (score: {result['relevance_score']:.4f})")
            preview = result['text'][:150] + "..." if len(result['text']) > 150 else result['text']
            wrapped_preview = textwrap.fill(preview, width=70, initial_indent="   ", subsequent_indent="   ")
            print(wrapped_preview)
            print()
        
        # Show raw search results if requested
        if show_raw_results:
            self.show_raw_search_results(response['search_results'], response['query'])
    
    def show_raw_search_results(self, results: List[Dict], query: str):
        """Display detailed raw search results."""
        print(f"\n🔬 RAW SEARCH RESULTS (DETAILED)")
        print("="*80)
        print(f"Query: {query}")
        print(f"Total Results: {len(results)}")
        print("="*80)
        
        # Group by source for better organization
        sources = {}
        for result in results:
            source = result['source_file']
            if source not in sources:
                sources[source] = []
            sources[source].append(result)
        
        print(f"\n📚 SOURCES BREAKDOWN:")
        for source, source_results in sources.items():
            print(f"  📄 {source}: {len(source_results)} excerpts")
        
        print(f"\n📝 DETAILED RESULTS:")
        print("-" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] 📖 {result['citation']}")
            print(f"    🎯 Relevance Score: {result['relevance_score']:.6f}")
            print(f"    📄 Chunk ID: {result['chunk_id']}")
            print(f"    📄 Source: {result['source_file']}")
            print(f"    📄 Pages: {result['page_start']}-{result['page_end']}")
            print("    " + "-" * 70)
            
            # Show full text with proper wrapping
            text = result['text']
            wrapped_text = textwrap.fill(text, width=75, initial_indent="    ", subsequent_indent="    ")
            print(wrapped_text)
            print("    " + "-" * 70)
            
            # Add separator between results
            if i < len(results):
                print()
    
    def interactive_rag_chat(self):
        """Interactive RAG chat with LLM summaries."""
        print("\n🎯 INTERACTIVE RAG CHAT WITH LLM SUMMARIES")
        print("="*80)
        print("Enter your questions (or 'quit' to exit)")
        print("Commands:")
        print("  - Type your question normally for LLM summary")
        print("  - Add '--raw' to see detailed raw search results")
        print("  - Type 'quit' or 'exit' to quit")
        
        # Check LLM status
        if self.check_llm_availability():
            print("✅ LLM is available for summaries")
        else:
            print("⚠️  LLM not available - will use fallback summaries")
        
        while True:
            user_input = input("\n> ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            # Check for --raw flag
            show_raw = False
            if user_input.endswith('--raw'):
                show_raw = True
                query = user_input[:-5].strip()  # Remove --raw flag
            else:
                query = user_input
            
            if not query:
                print("Please enter a question.")
                continue
            
            print("🔄 Processing...")
            response = self.rag_with_llm_summary(query)
            self.format_rag_response(response, show_raw_results=show_raw)
    
    def test_rag_with_llm(self):
        """Test the RAG system with LLM summaries."""
        test_queries = [
            "periodic boundary condition for ewfd simulation",
            "heat transfer in COMSOL",
            "mesh generation techniques",
            "boundary conditions in structural mechanics",
            "electromagnetic field simulation"
        ]
        
        print("\n🧪 TESTING RAG SYSTEM WITH LLM SUMMARIES")
        print("="*80)
        
        # Ask if user wants to see raw results
        show_raw_input = input("Show detailed raw search results for each query? (y/n): ").strip().lower()
        show_raw = show_raw_input in ['y', 'yes', '1', 'true']
        
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            response = self.rag_with_llm_summary(query)
            self.format_rag_response(response, show_raw_results=show_raw)
            
            # Wait for user input to continue
            input("\nPress Enter to continue to next query...")
    
    def single_query_test(self, query: str, show_raw: bool = False):
        """Test a single query and return the response."""
        print(f"🔍 Testing query: '{query}'")
        response = self.rag_with_llm_summary(query)
        self.format_rag_response(response, show_raw_results=show_raw)
        return response

def main():
    print("🚀 SMART RAG SYSTEM WITH LLM SUMMARIES")
    print("="*80)
    
    # Initialize system
    rag_system = SmartRAGWithLLM()
    
    # Check GPU memory status
    rag_system.check_gpu_memory()
    
    # Load existing system
    try:
        rag_system.load_existing_system()
    except FileNotFoundError:
        print("❌ Smart RAG system files not found!")
        print("Please run smart_pdf_processor.py first to generate the required files:")
        print("  - chunks_metadata.json")
        print("  - smart_faiss_index.index")
        return
    
    # Show system info
    print(f"\n📊 SYSTEM READY!")
    print(f"  📚 Total chunks: {len(rag_system.metadata)}")
    print(f"  🔍 Index size: {rag_system.index.ntotal}")
    print(f"  🎯 Embedding dimension: {rag_system.embed_dim}")
    print(f"  🖥️  Embedding device: {rag_system.device}")
    
    # Check LLM availability
    if rag_system.check_llm_availability():
        print(f"  🤖 LLM Status: ✅ Available at {rag_system.llm_url}")
    else:
        print(f"  🤖 LLM Status: ⚠️  Not available (will use fallback summaries)")
    
    # Ask user what to do
    print(f"\n🎮 CHOOSE MODE:")
    print("1. Test with sample queries")
    print("2. Interactive chat")
    print("3. Both")
    print("\nNote: In interactive mode, add '--raw' to any question to see detailed search results")
    print("Example: 'heat transfer in COMSOL --raw'")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        rag_system.test_rag_with_llm()
    
    if choice in ['2', '3']:
        rag_system.interactive_rag_chat()

    print(f"\n🎉 Session complete!")

if __name__ == "__main__":
    main() 