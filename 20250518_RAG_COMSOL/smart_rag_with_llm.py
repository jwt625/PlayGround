import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import faiss
import json
import requests
import textwrap
from typing import List, Dict, Tuple

class SmartRAG:
    def __init__(self, device="cuda:4", llm_url="http://localhost:8000"):
        self.device = device
        self.llm_url = llm_url
        self.tokenizer = None
        self.model = None
        self.index = None
        self.metadata = None
        
    def load_system(self):
        """Load existing RAG system."""
        with open("chunks_metadata.json", "r") as f:
            self.metadata = json.load(f)
        self.index = faiss.read_index("smart_faiss_index.index")
        print(f"✅ Loaded {len(self.metadata)} chunks")
        
    def load_embedding_model(self):
        """Load embedding model with GPU fallback."""
        model_name = "Alibaba-NLP/gte-Qwen2-7B-instruct"
        print(f"🤖 Loading {model_name} on {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.device == "cpu":
                self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32).eval()
            else:
                torch.cuda.empty_cache()
                self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).half().eval()
            self.model = self.model.to(self.device)
            print("✅ Model loaded")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("⚠️ GPU OOM, switching to CPU")
                self.device = "cpu"
                self.load_embedding_model()
            else:
                raise e
    
    def embed_query(self, text: str) -> np.ndarray:
        """Generate embedding for query."""
        if self.model is None:
            self.load_embedding_model()
            
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.cpu().numpy().astype("float32")
    
    def search(self, query: str, k: int = 15) -> List[Dict]:
        """Search and return results with citations."""
        query_embedding = self.embed_query(query)
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            chunk = self.metadata[idx]
            results.append({
                'text': chunk['text'],
                'citation': f"{chunk['source_file']}, page {chunk['page_start']}" if chunk['page_start'] == chunk['page_end'] 
                          else f"{chunk['source_file']}, pages {chunk['page_start']}-{chunk['page_end']}",
                'score': float(distance)
            })
        return results
    
    def check_llm(self) -> bool:
        """Check if LLM is available."""
        try:
            response = requests.get(f"{self.llm_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def query_llm(self, prompt: str) -> str:
        """Query LLM with proper context window management."""
        try:
            payload = {
                "prompt": prompt,
                "max_tokens": 2000,
                "temperature": 0.7,
                "top_p": 0.9
            }
            response = requests.post(f"{self.llm_url}/v1/completions", json=payload, timeout=60)
            if response.status_code == 200:
                return response.json().get("choices", [{}])[0].get("text", "").strip()
            return f"Error: LLM returned {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def create_context(self, results: List[Dict], max_tokens: int = 8000) -> Tuple[str, List[str]]:
        """Create context with proper token management for LLM context window."""
        context_parts = []
        citations = []
        
        # Estimate tokens (rough: 1 token ≈ 4 characters)
        current_tokens = 0
        max_chars = max_tokens * 4
        
        for i, result in enumerate(results, 1):
            text = result['text']
            citation = result['citation']
            
            # Format: [1] text content
            numbered_text = f"[{i}] {text}"
            
            # Check token limit
            if current_tokens + len(numbered_text) > max_chars:
                print(f"⚠️ Context truncated at {i-1} results due to token limit")
                break
                
            context_parts.append(numbered_text)
            citations.append(f"[{i}] {citation}")
            current_tokens += len(numbered_text)
        
        return "\n\n".join(context_parts), citations
    
    def rag_query(self, query: str, show_raw: bool = False) -> Dict:
        """Main RAG query function."""
        print(f"🔍 Searching: {query}")
        
        # Search
        results = self.search(query)
        if not results:
            return {'query': query, 'summary': "No results found", 'citations': [], 'results': []}
        
        # Create context with larger token limit
        context, citations = self.create_context(results, max_tokens=12000)  # Increased from 4000 chars
        
        # Query LLM
        llm_available = self.check_llm()
        if llm_available:
            prompt = f"""Answer the question based on the provided context. Use citation numbers [1], [2], etc.

Question: {query}

Context:
{context}

Citations:
{chr(10).join(citations)}

Answer with proper citations:"""
            
            print("🤖 Generating summary...")
            summary = self.query_llm(prompt)
        else:
            summary = f"LLM unavailable. Found {len(results)} results from {len(set(r['citation'].split(',')[0] for r in results))} sources."
        
        response = {
            'query': query,
            'summary': summary,
            'citations': citations,
            'results': results,
            'context_tokens': len(context) // 4,  # Rough token estimate
            'llm_available': llm_available
        }
        
        # Display results
        self.display_response(response, show_raw)
        return response
    
    def display_response(self, response: Dict, show_raw: bool = False):
        """Display response in clean format."""
        print("\n" + "="*60)
        print(f"🎯 {response['query']}")
        print("="*60)
        
        # Summary
        print(f"\n📝 SUMMARY:")
        print(textwrap.fill(response['summary'], width=70))
        
        # Citations
        if response['citations']:
            print(f"\n📚 CITATIONS:")
            for citation in response['citations']:
                print(f"  {citation}")
        
        # Stats
        print(f"\n📊 STATS:")
        print(f"  Results: {len(response['results'])}, Context: ~{response['context_tokens']} tokens, LLM: {'✅' if response['llm_available'] else '❌'}")
        
        # Raw results if requested
        if show_raw:
            print(f"\n🔬 RAW RESULTS:")
            for i, result in enumerate(response['results'], 1):
                print(f"\n[{i}] {result['citation']} (score: {result['score']:.4f})")
                print(textwrap.fill(result['text'], width=70, initial_indent="  ", subsequent_indent="  "))
    
    def chat(self):
        """Interactive chat mode."""
        print("\n🎯 RAG CHAT")
        print("Commands: 'query --raw' for detailed results, 'quit' to exit")
        
        while True:
            user_input = input("\n> ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if not user_input:
                continue
                
            show_raw = user_input.endswith('--raw')
            query = user_input[:-5].strip() if show_raw else user_input
            
            if query:
                self.rag_query(query, show_raw)

def main():
    print("🚀 SMART RAG WITH LLM")
    
    # Initialize
    rag = SmartRAG()
    try:
        rag.load_system()
    except FileNotFoundError:
        print("❌ Files not found. Run smart_pdf_processor.py first.")
        return
    
    # Check LLM
    if rag.check_llm():
        print("✅ LLM available")
    else:
        print("⚠️ LLM unavailable")
    
    # Mode selection
    print("\n1. Interactive chat\n2. Test query")
    choice = input("Choice (1/2): ").strip()
    
    if choice == "1":
        rag.chat()
    elif choice == "2":
        query = input("Enter test query: ").strip()
        if query:
            rag.rag_query(query, show_raw=True)
    
    print("🎉 Done!")

if __name__ == "__main__":
    main() 