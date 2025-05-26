import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import faiss
import json
from typing import List, Dict, Tuple
from tqdm import tqdm
import os

class SmartEmbeddingGenerator:
    def __init__(self, model_name="Alibaba-NLP/gte-Qwen2-7B-instruct", device="cuda:0"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.embed_dim = 3584  # For the Qwen2-7B model
        
    def load_model(self):
        """Load the embedding model."""
        print(f"Loading embedding model: {self.model_name}")
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
    
    def embed_chunks_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for multiple texts in batches."""
        all_embeddings = []
        
        print(f"Generating embeddings for {len(texts)} chunks...")
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = []
            
            for text in batch_texts:
                embedding = self.embed_text(text)
                batch_embeddings.append(embedding[0])  # Remove batch dimension
            
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings, dtype="float32")
    
    def load_processed_chunks(self, metadata_file: str, texts_file: str) -> Tuple[List[Dict], List[str]]:
        """Load processed chunks and their metadata."""
        print("Loading processed chunks...")
        
        # Load metadata
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # Load texts
        texts = np.load(texts_file, allow_pickle=True).tolist()
        
        if len(metadata) != len(texts):
            raise ValueError(f"Metadata count ({len(metadata)}) doesn't match texts count ({len(texts)})")
        
        print(f"✅ Loaded {len(metadata)} chunks with metadata")
        return metadata, texts
    
    def create_faiss_index(self, embeddings: np.ndarray, use_gpu: bool = True) -> faiss.Index:
        """Create FAISS index from embeddings."""
        print(f"Creating FAISS index for {embeddings.shape[0]} embeddings...")
        
        # Create index
        if use_gpu and faiss.get_num_gpus() > 0:
            print("Using GPU for FAISS index")
            index = faiss.IndexFlatIP(self.embed_dim)  # Inner product for cosine similarity
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
        else:
            print("Using CPU for FAISS index")
            index = faiss.IndexFlatIP(self.embed_dim)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        index.add(embeddings)
        
        print(f"✅ FAISS index created with {index.ntotal} vectors")
        return index
    
    def save_embedding_system(self, embeddings: np.ndarray, metadata: List[Dict], 
                             index: faiss.Index, output_dir: str = ".") -> Dict[str, str]:
        """Save the complete embedding system."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save embeddings
        embeddings_file = os.path.join(output_dir, "smart_embeddings.npy")
        np.save(embeddings_file, embeddings)
        
        # Save FAISS index
        index_file = os.path.join(output_dir, "smart_faiss_index.index")
        # Convert GPU index to CPU for saving
        if hasattr(index, 'index'):  # GPU index
            cpu_index = faiss.index_gpu_to_cpu(index)
            faiss.write_index(cpu_index, index_file)
        else:
            faiss.write_index(index, index_file)
        
        # Save enhanced metadata with embedding info
        enhanced_metadata = []
        for i, meta in enumerate(metadata):
            enhanced_meta = meta.copy()
            enhanced_meta.update({
                'embedding_index': i,
                'embedding_dim': self.embed_dim,
                'model_used': self.model_name
            })
            enhanced_metadata.append(enhanced_meta)
        
        metadata_file = os.path.join(output_dir, "smart_metadata.json")
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(enhanced_metadata, f, indent=2)
        
        # Save system info
        system_info = {
            'total_chunks': len(metadata),
            'embedding_dimension': self.embed_dim,
            'model_name': self.model_name,
            'index_type': 'IndexFlatIP',
            'files': {
                'embeddings': embeddings_file,
                'index': index_file,
                'metadata': metadata_file
            }
        }
        
        system_file = os.path.join(output_dir, "smart_system_info.json")
        with open(system_file, "w", encoding="utf-8") as f:
            json.dump(system_info, f, indent=2)
        
        print(f"\n🎉 Smart embedding system saved:")
        print(f"  🔢 Embeddings: {embeddings_file}")
        print(f"  🔍 FAISS index: {index_file}")
        print(f"  📊 Metadata: {metadata_file}")
        print(f"  ℹ️  System info: {system_file}")
        
        return {
            'embeddings_file': embeddings_file,
            'index_file': index_file,
            'metadata_file': metadata_file,
            'system_file': system_file
        }
    
    def process_from_chunks(self, chunks_metadata_file: str, chunks_texts_file: str, 
                           output_dir: str = ".", batch_size: int = 16) -> Dict[str, str]:
        """Complete pipeline: load chunks, generate embeddings, create index."""
        
        # Load model
        if self.model is None:
            self.load_model()
        
        # Load processed chunks
        metadata, texts = self.load_processed_chunks(chunks_metadata_file, chunks_texts_file)
        
        # Generate embeddings
        embeddings = self.embed_chunks_batch(texts, batch_size=batch_size)
        
        # Create FAISS index
        index = self.create_faiss_index(embeddings)
        
        # Save everything
        output_files = self.save_embedding_system(embeddings, metadata, index, output_dir)
        
        return output_files

class SmartRAGSearcher:
    """Enhanced RAG searcher that uses the smart embedding system."""
    
    def __init__(self, system_dir: str = "."):
        self.system_dir = system_dir
        self.index = None
        self.metadata = None
        self.tokenizer = None
        self.model = None
        self.system_info = None
        
    def load_system(self):
        """Load the complete smart embedding system."""
        print("Loading smart RAG system...")
        
        # Load system info
        system_file = os.path.join(self.system_dir, "smart_system_info.json")
        with open(system_file, "r") as f:
            self.system_info = json.load(f)
        
        # Load FAISS index
        index_file = os.path.join(self.system_dir, "smart_faiss_index.index")
        self.index = faiss.read_index(index_file)
        
        # Load metadata
        metadata_file = os.path.join(self.system_dir, "smart_metadata.json")
        with open(metadata_file, "r") as f:
            self.metadata = json.load(f)
        
        # Load embedding model
        model_name = self.system_info['model_name']
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).half().eval()
        self.model = self.model.to("cuda:0")
        
        print(f"✅ Smart RAG system loaded:")
        print(f"  📊 {len(self.metadata)} chunks")
        print(f"  🔍 {self.index.ntotal} embeddings")
        print(f"  🤖 Model: {model_name}")
    
    def search_with_citations(self, query: str, k: int = 10) -> List[Dict]:
        """Search and return results with proper citations."""
        # Generate query embedding
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            query_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().astype("float32")
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Prepare results with citations
        results = []
        for score, idx in zip(scores[0], indices[0]):
            chunk_meta = self.metadata[idx]
            
            # Create citation
            source = chunk_meta['source_file']
            page_start = chunk_meta['page_start']
            page_end = chunk_meta['page_end']
            
            if page_start == page_end:
                citation = f"{source}, page {page_start}"
            else:
                citation = f"{source}, pages {page_start}-{page_end}"
            
            result = {
                'text': chunk_meta['text'],
                'score': float(score),
                'citation': citation,
                'source_file': source,
                'pages': f"{page_start}-{page_end}",
                'chunk_id': chunk_meta['chunk_id'],
                'metadata': chunk_meta
            }
            results.append(result)
        
        return results

def main():
    # Check if we have processed chunks
    if not os.path.exists("chunks_metadata.json") or not os.path.exists("chunk_texts.npy"):
        print("❌ Processed chunks not found!")
        print("Please run smart_pdf_processor.py first to process the PDFs.")
        return
    
    # Generate embeddings
    generator = SmartEmbeddingGenerator()
    
    output_files = generator.process_from_chunks(
        chunks_metadata_file="chunks_metadata.json",
        chunks_texts_file="chunk_texts.npy",
        output_dir=".",
        batch_size=16  # Adjust based on GPU memory
    )
    
    print(f"\n🎉 Smart embedding system ready!")
    print("You can now use SmartRAGSearcher for searches with proper citations!")

if __name__ == "__main__":
    main() 