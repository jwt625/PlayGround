import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import numpy as np
import faiss
from typing import List, Tuple
import textwrap

# === CONFIG ===
MODEL_NAME = "Alibaba-NLP/gte-Qwen2-7B-instruct"
EMBED_DIM = 3584
K_NEAREST = 5  # Number of nearest neighbors to return
MAX_SUMMARY_LENGTH = 300  # Maximum length of summary in words

def load_models() -> Tuple[AutoTokenizer, AutoModel, pipeline]:
    """Load models for embedding and summarization."""
    # Load embedding model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).half().eval()
    model = model.to("cuda:0")
    
    # Load summarization pipeline
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",  # Good for extractive summarization
        device=4,  # Use GPU 0
        torch_dtype=torch.float16
    )
    
    return tokenizer, model, summarizer

def embed_query(query: str, tokenizer: AutoTokenizer, model: AutoModel) -> np.ndarray:
    """Embed a single query text."""
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy().astype("float32")

def search_index(query: str, index: faiss.Index, documents: np.ndarray, 
                tokenizer: AutoTokenizer, model: AutoModel) -> List[Tuple[str, float]]:
    """Search the index for the query and return top K results with scores."""
    # Embed the query
    query_embedding = embed_query(query, tokenizer, model)
    
    # Search the index
    distances, indices = index.search(query_embedding, K_NEAREST)
    
    # Get the corresponding documents and scores
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        results.append((documents[idx], float(distance)))
    
    return results

def summarize_results(results: List[Tuple[str, float]], summarizer: pipeline) -> str:
    """Summarize the search results."""
    # Combine all relevant documents
    combined_text = "\n".join([doc for doc, _ in results])
    
    # Generate summary
    summary = summarizer(combined_text, 
                        max_length=MAX_SUMMARY_LENGTH, 
                        min_length=100, 
                        do_sample=False)
    
    return summary[0]['summary_text']

def format_results(results: List[Tuple[str, float]], summary: str):
    """Format and print the results with summary."""
    print("\n=== Summary ===")
    print(textwrap.fill(summary, width=80))
    print("\n=== Detailed Results ===")
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n{i}. (Score: {score:.4f})")
        print("-" * 80)
        print(textwrap.fill(doc, width=80))
        print("-" * 80)

def main():
    # Load the index and documents
    print("Loading index and documents...")
    index = faiss.read_index("faiss_index.index")
    documents = np.load("documents.npy", allow_pickle=True)
    
    # Load models
    print("Loading models...")
    tokenizer, model, summarizer = load_models()
    
    # Interactive search loop
    print("\nEnter your search query (or 'quit' to exit):")
    while True:
        query = input("> ").strip()
        if query.lower() == 'quit':
            break
            
        print("\nSearching...")
        results = search_index(query, index, documents, tokenizer, model)
        
        print("\nGenerating summary...")
        summary = summarize_results(results, summarizer)
        
        format_results(results, summary)

if __name__ == "__main__":
    main() 