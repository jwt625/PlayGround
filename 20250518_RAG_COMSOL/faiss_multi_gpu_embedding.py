#%%

import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
import numpy as np
import faiss
from tqdm import tqdm
import os
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch.multiprocessing as mp
from typing import List, Tuple
import math

assert hasattr(faiss, "StandardGpuResources"), "Still using CPU-only FAISS"

# === CONFIG ===
MODEL_NAME = "Alibaba-NLP/gte-Qwen2-7B-instruct"
BATCH_SIZE = 64
CHUNK_SIZE = 512
EMBED_DIM = 3584  # Updated to match actual embedding dimension
DEVICE_COUNT = torch.cuda.device_count()

def load_model_and_tokenizer(device_id: int) -> Tuple[AutoTokenizer, AutoModel]:
    """Load model and tokenizer on a specific GPU."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).half().eval()
    model = model.to(f"cuda:{device_id}")
    return tokenizer, model

def embed_batch(batch_texts: List[str], device_id: int, tokenizer: AutoTokenizer, model: AutoModel) -> np.ndarray:
    """Embed a batch of texts using the specified GPU."""
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=CHUNK_SIZE)
    inputs = {k: v.to(f"cuda:{device_id}") for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()

def process_chunk(chunk_data: Tuple[List[str], int]) -> np.ndarray:
    """Process a chunk of documents on a specific GPU."""
    documents, device_id = chunk_data
    tokenizer, model = load_model_and_tokenizer(device_id)
    
    all_embeddings = []
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i + BATCH_SIZE]
        emb = embed_batch(batch, device_id, tokenizer, model)
        all_embeddings.append(emb)
    
    return np.vstack(all_embeddings).astype("float32")

#%%
def create_multi_gpu_index(embeddings: np.ndarray, n_gpus: int) -> faiss.Index:
    """Create a FAISS index distributed across multiple GPUs."""
    # Create CPU index
    cpu_index = faiss.IndexFlatL2(EMBED_DIM)
    
    # Create GPU resources
    gpu_resources = [faiss.StandardGpuResources() for _ in range(n_gpus)]
    
    # Create GPU indices
    gpu_indices = []
    for i in range(n_gpus):
        gpu_index = faiss.index_cpu_to_gpu(gpu_resources[i], i, cpu_index)
        gpu_indices.append(gpu_index)
    
    # Split data across GPUs
    chunk_size = math.ceil(len(embeddings) / n_gpus)
    for i in range(n_gpus):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(embeddings))
        gpu_indices[i].add(embeddings[start_idx:end_idx])
    
    # Convert all GPU indices back to CPU and add to the final index
    final_index = faiss.IndexFlatL2(EMBED_DIM)
    for i in range(n_gpus):
        cpu_index = faiss.index_gpu_to_cpu(gpu_indices[i])
        # Get the vectors from each index and add them to the final index
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(embeddings))
        final_index.add(embeddings[start_idx:end_idx])
    
    return final_index


#%% Load and chunk documents
# Read and chunk documents
with open("extracted_texts.txt", "r", encoding="utf-8") as f:
    texts = f.read().split("=== Document ")[1:]
    texts = [doc.split("\n", 1)[1] for doc in texts]

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=100)
documents = [chunk for text in texts for chunk in splitter.split_text(text)]

#%% Prepare chunks for parallel processing
# Split documents across GPUs
chunk_size = math.ceil(len(documents) / DEVICE_COUNT)
chunks = [(documents[i:i + chunk_size], i % DEVICE_COUNT) 
         for i in range(0, len(documents), chunk_size)]

#%% Generate embeddings
# Process chunks in parallel
with mp.Pool(DEVICE_COUNT) as pool:
    embeddings_chunks = list(tqdm(
        pool.imap(process_chunk, chunks),
        total=len(chunks),
        desc="Embedding chunks"
    ))

# Combine all embeddings
all_embeddings = np.vstack(embeddings_chunks).astype("float32")

#%% Create and save index
# Create multi-GPU index
print(f"Creating index using {DEVICE_COUNT} GPUs...")
index = create_multi_gpu_index(all_embeddings, DEVICE_COUNT)

# Save index and documents
faiss.write_index(index, "faiss_index.index")
np.save("documents_multi_gpu.npy", np.array(documents, dtype=object))
print("Index and documents saved successfully!")


# %%
