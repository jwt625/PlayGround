# rag_from_pdfs.py

#%%
import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# === 1. Extract Text from PDF Files ===
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])

def load_all_pdfs_from_directory(directory):
    texts = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            full_path = os.path.join(directory, filename)
            print(f"Extracting: {full_path}")
            text = extract_text_from_pdf(full_path)
            texts.append(text)
    return texts

# === 2. Chunk the Text ===
def chunk_texts(texts, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = splitter.create_documents(texts)
    return documents

# === 3. Embed the Chunks ===
def embed_documents(documents):
    embeddings = OpenAIEmbeddings()  # Requires OPENAI_API_KEY env var
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# === 4. Setup RAG ===
def setup_rag(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", k=5)
    llm = ChatOpenAI()  # Defaults to gpt-3.5-turbo
    qa_chain = RetrievalQA(llm=llm, retriever=retriever)
    return qa_chain

#%%
# Section 1: Extract and process PDFs
pdf_directory = "pdf"  # Change this to your PDF directory path
texts = load_all_pdfs_from_directory(pdf_directory)

#%%
# Save texts to a file
output_file = "extracted_texts.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for i, text in enumerate(texts):
        f.write(f"=== Document {i+1} ===\n")
        f.write(text)
        f.write("\n\n")
print(f"Texts saved to {output_file}")

#%% need openAI API key for this
# Section 2: Create chunks and embeddings
documents = chunk_texts(texts)
vectorstore = embed_documents(documents)



#%%
# Load texts from the extracted_texts.txt file
with open("extracted_texts.txt", "r", encoding="utf-8") as f:
    texts = f.read().split("=== Document ")[1:]  # Split on document markers and remove empty first element
    texts = [doc.split("\n", 1)[1] for doc in texts]  # Remove document number and keep content

from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # Open source model with good performance
    model_kwargs={"device": "mps"}  # use "cpu" if no GPU
)

# Then plug into your vector store
documents = chunk_texts(texts)
vectorstore = FAISS.from_documents(documents, embeddings)


#%%
# Section 3: Setup RAG and start QA
rag_chain = setup_rag(vectorstore)

while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() in ("exit", "quit"):
        break
    answer = rag_chain.run(query)
    print(f"\nAnswer:\n{answer}")
