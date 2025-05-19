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

# === 5. Main Pipeline ===
def main(pdf_directory):
    texts = load_all_pdfs_from_directory(pdf_directory)
    documents = chunk_texts(texts)
    vectorstore = embed_documents(documents)
    rag_chain = setup_rag(vectorstore)

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() in ("exit", "quit"):
            break
        answer = rag_chain.run(query)
        print(f"\nAnswer:\n{answer}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python rag_from_pdfs.py /path/to/pdf_directory")
        exit(1)
    main(sys.argv[1])
