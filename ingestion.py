import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# --- Config ---
DOCS_PATH = "docs"
VECTOR_STORE_FILE = "db/faiss_index.pkl"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 0

# --- Load documents ---
def load_documents(docs_path=DOCS_PATH):
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory {docs_path} does not exist.")

    documents = []
    for filename in os.listdir(docs_path):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(docs_path, filename)
            reader = PdfReader(filepath)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    documents.append({
                        "source": filename,
                        "page": page_num,
                        "content": text
                    })
    print(f"Loaded {len(documents)} pages from PDFs.")
    return documents

# --- Split documents into chunks ---
def split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    chunks = []
    for doc in documents:
        content = doc["content"]
        start = 0
        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunk_text = content[start:end]
            chunks.append({
                "source": doc["source"],
                "page": doc["page"],
                "content": chunk_text
            })
            start += chunk_size - chunk_overlap
    print(f"Split into {len(chunks)} chunks.")
    return chunks

# --- Create embeddings ---
def create_embeddings(chunks, model_name=EMBEDDING_MODEL_NAME):
    model = SentenceTransformer(model_name)
    embeddings = [model.encode(chunk["content"]) for chunk in chunks]
    print("Created embeddings for all chunks.")
    return np.array(embeddings, dtype="float32")

# --- Save FAISS index ---
def save_faiss_index(embeddings, chunks, file_path=VECTOR_STORE_FILE):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    # Save index and metadata
    with open(file_path, "wb") as f:
        pickle.dump({"index": index, "chunks": chunks}, f)
    print(f"FAISS index and metadata saved to {file_path}")

# --- Main pipeline ---
def main():
    documents = load_documents()
    chunks = split_documents(documents)
    embeddings = create_embeddings(chunks)
    save_faiss_index(embeddings, chunks)
    print("✅ Ingestion complete!")

if __name__ == "__main__":
    main()
