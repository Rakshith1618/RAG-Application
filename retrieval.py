import os
import pickle
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

# Limit PyTorch CPU threads to reduce memory overhead on Render (512MB RAM limit)
torch.set_num_threads(1)

# --- Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_STORE_FILE = os.path.join(BASE_DIR, "db", "faiss_index.pkl")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Cached Model Instance ---
_model_instance = None

def get_embedding_model():
    global _model_instance
    if _model_instance is None:
        _model_instance = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model_instance

# --- Load FAISS index and chunk metadata ---
def load_faiss_index(file_path=VECTOR_STORE_FILE):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"FAISS index file not found at {file_path}")
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data["index"], data["chunks"]

# --- Retrieve top-k relevant chunks ---
def get_relevant_documents(query, k=3):
    index, chunks = load_faiss_index()
    model = get_embedding_model()
    
    with torch.no_grad():
        query_vec = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks):
            chunk = chunks[idx]
            results.append({
                "source": chunk["source"],
                "page": chunk["page"],
                "content": chunk["content"],
                "score": float(distances[0][i])
            })
    return results

if __name__ == "__main__":
    query = "What is the main objective of basket ball?"
    docs = get_relevant_documents(query, k=3)
    print(f"Retrieved {len(docs)} documents for query: {query}")
    for d in docs:
        print(f"- Source: {d['source']} (Page {d['page']}): {d['content'][:100]}...")
