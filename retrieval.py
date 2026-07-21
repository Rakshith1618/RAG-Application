import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# --- Config ---
VECTOR_STORE_FILE = "db/faiss_index.pkl"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Load FAISS index and chunk metadata ---
def load_faiss_index(file_path=VECTOR_STORE_FILE):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    index = data["index"]
    chunks = data["chunks"]
    return index, chunks

# --- Retrieve top-k relevant chunks ---
def get_relevant_documents(query, k=3):
    # Load FAISS index and chunks
    index, chunks = load_faiss_index()
    
    # Embed the query
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    query_vec = model.encode([query]).astype("float32")
    
    # Search FAISS
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

# --- Example usage ---
if __name__ == "__main__":
    query = "Your query here"
    top_chunks = get_relevant_documents(query, k=3)
    for i, chunk in enumerate(top_chunks):
        print(f"\n--- Result {i+1} ---")
        print(f"Source: {chunk['source']} (Page {chunk['page']})")
        print(f"Score: {chunk['score']:.4f}")
        print(f"Content Preview: {chunk['content'][:200]}...")
