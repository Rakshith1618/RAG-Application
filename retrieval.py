import pickle
import numpy as np
import requests
import faiss

# --- Config ---
VECTOR_STORE_FILE = "db/faiss_index.pkl"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL_NAME}"

# --- Load FAISS index and chunk metadata ---
def load_faiss_index(file_path=VECTOR_STORE_FILE):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    index = data["index"]
    chunks = data["chunks"]
    return index, chunks

# --- Obtain 384-dim Query Vector via API (Low RAM) with Local Fallback ---
def get_query_vector(query):
    try:
        response = requests.post(
            HF_API_URL, 
            json={"inputs": query, "options": {"wait_for_model": True}},
            timeout=6
        )
        if response.status_code == 200:
            res = response.json()
            arr = np.array(res, dtype="float32")
            if arr.ndim == 2:
                # Token-level embeddings: mean pool across sequence length
                vec = np.mean(arr, axis=0, keepdims=True)
            elif arr.ndim == 1:
                vec = np.expand_dims(arr, axis=0)
            else:
                vec = arr[0] if arr.ndim > 2 else arr
            return vec.astype("float32")
    except Exception as e:
        print(f"API embedding fallback to local: {e}")

    # Fallback to local SentenceTransformer if network API is unreachable
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model.encode([query]).astype("float32")

# --- Retrieve top-k relevant chunks ---
def get_relevant_documents(query, k=3):
    # Load FAISS index and chunks
    index, chunks = load_faiss_index()
    
    # Embed the query
    query_vec = get_query_vector(query)
    
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
    query = "What is the main objective of basket ball?"
    top_chunks = get_relevant_documents(query, k=3)
    for i, chunk in enumerate(top_chunks):
        print(f"\n--- Result {i+1} ---")
        print(f"Source: {chunk['source']} (Page {chunk['page']})")
        print(f"Score: {chunk['score']:.4f}")
        print(f"Content Preview: {chunk['content'][:200]}...")
