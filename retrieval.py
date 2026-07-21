import os
import pickle
import re
import math
from collections import Counter

# --- Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_STORE_FILE = os.path.join(BASE_DIR, "db", "faiss_index.pkl")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Load FAISS index and chunk metadata ---
def load_faiss_index(file_path=VECTOR_STORE_FILE):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"FAISS index file not found at {file_path}")
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data.get("index"), data.get("chunks", [])

# --- Pure Python BM25 Search Fallback (Zero External Dependencies) ---
def bm25_retrieval(query, chunks, k=3):
    def tokenize(text):
        return re.findall(r'\w+', text.lower())

    query_tokens = tokenize(query)
    if not query_tokens or not chunks:
        return chunks[:k] if chunks else []

    doc_tokens = [tokenize(c.get("content", "")) for c in chunks]
    N = len(chunks)
    avgdl = sum(len(d) for d in doc_tokens) / max(N, 1)

    df = Counter()
    for d in doc_tokens:
        for term in set(d):
            df[term] += 1

    idf = {term: math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1) for term in df}

    scores = []
    k1, b = 1.5, 0.75

    for idx, (chunk, tokens) in enumerate(zip(chunks, doc_tokens)):
        doc_len = len(tokens)
        tf = Counter(tokens)
        score = 0.0
        for term in query_tokens:
            if term in tf:
                freq = tf[term]
                numer = idf.get(term, 0) * freq * (k1 + 1)
                denom = freq + k1 * (1 - b + b * (doc_len / max(avgdl, 1e-5)))
                score += numer / max(denom, 1e-5)
        scores.append((score, idx))

    scores.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, idx in scores[:k]:
        chunk = chunks[idx]
        results.append({
            "source": chunk.get("source", "Unknown"),
            "page": chunk.get("page", 0),
            "content": chunk.get("content", ""),
            "score": float(score)
        })
    return results

# --- Main Retrieval Function ---
def get_relevant_documents(query, k=3):
    index, chunks = load_faiss_index()
    
    try:
        import numpy as np
        import faiss
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        query_vec = model.encode([query]).astype("float32")
        distances, indices = index.search(query_vec, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(chunks):
                chunk = chunks[idx]
                results.append({
                    "source": chunk.get("source", "Unknown"),
                    "page": chunk.get("page", 0),
                    "content": chunk.get("content", ""),
                    "score": float(distances[0][i])
                })
        return results

    except Exception as e:
        print(f"FAISS/SentenceTransformers not active ({e}), using BM25 fallback retrieval.")
        return bm25_retrieval(query, chunks, k=k)

if __name__ == "__main__":
    query = "What is the main objective of basket ball?"
    top_chunks = get_relevant_documents(query, k=3)
    print(f"Retrieved {len(top_chunks)} chunks for: {query}")
    for i, c in enumerate(top_chunks):
        print(f"[{i+1}] {c['source']} (Page {c['page']}): {c['content'][:100]}...")
