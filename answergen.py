import os
from dotenv import load_dotenv
from groq import Groq
from retrieval import get_relevant_documents   # your FAISS retriever

# Load API key
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

# Your query
query = "What is the main objective of basket ball ?"

# Retrieve relevant docs
docs = get_relevant_documents(query, k=5)

if not docs:
    print("❌ No relevant information found in documents.")
    exit()

# Join contexts
context = "\n\n".join(chunk["content"] for chunk in docs)

# Build prompt messages
messages = [
    {"role": "system", "content": (
        "You are a strict RAG assistant.\n"
        "Answer ONLY using the provided context.\n"
        "If the answer is not present, say:\n"
        "'I don't have enough information based on the provided documents.'"
    )},
    {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
]

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Send to Groq Chat Completions API
response = client.chat.completions.create(
    messages=messages,
    model="llama-3.3-70b-versatile"  # choose a valid Groq model
)

print("\n=== Final Answer ===")
print(response.choices[0].message.content)
