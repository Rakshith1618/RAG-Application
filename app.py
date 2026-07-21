from flask import Flask, request, jsonify, render_template
from historygen import get_relevant_documents, Groq, GROQ_API_KEY

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    query = data.get('query')

    # Retrieve relevant documents
    docs = get_relevant_documents(query, k=5)
    if not docs:
        return jsonify({"answer": "❌ No relevant information found in documents."})

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
    response = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile"
    )

    # Return the answer
    answer = response.choices[0].message.content
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)