import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from groq import Groq
from retrieval import get_relevant_documents

load_dotenv()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json or {}
        query = data.get('query', '').strip()

        if not query:
            return jsonify({"answer": "Please enter a valid question."})

        groq_key = os.environ.get("GROQ_API_KEY")
        if not groq_key:
            return jsonify({"answer": "❌ GROQ_API_KEY environment variable is not configured. Please add GROQ_API_KEY in your deployment environment settings."})

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
        client = Groq(api_key=groq_key)
        response = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile"
        )

        answer = response.choices[0].message.content
        return jsonify({"answer": answer})

    except Exception as e:
        print(f"Error handling /ask request: {e}")
        return jsonify({"answer": f"❌ Error processing request: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)