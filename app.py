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

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "Sports RAG Assistant"})

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
        formatted_sources = [
            {
                "source": d.get("source", "Unknown"),
                "page": d.get("page", 0) + 1 if isinstance(d.get("page"), int) else d.get("page", 1),
                "snippet": d.get("content", "")[:250] + ("..." if len(d.get("content", "")) > 250 else ""),
                "score": round(float(d.get("score", 0.0)), 4)
            }
            for d in docs
        ]
        return jsonify({"answer": answer, "sources": formatted_sources})

    except Exception as e:
        print(f"Error handling /ask request: {e}")
        return jsonify({"answer": f"❌ Error processing request: {str(e)}", "sources": []})

@app.route('/api/documents', methods=['GET'])
def get_documents():
    try:
        docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
        files_info = []
        if os.path.exists(docs_dir):
            for filename in sorted(os.listdir(docs_dir)):
                if filename.lower().endswith(".pdf"):
                    filepath = os.path.join(docs_dir, filename)
                    size_kb = round(os.path.getsize(filepath) / 1024, 1)
                    files_info.append({
                        "filename": filename,
                        "size": f"{size_kb} KB"
                    })
        return jsonify({"documents": files_info})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Only PDF files are supported"}), 400

        docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
        os.makedirs(docs_dir, exist_ok=True)
        save_path = os.path.join(docs_dir, file.filename)
        file.save(save_path)

        # Trigger re-ingestion
        try:
            from ingestion import main as run_ingestion
            run_ingestion()
        except Exception as ingest_err:
            print(f"Re-ingestion notice: {ingest_err}")

        return jsonify({"message": f"Successfully uploaded and indexed '{file.filename}'!"})
    except Exception as e:
        return jsonify({"error": f"Failed to process file: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5050))
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    print(f"🚀 Starting RAG Web Assistant on http://localhost:{port}")
    app.run(debug=debug_mode, host='0.0.0.0', port=port)