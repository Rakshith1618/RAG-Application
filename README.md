# 🏆 Sports Knowledge RAG Assistant

A strict Retrieval-Augmented Generation (RAG) assistant designed for instant, fact-grounded Q&A against official sports rulebooks and PDFs. Built with Flask, FAISS, Sentence-Transformers, and Groq LLM (`llama-3.3-70b-versatile`).

---

## ✨ Features
- 🏀 **Strict Fact Grounding**: Answers queries exclusively using document context retrieved from indexed PDFs.
- ⚡ **Lightning-Fast LLM**: Powered by Groq API (`llama-3.3-70b-versatile`).
- 🔍 **FAISS Vector Store**: Semantic similarity search using `sentence-transformers/all-MiniLM-L6-v2`.
- 📁 **Dynamic PDF Ingestion**: Upload new rulebook PDFs directly from the UI for automatic chunking & re-indexing.
- 🎨 **Modern Dark Glassmorphic UI**: Includes markdown rendering, source snippet drawers, audio speech playback, and chat export.

---

## 🚀 Quickstart (Local)

1. **Clone Repository**:
   ```bash
   git clone https://github.com/Rakshith1618/RAG-Application.git
   cd RAG-Application
   ```

2. **Create Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Run Server**:
   ```bash
   python app.py
   ```
   Open `http://localhost:5050` in your browser.

---

## 🌐 Deploy to Cloud (Render / Railway)

### Option 1: Deploy on Render (Recommended Free Tier)
1. Go to [Render Dashboard](https://dashboard.render.com/) and click **New Web Service**.
2. Connect your GitHub repository `Rakshith1618/RAG-Application`.
3. Configure the settings:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT app:app`
4. In **Environment Variables**, add:
   - `GROQ_API_KEY`: `your_groq_api_key`
5. Click **Deploy Web Service**.

---

### Option 2: Deploy using Docker
Build and run locally or on any Docker container host:
```bash
docker build -t rag-assistant .
docker run -p 5050:5050 -e GROQ_API_KEY="your_groq_api_key" rag-assistant
```

---

## 📂 Project Structure
```
├── app.py              # Flask server routes & application logic
├── retrieval.py        # FAISS vector similarity search & embedding generator
├── ingestion.py        # PDF extraction, chunking, & vector store creator
├── db/                 # Serialized FAISS index & document chunk metadata
│   └── faiss_index.pkl
├── docs/               # PDF document rulebooks
├── templates/          # HTML view templates
│   └── index.html
├── Procfile            # Deployment WSGI configuration for Gunicorn
├── Dockerfile          # Container build definition
└── requirements.txt    # Python dependencies
```
