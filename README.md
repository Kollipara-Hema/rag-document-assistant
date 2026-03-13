# DocuMind AI --- RAG Document Assistant (Local LLM + Streamlit)

An LLM-powered document search and question-answering assistant built
using **Retrieval-Augmented Generation (RAG)**.\
It ingests PDFs, chunks them, generates embeddings, stores them in a
vector database (Chroma), retrieves the most relevant chunks at query
time, and generates grounded answers with citations.

------------------------------------------------------------------------

## Project Capabilities
- Multi-format document ingestion
- Recursive chunking with overlap
- Semantic embeddings generation
- Chroma vector database indexing
- Top-K similarity retrieval
- Context-grounded answer generation
- Citation tracking
- Streamlit interactive UI
- Local LLM inference via Ollama (Llama 3.1)

### Supported document formats
- PDF
- HTML
- TXT
- RTF
- CSV
- JSON
- DOCX


------------------------------------------------------------------------

## 🧠 Architecture Overview

PDFs → Loader → Chunking → Embeddings → Chroma Vector DB\
User Query → Embed → Retrieve Top-K → Prompt (Context + Question) → LLM
→ Answer + Sources

------------------------------------------------------------------------

## 📁 Repository Structure

rag-document-assistant/ 
│ ├── app/ 
│ └── app.py \# Streamlit UI 
│ ├── src/ 
│ ├── config.py \# Config constants 
│ ├── providers.py \# LLM + Embedding provider logic 
│ ├── ingest.py \#  # Multi-format ingestion pipeline 
│ ├── retriever.py \# Vector search logic 
│ ├── rag.py \# RAG chain logic 
│ ├── agent_graph.py \# Placeholder for LangGraph workflows 
│ └── utils.py \# Helper utilities 
│ ├── data/ 
│ ├── raw_docs/ \# Add PDFs here 
│ └── chroma_db/ \# Persisted vector database
│ ├── .env.example 
├── requirements.txt 
└── README.md

------------------------------------------------------------------------

## ⚙️ Local Setup

### 1️⃣ Clone and create environment

``` bash
git clone https://github.com/Kollipara-Hema/rag-document-assistant.git
cd rag-document-assistant
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2️⃣ Install Ollama (Local LLM)

Download from: https://ollama.com

Pull the model:

``` bash
ollama pull llama3.1
```

Test model:

``` bash
ollama run llama3.1
```

Exit with `Ctrl + D`

------------------------------------------------------------------------

## 📚 Add Documents

Place supported files inside:

data/raw_docs/

Supported formats:
- .pdf
- .html / .htm
- .txt
- .rtf
- .csv
- .json
- .docx


------------------------------------------------------------------------

## 🔄 Build the Vector Index

``` bash
python -u -m src.ingest
```

This will:
- Load supported documents from data/raw_docs
- Extract text based on file type
- Chunk text
- Generate embeddings
- Store vectors in Chroma


------------------------------------------------------------------------

## 💬 Ask Questions (CLI)

``` bash
python -u -m src.rag
```

------------------------------------------------------------------------

## 🖥 Run Streamlit App

``` bash
streamlit run app/app.py
```

Open in browser: http://localhost:8501

------------------------------------------------------------------------

## 🧩 Engineering Highlights

-   Modular provider abstraction (LLM & embeddings)
-   Grounded response prompting to reduce hallucination
-   Metadata-based citation tracking
-   Designed for local-first experimentation
-   Easily extendable to OpenAI or hosted LLM providers

------------------------------------------------------------------------

## 📌 Deployment Note

Streamlit Community Cloud does NOT support Ollama runtime.\
For public deployment, switch to OpenAI or another hosted provider.

------------------------------------------------------------------------

## 🛠 Future Improvements

-   Hybrid search (BM25 + vector)
-   Reranking layer
-   LangGraph multi-step reasoning
-   Evaluation metrics (faithfulness, retrieval recall)
-   Cloud-ready deployment pipeline


