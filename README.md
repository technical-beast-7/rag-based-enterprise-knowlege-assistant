# Enterprise Knowledge Assistant

> An AI-powered RAG system for querying enterprise documents with accurate, context-aware responses.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-green.svg)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq-LLM_Inference-orange.svg)](https://groq.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Organizations store large volumes of unstructured documents — HR policies, IT security guidelines, employee handbooks — that are time-consuming and error-prone to search manually. This project solves that by building an intelligent document assistant that retrieves relevant content and generates precise, grounded answers using **Retrieval-Augmented Generation (RAG)**.

---

## Features

- 📄 **Multi-document ingestion** — PDF support via PyPDFLoader
- 🔍 **Semantic search** — vector embeddings with FAISS
- 🧠 **Context-aware responses** — full RAG pipeline to reduce hallucinations
- ⚡ **Low-latency inference** — powered by Groq API
- 💬 **Interactive chat UI** — built with Streamlit
- 📊 **Evaluation pipeline** — custom accuracy & relevance metrics
- 🧾 **Source-grounded answers** — optional citation of source chunks

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM Inference | Groq API (Llama-3.3-70b-versatile) |
| RAG Orchestration | LangChain |
| Vector Database | FAISS |
| Embeddings | Hugging Face |
| Data Processing | PyPDFLoader, RecursiveCharacterTextSplitter |
| UI | Streamlit |
| Language | Python 3.9+ |

---

## Architecture

```
User Query
    │
    ▼
Query Embedding
    │
    ▼
FAISS Vector Search
    │
    ▼
Top-K Relevant Chunks
    │
    ▼
LLM (Groq API)
    │
    ▼
Generated Response
```

---

## Project Structure

```
enterprise-knowledge-assistant/
│
├── data/                   # Input PDF documents
├── vectorstore/            # FAISS index files
│
├── src/
│   ├── ingestion.py        # PDF loading and text extraction
│   ├── embeddings.py       # Embedding generation
│   ├── vectorstore.py      # FAISS index management
│   ├── retrieval.py        # Semantic search and chunk retrieval
│   ├── generation.py       # LLM response generation
│   └── app.py              # Streamlit UI
│
├── evaluation/             # Test dataset and eval scripts
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- A [Groq API key](https://console.groq.com)

### Installation

```bash
git clone https://github.com/your-username/enterprise-knowledge-assistant.git
cd enterprise-knowledge-assistant
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_api_key_here
```

### Run

```bash
streamlit run src/app.py
```

---

## How It Works

1. **Ingestion** — PDFs are loaded and converted to structured text using `PyPDFLoader`.
2. **Chunking** — Text is split into ~500-token chunks with overlap using `RecursiveCharacterTextSplitter`.
3. **Embedding** — Each chunk is converted to a vector using Hugging Face embeddings.
4. **Storage** — Embeddings are indexed and persisted in a FAISS vector store.
5. **Retrieval** — User queries are embedded and matched against the index to fetch top-k relevant chunks.
6. **Generation** — Retrieved chunks are passed as context to the Groq-hosted LLM, which generates a grounded response.

---

## API Usage

```python
from groq import Groq

client = Groq(api_key="YOUR_API_KEY")

response = client.chat.completions.create(
    model="mixtral-8x7b",
    messages=[{"role": "user", "content": query}]
)
```

---

## Evaluation

The system was evaluated against a custom dataset of 30+ enterprise queries.

| Metric | Result |
|---|---|
| Top-3 Retrieval Accuracy | ~90% |
| Hallucination Reduction | ~30% |
| Average Response Time | 2–4 seconds |

---

## Challenges & Solutions

| Challenge | Solution |
|---|---|
| Irrelevant retrieval | Optimized chunk size and top-k parameter |
| Hallucination | Prompt engineering with strict context grounding |
| High latency | Switched to Groq API for fast inference |

---

## Roadmap

- [ ] Source citations in responses
- [ ] Conversation memory across turns
- [ ] Hybrid search (keyword + semantic)
- [ ] Cloud deployment (AWS / GCP)
- [ ] Support for additional file formats (DOCX, XLSX)

---

## Contributing

Contributions are welcome! Fork the repo and open a pull request. For major changes, please open an issue first to discuss what you'd like to change.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

⭐ **Star this repo if you found it useful!**