# Medical Knowledge Assistant using RAG

A clinical-grade Medical AI Assistant powered by a **fine-tuned Llama 3.1 8B** model with **Retrieval-Augmented Generation (RAG)**, built on Django and running entirely locally on Apple Silicon.

---

## 🏗️ Architecture

```
Client (HTMX + Tailwind CSS)
        │
        ▼
Django Backend
├── chat_app        — User chat interface with SSE streaming
└── admin_app       — Admin dashboard (upload PDFs, tune LLM, audit logs)
        │
        ▼
RAG Engine
├── indexer.py      — PDF/TXT ingestion → ChromaDB
├── retriever.py    — Hybrid BM25 + Vector retrieval
├── guardrails.py   — Pre/Post query safety filters
└── ollama_client.py — LangChain → local Ollama streaming
        │
        ▼
Local Infrastructure
├── ChromaDB        — Vector store (BAAI/bge-small-en-v1.5 embeddings)
└── Ollama          — med-llama model (Llama 3.1 8B + QLoRA LoRA adapter)
```

---

## 🚀 Quick Start (MacBook Air M4 Local Inference)

### 1. Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed

### 2. Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r mac_requirements.txt
```

### 3. Set Up the LLM (after training on Colab)
Place your exported `med_llama3_adapter.gguf` in this directory, then:
```bash
ollama pull llama3.1:8b
ollama create med-llama -f Modelfile
```

### 4. Run Migrations & Start Server
```bash
python manage.py migrate
python manage.py createsuperuser   # for admin access
python manage.py runserver
```

### 5. Access the App
| Interface | URL |
|---|---|
| 💬 User Chat | http://127.0.0.1:8000/chat/ |
| 🛡️ Admin Panel | http://127.0.0.1:8000/app-admin/ |

---

## 🤖 Model Training (Google Colab)

Open `Colab_Training_Script.ipynb` in Google Colab (Free T4 or Pro) to:
1. Load `unsloth/Meta-Llama-3.1-8B-Instruct` in 4-bit
2. Fine-tune with QLoRA (r=16, α=32) on `medical_meadow_wikidoc_patient_information`
3. Export the adapter as `Q4_K_M.gguf`

---

## 📁 Project Structure

```
medical_assistant_project/
├── core_project/           # Django settings & root URLs
├── chat_app/               # User-facing streaming chat module
├── admin_app/              # Admin control panel (3 pages)
├── rag_engine/             # Core AI logic
│   ├── indexer.py          # Document ingestion → ChromaDB
│   ├── retriever.py        # Hybrid BM25 + vector search
│   ├── guardrails.py       # Pre/Post safety filters
│   └── ollama_client.py    # LangChain → Ollama streaming client
├── evaluation/             # Academic benchmark suite
│   ├── run_eval.py         # ROUGE-L evaluation runner
│   └── test_queries.json   # 8 USMLE-style benchmark questions
├── mac_requirements.txt    # Local Mac dependencies
└── colab_requirements.txt  # Google Colab training dependencies
```

---

## ⚙️ Admin Dashboard Features

| Page | Feature |
|---|---|
| Knowledge Base | Upload PDF/TXT → auto-index to ChromaDB, live status polling |
| LLM Config | Tune system prompt, temperature, top-k, similarity threshold |
| Audit Logs | Inspect AI responses, retrieved context, confidence scores, user feedback |

---

## 📊 Evaluation

Run the academic benchmark (requires `rouge-score`):
```bash
pip install rouge-score
python evaluation/run_eval.py
```
Results are saved to `evaluation/eval_results.json`.

---

## ⚠️ Disclaimer

> This system is designed for **academic research purposes only** and is **not a substitute for professional clinical medical advice**. Always consult a qualified healthcare provider.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Llama 3.1 8B + QLoRA (Unsloth) → GGUF via Ollama |
| RAG | LangChain + ChromaDB + BM25 |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Backend | Django 5.x + SQLite |
| Frontend | Tailwind CSS + HTMX + SSE |
| Training | Google Colab (T4 GPU) |
