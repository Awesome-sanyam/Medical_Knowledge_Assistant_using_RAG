# 🩺 MedAssist AI — Clinical RAG Assistant

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python&logoColor=white)
![Django](https://img.shields.io/badge/Django-5.0+-092E20.svg?logo=django&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-white.svg?logo=ollama)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6F61.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**MedAssist AI** is an enterprise-grade, locally hosted AI clinical consultant powered by a **fine-tuned LLaMA 3.1 (8B)** model and advanced **Retrieval-Augmented Generation (RAG)**. 

Designed for medical professionals, the system operates entirely offline on Apple Silicon (or standard Linux/Windows machines), ensuring strict **HIPAA compliance** and absolute data privacy. It features a stunning, unified glassmorphism UI, server-sent events (SSE) for fluid token streaming, and a dedicated suite of administrative dashboards.

---

## ✨ Key Features

- **💬 Real-Time Streaming Chat:** Lightning-fast inference via Ollama with fluid token-by-token rendering using SSE (Server-Sent Events) and HTMX.
- **📚 Advanced Hybrid RAG Engine:** Combines dense vector search (ChromaDB + `BAAI/bge-small-en-v1.5`) with sparse keyword retrieval (BM25) for highly accurate clinical context retrieval.
- **🛡️ Strict AI Guardrails:** Built-in hard thresholds to prevent hallucinations. The model is explicitly blocked from answering non-medical queries, acting purely as a clinical consultant.
- **⚙️ Unified Admin & Model Control:** A polished suite of administration tools to monitor hardware usage (RAM/CPU), hot-swap LLMs, tweak inference parameters (Temperature, Top-K), and ingest new medical literature instantly.
- **📄 Live Document Indexing:** Upload clinical PDFs/TXT files via the dashboard and watch them instantly chunked, vectorized, and indexed into the ChromaDB knowledge base.

---

## 🚀 Quick Start (Local Deployment)

### 1. Prerequisites
- **Python 3.11+**
- **[Ollama](https://ollama.com)** installed and running in the background.

### 2. Install Dependencies
Clone the repository and set up a virtual environment:
```bash
git clone https://github.com/Awesome-sanyam/Medical_Knowledge_Assistant_using_RAG.git
cd Medical_Knowledge_Assistant_using_RAG/medical_assistant_project
python3 -m venv venv
source venv/bin/activate
pip install -r mac_requirements.txt
```

### 3. Initialize the Local LLM
Ensure Ollama is running. Pull the base LLaMA 3.1 model and create our specialized medical adapter version:
```bash
ollama pull llama3.1:8b
ollama create med-llama -f Modelfile
```
*(Note: If you have trained a custom adapter via Google Colab, ensure `med_llama3_adapter.gguf` is placed in the project root before running `ollama create`)*.

### 4. Database Setup & Server Start
```bash
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser  # Create an admin account for the dashboards
python manage.py runserver
```

### 5. Access the Platform
| Interface | URL | Access Level |
|---|---|---|
| **Live Chat** | `http://127.0.0.1:8000/chat/` | Standard Users & Admins |
| **Admin Dashboard** | `http://127.0.0.1:8000/app-admin/` | Staff / Superuser Only |
| **Model Control** | `http://127.0.0.1:8000/model-control/` | Staff / Superuser Only |

---

## 🗺️ Project Architecture

For a deep dive into the system's architecture, data flows, and modular design, please refer to the comprehensive [ARCHITECTURE.md](ARCHITECTURE.md) guide.

### High-Level Folder Map
```text
medical_assistant_project/
├── core_project/           # Global Django settings, WSGI/ASGI, and Root URL routing
├── chat_app/               # User-facing chat UI, SSE streaming endpoint, Chat history models
├── admin_app/              # Base Admin UI, RAG Ingestion Portal, Audit Inspector, System Config
├── model_admin/            # LLM hardware control (RAM/CPU monitoring, Model load/unload)
├── rag_engine/             # The core AI logic layer
│   ├── indexer.py          # Document parsing (PyPDF2), chunking, and ChromaDB insertion
│   ├── retriever.py        # Hybrid search logic (Vector + BM25) and threshold enforcement
│   ├── guardrails.py       # Domain-specific constraints and safety prompt wrappers
│   └── ollama_client.py    # Direct HTTP streaming client to Ollama backend
├── evaluation/             # USMLE-style academic evaluation scripts (ROUGE-L scoring)
└── templates/              # Base HTML templates (Tailwind CSS, Inter Font, custom CSS tokens)
```

---

## 🧠 Model Fine-Tuning (Google Colab)

If you wish to re-train or fine-tune the model yourself:
1. Open `Colab_Training_Script.ipynb` in Google Colab (A T4 GPU is sufficient).
2. The script utilizes **Unsloth** for rapid 4-bit QLoRA fine-tuning of `Meta-Llama-3.1-8B-Instruct`.
3. The dataset used is `medical_meadow_wikidoc_patient_information`.
4. Export the resulting LoRA weights directly to GGUF format and load them via Ollama.

---

## 🤝 Contributing

We welcome contributions from developers, researchers, and medical professionals. Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to propose bug fixes, submit pull requests, and adhere to our coding standards.

---

## ⚠️ Medical Disclaimer

> **IMPORTANT:** MedAssist AI is designed strictly for **academic research, educational purposes, and clinical workflow experimentation**. It is **NOT** a certified medical device and must never be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
