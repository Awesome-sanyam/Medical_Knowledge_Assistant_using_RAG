# 🏛️ MedAssist AI — Architecture & Project Map

This document serves as a comprehensive technical guide to the MedAssist AI system. It maps out the directory structure, explains the flow of data, and details the responsibilities of each modular component.

---

## 1. System Overview

MedAssist AI follows a **modular monolithic** architecture built on Django 5.x. It separates concerns into distinct applications, ensuring the user-facing chat interface remains completely decoupled from the sensitive hardware and RAG administrative controls.

The application relies on three core pillars:
1. **The Web Layer (Django / HTMX / Tailwind CSS):** Handles HTTP routing, authentication, and Server-Sent Events (SSE) for streaming text.
2. **The RAG Engine (LangChain / ChromaDB):** Handles document parsing, dense/sparse indexing, and context retrieval.
3. **The LLM Layer (Ollama / LLaMA 3.1):** The local inference engine that generates the final responses based strictly on the retrieved context.

---

## 2. Directory Map & Component Breakdown

```text
medical_assistant_project/
├── manage.py                   # Django CLI utility
├── db.sqlite3                  # Local relational database (User accounts, Audit logs, Config)
├── mac_requirements.txt        # Local dependencies
├── colab_requirements.txt      # Remote training dependencies
│
├── core_project/               # System Foundation
│   ├── settings.py             # Global Django configurations and app registration
│   ├── urls.py                 # Root URL router (Maps /chat, /app-admin, /model-control)
│   ├── asgi.py                 # Async entrypoint (Future-proofing for WebSockets)
│   └── wsgi.py                 # Sync entrypoint for standard deployment
│
├── chat_app/                   # User-Facing Chat Interface
│   ├── models.py               # Conversation & Message data schemas
│   ├── views.py                # Chat logic, SSE streaming generator, LLM prompt wrapping
│   ├── auth_views.py           # Standard user login/registration
│   └── templates/chat/         # Client-side UI
│       └── interface.html      # HTMX-powered, Tailwind CSS streaming interface
│
├── admin_app/                  # RAG & System Administration
│   ├── models.py               # SystemConfig (Temp, Top-K), Document, AuditLog
│   ├── views.py                # Document upload handlers, Audit log viewers
│   └── templates/admin/        # Shared UI shell for all admin tools
│       ├── base_admin.html     # Master Layout (Sidebar, CSS tokens, Headers)
│       ├── dashboard.html      # Global stats and indexing health
│       ├── rag_upload.html     # XHR upload portal for PDFs/TXTs
│       ├── prompt_config.html  # Live system prompt and threshold editor
│       └── audit_logs.html     # Table view of all AI generations and retrieved context
│
├── model_admin/                # Hardware & LLM Control
│   ├── models.py               # ModelSession logs
│   ├── views.py                # CPU/RAM polling, Ollama API proxy (Load/Unload models)
│   └── templates/model_admin/  
│       └── dashboard.html      # Real-time hardware graphs and model hot-swapping
│
├── rag_engine/                 # Core AI Logic
│   ├── indexer.py              # PyPDF2 extraction → RecursiveCharacterTextSplitter → Chroma
│   ├── retriever.py            # Vector similarity + BM25 sparse retrieval. Returns 3-tuples
│   ├── guardrails.py           # Refusal generators and domain-safety checks
│   └── ollama_client.py        # LangChain wrapper for HTTP streaming to local Ollama port
│
├── chroma_db/                  # Local Vector Database (Auto-generated)
│   └── (SQLite/Parquet files)  # Stores the BAAI/bge-small-en-v1.5 embeddings
│
└── evaluation/                 # Academic Testing Suite
    ├── run_eval.py             # automated ROUGE-L scorer against ground truth
    └── test_queries.json       # Benchmark QA pairs
```

---

## 3. Data Flow: Answering a Clinical Query

When a user submits a query through the Chat Interface, the following sequence occurs:

1. **Request Intake:** `chat_app.views.stream_chat` receives the POST request via HTMX.
2. **Context Retrieval:** 
   - The query is passed to `rag_engine.retriever.retrieve_context(query, top_k)`.
   - The Retriever searches `ChromaDB` using dense vector embeddings.
   - It also performs a sparse keyword search (BM25) to catch specific medical acronyms.
   - It calculates a final relevance score.
3. **Guardrail Enforcement:** 
   - If the highest relevance score is below the `SystemConfig.similarity_threshold`, the query is flagged as **Out-of-Domain**.
   - `guardrails.py` is triggered, and a hard-coded refusal string ("Please be precised to the Context only.") is returned instantly. The LLM is **bypassed** to save compute.
4. **LLM Generation:** 
   - If the context is valid, it is formatted alongside the user query and the `SystemConfig.system_prompt`.
   - `rag_engine.ollama_client` opens an HTTP stream to the local Ollama instance (`http://localhost:11434/api/generate`).
5. **Streaming Response:** 
   - As Ollama generates tokens, Django yields them back to the client using Server-Sent Events (SSE).
   - The frontend (`interface.html`) receives the chunks and dynamically renders them into Markdown.
6. **Auditing:** 
   - Once generation is complete, the query, the raw context, the full response, and the confidence score are saved to `admin_app.models.AuditLog` for administrative review.

---

## 4. UI Design System

MedAssist AI utilizes a **Unified Design System** constructed with Vanilla CSS (Tailwind inspired variables). 

- **Master Template:** `admin_app/templates/admin/base_admin.html` acts as the single source of truth for the entire application (including `model_admin` and parts of `chat_app`).
- **Color Palette:** 
  - Clinical Teal (`#0d9488`)
  - Indigo (`#6366f1`)
  - Deep Slate (`#0f172a`)
- **Typography:** `Inter` for UI elements, `JetBrains Mono` for code and context chunks.

By extending `base_admin.html`, all administrative pages maintain perfect visual parity, shared drawer navigation, and responsive behavior without duplicating CSS.
