# 🤝 Contributing to MedAssist AI

First off, thank you for considering contributing to MedAssist AI! It's people like you that make open-source healthcare AI tools secure, robust, and accessible.

Whether you're fixing a bug, adding a new RAG feature, or improving our evaluation benchmarks, your help is deeply appreciated.

---

## 🛠️ How Can I Contribute?

### 1. Reporting Bugs
If you find a bug (e.g., an issue with HTMX SSE streaming, a ChromaDB indexing failure, or a UI glitch), please open an Issue. Include:
- A clear, descriptive title.
- Steps to reproduce the behavior.
- Your OS, Python version, and Ollama version.
- Any relevant logs from Django (`python manage.py runserver`).

### 2. Suggesting Enhancements
Have an idea for a better retrieval algorithm? Want to add support for a different local LLM backend (like vLLM or llama.cpp)? Open an Issue tagged as an `enhancement` and describe your proposal.

### 3. Submitting Pull Requests
1. **Fork the Repository** and create your branch from `main`.
2. **Setup your environment:** Ensure you have `mac_requirements.txt` installed.
3. **Write Code:** Make your changes.
    - If touching the RAG engine (`rag_engine/`), ensure you test against standard medical PDFs.
    - If modifying UI, stick to our Vanilla CSS tokens defined in `admin_app/templates/admin/base_admin.html`. **Do not introduce arbitrary colors.**
4. **Test:** Run the evaluation suite if you are changing LLM logic:
   ```bash
   python evaluation/run_eval.py
   ```
5. **Commit:** Write clear, concise commit messages.
6. **Push and PR:** Open a Pull Request against `main`. Provide a summary of what you did and why.

---

## 📐 Styleguides

### Python & Django
- We adhere strictly to **PEP 8**.
- Use Type Hints wherever possible, especially in the `rag_engine`.
- Keep Django Apps strictly modular. `chat_app` should not directly import from `admin_app` unless necessary; rely on the database or shared utility functions.

### Frontend (HTML/CSS)
- We do not use NPM or Node.js for this project to keep the deployment footprint small.
- CSS is written in Vanilla CSS utilizing custom properties (`var(--primary)`, etc.).
- HTMX handles all asynchronous interactions. Avoid writing custom Vanilla JS unless strictly necessary (like the SSE reader).

---

## 🏥 A Note on Clinical Safety

MedAssist AI implements strict Guardrails in `rag_engine/guardrails.py` and `retriever.py` to prevent hallucination. 

**DO NOT submit Pull Requests that weaken these thresholds.** The system must default to refusal if it cannot find relevant medical context. Pull Requests attempting to bypass these safety measures will be rejected. 

Thank you for helping us build safe AI!
