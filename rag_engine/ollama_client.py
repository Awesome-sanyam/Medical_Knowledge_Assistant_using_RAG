"""
Ollama Client — Enterprise-grade LLM integration with anti-repetition controls.

Key parameters:
- repeat_penalty=1.15: Penalizes repeated tokens to prevent looping
- top_k=40: Limits token sampling to top 40 candidates
- top_p=0.9: Nucleus sampling for diverse but coherent output
- num_predict=2048: Forces long-form clinical answers
- num_ctx=4096: Context window size for RAG passages
"""

import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton LLM Cache — avoids re-initializing on every request
# ---------------------------------------------------------------------------
_llm_cache: dict[str, ChatOllama] = {}

DEFAULT_MODEL = "med-llama"
OLLAMA_BASE_URL = "http://localhost:11434"

# ---------------------------------------------------------------------------
# Enterprise System Prompt — enforces structured markdown output
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = (
    "You are MedAssist AI, a senior clinical knowledge assistant with deep expertise "
    "in medicine, pharmacology, pathophysiology, and evidence-based clinical practice.\n\n"
    "## RESPONSE RULES (MANDATORY)\n"
    "1. You MUST structure your response using Markdown formatting.\n"
    "2. Use `### Headings` for each major section.\n"
    "3. Use **bold text** for key terms and emphasis.\n"
    "4. Use bullet points (`-`) or numbered lists for enumerations.\n"
    "5. NEVER repeat the same phrase, sentence, or paragraph twice.\n"
    "6. NEVER give one-sentence or brief replies. Minimum 200 words.\n"
    "7. If the user asks a non-medical question (coding, weather, politics, math, "
    "entertainment, etc.), you MUST politely refuse and redirect them.\n\n"
    "## RESPONSE STRUCTURE\n"
    "For every medical question, organize your answer into these sections:\n"
    "### Overview\n"
    "A clear definition and introduction to the topic.\n"
    "### Pathophysiology\n"
    "Underlying mechanism or disease process (when applicable).\n"
    "### Clinical Presentation\n"
    "Signs, symptoms, and how the condition manifests.\n"
    "### Diagnosis\n"
    "Diagnostic criteria, investigations, and differential diagnosis.\n"
    "### Management\n"
    "Pharmacological and non-pharmacological treatment approaches.\n"
    "### Key Considerations\n"
    "Safety warnings, contraindications, red flags, or when to refer.\n\n"
    "## CONTEXT USAGE\n"
    "Use the provided context passages to ground your answer with evidence. "
    "If context is limited or absent, draw on your parametric medical training "
    "to provide an accurate, empathetic, and clinically safe response."
)


def _get_llm(temperature: float = 0.2, model_name: str = DEFAULT_MODEL) -> ChatOllama:
    """
    Returns a cached ChatOllama instance.
    A new instance is only created when (model, temperature) combo changes.
    """
    key = f"{model_name}::{temperature}"
    if key not in _llm_cache:
        _llm_cache[key] = ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=OLLAMA_BASE_URL,
            # Context window — accommodates RAG passages + prompt
            num_ctx=4096,
            # Force long-form generation (critical fix for short replies)
            num_predict=2048,
            # === ANTI-REPETITION CONTROLS ===
            # Penalizes tokens that have already appeared, preventing loops
            repeat_penalty=1.15,
            # Limits sampling to top-40 tokens for coherence
            top_k=40,
            # Nucleus sampling — considers tokens within 90% cumulative probability
            top_p=0.9,
            # CPU optimization: use 6 of 8 M4 cores
            num_thread=6,
            # Keep model warm in memory for 10 min after last request
            keep_alive="10m",
        )
        logger.info(
            "Initialized LLM: %s (temp=%.2f, repeat_penalty=1.15, top_k=40, top_p=0.9)",
            model_name, temperature,
        )
    return _llm_cache[key]


def _build_prompt(system_prompt: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt + "\n\n---\n**Retrieved Context:**\n{context}"),
            ("human", "{question}"),
        ]
    )


def generate_stream(
    question: str,
    context: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.2,
    model_name: str = DEFAULT_MODEL,
):
    """
    Stream response tokens from local Ollama model via LangChain.
    Yields raw string chunks — caller handles SSE formatting and disclaimer.
    """
    llm = _get_llm(temperature=temperature, model_name=model_name)
    prompt = _build_prompt(system_prompt)
    chain = prompt | llm

    logger.info("Streaming for: '%s…' (model=%s)", question[:50], model_name)
    for chunk in chain.stream({"context": context, "question": question}):
        if chunk.content:
            yield chunk.content


def generate_response(question: str, context_chunks: list[str]) -> str:
    """Non-streaming response for the evaluation suite."""
    context = "\n".join(context_chunks)
    return "".join(generate_stream(question, context))


def evict_model_cache():
    """Clear the LLM singleton cache (called when admin changes model settings)."""
    _llm_cache.clear()
    logger.info("LLM cache cleared.")


def get_ollama_status() -> dict:
    """Check if Ollama server is reachable and return model list."""
    try:
        import httpx
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3.0)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            return {"online": True, "models": models}
        return {"online": False, "models": []}
    except Exception as e:
        logger.warning("Ollama unreachable: %s", e)
        return {"online": False, "models": [], "error": str(e)}
