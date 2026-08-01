"""
Ollama Client with connection pooling and singleton LLM instance.

Key optimizations:
- Single LLM instance per (model, temperature) combination cached in a dict
- num_ctx=4096 limits context window RAM usage
- num_predict=2048 forces long-form clinical answers (critical fix)
- num_thread=6 uses only 6 CPU threads (leaves 2 free on M4)
- keep_alive="10m" keeps model warm but releases RAM after 10 min idle
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
# Enterprise System Prompt — forces comprehensive, long-form clinical answers
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = (
    "You are a senior clinical AI assistant with deep expertise in medicine, "
    "pharmacology, pathophysiology, and evidence-based clinical practice. "
    "You MUST provide highly detailed, comprehensive, and well-structured answers. "
    "NEVER give one-sentence or brief replies. "
    "Always expand your response with:\n"
    "1. A clear definition or overview of the topic\n"
    "2. Detailed pathophysiology or mechanism when relevant\n"
    "3. Clinical presentation, signs and symptoms\n"
    "4. Diagnostic criteria and investigations\n"
    "5. Management approach including pharmacological and non-pharmacological options\n"
    "6. Important safety considerations, contraindications, or red flags\n\n"
    "Use the provided context to ground your answer. If the retrieved context covers "
    "the topic, integrate it thoroughly. If context is limited or absent, draw on your "
    "extensive parametric medical training to provide an accurate, empathetic, and "
    "clinically safe response. Use markdown formatting with headers, bullet points, "
    "and bold text for readability."
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
            # RAM optimization: limit context window
            num_ctx=4096,
            # *** CRITICAL FIX: Force long-form generation ***
            # Without this, Ollama defaults to ~128 tokens → short replies
            num_predict=2048,
            # CPU optimization: use 6 of 8 M4 cores, leave 2 for system
            num_thread=6,
            # Keep model warm in memory for 10 min after last request
            keep_alive="10m",
        )
        logger.info("Initialized LLM instance: %s (temp=%.2f, num_predict=2048)", model_name, temperature)
    return _llm_cache[key]


def _build_prompt(system_prompt: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt + "\n\nContext:\n{context}"),
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
    Uses cached LLM instance for efficiency.
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
    """
    Non-streaming response for the evaluation suite.
    Returns the full assembled response string.
    """
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
