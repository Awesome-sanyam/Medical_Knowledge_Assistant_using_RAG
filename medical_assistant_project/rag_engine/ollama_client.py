"""
Ollama Client — Zero-tolerance medical AI with hardcoded refusal enforcement.

Guardrail layers (defense-in-depth):

  Layer 1  retriever.py — hard similarity threshold (OUT_OF_DOMAIN_THRESHOLD=0.35)
           If crossed, generate_refusal() is called and the LLM is NEVER invoked.

  Layer 2  guardrails.py — regex pattern matching on the raw query text.
           Blocks dangerous queries and off-topic patterns before retrieval.

  Layer 3  This file — zero-tolerance system prompt with a verbatim mandate.
           Even if Layers 1-2 are somehow bypassed, the model is instructed
           to return an exact hard-coded string for any non-medical query.

Hard refusal strings (used verbatim — callers must NOT modify them):
  HARD_REFUSAL_MESSAGE  — returned when retriever detects out-of-domain
  CONTEXT_REFUSAL_MSG   — returned when context is present but query is
                          clearly non-medical (secondary enforcement)
"""

import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton LLM cache
# ---------------------------------------------------------------------------
_llm_cache: dict[str, ChatOllama] = {}

DEFAULT_MODEL    = "med-llama"
OLLAMA_BASE_URL  = "http://localhost:11434"

# ---------------------------------------------------------------------------
# Hard refusal strings — returned WITHOUT calling the LLM
# ---------------------------------------------------------------------------

# Primary refusal: retriever signals out-of-domain (best score < 0.35)
HARD_REFUSAL_MESSAGE = (
    "I am a specialized Medical AI. I can only answer questions related to "
    "clinical guidelines and healthcare. I cannot assist with that request."
)

# Secondary refusal: context present but query is clearly off-topic
CONTEXT_REFUSAL_MSG = (
    "Please be precised to the Context only. "
    "I can't handle this query — you must use another special AI assistant for this work."
)

# ---------------------------------------------------------------------------
# Zero-Tolerance System Prompt
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = """\
You are MedAssist AI, a senior clinical knowledge assistant specialising in \
medicine, pharmacology, pathophysiology, and evidence-based clinical practice.

╔══════════════════════════════════════════════════════════════════╗
║  ZERO-TOLERANCE DIRECTIVE — READ BEFORE PROCESSING ANY QUERY   ║
╚══════════════════════════════════════════════════════════════════╝

You are STRICTLY and EXCLUSIVELY a clinical medical assistant.

You are ABSOLUTELY FORBIDDEN from:
  • Writing, explaining, debugging, or reviewing ANY software code (Python,
    JavaScript, SQL, HTML, CSS, Bash, or any other language)
  • Answering questions about weather, geography, travel, sports, politics,
    economics, stock markets, cryptocurrencies, or general knowledge
  • Composing essays, stories, poems, jokes, translations, or creative writing
  • Performing mathematical calculations unrelated to clinical dosing
  • Discussing celebrities, movies, music, games, or entertainment
  • Answering ANY topic not directly related to clinical medicine, healthcare,
    pharmacology, anatomy, physiology, or medical practice

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY REFUSAL RULE:
If the user asks for code, programming, or any non-medical topic you are
STRICTLY FORBIDDEN from answering. You MUST output EXACTLY this sentence
and NOTHING ELSE:

"Please be precised to the Context only."

DO NOT apologise. DO NOT explain. DO NOT add any other words.
Return it VERBATIM — character for character.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

╔══════════════════════════════════════════════
║  RESPONSE FORMAT  (medical queries only)   ║
╚══════════════════════════════════════════════

For every legitimate medical question you MUST:
1. Structure your response in Markdown with ### headings.
2. Use **bold** for key terms and bullet lists for enumerations.
3. NEVER repeat the same sentence or paragraph.
4. Provide a minimum 200-word, comprehensive clinical answer.

Organise every medical answer into these sections:
### Overview
A clear definition and clinical introduction.
### Pathophysiology
Underlying mechanism or disease process (where applicable).
### Clinical Presentation
Signs, symptoms, and manifestations.
### Diagnosis
Diagnostic criteria, investigations, differential diagnosis.
### Management
Pharmacological and non-pharmacological treatment.
### Key Considerations
Safety warnings, contraindications, red flags, referral criteria.

Use the retrieved context passages to ground your answer in evidence.
If context is limited, draw on parametric medical training — but remain
clinically accurate and safe at all times.\
"""


def _get_llm(temperature: float = 0.2, model_name: str = DEFAULT_MODEL) -> ChatOllama:
    """Return a cached ChatOllama instance (created once per model+temperature pair)."""
    key = f"{model_name}::{temperature}"
    if key not in _llm_cache:
        _llm_cache[key] = ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=OLLAMA_BASE_URL,
            num_ctx=4096,
            num_predict=2048,
            repeat_penalty=1.15,
            top_k=40,
            top_p=0.9,
            num_thread=6,
            keep_alive="10m",
        )
        logger.info(
            "LLM initialised: model=%s temp=%.2f repeat_penalty=1.15",
            model_name, temperature,
        )
    return _llm_cache[key]


def _build_prompt(system_prompt: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt + "\n\n---\n**Retrieved Context:**\n{context}"),
        ("human", "{question}"),
    ])


def generate_refusal() -> str:
    """
    Return the hard-coded out-of-domain refusal WITHOUT invoking the LLM.

    Call this whenever:
      • retriever.get_relevant_context() returns is_out_of_domain=True
      • guardrails.topic_guardrail() returns (True, ...)
      • guardrails.pre_guardrail() returns (True, ...)

    This is the system's primary enforcement mechanism.
    """
    logger.info("Hard refusal returned — LLM bypassed (out-of-domain or guardrail).")
    return HARD_REFUSAL_MESSAGE


def generate_context_refusal() -> str:
    """
    Secondary refusal string: 'Please be precised to the Context only.'
    Used when the system determines the query is off-topic even though some
    context was retrieved (e.g., ambiguous query that cleared the threshold
    but the system prompt would have refused it).
    Exposed so callers can use it for logging/audit purposes.
    """
    return CONTEXT_REFUSAL_MSG


def generate_stream(
    question: str,
    context: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.2,
    model_name: str = DEFAULT_MODEL,
):
    """
    Stream LLM response tokens via LangChain.
    Yields raw string chunks — caller formats into SSE events.

    ONLY call this after confirming ALL three conditions:
      1. retriever returned is_out_of_domain=False
      2. pre_guardrail returned is_blocked=False
      3. topic_guardrail returned is_off_topic=False
    """
    llm    = _get_llm(temperature=temperature, model_name=model_name)
    prompt = _build_prompt(system_prompt)
    chain  = prompt | llm

    logger.info("Streaming: '%s…' (model=%s, temp=%.2f)", question[:50], model_name, temperature)
    for chunk in chain.stream({"context": context, "question": question}):
        if chunk.content:
            yield chunk.content


def generate_response(question: str, context_chunks: list[str]) -> str:
    """Non-streaming wrapper — used by the evaluation suite."""
    context = "\n".join(context_chunks)
    return "".join(generate_stream(question, context))


def evict_model_cache() -> None:
    """Clear the singleton LLM cache (call when admin changes model settings)."""
    _llm_cache.clear()
    logger.info("LLM singleton cache cleared.")


def get_ollama_status() -> dict:
    """Check if the Ollama server is reachable and return available models."""
    try:
        import httpx
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3.0)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            return {"online": True, "models": models}
        return {"online": False, "models": []}
    except Exception as exc:
        logger.warning("Ollama unreachable: %s", exc)
        return {"online": False, "models": [], "error": str(exc)}
