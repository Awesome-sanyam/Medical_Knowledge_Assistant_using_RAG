import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from .guardrails import CLINICAL_DISCLAIMER

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM client — model name and params are loaded fresh per-request from
# SystemConfig so admin can tune them without a server restart.
# ---------------------------------------------------------------------------


def _get_llm(temperature: float = 0.2, model_name: str = "med-llama") -> ChatOllama:
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        base_url="http://localhost:11434",
    )


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
    system_prompt: str = (
        "You are a clinical-grade medical AI assistant. "
        "Answer using only the provided context. "
        "If the context is insufficient, state that you do not know."
    ),
    temperature: float = 0.2,
    model_name: str = "med-llama",
):
    """
    Stream response tokens from local Ollama model via LangChain.
    Yields raw string chunks — caller is responsible for SSE formatting.
    Does NOT append disclaimer here; post_guardrail() should be called by the view.
    """
    llm = _get_llm(temperature=temperature, model_name=model_name)
    prompt = _build_prompt(system_prompt)
    chain = prompt | llm

    logger.info("Streaming response for question: '%s…'", question[:60])
    for chunk in chain.stream({"context": context, "question": question}):
        if chunk.content:
            yield chunk.content


def generate_response(question: str, context_chunks: list[str]) -> str:
    """
    Non-streaming response for evaluation suite.
    Returns the full assembled response string (without disclaimer).
    """
    context = "\n".join(context_chunks)
    full_response = ""
    for chunk in generate_stream(question, context):
        full_response += chunk
    return full_response
