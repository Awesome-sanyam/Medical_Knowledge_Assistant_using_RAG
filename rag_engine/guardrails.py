import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-Guardrail: Blocked intent patterns
# ---------------------------------------------------------------------------
BLOCKED_PATTERNS = [
    # Self-harm / suicidal intent
    r"\b(how to|ways to|help me)\s+(kill|harm|hurt|end|overdose|suicide)\b",
    # Illegal substance manufacturing
    r"\b(synthesize|manufacture|make|produce)\s+(meth|heroin|fentanyl|cocaine|drugs)\b",
    # Explicit dosage hallucination fishing
    r"\bgive me the (exact|specific|precise) (lethal|maximum|minimum)\s+dose\b",
    # Drug diversion / abuse
    r"\bhow (do i|can i|to)\s+(abuse|misuse|get high on|snort|inject)\b",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in BLOCKED_PATTERNS]

CLINICAL_DISCLAIMER = (
    "\n\n---\n"
    "⚠️ *Disclaimer: This response is generated for academic research purposes only "
    "and is not a substitute for professional clinical medical advice. "
    "Always consult a qualified healthcare provider.*"
)

LOW_CONTEXT_WARNING = (
    "I do not have sufficient verified medical literature in my knowledge index "
    "to answer this query accurately. Please consult a qualified medical professional "
    "or refer to peer-reviewed clinical sources."
)


def pre_guardrail(query: str) -> tuple[bool, str]:
    """
    Check the user query before sending to the RAG pipeline.

    Returns:
        (is_blocked: bool, reason: str)
        If is_blocked=True, 'reason' contains the message to return to the user.
    """
    query_stripped = query.strip()

    if not query_stripped:
        return True, "Please provide a valid medical question."

    if len(query_stripped) < 3:
        return True, "Your query is too short. Please provide more detail."

    for pattern in _COMPILED_PATTERNS:
        if pattern.search(query_stripped):
            logger.warning("Blocked query matched guardrail pattern: '%s'", query_stripped[:80])
            return True, (
                "I'm unable to respond to this type of query. "
                "If you or someone you know is in crisis, please contact emergency services "
                "or a mental health helpline immediately."
            )

    return False, ""


def low_context_fallback(low_confidence: bool, chunks: list) -> tuple[bool, str]:
    """
    Determine if we have enough context to answer safely.

    Returns:
        (should_fallback: bool, fallback_message: str)
    """
    if not chunks:
        return True, LOW_CONTEXT_WARNING

    if low_confidence and all(c.get("low_confidence", False) for c in chunks):
        return True, LOW_CONTEXT_WARNING

    return False, ""


def post_guardrail(response: str) -> str:
    """
    Append the mandatory clinical disclaimer to any generated response.
    """
    return response + CLINICAL_DISCLAIMER
