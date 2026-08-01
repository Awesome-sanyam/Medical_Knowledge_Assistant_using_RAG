"""
Guardrails — Pre/post processing for medical query safety.

Three layers:
1. pre_guardrail(): Blocks dangerous queries (self-harm, drug manufacture)
2. topic_guardrail(): Detects non-medical queries and politely refuses
3. post_guardrail(): Appends clinical disclaimer to every response
"""

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-Guardrail: Blocked intent patterns (dangerous queries)
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

# ---------------------------------------------------------------------------
# Topic Guardrail: Non-medical topic detection
# ---------------------------------------------------------------------------
# Keywords that strongly indicate a NON-medical question
_NON_MEDICAL_INDICATORS = [
    r"\b(write|generate|create|build|fix|debug)\s+(code|program|script|function|app|website|html|css|python|javascript|java|sql)\b",
    r"\b(weather|forecast|temperature outside|rain today|sunny)\b",
    r"\b(stock market|bitcoin|cryptocurrency|trading|forex|invest in)\b",
    r"\b(recipe|cook|bake|ingredients for)\b",
    r"\b(who won|score|game|match|tournament|world cup|olympics)\b",
    r"\b(president|prime minister|election|democrat|republican|political|politics|vote for)\b",
    r"\b(movie|film|tv show|netflix|actor|actress|celebrity)\b",
    r"\b(joke|funny|meme|riddle|trivia|quiz)\b",
    r"\b(essay|homework|assignment|write me a|compose a)\b",
    r"\b(translate|translation|convert .* to .*language)\b",
    r"\b(math problem|calculate|equation|algebra|calculus)\b",
    r"\b(travel|flight|hotel|booking|vacation|tourist)\b",
]

_COMPILED_NON_MEDICAL = [re.compile(p, re.IGNORECASE) for p in _NON_MEDICAL_INDICATORS]

# Positive medical keywords — if present, allow the query even if ambiguous
_MEDICAL_KEYWORDS = re.compile(
    r"\b(symptom|disease|diagnosis|treatment|medication|drug|dose|dosage|"
    r"patient|clinical|therapy|surgery|pathology|anatomy|physiology|"
    r"infection|virus|bacteria|cancer|tumor|diabetes|hypertension|"
    r"heart|lung|liver|kidney|brain|blood|bone|muscle|nerve|"
    r"prescription|antibiotic|vaccine|immunization|allergy|"
    r"pain|fever|cough|nausea|headache|fatigue|rash|swelling|"
    r"emergency|ICU|hospital|clinic|doctor|nurse|physician|"
    r"medical|healthcare|medicine|pharma|health|wellness|"
    r"pregnancy|prenatal|pediatric|geriatric|oncology|cardiology|"
    r"radiology|MRI|CT scan|X-ray|ultrasound|ECG|EKG|"
    r"mental health|depression|anxiety|PTSD|bipolar|schizophrenia|"
    r"first aid|CPR|AED|trauma|wound|fracture|sprain|burn|"
    r"nutrition|diet|BMI|obesity|cholesterol|glucose|insulin|"
    r"guideline|protocol|management|prognosis|etiology|epidemiology)\b",
    re.IGNORECASE,
)


CLINICAL_DISCLAIMER = (
    "\n\n---\n"
    "⚠️ *Disclaimer: This response is generated for academic research purposes only "
    "and is not a substitute for professional clinical medical advice. "
    "Always consult a qualified healthcare provider.*"
)

NON_MEDICAL_REFUSAL = (
    "I am **MedAssist AI**, a specialized Clinical Knowledge Assistant. "
    "I can only provide information related to **medical guidelines, healthcare, "
    "anatomy, pharmacology, and clinical practice** based on my verified database.\n\n"
    "I'm not able to answer questions about non-medical topics such as coding, "
    "weather, politics, entertainment, or general knowledge.\n\n"
    "**How can I help you with a medical question today?**"
)


def pre_guardrail(query: str) -> tuple[bool, str]:
    """
    Check the user query before sending to the RAG pipeline.
    Returns: (is_blocked, reason)
    """
    query_stripped = query.strip()

    if not query_stripped:
        return True, "Please provide a valid medical question."

    if len(query_stripped) < 3:
        return True, "Your query is too short. Please provide more detail."

    for pattern in _COMPILED_PATTERNS:
        if pattern.search(query_stripped):
            logger.warning("Blocked dangerous query: '%s'", query_stripped[:80])
            return True, (
                "I'm unable to respond to this type of query. "
                "If you or someone you know is in crisis, please contact emergency services "
                "or a mental health helpline immediately."
            )

    return False, ""


def topic_guardrail(query: str) -> tuple[bool, str]:
    """
    Detect non-medical queries and return a polite refusal.
    Returns: (is_off_topic, refusal_message)

    Logic:
    1. If query contains strong medical keywords → ALLOW (even if ambiguous)
    2. If query matches a non-medical pattern → BLOCK
    3. Otherwise → ALLOW (benefit of the doubt)
    """
    query_lower = query.strip().lower()

    # Short queries get benefit of the doubt
    if len(query_lower) < 10:
        return False, ""

    # If strong medical keywords are present, always allow
    if _MEDICAL_KEYWORDS.search(query_lower):
        return False, ""

    # Check against non-medical indicators
    for pattern in _COMPILED_NON_MEDICAL:
        if pattern.search(query_lower):
            logger.info("Off-topic query detected: '%s'", query_lower[:80])
            return True, NON_MEDICAL_REFUSAL

    return False, ""


def low_context_fallback(low_confidence: bool, chunks: list) -> tuple[bool, str]:
    """
    Determine if we have enough context to answer safely.
    We let the model use its parametric knowledge if RAG context is empty.
    The UI will still indicate if retrieved chunks are low confidence.
    """
    return False, ""


def post_guardrail(response: str) -> str:
    """Append the mandatory clinical disclaimer to any generated response."""
    return response + CLINICAL_DISCLAIMER
