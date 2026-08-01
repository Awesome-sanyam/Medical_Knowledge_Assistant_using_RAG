"""
Retriever — Hybrid Dense + BM25 retrieval with hard out-of-domain guardrail.

Architecture:
  Stage 1  — ChromaDB dense vector search (BGE embeddings)
  Stage 2  — BM25 keyword re-ranking over the dense candidate set
  Stage 3  — Hard similarity threshold: if best hybrid score < OUT_OF_DOMAIN_THRESHOLD
              the function short-circuits, returns is_out_of_domain=True, and the
              LLM is NEVER called.  A hardcoded refusal is returned instead.

Thresholds (tunable):
  OUT_OF_DOMAIN_THRESHOLD = 0.35  — triggers hard refusal (LLM bypassed)
  LOW_CONFIDENCE_THRESHOLD = 0.65 — soft flag shown in Audit Inspector
"""

import logging

import chromadb
from rank_bm25 import BM25Okapi

from .indexer import VECTOR_STORE_PATH, get_embeddings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "medical_knowledge"

# ---------------------------------------------------------------------------
# Hard out-of-domain guardrail
# If the best combined score is below this value the retriever short-circuits.
# A pure coding/weather/trivia query against a clinical KB typically scores
# 0.05–0.20; a relevant medical query typically scores 0.40–0.90.
# ---------------------------------------------------------------------------
OUT_OF_DOMAIN_THRESHOLD: float = 0.35

# Soft low-confidence threshold — used for Audit Inspector flagging only.
# Does NOT block the LLM call; it just annotates the chunk dict.
LOW_CONFIDENCE_THRESHOLD: float = 0.65

# ---------------------------------------------------------------------------
# Singleton ChromaDB client
# ---------------------------------------------------------------------------
_CHROMA_CLIENT: chromadb.PersistentClient | None = None


def _get_client() -> chromadb.PersistentClient:
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        _CHROMA_CLIENT = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
        logger.info("ChromaDB client initialised at: %s", VECTOR_STORE_PATH)
    return _CHROMA_CLIENT


def reset_client() -> None:
    """Force a singleton reset — call after new documents are indexed."""
    global _CHROMA_CLIENT
    _CHROMA_CLIENT = None
    logger.info("ChromaDB client reset.")


def get_relevant_context(
    query: str,
    top_k: int = 3,
    similarity_threshold: float = LOW_CONFIDENCE_THRESHOLD,
) -> tuple[list[dict], bool, bool]:
    """
    Hybrid retrieval with a hard out-of-domain short-circuit.

    Returns:
        (context_chunks, low_confidence, is_out_of_domain)

        context_chunks   — list of dicts:
                           {'content', 'source', 'score', 'chunk_index',
                            'low_confidence', 'out_of_domain'}
        low_confidence   — True if any chunk scored below similarity_threshold
        is_out_of_domain — True if the query has NO meaningful overlap with
                           the medical KB (best score < OUT_OF_DOMAIN_THRESHOLD).
                           When True, context_chunks is [] and the caller MUST
                           NOT invoke the LLM — use generate_refusal() instead.
    """
    from langchain_chroma import Chroma

    # ── 0. Collection sanity check ──────────────────────────────────────────
    try:
        client = _get_client()
        try:
            collection = client.get_collection(COLLECTION_NAME)
            count = collection.count()
        except Exception:
            logger.info("Collection '%s' not found — KB is empty.", COLLECTION_NAME)
            # Empty KB: cannot determine domain; let LLM use parametric knowledge
            return [], True, False

        if count == 0:
            return [], True, False

        store = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
        )
    except Exception as exc:
        logger.warning("ChromaDB unavailable: %s", exc)
        return [], True, False

    # ── 1. Dense vector retrieval ────────────────────────────────────────────
    fetch_k = min(top_k * 3, count)
    try:
        results = store.similarity_search_with_relevance_scores(query, k=fetch_k)
    except Exception as exc:
        logger.error("Vector search failed: %s", exc)
        return [], True, False

    if not results:
        return [], True, False

    # ── 2. BM25 re-ranking over the dense candidate set ─────────────────────
    docs = [doc for doc, _ in results]
    tokenized = [doc.page_content.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query.lower().split())

    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
    combined = []
    for i, (doc, vec_score) in enumerate(results):
        norm_bm25  = bm25_scores[i] / max_bm25
        # Weighted blend: 60 % dense vector + 40 % BM25
        final_score = 0.6 * vec_score + 0.4 * norm_bm25
        combined.append((doc, final_score))

    combined.sort(key=lambda x: x[1], reverse=True)

    # ── 3. HARD OUT-OF-DOMAIN GUARDRAIL ─────────────────────────────────────
    best_score = combined[0][1] if combined else 0.0
    if best_score < OUT_OF_DOMAIN_THRESHOLD:
        logger.warning(
            "OUT-OF-DOMAIN: query='%s…' best_score=%.4f < threshold=%.4f — LLM bypassed.",
            query[:60], best_score, OUT_OF_DOMAIN_THRESHOLD,
        )
        return [], True, True   # (no_chunks, low_conf=True, out_of_domain=True)

    # ── 4. Build structured chunk list ──────────────────────────────────────
    top_results    = combined[:top_k]
    context_chunks = []
    low_confidence = False

    for doc, score in top_results:
        is_low = score < similarity_threshold
        if is_low:
            low_confidence = True
        context_chunks.append({
            "content":      doc.page_content,
            "source":       doc.metadata.get("source_title", doc.metadata.get("source", "Unknown")),
            "chunk_index":  int(doc.metadata.get("chunk_index", -1)),
            "score":        round(float(score), 4),
            "low_confidence": bool(is_low),
            "out_of_domain":  False,
        })

    logger.info(
        "Retrieved %d chunks for '%s…' (best=%.4f, low_conf=%s)",
        len(context_chunks), query[:40], best_score, low_confidence,
    )
    return context_chunks, low_confidence, False


def fetch_context(query: str, top_k: int = 3) -> list[str]:
    """Backward-compatible plain-text wrapper (used by eval suite)."""
    chunks, _, _ = get_relevant_context(query, top_k=top_k)
    return [c["content"] for c in chunks]
