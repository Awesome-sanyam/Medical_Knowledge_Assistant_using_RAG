"""
Retriever with persistent ChromaDB client singleton.

Key optimizations:
- _CHROMA_CLIENT is created once at import time, not per request
- Embedding model is loaded once via indexer.get_embeddings() singleton
- BM25 re-ranking only runs on the dense candidate set (not entire corpus)
"""

import logging
from functools import lru_cache

import chromadb
from rank_bm25 import BM25Okapi

from .indexer import VECTOR_STORE_PATH, get_embeddings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "medical_knowledge"

# ---------------------------------------------------------------------------
# Singleton ChromaDB client — created once, reused for every request
# ---------------------------------------------------------------------------
_CHROMA_CLIENT: chromadb.PersistentClient | None = None


def _get_client() -> chromadb.PersistentClient:
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        _CHROMA_CLIENT = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
        logger.info("ChromaDB persistent client initialized at: %s", VECTOR_STORE_PATH)
    return _CHROMA_CLIENT


def reset_client():
    """Force a client reset (called after new documents are indexed)."""
    global _CHROMA_CLIENT
    _CHROMA_CLIENT = None
    logger.info("ChromaDB client reset — will reinitialize on next request.")


def get_relevant_context(
    query: str, top_k: int = 3, similarity_threshold: float = 0.65
) -> tuple[list[dict], bool]:
    """
    Hybrid retrieval: Dense vector search (ChromaDB) + BM25 keyword re-ranking.

    Returns:
        (context_chunks, low_confidence_flag)
        Each chunk: {'content', 'source', 'score', 'chunk_index', 'low_confidence'}
    """
    from langchain_chroma import Chroma

    try:
        client = _get_client()
        # Check collection exists and has documents
        try:
            collection = client.get_collection(COLLECTION_NAME)
            count = collection.count()
        except Exception:
            logger.info("Collection '%s' not found — vector store empty.", COLLECTION_NAME)
            return [], True

        if count == 0:
            return [], True

        store = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
        )
    except Exception as e:
        logger.warning("ChromaDB unavailable: %s", e)
        return [], True

    # --- Stage 1: Dense vector retrieval ---
    fetch_k = min(top_k * 3, count)
    try:
        results = store.similarity_search_with_relevance_scores(query, k=fetch_k)
    except Exception as e:
        logger.error("Vector search failed: %s", e)
        return [], True

    if not results:
        return [], True

    # --- Stage 2: BM25 re-ranking over dense candidates ---
    docs = [doc for doc, _ in results]
    tokenized_corpus = [doc.page_content.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.lower().split())

    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
    combined = []
    for i, (doc, vec_score) in enumerate(results):
        norm_bm25 = bm25_scores[i] / max_bm25
        # 60% vector similarity + 40% BM25
        final_score = 0.6 * vec_score + 0.4 * norm_bm25
        combined.append((doc, final_score))

    combined.sort(key=lambda x: x[1], reverse=True)
    top_results = combined[:top_k]

    # --- Stage 3: Build structured output ---
    context_chunks = []
    low_confidence = False

    for doc, score in top_results:
        is_low = score < similarity_threshold
        if is_low:
            low_confidence = True
        context_chunks.append(
            {
                "content": doc.page_content,
                "source": doc.metadata.get(
                    "source_title", doc.metadata.get("source", "Unknown")
                ),
                "chunk_index": doc.metadata.get("chunk_index", -1),
                "score": round(float(score), 4),
                "low_confidence": is_low,
            }
        )

    logger.info(
        "Retrieved %d chunks for '%s…' (low_confidence=%s)",
        len(context_chunks),
        query[:40],
        low_confidence,
    )
    return context_chunks, low_confidence


def fetch_context(query: str, top_k: int = 3) -> list[str]:
    """Backward-compatible plain-text wrapper."""
    chunks, _ = get_relevant_context(query, top_k=top_k)
    return [c["content"] for c in chunks]
