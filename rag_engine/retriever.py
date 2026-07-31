import os
import logging
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma

from .indexer import get_embeddings, VECTOR_STORE_PATH

logger = logging.getLogger(__name__)

COLLECTION_NAME = "medical_knowledge"


def _get_vector_store() -> Chroma:
    client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
    )


def get_relevant_context(
    query: str, top_k: int = 3, similarity_threshold: float = 0.65
) -> tuple[list[dict], bool]:
    """
    Hybrid retrieval: Vector similarity (ChromaDB) + BM25 keyword ranking.

    Returns:
        (context_chunks, low_confidence_flag)
        Each chunk dict has: {'content', 'source', 'score', 'chunk_index'}
    """
    try:
        store = _get_vector_store()
        collection = store._collection
        count = collection.count()
    except Exception as e:
        logger.warning("ChromaDB unavailable: %s", e)
        return [], True

    if count == 0:
        logger.info("Vector store is empty. No context available.")
        return [], True

    # --- Stage 1: Dense vector retrieval ---
    results = store.similarity_search_with_relevance_scores(query, k=min(top_k * 3, count))

    if not results:
        return [], True

    # --- Stage 2: BM25 re-ranking over the dense candidates ---
    docs = [doc for doc, _ in results]
    tokenized_corpus = [doc.page_content.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.lower().split())

    # Combine: 60% vector score + 40% BM25 (normalized)
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
    combined = []
    for i, (doc, vec_score) in enumerate(results):
        norm_bm25 = bm25_scores[i] / max_bm25
        final_score = 0.6 * vec_score + 0.4 * norm_bm25
        combined.append((doc, final_score))

    # Sort by hybrid score descending
    combined.sort(key=lambda x: x[1], reverse=True)
    top_results = combined[:top_k]

    # --- Stage 3: Build output with confidence flag ---
    context_chunks = []
    low_confidence = False

    for doc, score in top_results:
        if score < similarity_threshold:
            low_confidence = True
        context_chunks.append(
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source_title", doc.metadata.get("source", "Unknown")),
                "chunk_index": doc.metadata.get("chunk_index", -1),
                "score": round(float(score), 4),
                "low_confidence": score < similarity_threshold,
            }
        )

    logger.info(
        "Retrieved %d chunks for query '%s…' (low_confidence=%s)",
        len(context_chunks),
        query[:40],
        low_confidence,
    )
    return context_chunks, low_confidence


def fetch_context(query: str, top_k: int = 3) -> list[str]:
    """
    Backward-compatible wrapper that returns plain text chunks.
    Used by the streaming view.
    """
    chunks, _ = get_relevant_context(query, top_k=top_k)
    return [c["content"] for c in chunks]
