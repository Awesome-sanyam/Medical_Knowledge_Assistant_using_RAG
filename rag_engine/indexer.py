"""
Fixed indexer.py:
- Singleton embedding model (loaded once, never reloaded between requests)
- Calls retriever.reset_client() after indexing so the new docs are visible immediately
- Supports PDF and TXT files
- Returns chunk_count on success, raises on failure
"""

import logging
import os

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "medical_knowledge"

# ---------------------------------------------------------------------------
# Singleton embedding model — loaded once at module import time
# Model is small (33MB) so startup cost is acceptable
# ---------------------------------------------------------------------------
_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Returns the singleton BGE embedding model.
    Loads on first call, cached on subsequent calls.
    Uses CPU explicitly so it doesn't compete with Ollama for GPU/MPS memory.
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("BGE embedding model loaded (CPU).")
    return _embeddings


def index_document(file_path: str, doc_title: str = "") -> int:
    """
    Load a PDF or TXT, chunk it, and upsert into ChromaDB.
    After indexing, resets the retriever's singleton client so the new
    documents are immediately visible to queries.

    Returns: number of chunks indexed
    Raises: ValueError or Exception on failure
    """
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: '{ext}'. Only PDF and TXT are accepted.")

    documents = loader.load()
    if not documents:
        raise ValueError(f"No extractable text found in: {file_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Document produced zero text chunks after splitting.")

    title = doc_title or os.path.basename(file_path)
    for i, chunk in enumerate(chunks):
        chunk.metadata["source_title"] = title
        chunk.metadata["chunk_index"] = i

    client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        client=client,
        collection_name=COLLECTION_NAME,
    )

    logger.info("Indexed %d chunks from '%s'", len(chunks), file_path)

    # Reset retriever singleton so new docs are immediately queryable
    try:
        from rag_engine.retriever import reset_client
        reset_client()
    except Exception:
        pass

    return len(chunks)


# Backward compatibility alias
def index_pdf(pdf_path: str) -> int:
    return index_document(pdf_path)
