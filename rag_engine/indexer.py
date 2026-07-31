import os
import logging
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "chroma_db")

# Singleton embedding model (CPU/MPS-optimized for Apple Silicon)
_embeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def index_document(file_path: str, doc_title: str = "") -> int:
    """
    Load a PDF or TXT file, chunk it, and index into ChromaDB.
    Returns the number of chunks indexed.
    Raises on failure so the caller can update Document.status = 'failed'.
    """
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}. Only PDF and TXT are supported.")

    documents = loader.load()
    if not documents:
        raise ValueError(f"No content extracted from {file_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Document produced zero chunks after splitting.")

    # Enrich metadata on each chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata["source_title"] = doc_title or os.path.basename(file_path)
        chunk.metadata["chunk_index"] = i

    client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        client=client,
        collection_name="medical_knowledge",
    )

    logger.info("Indexed %d chunks from '%s'", len(chunks), file_path)
    return len(chunks)


# Backward compatibility alias used by admin_app
def index_pdf(pdf_path: str) -> int:
    return index_document(pdf_path)
