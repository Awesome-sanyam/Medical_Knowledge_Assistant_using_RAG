import json
import logging
import os
import threading

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from chat_app.models import Conversation, Message, UserFeedback

from .models import Document, SystemConfig

logger = logging.getLogger(__name__)

try:
    from rag_engine.indexer import index_document
    INDEXER_AVAILABLE = True
except ImportError:
    INDEXER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Access Control — staff or model_admin_profile only
# ---------------------------------------------------------------------------

def _admin_required(view_func):
    """Decorator: restricts view to is_staff or model_admin_profile users."""
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect('/chat/login/?next=' + request.path)
        if not (request.user.is_staff or hasattr(request.user, 'model_admin_profile')):
            messages.error(request, 'You do not have permission to access the Admin panel. '
                           'Please contact your system administrator.')
            return redirect('chat_interface')
        return view_func(request, *args, **kwargs)
    wrapper.__name__ = view_func.__name__
    return wrapper


# ---------------------------------------------------------------------------
# Helper: background indexing thread
# ---------------------------------------------------------------------------

def _run_indexer_thread(doc_id: int):
    """Spawns a daemon thread to index a document without blocking the request."""
    def _worker(doc_id):
        from .models import Document  # re-import inside thread for Django ORM safety
        try:
            doc = Document.objects.get(id=doc_id)
            doc.status = "indexing"
            doc.save(update_fields=["status"])
            chunk_count = index_document(doc.file.path, doc_title=doc.title)
            doc.status = "completed"
            doc.chunk_count = chunk_count
            doc.save(update_fields=["status", "chunk_count"])
            logger.info("Indexed '%s' → %d chunks", doc.title, chunk_count)
        except Exception as e:
            try:
                doc = Document.objects.get(id=doc_id)
                doc.status = "failed"
                doc.error_message = str(e)[:500]
                doc.save(update_fields=["status", "error_message"])
            except Exception:
                pass
            logger.error("Indexing failed for doc_id=%d: %s", doc_id, e)

    threading.Thread(target=_worker, args=(doc_id,), daemon=True).start()


# ---------------------------------------------------------------------------
# Dashboard — Summary statistics only (no upload here)
# ---------------------------------------------------------------------------

@_admin_required
def admin_dashboard(request):
    """
    Main admin landing page.
    Shows document count, chunk stats, and quick links.
    Upload functionality has its own dedicated page: rag_upload.
    """
    documents = Document.objects.all().order_by("-uploaded_at")
    total_chunks = sum(d.chunk_count for d in documents if d.status == "completed")
    completed = sum(1 for d in documents if d.status == "completed")
    failed = sum(1 for d in documents if d.status == "failed")
    indexing = sum(1 for d in documents if d.status in ("pending", "indexing"))

    try:
        from rag_engine.ollama_client import get_ollama_status
        ollama = get_ollama_status()
    except Exception:
        ollama = {"online": False, "models": []}

    return render(request, "admin/dashboard.html", {
        "documents": documents,
        "total_chunks": total_chunks,
        "completed": completed,
        "failed": failed,
        "indexing": indexing,
        "ollama": ollama,
    })


# ---------------------------------------------------------------------------
# RAG Ingestion Portal — dedicated document upload & indexing page
# ---------------------------------------------------------------------------

@_admin_required
def rag_upload(request):
    """
    Dedicated RAG ingestion portal.
    POST: upload a PDF/TXT, persist it, trigger background indexing.
    GET:  render the drag-and-drop upload UI with document list.
    """
    if request.method == "POST":
        uploaded_file = request.FILES.get("document")

        if not uploaded_file:
            return JsonResponse({"status": "error", "message": "No file received."}, status=400)

        fname = uploaded_file.name.lower()
        if not (fname.endswith(".pdf") or fname.endswith(".txt")):
            return JsonResponse(
                {"status": "error", "message": "Invalid file type. Only PDF and TXT are accepted."},
                status=400,
            )

        max_bytes = 50 * 1024 * 1024  # 50 MB
        if uploaded_file.size > max_bytes:
            return JsonResponse(
                {"status": "error", "message": "File too large. Maximum size is 50 MB."},
                status=400,
            )

        # Ensure media directory exists
        from django.conf import settings
        os.makedirs(os.path.join(settings.MEDIA_ROOT, "medical_docs"), exist_ok=True)

        doc = Document.objects.create(title=uploaded_file.name, file=uploaded_file)
        logger.info("Document created: id=%d title='%s'", doc.id, doc.title)

        if INDEXER_AVAILABLE:
            _run_indexer_thread(doc.id)
        else:
            doc.status = "failed"
            doc.error_message = "Indexer not available (rag_engine import failed)."
            doc.save(update_fields=["status", "error_message"])

        return JsonResponse({
            "status": "ok",
            "message": f"'{uploaded_file.name}' uploaded. Indexing started.",
            "doc_id": doc.id,
            "doc_title": doc.title,
        })

    # GET — render the upload portal
    documents = Document.objects.all().order_by("-uploaded_at")
    total_chunks = sum(d.chunk_count for d in documents if d.status == "completed")

    return render(request, "admin/rag_upload.html", {
        "documents": documents,
        "total_chunks": total_chunks,
        "indexer_available": INDEXER_AVAILABLE,
    })


# ---------------------------------------------------------------------------
# Document lifecycle
# ---------------------------------------------------------------------------

@_admin_required
@require_POST
def delete_document(request, doc_id):
    """Remove a document record and its media file."""
    doc = get_object_or_404(Document, id=doc_id)
    title = doc.title
    try:
        doc.file.delete(save=False)
    except Exception:
        pass
    doc.delete()
    logger.info("Document deleted: '%s'", title)
    return redirect("rag_upload")


@_admin_required
def document_status(request, doc_id):
    """AJAX polling endpoint — returns current indexing status for a document."""
    doc = get_object_or_404(Document, id=doc_id)
    return JsonResponse({
        "status": doc.status,
        "chunk_count": doc.chunk_count,
        "error": doc.error_message or "",
    })


# ---------------------------------------------------------------------------
# System Prompt & Parameter Configurator
# ---------------------------------------------------------------------------

@_admin_required
def prompt_config(request):
    config = SystemConfig.get_active()

    if request.method == "POST":
        config.system_prompt = request.POST.get("system_prompt", config.system_prompt)
        try:
            config.temperature = float(request.POST.get("temperature", config.temperature))
            config.top_k_retrieval = int(request.POST.get("top_k_retrieval", config.top_k_retrieval))
            config.similarity_threshold = float(
                request.POST.get("similarity_threshold", config.similarity_threshold)
            )
        except (ValueError, TypeError):
            messages.error(request, "Invalid parameter value. Please check inputs.")
            return redirect("prompt_config")

        config.save()
        messages.success(request, "Configuration saved successfully.")
        return redirect("prompt_config")

    return render(request, "admin/prompt_config.html", {"config": config})


# ---------------------------------------------------------------------------
# Audit Log & Hallucination Inspector
# ---------------------------------------------------------------------------

@_admin_required
def audit_logs(request):
    filter_type = request.GET.get("filter", "all")

    audit_messages = (
        Message.objects.select_related("conversation", "feedback")
        .filter(role="assistant")
        .order_by("-created_at")[:200]
    )

    if filter_type == "flagged":
        flagged = []
        for m in audit_messages:
            is_flagged = False
            try:
                if hasattr(m, "feedback") and m.feedback and not m.feedback.is_positive:
                    is_flagged = True
            except Exception:
                pass
            if not is_flagged and (
                m.retrieved_context
                and isinstance(m.retrieved_context, list)
                and any(c.get("low_confidence") for c in m.retrieved_context)
            ):
                is_flagged = True
            if is_flagged:
                flagged.append(m)
        audit_messages = flagged

    return render(
        request,
        "admin/audit_logs.html",
        {"audit_messages": audit_messages, "filter_type": filter_type},
    )
