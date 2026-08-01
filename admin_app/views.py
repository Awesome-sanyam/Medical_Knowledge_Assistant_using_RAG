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
# Dashboard — Document Manager
# ---------------------------------------------------------------------------


@_admin_required
def admin_dashboard(request):
    if request.method == "POST":
        uploaded_file = request.FILES.get("document")
        if not uploaded_file:
            messages.error(request, "No file selected. Please choose a PDF or TXT file.")
            return redirect("admin_dashboard")

        # Validate file type
        fname = uploaded_file.name.lower()
        if not (fname.endswith(".pdf") or fname.endswith(".txt")):
            messages.error(request, "Invalid file type. Only PDF and TXT files are accepted.")
            return redirect("admin_dashboard")

        # Ensure media directory exists
        from django.conf import settings
        upload_dir = os.path.join(settings.MEDIA_ROOT, "medical_docs")
        os.makedirs(upload_dir, exist_ok=True)

        doc = Document.objects.create(title=uploaded_file.name, file=uploaded_file)
        messages.success(request, f"'{uploaded_file.name}' uploaded successfully. Indexing started...")

        if INDEXER_AVAILABLE:
            def run_indexer(doc_id):
                from .models import Document
                doc = Document.objects.get(id=doc_id)
                doc.status = "indexing"
                doc.save(update_fields=["status"])
                try:
                    chunk_count = index_document(doc.file.path, doc_title=doc.title)
                    doc.status = "completed"
                    doc.chunk_count = chunk_count
                    doc.save(update_fields=["status", "chunk_count"])
                    logger.info("Indexed '%s' → %d chunks", doc.title, chunk_count)
                except Exception as e:
                    doc.status = "failed"
                    doc.error_message = str(e)
                    doc.save(update_fields=["status", "error_message"])
                    logger.error("Indexing failed for '%s': %s", doc.title, e)

            threading.Thread(target=run_indexer, args=(doc.id,), daemon=True).start()
        else:
            doc.status = "failed"
            doc.error_message = "Indexer not available."
            doc.save()

        return redirect("admin_dashboard")

    documents = Document.objects.all().order_by("-uploaded_at")
    return render(request, "admin/dashboard.html", {"documents": documents})


@_admin_required
@require_POST
def delete_document(request, doc_id):
    """Remove a document record (does not purge from ChromaDB — see note)."""
    doc = get_object_or_404(Document, id=doc_id)
    doc.file.delete(save=False)
    doc.delete()
    return redirect("admin_dashboard")


@_admin_required
def document_status(request, doc_id):
    """AJAX polling endpoint for live status updates."""
    doc = get_object_or_404(Document, id=doc_id)
    return JsonResponse(
        {
            "status": doc.status,
            "chunk_count": doc.chunk_count,
            "error": doc.error_message or "",
        }
    )


# ---------------------------------------------------------------------------
# Prompt & Parameter Configurator
# ---------------------------------------------------------------------------


@_admin_required
def prompt_config(request):
    config = SystemConfig.get_active()

    if request.method == "POST":
        config.system_prompt = request.POST.get("system_prompt", config.system_prompt)
        config.temperature = float(request.POST.get("temperature", config.temperature))
        config.top_k_retrieval = int(request.POST.get("top_k_retrieval", config.top_k_retrieval))
        config.similarity_threshold = float(
            request.POST.get("similarity_threshold", config.similarity_threshold)
        )
        config.save()
        return redirect("prompt_config")

    return render(request, "admin/prompt_config.html", {"config": config})


# ---------------------------------------------------------------------------
# Audit Log & Hallucination Inspector
# ---------------------------------------------------------------------------


@_admin_required
def audit_logs(request):
    # Filter: flagged = messages with negative feedback OR low-confidence context
    filter_type = request.GET.get("filter", "all")

    audit_messages = (
        Message.objects.select_related("conversation", "feedback")
        .filter(role="assistant")
        .order_by("-created_at")[:200]
    )

    if filter_type == "flagged":
        audit_messages = [
            m
            for m in audit_messages
            if (
                hasattr(m, "feedback") and not m.feedback.is_positive
            )
            or (
                m.retrieved_context
                and isinstance(m.retrieved_context, list)
                and any(c.get("low_confidence") for c in m.retrieved_context)
            )
        ]

    return render(
        request,
        "admin/audit_logs.html",
        {"audit_messages": audit_messages, "filter_type": filter_type},
    )
