import json
import logging
import threading

from django.contrib.admin.views.decorators import staff_member_required
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
# Dashboard — Document Manager
# ---------------------------------------------------------------------------


@staff_member_required
def admin_dashboard(request):
    if request.method == "POST":
        uploaded_file = request.FILES.get("document")
        if uploaded_file:
            doc = Document.objects.create(title=uploaded_file.name, file=uploaded_file)

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


@staff_member_required
@require_POST
def delete_document(request, doc_id):
    """Remove a document record (does not purge from ChromaDB — see note)."""
    doc = get_object_or_404(Document, id=doc_id)
    doc.file.delete(save=False)
    doc.delete()
    return redirect("admin_dashboard")


@staff_member_required
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


@staff_member_required
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


@staff_member_required
def audit_logs(request):
    # Filter: flagged = messages with negative feedback OR low-confidence context
    filter_type = request.GET.get("filter", "all")

    messages = (
        Message.objects.select_related("conversation", "feedback")
        .filter(role="assistant")
        .order_by("-created_at")[:200]
    )

    if filter_type == "flagged":
        messages = [
            m
            for m in messages
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
        {"messages": messages, "filter_type": filter_type},
    )
