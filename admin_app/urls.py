from django.urls import path

from . import views

urlpatterns = [
    # Dashboard — document index & stats
    path("", views.admin_dashboard, name="admin_dashboard"),

    # RAG Ingestion Portal — dedicated upload & indexing page
    path("rag/", views.rag_upload, name="rag_upload"),

    # Document lifecycle endpoints
    path("doc/<int:doc_id>/delete/", views.delete_document, name="delete_document"),
    path("doc/<int:doc_id>/status/", views.document_status, name="document_status"),

    # LLM / System configurator
    path("config/", views.prompt_config, name="prompt_config"),

    # Audit & hallucination inspector
    path("audit/", views.audit_logs, name="audit_logs"),
]
