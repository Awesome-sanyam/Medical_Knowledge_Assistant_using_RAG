from django.urls import path

from . import views

urlpatterns = [
    path("", views.admin_dashboard, name="admin_dashboard"),
    path("doc/<int:doc_id>/delete/", views.delete_document, name="delete_document"),
    path("doc/<int:doc_id>/status/", views.document_status, name="document_status"),
    path("config/", views.prompt_config, name="prompt_config"),
    path("audit/", views.audit_logs, name="audit_logs"),
]
