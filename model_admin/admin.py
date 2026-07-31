from django.contrib import admin

from .models import AdminUser, ModelSession


@admin.register(AdminUser)
class AdminUserAdmin(admin.ModelAdmin):
    """
    Model Controlling Admin registration — only creatable by Django superusers.
    This is the sole path for granting model admin access.
    """

    list_display = ["user", "created_at"]
    search_fields = ["user__username", "user__email"]
    readonly_fields = ["created_at"]


@admin.register(ModelSession)
class ModelSessionAdmin(admin.ModelAdmin):
    list_display = ["model_name", "status", "action", "timestamp"]
    list_filter = ["status", "action"]
    readonly_fields = ["model_name", "status", "action", "note", "timestamp"]
    ordering = ["-timestamp"]
