from django.contrib import admin
from django.contrib.auth.models import User

from .models import AdminUser, ModelSession


def activate_accounts(modeladmin, request, queryset):
    """Action: approve pending model admin registrations."""
    for admin_user in queryset:
        admin_user.user.is_active = True
        admin_user.user.save(update_fields=['is_active'])
    count = queryset.count()
    modeladmin.message_user(request, f"{count} admin account(s) approved and activated.")
activate_accounts.short_description = "✓ Approve selected registrations (activate)"


def reject_accounts(modeladmin, request, queryset):
    """Action: reject + delete pending model admin registrations."""
    for admin_user in queryset:
        user = admin_user.user
        admin_user.delete()
        user.delete()
    modeladmin.message_user(request, "Selected registrations rejected and deleted.")
reject_accounts.short_description = "✗ Reject & delete selected registrations"


@admin.register(AdminUser)
class AdminUserAdmin(admin.ModelAdmin):
    """
    Model Controlling Admin management.
    Approve pending self-registrations using the 'Approve' action.
    Reject with the 'Reject' action.
    """
    list_display = ["user", "get_email", "get_is_active", "created_at"]
    list_filter = ["user__is_active"]
    search_fields = ["user__username", "user__email"]
    readonly_fields = ["created_at"]
    actions = [activate_accounts, reject_accounts]

    @admin.display(description="Email")
    def get_email(self, obj):
        return obj.user.email

    @admin.display(description="Status", boolean=False)
    def get_is_active(self, obj):
        return "✓ Active" if obj.user.is_active else "⏳ Pending Approval"


@admin.register(ModelSession)
class ModelSessionAdmin(admin.ModelAdmin):
    list_display = ["model_name", "status", "action", "timestamp"]
    list_filter = ["status", "action"]
    readonly_fields = ["model_name", "status", "action", "note", "timestamp"]
    ordering = ["-timestamp"]
