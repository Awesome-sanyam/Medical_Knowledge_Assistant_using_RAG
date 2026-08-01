from django.db import models
from django.contrib.auth.models import User


class AdminUser(models.Model):
    """
    Standalone admin user for the Model Control Panel.
    Completely independent from Django's built-in admin system.
    """
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="model_admin_profile")
    pin = models.CharField(max_length=128)  # hashed PIN for quick access
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"ModelAdmin: {self.user.username}"


class ModelSession(models.Model):
    """Tracks active Ollama model load/unload events."""
    STATUS_CHOICES = [
        ("loaded", "Loaded"),
        ("unloaded", "Unloaded"),
        ("error", "Error"),
    ]
    model_name = models.CharField(max_length=100)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="unloaded")
    action = models.CharField(max_length=50, default="manual")
    note = models.TextField(blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-timestamp"]

    def __str__(self):
        return f"[{self.status}] {self.model_name} @ {self.timestamp}"
