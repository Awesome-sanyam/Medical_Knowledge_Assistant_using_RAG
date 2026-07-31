from django.db import models


class SystemConfig(models.Model):
    """Allows Admin to tune LLM parameters dynamically without code changes."""

    key = models.CharField(max_length=50, unique=True, default="default")
    system_prompt = models.TextField(
        default=(
            "You are a clinical-grade medical AI assistant. "
            "Use the provided context to ground your answer if available. "
            "If no context is provided, rely on your extensive medical training to provide an accurate, empathetic, and safe response."
        )
    )
    temperature = models.FloatField(default=0.2)
    top_k_retrieval = models.IntegerField(default=3)
    # Rejection threshold: chunks with distance > this are flagged low-confidence
    similarity_threshold = models.FloatField(default=0.65)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "System Configuration"

    def __str__(self):
        return f"SystemConfig[{self.key}] – temp={self.temperature}, top_k={self.top_k_retrieval}"

    @classmethod
    def get_active(cls):
        """Return (or create) the singleton active configuration."""
        config, _ = cls.objects.get_or_create(key="default")
        return config


class Document(models.Model):
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("indexing", "Indexing"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    ]

    title = models.CharField(max_length=255)
    file = models.FileField(upload_to="medical_docs/")
    chunk_count = models.IntegerField(default=0)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    error_message = models.TextField(blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self):
        return f"{self.title} [{self.status}]"
