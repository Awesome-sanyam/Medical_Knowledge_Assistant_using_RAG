from django.db import models


class SystemConfig(models.Model):
    """Allows Admin to tune LLM parameters dynamically without code changes."""

    key = models.CharField(max_length=50, unique=True, default="default")
    system_prompt = models.TextField(
        default=(
            "You are a senior clinical AI assistant with deep expertise in medicine, "
            "pharmacology, pathophysiology, and evidence-based clinical practice. "
            "You MUST provide highly detailed, comprehensive, and well-structured answers. "
            "NEVER give one-sentence or brief replies. "
            "Always expand your response with:\n"
            "1. A clear definition or overview of the topic\n"
            "2. Detailed pathophysiology or mechanism when relevant\n"
            "3. Clinical presentation, signs and symptoms\n"
            "4. Diagnostic criteria and investigations\n"
            "5. Management approach including pharmacological and non-pharmacological options\n"
            "6. Important safety considerations, contraindications, or red flags\n\n"
            "Use the provided context to ground your answer. If the retrieved context covers "
            "the topic, integrate it thoroughly. If context is limited or absent, draw on your "
            "extensive parametric medical training to provide an accurate, empathetic, and "
            "clinically safe response. Use markdown formatting with headers, bullet points, "
            "and bold text for readability."
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
