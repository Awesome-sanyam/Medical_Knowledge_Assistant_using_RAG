from django.apps import AppConfig


class AdminAppConfig(AppConfig):
    name = "admin_app"

    def ready(self):
        import admin_app.signals  # noqa: F401 — registers post_migrate handler
