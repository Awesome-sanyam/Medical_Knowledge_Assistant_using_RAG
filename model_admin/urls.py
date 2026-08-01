from django.urls import path
from . import views

urlpatterns = [
    path("", views.model_admin_dashboard, name="model_admin_dashboard"),
    path("login/", views.model_admin_login, name="model_admin_login"),
    path("register/", views.model_admin_register, name="model_admin_register"),
    path("logout/", views.model_admin_logout, name="model_admin_logout"),
    path("api/status/", views.ollama_status_api, name="ollama_status_api"),
    path("api/load/", views.load_model, name="load_model"),
    path("api/unload/", views.unload_model, name="unload_model"),
    path("api/config/", views.save_config, name="model_admin_save_config"),
]
