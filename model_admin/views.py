import json
import logging

import httpx
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_POST

from admin_app.models import SystemConfig
from rag_engine.ollama_client import evict_model_cache, get_ollama_status

from .models import ModelSession

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"


# ---------------------------------------------------------------------------
# Auth helpers — standalone, no Django admin required
# ---------------------------------------------------------------------------


def model_admin_login(request):
    if request.user.is_authenticated and hasattr(request.user, "model_admin_profile"):
        return redirect("model_admin_dashboard")

    error = None
    if request.method == "POST":
        username = request.POST.get("username", "")
        password = request.POST.get("password", "")
        user = authenticate(request, username=username, password=password)
        if user is not None and (user.is_staff or hasattr(user, "model_admin_profile")):
            login(request, user)
            return redirect("model_admin_dashboard")
        else:
            error = "Invalid credentials or insufficient permissions."

    return render(request, "model_admin/login.html", {"error": error})


def model_admin_logout(request):
    logout(request)
    return redirect("model_admin_login")


def _model_admin_required(view_func):
    """Custom decorator — requires login + staff or model_admin_profile."""
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect("model_admin_login")
        if not (request.user.is_staff or hasattr(request.user, "model_admin_profile")):
            return redirect("model_admin_login")
        return view_func(request, *args, **kwargs)
    wrapper.__name__ = view_func.__name__
    return wrapper


# ---------------------------------------------------------------------------
# Dashboard — model status + controls
# ---------------------------------------------------------------------------


@_model_admin_required
def model_admin_dashboard(request):
    ollama_status = get_ollama_status()
    config = SystemConfig.get_active()
    recent_sessions = ModelSession.objects.all()[:10]

    # CPU/RAM stats via psutil if available
    system_stats = _get_system_stats()

    return render(
        request,
        "model_admin/dashboard.html",
        {
            "ollama_status": ollama_status,
            "config": config,
            "recent_sessions": recent_sessions,
            "system_stats": system_stats,
        },
    )


# ---------------------------------------------------------------------------
# Model Control API endpoints
# ---------------------------------------------------------------------------


@_model_admin_required
@require_POST
def load_model(request):
    """Warm the model by sending a short ping request."""
    model_name = request.POST.get("model_name", "med-llama")
    try:
        resp = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": model_name, "prompt": "ping", "stream": False},
            timeout=30.0,
        )
        if resp.status_code == 200:
            ModelSession.objects.create(model_name=model_name, status="loaded", action="manual_load")
            # Evict LLM cache so next request uses fresh config
            evict_model_cache()
            return JsonResponse({"status": "ok", "message": f"Model '{model_name}' warmed."})
        return JsonResponse({"status": "error", "message": resp.text}, status=500)
    except Exception as e:
        ModelSession.objects.create(model_name=model_name, status="error", note=str(e), action="manual_load")
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


@_model_admin_required
@require_POST
def unload_model(request):
    """Tell Ollama to evict the model from memory (keep_alive=0)."""
    model_name = request.POST.get("model_name", "med-llama")
    try:
        resp = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": model_name, "prompt": "", "keep_alive": 0},
            timeout=15.0,
        )
        evict_model_cache()
        ModelSession.objects.create(model_name=model_name, status="unloaded", action="manual_unload")
        return JsonResponse({"status": "ok", "message": f"Model '{model_name}' evicted from RAM."})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


@_model_admin_required
def ollama_status_api(request):
    """Live status JSON for the dashboard to poll."""
    status = get_ollama_status()
    system_stats = _get_system_stats()
    return JsonResponse({**status, "system": system_stats})


@_model_admin_required
@require_POST
def save_config(request):
    """Save SystemConfig from the model admin panel."""
    config = SystemConfig.get_active()
    try:
        config.system_prompt = request.POST.get("system_prompt", config.system_prompt)
        config.temperature = float(request.POST.get("temperature", config.temperature))
        config.top_k_retrieval = int(request.POST.get("top_k_retrieval", config.top_k_retrieval))
        config.similarity_threshold = float(
            request.POST.get("similarity_threshold", config.similarity_threshold)
        )
        config.save()
        # Evict LLM cache so new temp takes effect immediately
        evict_model_cache()
        return JsonResponse({"status": "ok", "message": "Configuration saved. LLM cache cleared."})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=400)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_system_stats() -> dict:
    try:
        import psutil
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.1)
        return {
            "ram_used_gb": round(mem.used / 1e9, 1),
            "ram_total_gb": round(mem.total / 1e9, 1),
            "ram_percent": mem.percent,
            "cpu_percent": cpu,
        }
    except ImportError:
        return {}
