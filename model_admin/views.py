import json
import logging

import httpx
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_POST

from admin_app.models import SystemConfig
from rag_engine.ollama_client import evict_model_cache, get_ollama_status

from .models import AdminUser, ModelSession

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"


# ---------------------------------------------------------------------------
# Auth helpers — standalone, no Django admin required
# ---------------------------------------------------------------------------


def model_admin_login(request):
    if request.user.is_authenticated and (request.user.is_staff or hasattr(request.user, 'model_admin_profile')):
        return redirect('model_admin_dashboard')

    error = None
    if request.method == 'POST':
        username = request.POST.get('username', '')
        password = request.POST.get('password', '')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            if not (user.is_staff or hasattr(user, 'model_admin_profile')):
                error = 'This account does not have Model Control access.'
            elif not user.is_active:
                error = 'Your account is pending approval by the Django Administrator.'
            else:
                login(request, user)
                return redirect('model_admin_dashboard')
        else:
            error = 'Invalid credentials. Please try again.'

    return render(request, 'model_admin/login.html', {'error': error})


def model_admin_register(request):
    """Self-registration for Model Admins — creates pending account, Django Admin approves."""
    if request.user.is_authenticated:
        return redirect('model_admin_dashboard')

    success = False
    errors = []
    if request.method == 'POST':
        full_name = request.POST.get('full_name', '').strip()
        email = request.POST.get('email', '').strip()
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        confirm = request.POST.get('confirm_password', '')
        reason = request.POST.get('reason', '').strip()

        if not username or not password:
            errors.append('Username and password are required.')
        if password != confirm:
            errors.append('Passwords do not match.')
        if len(password) < 8:
            errors.append('Password must be at least 8 characters.')
        if User.objects.filter(username=username).exists():
            errors.append('Username is already taken.')
        if email and User.objects.filter(email=email).exists():
            errors.append('An account with this email already exists.')
        if not reason:
            errors.append('Please provide a reason for requesting admin access.')

        if not errors:
            first, _, last = full_name.partition(' ')
            # Create user as inactive — Django Admin must activate
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password,
                first_name=first,
                last_name=last,
                is_active=False,   # pending approval
                is_staff=False,
            )
            AdminUser.objects.create(user=user, pin='pending')
            logger.info("Model Admin registration pending approval: %s — %s", username, reason)
            success = True

    return render(request, 'model_admin/register.html', {
        'success': success,
        'errors': errors,
    })


def model_admin_logout(request):
    logout(request)
    return redirect('model_admin_login')


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
