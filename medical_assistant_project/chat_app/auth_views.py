"""
Authentication views for chat users (Doctors/Patients).

- Login and self-registration are available to anyone.
- Model Admin registration is NOT available here — only through Django Admin.
"""

import logging

from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.shortcuts import redirect, render

logger = logging.getLogger(__name__)


def user_login(request):
    """Render login page and authenticate users."""
    if request.user.is_authenticated:
        return redirect("chat_interface")

    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            next_url = request.GET.get("next", "/chat/")
            return redirect(next_url)
        else:
            return render(request, "chat/login.html", {"error": "Invalid username or password."})

    return render(request, "chat/login.html")


def user_register(request):
    """Self-registration for chat users."""
    if request.user.is_authenticated:
        return redirect("chat_interface")

    if request.method == "POST":
        full_name = request.POST.get("full_name", "").strip()
        email = request.POST.get("email", "").strip()
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")
        confirm_password = request.POST.get("confirm_password", "")

        # Validation
        errors = []
        if not username or not password:
            errors.append("Username and password are required.")
        if password != confirm_password:
            errors.append("Passwords do not match.")
        if len(password) < 8:
            errors.append("Password must be at least 8 characters.")
        if User.objects.filter(username=username).exists():
            errors.append("Username is already taken.")
        if email and User.objects.filter(email=email).exists():
            errors.append("An account with this email already exists.")

        if errors:
            return render(request, "chat/register.html", {
                "errors": errors,
                "full_name": full_name,
                "email": email,
                "username": username,
            })

        # Create user
        first_name, _, last_name = full_name.partition(" ")
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name,
        )
        logger.info("New user registered: %s", username)

        # Auto-login after registration
        login(request, user)
        return redirect("chat_interface")

    return render(request, "chat/register.html")


def user_logout(request):
    """Log out and redirect to login."""
    logout(request)
    return redirect("user_login")
