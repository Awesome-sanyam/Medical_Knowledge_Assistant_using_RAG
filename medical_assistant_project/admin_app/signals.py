"""
Auto-create the Django superuser on first migration.

Creates or updates:
  username: admin
  password: admin

This runs once after migrations via the post_migrate signal.
"""

import logging

from django.contrib.auth.models import User
from django.db.models.signals import post_migrate
from django.dispatch import receiver

logger = logging.getLogger(__name__)


@receiver(post_migrate)
def create_default_superuser(sender, **kwargs):
    """Ensure the default superuser exists with known credentials."""
    if sender.name != "admin_app":
        return

    username = "admin"
    password = "admin"

    try:
        user, created = User.objects.get_or_create(
            username=username,
            defaults={
                "is_superuser": True,
                "is_staff": True,
                "is_active": True,
            },
        )
        if created:
            user.set_password(password)
            user.save()
            logger.info("Created default superuser: %s", username)
        else:
            # Ensure the existing user has the right password and perms
            user.set_password(password)
            user.is_superuser = True
            user.is_staff = True
            user.is_active = True
            user.save()
            logger.info("Updated default superuser credentials: %s", username)
    except Exception as e:
        logger.warning("Could not create/update default superuser: %s", e)
