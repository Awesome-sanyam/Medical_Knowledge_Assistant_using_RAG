from django.urls import path

from . import views

urlpatterns = [
    path("", views.chat_interface, name="chat_interface"),
    path("stream/", views.stream_chat, name="stream_chat"),
    path("feedback/", views.submit_feedback, name="submit_feedback"),
]
