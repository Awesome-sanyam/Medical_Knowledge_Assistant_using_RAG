from django.urls import path

from . import auth_views, views

urlpatterns = [
    # Auth
    path("login/", auth_views.user_login, name="user_login"),
    path("register/", auth_views.user_register, name="user_register"),
    path("logout/", auth_views.user_logout, name="user_logout"),
    # Chat
    path("", views.chat_interface, name="chat_interface"),
    path("stream/", views.stream_chat, name="stream_chat"),
    path("feedback/", views.submit_feedback, name="submit_feedback"),
    # Conversation API
    path("new/", views.new_conversation, name="new_conversation"),
    path("conversations/", views.conversation_list, name="conversation_list"),
    path("messages/<str:conv_id>/", views.conversation_messages, name="conversation_messages"),
]
