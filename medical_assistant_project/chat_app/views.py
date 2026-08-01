import json
import logging
import time

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

from .models import Conversation, Message, UserFeedback

logger = logging.getLogger(__name__)

# Graceful import — server still boots without Ollama running
try:
    from admin_app.models import SystemConfig
    from rag_engine.guardrails import (
        CLINICAL_DISCLAIMER,
        low_context_fallback,
        post_guardrail,
        pre_guardrail,
        topic_guardrail,
    )
    from rag_engine.ollama_client import (
        CONTEXT_REFUSAL_MSG,
        HARD_REFUSAL_MESSAGE,
        generate_context_refusal,
        generate_refusal,
        generate_stream,
    )
    from rag_engine.retriever import get_relevant_context

    RAG_ENABLED = True
except ImportError as exc:
    logger.warning("RAG Engine not fully loaded: %s", exc)
    RAG_ENABLED = False


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


@login_required
def chat_interface(request):
    """Render the main chat SPA."""
    conv_id = request.GET.get("c")
    conversation = None

    if conv_id:
        try:
            conversation = Conversation.objects.get(id=conv_id, user=request.user)
        except Conversation.DoesNotExist:
            conversation = None

    if conversation is None:
        # Get or create the most recent conversation for this user
        conversation = Conversation.objects.filter(user=request.user).first()
        if conversation is None:
            conversation = Conversation.objects.create(user=request.user)

    conversations = Conversation.objects.filter(user=request.user)[:20]

    return render(
        request,
        "chat/interface.html",
        {
            "conversation_id": str(conversation.id),
            "conversations": conversations,
            "active_conversation": conversation,
        },
    )


@login_required
def new_conversation(request):
    """Create a new conversation and return its ID."""
    conversation = Conversation.objects.create(user=request.user)
    return JsonResponse({
        "id": str(conversation.id),
        "title": conversation.title,
    })


@login_required
def conversation_list(request):
    """Return JSON list of user's conversations for sidebar."""
    conversations = Conversation.objects.filter(user=request.user)[:30]
    data = [
        {
            "id": str(c.id),
            "title": c.title,
            "updated_at": c.updated_at.strftime("%b %d"),
        }
        for c in conversations
    ]
    return JsonResponse({"conversations": data})


@login_required
@require_POST
def stream_chat(request):
    """
    SSE streaming endpoint.
    Expects POST body: { message, conversation_id }
    Streams: data: <token>\n\n
    Closes: event: close\ndata: \n\n
    """
    user_input = request.POST.get("message", "").strip()
    conversation_id = request.POST.get("conversation_id", "")

    if not user_input:
        return StreamingHttpResponse(
            _error_stream("Please enter a message."), content_type="text/event-stream"
        )

    try:
        conversation = Conversation.objects.get(id=conversation_id, user=request.user)
    except (Conversation.DoesNotExist, Exception):
        conversation = Conversation.objects.create(user=request.user)

    # Auto-title from first message
    if conversation.title == "New Consultation":
        conversation.title = user_input[:60] + ("…" if len(user_input) > 60 else "")
        conversation.save(update_fields=["title"])

    # Persist user message immediately
    user_msg = Message.objects.create(
        conversation=conversation, role="user", content=user_input
    )

    return StreamingHttpResponse(
        _event_generator(user_input, conversation, user_msg),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _error_stream(msg: str):
    yield f"data: {msg}\n\n"
    yield "event: close\ndata: \n\n"


def _event_generator(user_input: str, conversation: Conversation, user_msg: Message):
    """Core generator — yield SSE-formatted tokens."""
    if not RAG_ENABLED:
        yield "data: ⚠️ RAG Engine not initialized. Please check server logs.\n\n"
        yield "event: close\ndata: \n\n"
        return

    start_time = time.time()

    # --- Pre-Guardrail: dangerous content ---
    is_blocked, block_msg = pre_guardrail(user_input)
    if is_blocked:
        safe_msg = block_msg.replace("\n", " ")
        yield f"data: {safe_msg}\n\n"
        yield "event: close\ndata: \n\n"
        _save_assistant_message(conversation, block_msg, [], time.time() - start_time)
        return

    # --- Topic Guardrail: non-medical queries ---
    is_off_topic, refusal_msg = topic_guardrail(user_input)
    if is_off_topic:
        final = post_guardrail(refusal_msg)
        for line in final.splitlines(keepends=True):
            safe = line.replace("\n", "⏎")
            yield f"data: {safe}\n\n"
        yield "event: close\ndata: \n\n"
        _save_assistant_message(conversation, final, [], time.time() - start_time)
        return

    # --- Load live config from admin panel ---
    try:
        config = SystemConfig.get_active()
        top_k = config.top_k_retrieval
        temperature = config.temperature
        system_prompt = config.system_prompt
        threshold = config.similarity_threshold
    except Exception:
        top_k, temperature, threshold = 3, 0.2, 0.65
        system_prompt = (
            "You are MedAssist AI, a senior clinical knowledge assistant. "
            "Structure your response using Markdown with ### Headings, **bold**, and bullet points. "
            "NEVER repeat yourself. Provide detailed, comprehensive clinical answers."
        )

    # --- Hybrid Retrieval (with hard out-of-domain guardrail) ---
    try:
        context_chunks, low_confidence, is_out_of_domain = get_relevant_context(
            user_input, top_k=top_k, similarity_threshold=threshold
        )
    except Exception as e:
        logger.error("Retrieval failed: %s", e)
        context_chunks, low_confidence, is_out_of_domain = [], True, False

    # --- Hard Out-of-Domain Guardrail (retriever-level) ---
    # If the best similarity score is below OUT_OF_DOMAIN_THRESHOLD, we never
    # call the LLM. The refusal is returned verbatim and the query is logged.
    if is_out_of_domain:
        refusal = generate_refusal()
        logger.warning(
            "Retriever-level domain rejection for query: '%s…'", user_input[:60]
        )
        safe_refusal = refusal.replace("\n", "⏎")
        yield f"data: {safe_refusal}\n\n"
        yield "event: close\ndata: \n\n"
        _save_assistant_message(conversation, refusal, [], time.time() - start_time)
        return

    # --- Low-context fallback ---
    should_fallback, fallback_msg = low_context_fallback(low_confidence, context_chunks)
    if should_fallback:
        final_response = post_guardrail(fallback_msg)
        for line in final_response.splitlines(keepends=True):
            safe = line.replace("\n", " ")
            yield f"data: {safe}\n\n"
        yield "event: close\ndata: \n\n"
        _save_assistant_message(
            conversation, final_response, context_chunks, time.time() - start_time
        )
        return

    # --- Stream LLM Response ---
    context_text = "\n\n".join(c["content"] for c in context_chunks)
    full_response = ""

    try:
        for token in generate_stream(
            question=user_input,
            context=context_text,
            system_prompt=system_prompt,
            temperature=temperature,
        ):
            full_response += token
            # Escape newlines for SSE transport
            safe_token = token.replace("\n", "⏎")
            yield f"data: {safe_token}\n\n"
    except Exception as e:
        err = f"Model error: {str(e)}"
        logger.error(err)
        full_response += err
        yield f"data: ⚠️ {err}\n\n"

    # --- Post-Guardrail: append disclaimer ---
    disclaimer = CLINICAL_DISCLAIMER
    yield f"data: {disclaimer.replace(chr(10), '⏎')}\n\n"
    full_response = post_guardrail(full_response)

    # --- Send source citations as a special JSON event ---
    sources_payload = json.dumps(context_chunks)
    yield f"event: sources\ndata: {sources_payload}\n\n"

    yield "event: close\ndata: \n\n"

    # Persist the full AI response
    latency = time.time() - start_time
    _save_assistant_message(conversation, full_response, context_chunks, latency)


def _save_assistant_message(
    conversation: Conversation,
    content: str,
    context_chunks: list,
    latency: float,
):
    msg = Message.objects.create(
        conversation=conversation,
        role="assistant",
        content=content,
        retrieved_context=context_chunks,
        latency_seconds=round(latency, 3),
    )
    return msg


# ---------------------------------------------------------------------------
# Feedback endpoint
# ---------------------------------------------------------------------------


@login_required
@require_POST
def submit_feedback(request):
    """Thumbs up/down feedback on a message."""
    try:
        data = json.loads(request.body)
        message_id = data.get("message_id")
        is_positive = bool(data.get("is_positive"))
        feedback_text = data.get("feedback_text", "")

        msg = Message.objects.get(id=message_id)
        UserFeedback.objects.update_or_create(
            message=msg,
            defaults={"is_positive": is_positive, "feedback_text": feedback_text},
        )
        return JsonResponse({"status": "ok"})
    except Exception as e:
        return JsonResponse({"status": "error", "detail": str(e)}, status=400)


# ---------------------------------------------------------------------------
# Conversation messages API (for loading chat history)
# ---------------------------------------------------------------------------


@login_required
def conversation_messages(request, conv_id):
    """Return messages for a specific conversation."""
    try:
        conversation = Conversation.objects.get(id=conv_id, user=request.user)
    except Conversation.DoesNotExist:
        return JsonResponse({"messages": []})

    msgs = Message.objects.filter(conversation=conversation).order_by("created_at")
    data = [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "retrieved_context": m.retrieved_context,
            "created_at": m.created_at.strftime("%H:%M"),
        }
        for m in msgs
    ]
    return JsonResponse({"messages": data})
