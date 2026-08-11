"""
Model dispatcher — sends the query to the selected provider + model.

Two dispatch modes per provider:
  • Standard (await)  — full response returned as a dict
  • Streaming (async generator) — yields OpenAI-format SSE strings,
    then writes final stats (tokens, timing) into a mutable `stats` dict

Anthropic: supports both plain {role, content} messages and the
content-block format required for prompt caching. Returns cache token
counts from usage so cost.py can price them correctly.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional


# ---------------------------------------------------------------------------
# Standard (non-streaming)
# ---------------------------------------------------------------------------

async def dispatch_anthropic(
    client: Any,                    # anthropic.AsyncAnthropic
    model: str,
    messages: List[Dict[str, Any]],
    system: Any = None,             # str | List[Dict] (content blocks)
    max_tokens: int = 4096,
    temperature: float = 1.0,
    **kwargs,
) -> Dict[str, Any]:
    """
    Call Anthropic Messages API and return response + timing + cache stats.

    system may be:
      - None        → extract from messages if a {role:system} entry exists
      - str         → plain text system prompt
      - list        → content blocks (used by CacheManager for prompt caching)
    """
    start = time.monotonic()

    if system is None:
        system_msgs = [m for m in messages if m.get("role") == "system"]
        if system_msgs:
            system = system_msgs[0]["content"]
        messages = [m for m in messages if m.get("role") != "system"]

    create_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        **kwargs,
    }
    if system:
        create_kwargs["system"] = system

    response = await client.messages.create(**create_kwargs)

    elapsed_ms = (time.monotonic() - start) * 1000
    usage = response.usage

    return {
        "response": response,
        "response_ms": round(elapsed_ms, 1),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_write_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
    }


async def dispatch_openai(
    client: Any,                    # openai.AsyncOpenAI
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 4096,
    temperature: float = 1.0,
    **kwargs,
) -> Dict[str, Any]:
    """
    Call OpenAI Chat Completions API and return response + timing.

    OpenAI uses automatic prefix caching (≥1024 token prefixes, 50% discount).
    """
    start = time.monotonic()

    plain_messages = [_flatten_to_openai(m) for m in messages]

    response = await client.chat.completions.create(
        model=model,
        messages=plain_messages,
        **_openai_token_kwargs(model, max_tokens, temperature),
        **kwargs,
    )

    elapsed_ms = (time.monotonic() - start) * 1000

    return {
        "response": response,
        "response_ms": round(elapsed_ms, 1),
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "cache_write_tokens": 0,
        "cache_read_tokens": 0,
    }


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

async def stream_anthropic(
    client: Any,
    model: str,
    messages: List[Dict[str, Any]],
    system: Any = None,
    max_tokens: int = 4096,
    temperature: float = 1.0,
    stats: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> AsyncGenerator[str, None]:
    """
    Stream from Anthropic and yield OpenAI-compatible SSE strings.

    After the stream ends, `stats` dict is populated with:
      input_tokens, output_tokens, cache_write_tokens, cache_read_tokens,
      response_ms

    Usage:
        stats = {}
        async for line in stream_anthropic(client, model, messages, stats=stats):
            yield line
        # stats is now populated
    """
    if stats is None:
        stats = {}

    if system is None:
        system_msgs = [m for m in messages if m.get("role") == "system"]
        if system_msgs:
            system = system_msgs[0]["content"]
        messages = [m for m in messages if m.get("role") != "system"]

    create_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        **kwargs,
    }
    if system:
        create_kwargs["system"] = system

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    start = time.monotonic()
    output_tokens = 0
    ttft_ms: Optional[float] = None

    # Opening chunk — role delta
    yield _sse({"id": completion_id, "object": "chat.completion.chunk",
                "created": int(time.time()), "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})

    async with client.messages.stream(**create_kwargs) as stream:
        async for event in stream:
            event_type = getattr(event, "type", None)

            if event_type == "content_block_delta":
                delta = getattr(event, "delta", None)
                if delta and getattr(delta, "type", None) == "text_delta":
                    if ttft_ms is None:
                        ttft_ms = round((time.monotonic() - start) * 1000, 1)
                    yield _sse({
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": delta.text},
                            "finish_reason": None,
                        }],
                    })

            elif event_type == "message_delta":
                delta = getattr(event, "delta", None)
                usage = getattr(event, "usage", None)
                if usage:
                    output_tokens = getattr(usage, "output_tokens", 0)
                stop_reason = getattr(delta, "stop_reason", "stop") if delta else "stop"
                finish = "stop" if stop_reason in ("end_turn", "stop") else stop_reason

                # Final chunk with finish_reason
                yield _sse({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                })

        final_msg = await stream.get_final_message()
        final_usage = final_msg.usage

    elapsed_ms = (time.monotonic() - start) * 1000
    stats.update({
        "completion_id": completion_id,
        "response_ms": round(elapsed_ms, 1),
        "ttft_ms": ttft_ms,
        "input_tokens": getattr(final_usage, "input_tokens", 0),
        "output_tokens": getattr(final_usage, "output_tokens", output_tokens),
        "cache_write_tokens": getattr(final_usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_tokens": getattr(final_usage, "cache_read_input_tokens", 0) or 0,
    })

    yield "data: [DONE]\n\n"


async def stream_openai(
    client: Any,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 4096,
    temperature: float = 1.0,
    stats: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> AsyncGenerator[str, None]:
    """
    Stream from OpenAI and yield OpenAI-compatible SSE strings.

    OpenAI chunks are already in the right format; we just re-serialise them
    so both providers go through the same interface.
    """
    if stats is None:
        stats = {}

    plain_messages = [_flatten_to_openai(m) for m in messages]
    start = time.monotonic()
    ttft_ms: Optional[float] = None

    async with client.chat.completions.stream(
        model=model,
        messages=plain_messages,
        stream_options={"include_usage": True},
        **_openai_token_kwargs(model, max_tokens, temperature),
        **kwargs,
    ) as stream:
        async for chunk in stream:
            if chunk.choices or getattr(chunk, "usage", None):
                if ttft_ms is None and chunk.choices:
                    ttft_ms = round((time.monotonic() - start) * 1000, 1)
                yield _sse(chunk.model_dump())

    elapsed_ms = (time.monotonic() - start) * 1000
    final = await stream.get_final_completion()
    usage = getattr(final, "usage", None)

    stats.update({
        "completion_id": getattr(final, "id", ""),
        "response_ms": round(elapsed_ms, 1),
        "ttft_ms": ttft_ms,
        "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
        "cache_write_tokens": 0,
        "cache_read_tokens": 0,
    })

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sse(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _is_reasoning_model(model: str) -> bool:
    """OpenAI reasoning models (o1/o3/o4 families) use a different call contract
    than chat models: they take `max_completion_tokens` instead of `max_tokens`
    and reject any non-default `temperature`."""
    return model.startswith(("o1", "o3", "o4"))


def _openai_stream_text(event: Any) -> str:
    """Text delta from one `chat.completions.stream()` event, or "".

    That stream yields typed ChatCompletionStreamEvent objects — content deltas,
    chunks, done/refusal/tool-call/logprobs variants — and *none* of them
    exposes `.choices`. Reading chunk-shaped fields straight off the event
    raises AttributeError on the very first one. Unknown event types yield no
    text rather than raising, so event kinds added by future openai releases
    can't break streaming.
    """
    etype = getattr(event, "type", None)
    if etype == "content.delta":
        return getattr(event, "delta", "") or ""
    if etype == "chunk":
        choices = getattr(getattr(event, "chunk", None), "choices", None)
        if choices:
            return getattr(choices[0].delta, "content", None) or ""
    return ""


def _openai_token_kwargs(model: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
    """Token/sampling kwargs for chat.completions, adapted per model family.

    Reasoning models bill hidden reasoning tokens against the completion budget,
    so `max_tokens` maps to `max_completion_tokens`; `temperature` is omitted
    because only the default is accepted."""
    if _is_reasoning_model(model):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens, "temperature": temperature}


def _flatten_to_openai(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Anthropic content-block format back to plain OpenAI string."""
    content = msg.get("content")
    if isinstance(content, list):
        text = " ".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        return {"role": msg["role"], "content": text}
    return msg
