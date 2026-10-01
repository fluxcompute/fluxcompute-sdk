"""
Model dispatcher — sends the query to the selected provider + model.

Two dispatch modes per provider:
  • Standard (await)  — full response returned as a dict
  • Streaming (async generator) — yields OpenAI-format SSE strings,
    then writes final stats (tokens, timing) into a mutable `stats` dict

Anthropic: supports both plain {role, content} messages and the
content-block format required for prompt caching. Returns cache token
counts from usage so cost.py can price them correctly.

Gemini: generateContent is not OpenAI-compatible — no "assistant" role
(it's "model"), no "system" message in the turn list (system text is a
separate `system_instruction` config field), and usage field names differ
(prompt_token_count / candidates_token_count / cached_content_token_count).
dispatch_gemini normalises all of that to the same shape the other two
dispatch functions return. There is no stream_gemini yet — Gemini is only
wired up for the non-streaming path (see FluxClient._stream_execute).
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from contextlib import AsyncExitStack
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Set

logger = logging.getLogger("fluxcompute.dispatcher")


# ---------------------------------------------------------------------------
# Standard (non-streaming)
# ---------------------------------------------------------------------------

async def dispatch_anthropic(
    client: Any,                    # anthropic.AsyncAnthropic
    model: str,
    messages: List[Dict[str, Any]],
    system: Any = None,             # str | List[Dict] (content blocks)
    max_tokens: int = 4096,
    temperature: Optional[float] = None,
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

    create_kwargs = anthropic_create_kwargs(model, messages, system, max_tokens, temperature, kwargs)
    response = await _create_with_param_fallback(client.messages.create, model, create_kwargs)

    elapsed_ms = (time.monotonic() - start) * 1000
    usage = response.usage

    return {
        "response": response,
        "response_ms": round(elapsed_ms, 1),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_write_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
        **anthropic_finish_state(response),
    }


async def dispatch_openai(
    client: Any,                    # openai.AsyncOpenAI
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 4096,
    temperature: Optional[float] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Call OpenAI Chat Completions API and return response + timing.

    OpenAI uses automatic prefix caching (≥1024 token prefixes, 50% discount).
    """
    start = time.monotonic()

    plain_messages = [_flatten_to_openai(m) for m in messages]

    create_kwargs = {
        "model": model,
        "messages": plain_messages,
        **_openai_token_kwargs(model, max_tokens, temperature),
        **kwargs,
    }
    response = await _create_with_param_fallback(client.chat.completions.create, model, create_kwargs)

    elapsed_ms = (time.monotonic() - start) * 1000

    return {
        "response": response,
        "response_ms": round(elapsed_ms, 1),
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "cache_write_tokens": 0,
        "cache_read_tokens": 0,
        **openai_finish_state(response),
    }


async def dispatch_gemini(
    client: Any,                    # google.genai.Client
    model: str,
    messages: List[Dict[str, Any]],
    system: Any = None,             # str — Gemini has no content-block system format
    max_tokens: int = 4096,
    temperature: Optional[float] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Call Gemini's generateContent API (via the google-genai SDK's async client)
    and return response + timing + cache stats.

    No import of google.genai here — `client` arrives already constructed
    (fluxcompute/client.py), and both `contents` and `config` accept plain
    dicts, so this stays duck-typed like dispatch_anthropic/dispatch_openai.
    """
    start = time.monotonic()

    if system is None:
        system_msgs = [m for m in messages if m.get("role") == "system"]
        if system_msgs:
            system = system_msgs[0]["content"]
        messages = [m for m in messages if m.get("role") != "system"]

    contents = [_to_gemini_content(m) for m in messages]

    config: Dict[str, Any] = {"max_output_tokens": max_tokens, **kwargs}
    if temperature is not None:
        config["temperature"] = temperature
    if system:
        config["system_instruction"] = system

    response = await client.aio.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    elapsed_ms = (time.monotonic() - start) * 1000
    usage = response.usage_metadata

    return {
        "response": response,
        "response_ms": round(elapsed_ms, 1),
        "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
        "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
        # Gemini's implicit context caching is automatic (no explicit "write"
        # call/cost the way Anthropic's is) — only a read count comes back.
        "cache_write_tokens": 0,
        "cache_read_tokens": getattr(usage, "cached_content_token_count", 0) or 0,
        **gemini_finish_state(response),
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
    temperature: Optional[float] = None,
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

    create_kwargs = anthropic_create_kwargs(model, messages, system, max_tokens, temperature, kwargs)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    start = time.monotonic()
    output_tokens = 0
    ttft_ms: Optional[float] = None

    # Opening chunk — role delta
    yield _sse({"id": completion_id, "object": "chat.completion.chunk",
                "created": int(time.time()), "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})

    async with AsyncExitStack() as stack:
        stream = await open_stream_with_param_fallback(stack, client.messages.stream, model, create_kwargs)
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
    temperature: Optional[float] = None,
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

    create_kwargs = {
        "model": model,
        "messages": plain_messages,
        "stream_options": {"include_usage": True},
        **_openai_token_kwargs(model, max_tokens, temperature),
        **kwargs,
    }
    async with AsyncExitStack() as stack:
        stream = await open_stream_with_param_fallback(
            stack, client.chat.completions.stream, model, create_kwargs,
        )
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


def _openai_token_kwargs(model: str, max_tokens: int, temperature: Optional[float]) -> Dict[str, Any]:
    """Token/sampling kwargs for chat.completions.

    No per-family branching on the model name: a name list goes stale with every
    release (gpt-5.x and gpt-6-luna were sent `max_tokens` + `temperature` and
    400'd). `max_completion_tokens` is accepted by chat and reasoning models alike
    (checked live on gpt-4o-mini, gpt-5.6-luna, gpt-6-luna), while `max_tokens`
    is rejected by reasoning models. `temperature` goes out only when the caller
    set one; a model that rejects it is handled by the param fallback below."""
    kwargs: Dict[str, Any] = {"max_completion_tokens": max_tokens}
    if temperature is not None:
        kwargs["temperature"] = temperature
    return _without_rejected(model, kwargs)


def anthropic_create_kwargs(
    model: str,
    messages: List[Dict[str, Any]],
    system: Any,
    max_tokens: int,
    temperature: Optional[float],
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    """messages.create/stream kwargs, lifting a {role: system} message into `system`
    when no explicit system is given (the Messages API has no system role)."""
    if system is None:
        system_msgs = [m for m in messages if m.get("role") == "system"]
        if system_msgs:
            system = system_msgs[0]["content"]
        messages = [m for m in messages if m.get("role") != "system"]

    create_kwargs: Dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens, **extra}
    if temperature is not None:
        create_kwargs["temperature"] = temperature
    if system:
        create_kwargs["system"] = system
    return _without_rejected(model, create_kwargs)


# ---------------------------------------------------------------------------
# Param fallback: a model that rejects one request parameter
# ---------------------------------------------------------------------------

#: Parameters a model has rejected, per model id, learned from its 400s. Later
#: calls leave them out up front instead of paying the round trip again.
_REJECTED_PARAMS: Dict[str, Set[str]] = {}

#: Rejected params that have a replacement spelling, rather than being dropped.
_RENAMES = {"max_tokens": "max_completion_tokens", "max_completion_tokens": "max_tokens"}
_DROPPABLE = frozenset({"temperature", "top_p"})

#: OpenAI's error codes for "this model doesn't take that parameter/value". Other
#: codes on the same param (e.g. a max_completion_tokens above the model's limit)
#: are real caller errors and must surface unchanged.
_OPENAI_UNSUPPORTED_CODES = frozenset({"unsupported_parameter", "unsupported_value"})

#: Anthropic's 400 carries no `param` field, only a message, e.g.
#: "`temperature` is deprecated for this model." (claude-opus-4-8, 2026-09).
_ANTHROPIC_UNSUPPORTED_RE = re.compile(
    r"`?(temperature|top_p)`?\s+is\s+(?:deprecated|not supported|unsupported)",
    re.IGNORECASE,
)


def _without_rejected(model: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """kwargs with every param this model has rejected dropped or renamed."""
    for param in _REJECTED_PARAMS.get(model, ()):
        if param not in kwargs:
            continue
        value = kwargs.pop(param)
        renamed = _RENAMES.get(param)
        if renamed and renamed not in kwargs and renamed not in _REJECTED_PARAMS[model]:
            kwargs[renamed] = value
    return kwargs


def _rejected_param(exc: BaseException, kwargs: Dict[str, Any]) -> Optional[str]:
    """The param a 400 says this model doesn't support, if it is one we can drop or
    rename and the request actually sent it; None for every other error."""
    if getattr(exc, "status_code", None) != 400:
        return None
    param = getattr(exc, "param", None)  # set on openai errors, absent on anthropic's
    if param is not None:
        fixable = param in _DROPPABLE or param in _RENAMES
        if fixable and param in kwargs and getattr(exc, "code", None) in _OPENAI_UNSUPPORTED_CODES:
            return param
        return None
    match = _ANTHROPIC_UNSUPPORTED_RE.search(str(getattr(exc, "message", None) or exc))
    if match and match.group(1).lower() in kwargs:
        return match.group(1).lower()
    return None


def _remember_rejection(model: str, param: str) -> None:
    rejected = _REJECTED_PARAMS.setdefault(model, set())
    if param not in rejected:
        rejected.add(param)
        logger.warning(
            "%s rejected %r; resending without it, and leaving it out of later calls "
            "to this model in this process.", model, param,
        )


def _retry_kwargs(exc: BaseException, model: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """The kwargs to resend after `exc`, or re-raise it if no param is to blame.

    One resend, only for a 400: the provider rejected the request before doing
    any work, so nothing was billed and nothing ran twice."""
    param = _rejected_param(exc, kwargs)
    if param is None:
        raise exc
    _remember_rejection(model, param)
    return _without_rejected(model, dict(kwargs))


async def _create_with_param_fallback(
    create: Callable[..., Awaitable[Any]], model: str, kwargs: Dict[str, Any],
) -> Any:
    """`await create(**kwargs)`, resent once without a param the model rejects."""
    try:
        return await create(**kwargs)
    except Exception as exc:
        retry = _retry_kwargs(exc, model, kwargs)
    return await create(**retry)


async def open_stream_with_param_fallback(
    stack: AsyncExitStack, open_stream: Callable[..., Any], model: str, kwargs: Dict[str, Any],
) -> Any:
    """Enter `open_stream(**kwargs)` on `stack`, resent once without a param the model
    rejects. The 400 surfaces on entering the stream, before any chunk is yielded,
    so resending can't duplicate output."""
    try:
        return await stack.enter_async_context(open_stream(**kwargs))
    except Exception as exc:
        retry = _retry_kwargs(exc, model, kwargs)
    return await stack.enter_async_context(open_stream(**retry))


# ---------------------------------------------------------------------------
# Finish state: did the model stop because it ran out of output tokens?
# ---------------------------------------------------------------------------

def _count(value: Any) -> int:
    return value if isinstance(value, int) else 0


def openai_finish_state(completion: Any) -> Dict[str, Any]:
    """`truncated` / `finish_reason` / `reasoning_tokens` / `has_tool_calls` from a
    chat completion. `reasoning_tokens` is the hidden reasoning billed inside
    `completion_tokens`, so all of them going to reasoning is visible."""
    choices = getattr(completion, "choices", None) or []
    choice = choices[0] if isinstance(choices, list) and choices else None
    finish = getattr(choice, "finish_reason", None)
    tool_calls = getattr(getattr(choice, "message", None), "tool_calls", None)
    details = getattr(getattr(completion, "usage", None), "completion_tokens_details", None)
    return {
        "truncated": finish == "length",
        "finish_reason": finish if isinstance(finish, str) else None,
        "reasoning_tokens": _count(getattr(details, "reasoning_tokens", None)),
        "has_tool_calls": isinstance(tool_calls, list) and bool(tool_calls),
    }


def anthropic_finish_state(message: Any) -> Dict[str, Any]:
    """As openai_finish_state, for an Anthropic Message. Thinking tokens are part of
    `output_tokens` and not broken out, so reasoning_tokens stays 0."""
    stop = getattr(message, "stop_reason", None)
    content = getattr(message, "content", None)
    blocks = content if isinstance(content, list) else []
    return {
        "truncated": stop == "max_tokens",
        "finish_reason": stop if isinstance(stop, str) else None,
        "reasoning_tokens": 0,
        "has_tool_calls": any(getattr(b, "type", None) == "tool_use" for b in blocks),
    }


def gemini_finish_state(response: Any) -> Dict[str, Any]:
    """As openai_finish_state, for a Gemini GenerateContentResponse."""
    candidates = getattr(response, "candidates", None)
    candidate = candidates[0] if isinstance(candidates, list) and candidates else None
    finish = getattr(candidate, "finish_reason", None)
    finish_name = getattr(finish, "name", finish)   # FinishReason enum or plain string
    parts = getattr(getattr(candidate, "content", None), "parts", None)
    usage = getattr(response, "usage_metadata", None)
    return {
        "truncated": finish_name == "MAX_TOKENS",
        "finish_reason": finish_name if isinstance(finish_name, str) else None,
        "reasoning_tokens": _count(getattr(usage, "thoughts_token_count", None)),
        "has_tool_calls": isinstance(parts, list)
        and any(getattr(p, "function_call", None) is not None for p in parts),
    }


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


def _to_gemini_content(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an Anthropic/OpenAI-style message to Gemini's {role, parts} shape.

    Gemini's Content type only has two roles, "user" and "model" — there is
    no third "tool" role the way OpenAI has, so any non-"assistant" role
    (including a literal role="tool" message, or a role="user" message
    carrying Anthropic-style tool_result blocks) maps to "user", matching
    Gemini's own function-calling contract where a function's result is
    sent back as a functionResponse part on a "user" turn.

    tool_use/tool_result blocks are not dropped: tool_use blocks (an
    assistant's function call) become functionCall parts, and tool_result
    blocks (or a plain role="tool" message) become functionResponse parts,
    so tool-call turns survive a mid-session handoff to Gemini instead of
    silently going empty."""
    role = "model" if msg.get("role") == "assistant" else "user"
    content = msg.get("content")

    if isinstance(content, list):
        parts: List[Dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "tool_use":
                parts.append({
                    "functionCall": {
                        "name": block.get("name", ""),
                        "args": block.get("input") or {},
                    }
                })
            elif block_type == "tool_result":
                parts.append({
                    "functionResponse": {
                        "name": block.get("tool_use_id") or block.get("name", ""),
                        "response": {"result": block.get("content", "")},
                    }
                })
        text = " ".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        if text:
            parts.append({"text": text})
        if not parts:
            parts = [{"text": ""}]
        return {"role": role, "parts": parts}

    if msg.get("role") == "tool":
        return {
            "role": role,
            "parts": [{
                "functionResponse": {
                    "name": msg.get("name") or msg.get("tool_call_id") or "tool",
                    "response": {"result": content or ""},
                }
            }],
        }

    return {"role": role, "parts": [{"text": content or ""}]}


# ---------------------------------------------------------------------------
# Response extraction — small provider-specific readers so client.py/models.py
# never need their own anthropic/openai/google branch to pull the assistant's
# text/content/usage back out of a raw SDK response object.
# ---------------------------------------------------------------------------

def extract_gemini_text(response: Any) -> str:
    for candidate in getattr(response, "candidates", None) or []:
        for part in getattr(candidate.content, "parts", None) or []:
            text = getattr(part, "text", None)
            if text:
                return text
    return ""


def extract_gemini_content(response: Any) -> Any:
    return response.candidates


def extract_gemini_usage(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage_metadata", None)
    return {
        "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
        "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
    }
