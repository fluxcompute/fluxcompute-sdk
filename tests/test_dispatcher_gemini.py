"""Gemini's generateContent API isn't OpenAI-compatible: no "assistant" role
(it's "model"), no "system" message in the turn list (system text is a
separate config field), and usage field names differ from both Anthropic's
and OpenAI's. dispatch_gemini must normalise all of that to the same shape
dispatch_anthropic/dispatch_openai return, without importing google.genai
(client arrives pre-constructed; contents/config are plain dicts)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from fluxcompute.router.dispatcher import (
    _to_gemini_content,
    dispatch_gemini,
    extract_gemini_content,
    extract_gemini_text,
    extract_gemini_usage,
)


def _fake_gemini_response(input_tokens=11, output_tokens=7, cached_tokens=0):
    usage = MagicMock()
    usage.prompt_token_count = input_tokens
    usage.candidates_token_count = output_tokens
    usage.cached_content_token_count = cached_tokens
    response = MagicMock()
    response.usage_metadata = usage
    return response


def _fake_client(response=None):
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value=response or _fake_gemini_response())
    return client


# ─── message/role conversion ───────────────────────────────────────────────

def test_to_gemini_content_maps_assistant_to_model_role():
    assert _to_gemini_content({"role": "assistant", "content": "hi"}) == {
        "role": "model",
        "parts": [{"text": "hi"}],
    }


def test_to_gemini_content_keeps_user_role():
    assert _to_gemini_content({"role": "user", "content": "hi"}) == {
        "role": "user",
        "parts": [{"text": "hi"}],
    }


def test_to_gemini_content_flattens_content_blocks():
    msg = {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
    assert _to_gemini_content(msg) == {"role": "user", "parts": [{"text": "a b"}]}


def test_to_gemini_content_converts_tool_use_block_to_function_call():
    """An assistant's tool_use block must survive the conversion as a
    functionCall part -- the old implementation only kept type=="text"
    blocks and silently dropped everything else."""
    msg = {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "t1", "name": "calc", "input": {"expression": "2+2"}}],
    }
    assert _to_gemini_content(msg) == {
        "role": "model",
        "parts": [{"functionCall": {"name": "calc", "args": {"expression": "2+2"}}}],
    }


def test_to_gemini_content_converts_tool_result_block_to_function_response():
    """A tool_result content block (Anthropic's tool-result shape, sent on a
    role="user" turn) must survive as a functionResponse part instead of
    being dropped because it isn't a type=="text" block."""
    msg = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "4"}],
    }
    assert _to_gemini_content(msg) == {
        "role": "user",
        "parts": [{"functionResponse": {"name": "t1", "response": {"result": "4"}}}],
    }


def test_to_gemini_content_preserves_plain_tool_role_message():
    """A literal role="tool" message (OpenAI-style function-result message,
    plain string content) must not be silently relabeled "user" with its
    content lost -- it becomes a functionResponse part, Gemini's own shape
    for returning a function's result."""
    msg = {"role": "tool", "tool_call_id": "t1", "name": "calc", "content": "4"}
    assert _to_gemini_content(msg) == {
        "role": "user",
        "parts": [{"functionResponse": {"name": "calc", "response": {"result": "4"}}}],
    }


# ─── dispatch_gemini ────────────────────────────────────────────────────────

async def test_dispatch_gemini_sends_model_and_contents():
    client = _fake_client()
    await dispatch_gemini(
        client=client,
        model="gemini-3.6-flash",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=2048,
        temperature=0.5,
    )
    _, kwargs = client.aio.models.generate_content.call_args
    assert kwargs["model"] == "gemini-3.6-flash"
    assert kwargs["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
    assert kwargs["config"]["max_output_tokens"] == 2048
    assert kwargs["config"]["temperature"] == 0.5
    assert "system_instruction" not in kwargs["config"]


async def test_dispatch_gemini_extracts_system_message_from_turn_list():
    """Gemini has no "system" role in `contents` -- a {role: system} entry
    must be pulled out into config.system_instruction and dropped from the
    turn list, mirroring dispatch_anthropic's system extraction."""
    client = _fake_client()
    await dispatch_gemini(
        client=client,
        model="gemini-3.6-flash",
        messages=[
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hello"},
        ],
    )
    _, kwargs = client.aio.models.generate_content.call_args
    assert kwargs["config"]["system_instruction"] == "be terse"
    assert kwargs["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]


async def test_dispatch_gemini_explicit_system_kwarg_wins():
    client = _fake_client()
    await dispatch_gemini(
        client=client,
        model="gemini-3.6-flash",
        messages=[{"role": "user", "content": "hello"}],
        system="explicit system",
    )
    _, kwargs = client.aio.models.generate_content.call_args
    assert kwargs["config"]["system_instruction"] == "explicit system"


async def test_dispatch_gemini_maps_usage_fields():
    """Gemini's usage field names (prompt_token_count/candidates_token_count/
    cached_content_token_count) differ from Anthropic's and OpenAI's -- they
    must map onto the shared normalised dict, not pass through unnamed."""
    client = _fake_client(_fake_gemini_response(input_tokens=100, output_tokens=42, cached_tokens=30))
    result = await dispatch_gemini(
        client=client,
        model="gemini-3.6-flash",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert result["input_tokens"] == 100
    assert result["output_tokens"] == 42
    assert result["cache_read_tokens"] == 30
    # Gemini's implicit caching has no per-call "write" cost the way
    # Anthropic's explicit caching does.
    assert result["cache_write_tokens"] == 0
    assert "response_ms" in result
    assert result["response"] is client.aio.models.generate_content.return_value


async def test_dispatch_gemini_preserves_tool_call_turn_mid_session():
    """A multi-turn session that used tool calls gets routed to Gemini
    mid-session. The tool_use/tool_result turns must reach Gemini as
    functionCall/functionResponse parts, not as mislabeled or empty user
    turns."""
    client = _fake_client()
    await dispatch_gemini(
        client=client,
        model="gemini-3.6-flash",
        messages=[
            {"role": "user", "content": "what is 2+2"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "calc", "input": {"expression": "2+2"}}],
            },
            {"role": "tool", "tool_call_id": "t1", "name": "calc", "content": "4"},
        ],
    )
    _, kwargs = client.aio.models.generate_content.call_args
    contents = kwargs["contents"]
    assert contents[0] == {"role": "user", "parts": [{"text": "what is 2+2"}]}
    assert contents[1] == {
        "role": "model",
        "parts": [{"functionCall": {"name": "calc", "args": {"expression": "2+2"}}}],
    }
    assert contents[2] == {
        "role": "user",
        "parts": [{"functionResponse": {"name": "calc", "response": {"result": "4"}}}],
    }


async def test_dispatch_gemini_defaults_missing_usage_fields_to_zero():
    usage = MagicMock(spec=[])  # no attributes at all
    response = MagicMock()
    response.usage_metadata = usage
    client = _fake_client(response)
    result = await dispatch_gemini(
        client=client, model="gemini-3.6-flash", messages=[{"role": "user", "content": "hi"}],
    )
    assert result["input_tokens"] == 0
    assert result["output_tokens"] == 0
    assert result["cache_read_tokens"] == 0


# ─── extract_gemini_text ────────────────────────────────────────────────────

def test_extract_gemini_text_reads_first_text_part():
    part = MagicMock()
    part.text = "the answer"
    candidate = MagicMock()
    candidate.content.parts = [part]
    response = MagicMock()
    response.candidates = [candidate]
    assert extract_gemini_text(response) == "the answer"


def test_extract_gemini_text_handles_no_candidates():
    response = MagicMock()
    response.candidates = []
    assert extract_gemini_text(response) == ""


# ─── extract_gemini_content / extract_gemini_usage ─────────────────────────

def test_extract_gemini_content_returns_candidates():
    response = MagicMock()
    response.candidates = ["c1", "c2"]
    assert extract_gemini_content(response) is response.candidates


def test_extract_gemini_usage_maps_field_names():
    response = _fake_gemini_response(input_tokens=100, output_tokens=42, cached_tokens=30)
    assert extract_gemini_usage(response) == {"input_tokens": 100, "output_tokens": 42}


def test_extract_gemini_usage_defaults_missing_fields_to_zero():
    response = MagicMock()
    response.usage_metadata = MagicMock(spec=[])
    assert extract_gemini_usage(response) == {"input_tokens": 0, "output_tokens": 0}
