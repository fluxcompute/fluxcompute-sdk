"""The OpenAI call contract no longer depends on the model's name.

Every OpenAI model gets `max_completion_tokens` (accepted by chat and reasoning
models alike) and `temperature` only when the caller set one. A model that
rejects a param it was sent gets one resend without it, and the rejection is
remembered for that model. The old o1/o3/o4 name check sent gpt-5.x and
gpt-6-luna `max_tokens` + `temperature`, which both reject with a 400.

The 400s below are built from the installed provider SDKs' own error classes,
with the bodies the APIs returned when checked live on 2026-10-01.
"""
from __future__ import annotations

from contextlib import AsyncExitStack
from unittest.mock import AsyncMock, MagicMock

import anthropic
import httpx
import openai
import pytest

from fluxcompute.router import dispatcher
from fluxcompute.router.dispatcher import (
    _openai_token_kwargs,
    anthropic_finish_state,
    dispatch_anthropic,
    dispatch_openai,
    gemini_finish_state,
    open_stream_with_param_fallback,
    openai_finish_state,
)

MESSAGES = [{"role": "user", "content": "hi"}]


def _openai_400(param, code, message="rejected"):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    body = {"message": message, "type": "invalid_request_error", "param": param, "code": code}
    return openai.BadRequestError(message, response=httpx.Response(400, request=request), body=body)


def _anthropic_400(message):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
    return anthropic.BadRequestError(message, response=httpx.Response(400, request=request), body=body)


def _fake_openai_client(*errors):
    """create() raises `errors` in order, then succeeds on every later call."""
    client = MagicMock()
    response = MagicMock()
    response.usage.prompt_tokens = 11
    response.usage.completion_tokens = 7
    pending = list(errors)

    async def create(**kwargs):
        if pending:
            raise pending.pop(0)
        return response

    client.chat.completions.create = AsyncMock(side_effect=create)
    return client


def _sent(client, call=-1):
    return client.chat.completions.create.call_args_list[call].kwargs


class TestTokenKwargs:
    @pytest.mark.parametrize("model", ["gpt-4o-mini", "o1", "gpt-5.6-luna", "gpt-6-luna", "gpt-7-nova"])
    def test_every_openai_model_gets_max_completion_tokens(self, model):
        """Including a name no release has seen yet: new models need no patch."""
        assert _openai_token_kwargs(model, 1500, None) == {"max_completion_tokens": 1500}

    def test_temperature_is_sent_only_when_the_caller_set_one(self):
        assert _openai_token_kwargs("gpt-4o", 4096, 0.7) == {"max_completion_tokens": 4096, "temperature": 0.7}
        assert "temperature" not in _openai_token_kwargs("gpt-4o", 4096, None)


class TestOpenAIParamFallback:
    async def test_rejected_temperature_is_dropped_and_the_call_resent_once(self):
        client = _fake_openai_client(_openai_400("temperature", "unsupported_value"))
        await dispatch_openai(client=client, model="gpt-6-luna", messages=MESSAGES, temperature=0.0)

        assert client.chat.completions.create.await_count == 2
        assert _sent(client, 0)["temperature"] == 0.0
        assert "temperature" not in _sent(client, 1)
        assert _sent(client, 1)["max_completion_tokens"] == 4096

    async def test_the_rejection_is_remembered_for_that_model(self):
        client = _fake_openai_client(_openai_400("temperature", "unsupported_value"))
        await dispatch_openai(client=client, model="gpt-6-luna", messages=MESSAGES, temperature=0.0)
        await dispatch_openai(client=client, model="gpt-6-luna", messages=MESSAGES, temperature=0.0)

        assert client.chat.completions.create.await_count == 3   # 400 + resend + one clean call
        assert "temperature" not in _sent(client, 2)

    async def test_the_rejection_does_not_leak_to_other_models(self):
        client = _fake_openai_client(_openai_400("temperature", "unsupported_value"))
        await dispatch_openai(client=client, model="gpt-6-luna", messages=MESSAGES, temperature=0.0)
        await dispatch_openai(client=client, model="gpt-4o", messages=MESSAGES, temperature=0.0)

        assert _sent(client)["temperature"] == 0.0

    async def test_a_model_that_rejects_max_completion_tokens_gets_max_tokens(self):
        client = _fake_openai_client(_openai_400("max_completion_tokens", "unsupported_parameter"))
        await dispatch_openai(client=client, model="some-legacy-model", messages=MESSAGES, max_tokens=900)

        assert _sent(client)["max_tokens"] == 900
        assert "max_completion_tokens" not in _sent(client)

    @pytest.mark.parametrize("error", [
        _openai_400("messages", "invalid_value"),
        # Same param, but a real caller error (over the model's limit): renaming fixes nothing.
        _openai_400("max_completion_tokens", "invalid_value"),
        _openai_400(None, None, "temperature is broken"),
    ])
    async def test_other_400s_surface_unchanged_without_a_resend(self, error):
        client = _fake_openai_client(error)
        with pytest.raises(openai.BadRequestError) as raised:
            await dispatch_openai(client=client, model="gpt-6-luna", messages=MESSAGES, temperature=0.0)
        assert raised.value is error
        assert client.chat.completions.create.await_count == 1
        assert dispatcher._REJECTED_PARAMS == {}

    async def test_a_param_the_request_did_not_send_is_not_resent(self):
        """Nothing to drop, so a resend would fail identically."""
        client = _fake_openai_client(_openai_400("temperature", "unsupported_value"))
        with pytest.raises(openai.BadRequestError):
            await dispatch_openai(client=client, model="gpt-6-luna", messages=MESSAGES)
        assert client.chat.completions.create.await_count == 1

    async def test_rate_limits_are_not_resent(self):
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        error = openai.RateLimitError("slow down", response=httpx.Response(429, request=request),
                                      body={"param": "temperature", "code": "unsupported_value"})
        client = _fake_openai_client(error)
        with pytest.raises(openai.RateLimitError):
            await dispatch_openai(client=client, model="gpt-6-luna", messages=MESSAGES, temperature=0.0)
        assert client.chat.completions.create.await_count == 1


class TestAnthropicParamFallback:
    def _client(self, *side_effect):
        client = MagicMock()
        response = MagicMock()
        response.usage.input_tokens = 3
        response.usage.output_tokens = 2
        response.usage.cache_creation_input_tokens = 0
        response.usage.cache_read_input_tokens = 0
        client.messages.create = AsyncMock(side_effect=[*side_effect, response])
        return client

    async def test_deprecated_temperature_is_dropped_and_resent(self):
        """claude-opus-4-8 answers a non-default temperature with this 400."""
        client = self._client(_anthropic_400("`temperature` is deprecated for this model."))
        await dispatch_anthropic(client=client, model="claude-opus-4-8", messages=MESSAGES, temperature=0.0)

        assert client.messages.create.await_count == 2
        assert "temperature" not in client.messages.create.call_args.kwargs

    async def test_an_out_of_range_temperature_is_the_callers_error(self):
        error = _anthropic_400("temperature: range: 0..1")
        client = self._client(error)
        with pytest.raises(anthropic.BadRequestError):
            await dispatch_anthropic(client=client, model="claude-sonnet-5", messages=MESSAGES, temperature=3.0)
        assert client.messages.create.await_count == 1

    async def test_temperature_is_left_out_unless_set(self):
        client = self._client()
        await dispatch_anthropic(client=client, model="claude-sonnet-5", messages=MESSAGES)
        assert "temperature" not in client.messages.create.call_args.kwargs


class TestStreamFallback:
    async def test_a_stream_rejected_on_open_is_reopened_without_the_param(self):
        opened = MagicMock()
        good = MagicMock()
        good.__aenter__ = AsyncMock(return_value=opened)
        good.__aexit__ = AsyncMock(return_value=False)
        bad = MagicMock()
        bad.__aenter__ = AsyncMock(side_effect=_openai_400("temperature", "unsupported_value"))
        bad.__aexit__ = AsyncMock(return_value=False)
        open_stream = MagicMock(side_effect=[bad, good])

        async with AsyncExitStack() as stack:
            stream = await open_stream_with_param_fallback(
                stack, open_stream, "gpt-6-luna", {"model": "gpt-6-luna", "temperature": 0.0},
            )
        assert stream is opened
        assert open_stream.call_args_list[1].kwargs == {"model": "gpt-6-luna"}


class TestFinishState:
    def test_openai_reasoning_spent_the_whole_budget(self):
        """The shape gpt-6-luna returned live at max_completion_tokens=64."""
        completion = openai.types.chat.ChatCompletion.model_validate({
            "id": "c", "object": "chat.completion", "created": 0, "model": "gpt-6-luna",
            "choices": [{"index": 0, "finish_reason": "length",
                         "message": {"role": "assistant", "content": ""}}],
            "usage": {"prompt_tokens": 80, "completion_tokens": 64, "total_tokens": 144,
                      "completion_tokens_details": {"reasoning_tokens": 64}},
        })
        assert openai_finish_state(completion) == {
            "truncated": True, "finish_reason": "length", "reasoning_tokens": 64, "has_tool_calls": False,
        }

    def test_openai_tool_call(self):
        completion = openai.types.chat.ChatCompletion.model_validate({
            "id": "c", "object": "chat.completion", "created": 0, "model": "gpt-6-luna",
            "choices": [{"index": 0, "finish_reason": "length", "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "t", "type": "function",
                                "function": {"name": "lookup", "arguments": "{}"}}]}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })
        assert openai_finish_state(completion)["has_tool_calls"] is True

    def test_anthropic_max_tokens(self):
        message = MagicMock(stop_reason="max_tokens", content=[MagicMock(type="tool_use")])
        assert anthropic_finish_state(message) == {
            "truncated": True, "finish_reason": "max_tokens", "reasoning_tokens": 0, "has_tool_calls": True,
        }

    def test_gemini_max_tokens(self):
        from google.genai import types
        response = types.GenerateContentResponse(
            candidates=[types.Candidate(finish_reason=types.FinishReason.MAX_TOKENS,
                                        content=types.Content(role="model", parts=[]))],
            usage_metadata=types.GenerateContentResponseUsageMetadata(thoughts_token_count=50),
        )
        assert gemini_finish_state(response) == {
            "truncated": True, "finish_reason": "MAX_TOKENS", "reasoning_tokens": 50, "has_tool_calls": False,
        }

    def test_mocks_and_missing_fields_read_as_not_truncated(self):
        assert openai_finish_state(MagicMock())["truncated"] is False
        assert anthropic_finish_state(MagicMock())["truncated"] is False
        assert gemini_finish_state(MagicMock())["truncated"] is False
