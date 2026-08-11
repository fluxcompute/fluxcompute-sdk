"""Reasoning models (o1/o3/o4) use `max_completion_tokens` and reject
`temperature`; chat models keep `max_tokens` + `temperature`. dispatch_openai
must adapt the call contract per model family."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from fluxcompute.router.dispatcher import (
    _is_reasoning_model,
    _openai_token_kwargs,
    dispatch_openai,
)


def _fake_openai_client():
    client = MagicMock()
    response = MagicMock()
    response.usage.prompt_tokens = 11
    response.usage.completion_tokens = 7
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


def test_is_reasoning_model():
    assert _is_reasoning_model("o1")
    assert _is_reasoning_model("o3-mini")
    assert _is_reasoning_model("o4-mini")
    assert not _is_reasoning_model("gpt-4o")
    assert not _is_reasoning_model("gpt-4o-mini")


def test_token_kwargs_reasoning_vs_chat():
    reasoning = _openai_token_kwargs("o1", 1500, 1.0)
    assert reasoning == {"max_completion_tokens": 1500}
    assert "temperature" not in reasoning
    assert "max_tokens" not in reasoning

    chat = _openai_token_kwargs("gpt-4o", 4096, 0.7)
    assert chat == {"max_tokens": 4096, "temperature": 0.7}
    assert "max_completion_tokens" not in chat


async def test_dispatch_openai_reasoning_model_uses_max_completion_tokens():
    client = _fake_openai_client()
    await dispatch_openai(
        client=client,
        model="o1",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=1500,
        temperature=1.0,
    )
    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["max_completion_tokens"] == 1500
    assert "max_tokens" not in kwargs
    assert "temperature" not in kwargs


async def test_dispatch_openai_chat_model_uses_max_tokens_and_temperature():
    client = _fake_openai_client()
    await dispatch_openai(
        client=client,
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=4096,
        temperature=0.5,
    )
    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["max_tokens"] == 4096
    assert kwargs["temperature"] == 0.5
    assert "max_completion_tokens" not in kwargs
