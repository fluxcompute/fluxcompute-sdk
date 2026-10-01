"""An answer the token limit emptied raises instead of returning "".

A reasoning model can spend its whole output limit on hidden reasoning and
return empty text with finish_reason "length" and no error. Checked live on
2026-10-01: gpt-6-luna at max_completion_tokens=64 and 256 returned "" with
every output token counted as reasoning. Returned as "", that looks like a
real (blank) answer, so FluxClient raises FluxEmptyResponseError and records
the graph node as failed, keeping the billed tokens and cost.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import openai
import pytest

from fluxcompute import FluxClient, FluxEmptyResponseError

MESSAGES = [{"role": "user", "content": "Classify: 'Your invoice #4471 is overdue.'"}]


def _completion(content, finish_reason, completion_tokens=64, reasoning_tokens=64, tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return openai.types.chat.ChatCompletion.model_validate({
        "id": "c", "object": "chat.completion", "created": 0, "model": "gpt-6-luna",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
        "usage": {"prompt_tokens": 80, "completion_tokens": completion_tokens,
                  "total_tokens": 80 + completion_tokens,
                  "completion_tokens_details": {"reasoning_tokens": reasoning_tokens}},
    })


def _client(completion):
    fc = FluxClient(openai_key="sk-test")
    fc._openai_client = MagicMock()
    fc._openai_client.chat.completions.create = AsyncMock(return_value=completion)
    return fc


async def test_empty_answer_at_the_limit_raises_and_fails_the_node():
    fc = _client(_completion("", "length"))
    with fc.task("classify-email") as t:
        with pytest.raises(FluxEmptyResponseError) as raised:
            await fc.messages.create(model="gpt-6-luna", messages=MESSAGES, max_tokens=64)

    error = raised.value
    assert (error.model, error.max_tokens, error.output_tokens, error.reasoning_tokens) == ("gpt-6-luna", 64, 64, 64)
    assert "Raise max_tokens" in str(error)
    assert error.raw is not None

    [node] = [n for n in fc.get_task_graph(t.task_id).in_order() if n.node_type == "llm_call"]
    assert node.status == "failed"
    assert node.failure_reason == "token_budget"
    assert node.output_tokens == 64
    assert node.cost_usd > 0          # the reasoning tokens were billed


async def test_empty_answer_is_not_kept_as_session_history():
    fc = _client(_completion("", "length"))
    with pytest.raises(FluxEmptyResponseError):
        await fc.messages.create(model="gpt-6-luna", messages=MESSAGES, session_id="s1")
    assert fc._sessions.get_or_create("s1").conversation_history == []


async def test_a_tool_call_with_no_text_is_a_complete_answer():
    tool_calls = [{"id": "t", "type": "function", "function": {"name": "label", "arguments": "{}"}}]
    fc = _client(_completion(None, "length", tool_calls=tool_calls))
    response = await fc.messages.create(model="gpt-6-luna", messages=MESSAGES)
    assert response.fluxcompute.truncated is True


async def test_a_cut_off_answer_with_text_is_returned_and_flagged(caplog):
    fc = _client(_completion("primary — the invoice", "length", reasoning_tokens=50))
    response = await fc.messages.create(model="gpt-6-luna", messages=MESSAGES)

    assert response.text == "primary — the invoice"
    assert response.fluxcompute.truncated is True
    assert response.fluxcompute.reasoning_tokens == 50
    assert "cut off" in caplog.text


async def test_a_complete_answer_is_not_flagged():
    fc = _client(_completion("primary", "stop", completion_tokens=120, reasoning_tokens=99))
    response = await fc.messages.create(model="gpt-6-luna", messages=MESSAGES)
    assert response.fluxcompute.truncated is False
    assert response.fluxcompute.reasoning_tokens == 99


async def test_streaming_an_empty_answer_raises():
    fc = FluxClient(openai_key="sk-test")

    async def no_events():
        return
        yield  # pragma: no cover  (makes this an async generator)

    stream = MagicMock()
    stream.__aiter__ = lambda self: no_events()
    stream.get_final_completion = AsyncMock(return_value=_completion("", "length"))
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=stream)
    ctx.__aexit__ = AsyncMock(return_value=False)
    fc._openai_client = MagicMock()
    fc._openai_client.chat.completions.stream = MagicMock(return_value=ctx)

    with pytest.raises(FluxEmptyResponseError):
        async with fc.messages.stream(model="gpt-6-luna", messages=MESSAGES, max_tokens=64) as s:
            async for _ in s:
                pass
