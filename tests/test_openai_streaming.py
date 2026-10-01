"""OpenAI streaming event handling.

`chat.completions.stream()` yields typed ChatCompletionStreamEvent objects,
not raw chunks. These tests build events from the installed openai package's
own classes so they track the real shape.
"""

from __future__ import annotations

import pytest

from fluxcompute.router.dispatcher import _openai_stream_text, _openai_token_kwargs

openai = pytest.importorskip("openai")


def _content_delta(text: str):
    from openai.lib.streaming.chat import ContentDeltaEvent
    return ContentDeltaEvent(type="content.delta", delta=text, snapshot=text)


class TestOpenAIStreamText:
    def test_extracts_text_from_content_delta_events(self):
        assert _openai_stream_text(_content_delta("Hello")) == "Hello"

    def test_real_event_objects_have_no_choices_attribute(self):
        """Pins the reason the old code failed, so a regression is obvious."""
        event = _content_delta("Hello")
        assert not hasattr(event, "choices")

    def test_unknown_event_types_yield_no_text_instead_of_raising(self):
        class _FutureEvent:
            type = "something.new.in.openai"

        assert _openai_stream_text(_FutureEvent()) == ""

    def test_content_done_event_yields_no_text(self):
        """Done events carry the full snapshot; counting them would duplicate
        the whole response."""
        class _Done:
            type = "content.done"
            content = "Hello world"

        assert _openai_stream_text(_Done()) == ""

    def test_empty_delta_yields_empty_string_not_none(self):
        assert _openai_stream_text(_content_delta("")) == ""

    def test_accumulating_deltas_reconstructs_the_response(self):
        events = [_content_delta(t) for t in ("Hel", "lo ", "world")]
        assert "".join(_openai_stream_text(e) for e in events) == "Hello world"


class TestStreamingUsesReasoningModelContract:
    def test_streaming_uses_max_completion_tokens(self):
        """Streaming once hardcoded max_tokens, which reasoning models reject.
        gpt-6-luna is the OpenAI *easy* tier, so `auto` streams to it."""
        kwargs = _openai_token_kwargs("gpt-6-luna", 1500, None)
        assert kwargs == {"max_completion_tokens": 1500}

    def test_an_explicit_temperature_is_passed_through(self):
        kwargs = _openai_token_kwargs("gpt-4o", 4096, 0.7)
        assert kwargs == {"max_completion_tokens": 4096, "temperature": 0.7}
