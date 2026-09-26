"""FluxClient(google_key=...) end-to-end, same shape as the existing
Anthropic/OpenAI FluxClient tests -- Gemini must route through all three
tiers exactly like the other two providers."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fluxcompute.client import FluxClient


def _fake_gemini_result(text="ok", input_tokens=10, output_tokens=5):
    part = MagicMock()
    part.text = text
    candidate = MagicMock()
    candidate.content.parts = [part]
    usage = MagicMock()
    usage.prompt_token_count = input_tokens
    usage.candidates_token_count = output_tokens
    usage.cached_content_token_count = 0
    response = MagicMock()
    response.candidates = [candidate]
    response.usage_metadata = usage
    response.model = "gemini-3.6-flash"
    return {
        "response": response,
        "response_ms": 12.0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_write_tokens": 0,
        "cache_read_tokens": 0,
    }


def test_google_key_selects_google_provider():
    client = FluxClient(google_key="fake-google-key", telemetry=False)
    assert client._provider == "google"
    assert client._google_client is not None
    assert client._anthropic_client is None
    assert client._openai_client is None


def test_google_key_falls_back_to_env_var(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "env-google-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = FluxClient(telemetry=False)
    assert client._provider == "google"


def test_anthropic_still_wins_precedence_when_multiple_keys_given():
    """Generalizing two-provider precedence to three must not change the
    documented two-provider behavior: anthropic wins when multiple keys are
    passed and no explicit provider= override is given."""
    with patch("fluxcompute.client.anthropic"):
        client = FluxClient(
            anthropic_key="sk-ant-test", google_key="fake-google-key", telemetry=False,
        )
    assert client._provider == "anthropic"


async def test_route_and_execute_dispatches_through_google_provider():
    client = FluxClient(google_key="fake-google-key", telemetry=False)

    with patch(
        "fluxcompute.client.dispatch_gemini",
        new_callable=AsyncMock, return_value=_fake_gemini_result("four"),
    ) as mock_dispatch:
        response = await client.messages.create(
            model="auto",
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )

    assert mock_dispatch.await_args.kwargs["client"] is client._google_client
    assert response.fluxcompute.model_selected in (
        "gemini-3.1-flash-lite", "gemini-3.6-flash", "gemini-3.1-pro-preview",
    )
    assert response.provider == "google"


async def test_route_and_execute_records_gemini_text_via_extractor():
    """Regression guard: client.py's session-history update assumed either
    Anthropic's `.content` blocks or OpenAI's `.choices[0].message`, neither
    of which a real Gemini response object has -- this used to raise
    AttributeError instead of updating session state."""
    client = FluxClient(google_key="fake-google-key", telemetry=False)

    with patch(
        "fluxcompute.client.dispatch_gemini",
        new_callable=AsyncMock, return_value=_fake_gemini_result("the answer is four"),
    ):
        response = await client.messages.create(
            model="auto",
            session_id="sess-1",
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )

    session = client._sessions.get_or_create("sess-1")
    assert session.conversation_history[-1]["content"] == "the answer is four"
    assert response.fluxcompute.session_id == "sess-1"


async def test_flux_response_text_content_usage_work_for_google_provider():
    """The bug this guards: FluxResponse.text/.content/.usage used to hardcode
    an anthropic/openai check and fall into the "OpenAI" else-branch for any
    other provider, so response.raw.choices[...] raised AttributeError on a
    real google.genai response object (which has .candidates, not .choices).
    No prior test exercised these convenience properties on a Gemini-backed
    FluxResponse."""
    client = FluxClient(google_key="fake-google-key", telemetry=False)

    with patch(
        "fluxcompute.client.dispatch_gemini",
        new_callable=AsyncMock, return_value=_fake_gemini_result("four", input_tokens=11, output_tokens=3),
    ):
        response = await client.messages.create(
            model="auto",
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )

    assert response.text == "four"
    assert response.content is response.raw.candidates
    assert response.usage == {"input_tokens": 11, "output_tokens": 3}


async def test_no_api_key_still_raises_with_google_mentioned():
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        with patch.dict("os.environ", {}, clear=True):
            FluxClient(telemetry=False)


async def test_stream_raises_clear_error_for_google_provider():
    """Streaming has no Gemini dispatch yet -- this must fail loudly and
    clearly rather than falling through to the OpenAI branch and raising a
    confusing "OpenAI client not initialised" error."""
    client = FluxClient(google_key="fake-google-key", telemetry=False)

    with pytest.raises(NotImplementedError, match="Google"):
        async with client.messages.stream(
            model="auto",
            messages=[{"role": "user", "content": "hi"}],
        ) as stream:
            async for _ in stream:
                pass
