"""Tests for FluxClient SDK internals."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _fake_anthropic_client(text: str = "ok"):
    client = MagicMock()
    response = MagicMock()
    response.model = "claude-haiku-4-5-20251001"
    response.stop_reason = "end_turn"
    block = MagicMock()
    block.text = text
    response.content = [block]
    response.usage.input_tokens = 10
    response.usage.output_tokens = 5
    response.usage.cache_creation_input_tokens = 0
    response.usage.cache_read_input_tokens = 0
    client.messages.create = AsyncMock(return_value=response)
    return client


async def test_verify_raises_without_fluxcompute_key(monkeypatch):
    monkeypatch.delenv("FLUXCOMPUTE_KEY", raising=False)
    from fluxcompute.client import FluxClient
    with patch("fluxcompute.client.anthropic") as mock_anth:
        mock_anth.AsyncAnthropic.return_value = _fake_anthropic_client()
        fc = FluxClient(anthropic_key="sk-ant-test")
    with pytest.raises(ValueError, match="requires fluxcompute_key"):
        await fc.verify()


def test_fluxcompute_key_falls_back_to_env(monkeypatch):
    """The error messages tell users to set FLUXCOMPUTE_KEY, so the
    constructor has to actually read it."""
    monkeypatch.setenv("FLUXCOMPUTE_KEY", "flx_from_env")
    from fluxcompute.client import FluxClient
    with patch("fluxcompute.client.anthropic") as mock_anth:
        mock_anth.AsyncAnthropic.return_value = _fake_anthropic_client()
        fc = FluxClient(anthropic_key="sk-ant-test")
    assert fc._fluxcompute_key == "flx_from_env"
    assert fc._graph_emitter._enabled  # telemetry actually switches on


def test_explicit_fluxcompute_key_beats_env(monkeypatch):
    monkeypatch.setenv("FLUXCOMPUTE_KEY", "flx_from_env")
    from fluxcompute.client import FluxClient
    with patch("fluxcompute.client.anthropic") as mock_anth:
        mock_anth.AsyncAnthropic.return_value = _fake_anthropic_client()
        fc = FluxClient(anthropic_key="sk-ant-test", fluxcompute_key="flx_explicit")
    assert fc._fluxcompute_key == "flx_explicit"


async def test_verify_success_returns_identity():
    """verify() hits GET /v1/whoami on the graph-emitter's shared client --
    same host FLUX_GRAPH_EVENTS_URL already governs, no new client spun up."""
    from fluxcompute.client import FluxClient

    class _StubHTTP:
        async def get(self, url, headers=None):
            self.url = url
            self.headers = headers

            class R:
                status_code = 200

                def raise_for_status(self):
                    pass

                def json(self):
                    return {"customer_id": "cust-1", "name": "Partner"}
            return R()

    with patch("fluxcompute.client.anthropic") as mock_anth:
        mock_anth.AsyncAnthropic.return_value = _fake_anthropic_client()
        fc = FluxClient(anthropic_key="sk-ant-test", fluxcompute_key="flx_test")
    stub = _StubHTTP()
    fc._graph_emitter._client = stub
    result = await fc.verify()
    assert result == {"customer_id": "cust-1", "name": "Partner"}
    assert stub.headers["Authorization"] == "Bearer flx_test"
    assert stub.url.endswith("/v1/whoami")


async def test_verify_raises_on_401():
    from fluxcompute.client import FluxClient

    class _StubHTTP401:
        async def get(self, url, headers=None):
            class R:
                status_code = 401
            return R()

    with patch("fluxcompute.client.anthropic") as mock_anth:
        mock_anth.AsyncAnthropic.return_value = _fake_anthropic_client()
        fc = FluxClient(anthropic_key="sk-ant-test", fluxcompute_key="flx_bad")
    fc._graph_emitter._client = _StubHTTP401()
    with pytest.raises(ValueError, match="Invalid FluxCompute key"):
        await fc.verify()


async def test_context_builder_called_before_dispatch():
    """ContextBuilder.build must be called during _route_and_execute."""
    from fluxcompute.client import FluxClient
    with patch("fluxcompute.client.anthropic") as mock_anth:
        mock_anth.AsyncAnthropic.return_value = _fake_anthropic_client()
        fc = FluxClient(anthropic_key="sk-ant-test")

        with patch.object(fc._context_builder, "build", wraps=fc._context_builder.build) as mock_build:
            await fc.messages.create(
                model="auto",
                messages=[{"role": "user", "content": "hello"}],
            )
        mock_build.assert_called_once()


async def test_cache_manager_called_for_anthropic():
    """CacheManager.prepare_for_anthropic must be called for Anthropic provider."""
    from fluxcompute.client import FluxClient
    with patch("fluxcompute.client.anthropic") as mock_anth:
        mock_anth.AsyncAnthropic.return_value = _fake_anthropic_client()
        fc = FluxClient(anthropic_key="sk-ant-test")

        with patch.object(fc._cache_manager, "prepare_for_anthropic", wraps=fc._cache_manager.prepare_for_anthropic) as mock_cache:
            await fc.messages.create(
                model="auto",
                messages=[{"role": "user", "content": "hello"}],
            )
        mock_cache.assert_called_once()


async def test_response_has_context_compression_field():
    """FluxMetadata.context_compression must be a non-negative float."""
    from fluxcompute.client import FluxClient
    with patch("fluxcompute.client.anthropic") as mock_anth:
        mock_anth.AsyncAnthropic.return_value = _fake_anthropic_client()
        fc = FluxClient(anthropic_key="sk-ant-test")
        result = await fc.messages.create(
            model="auto",
            messages=[{"role": "user", "content": "hello"}],
        )
    assert isinstance(result.fluxcompute.context_compression, float)
    assert result.fluxcompute.context_compression >= 0.0


async def test_sdk_streaming_yields_text_chunks():
    """stream() must yield FluxStreamChunk objects with non-empty text."""
    from fluxcompute.client import FluxClient
    from fluxcompute.models import FluxStreamChunk

    # Build a fake Anthropic streaming client
    async def fake_text_stream():
        for word in ["Hello", " ", "world"]:
            yield word

    fake_final = MagicMock()
    fake_final.usage.input_tokens = 10
    fake_final.usage.output_tokens = 3
    fake_final.usage.cache_creation_input_tokens = 0
    fake_final.usage.cache_read_input_tokens = 0
    fake_final.content = [MagicMock(text="Hello world")]

    fake_stream_ctx = MagicMock()
    fake_stream_ctx.__aenter__ = AsyncMock(return_value=fake_stream_ctx)
    fake_stream_ctx.__aexit__ = AsyncMock(return_value=False)
    fake_stream_ctx.text_stream = fake_text_stream()
    fake_stream_ctx.get_final_message = AsyncMock(return_value=fake_final)

    with patch("fluxcompute.client.anthropic") as mock_anth:
        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=fake_stream_ctx)
        mock_anth.AsyncAnthropic.return_value = mock_client

        fc = FluxClient(anthropic_key="sk-ant-test")
        chunks = []
        async with fc.messages.stream(
            model="auto",
            messages=[{"role": "user", "content": "hi"}],
        ) as s:
            async for chunk in s:
                chunks.append(chunk)

    assert len(chunks) >= 1
    assert all(isinstance(c, FluxStreamChunk) for c in chunks)
    assert all(isinstance(c.text, str) for c in chunks)
    assert s.fluxcompute is not None
    assert s.fluxcompute.model_selected != ""


async def test_sdk_streaming_metadata_populated_after_iteration():
    """fluxcompute metadata on the stream context must be set after iteration ends."""
    from fluxcompute.client import FluxClient
    from fluxcompute.models import FluxMetadata

    async def fake_text_stream():
        yield "test"

    fake_final = MagicMock()
    fake_final.usage.input_tokens = 5
    fake_final.usage.output_tokens = 2
    fake_final.usage.cache_creation_input_tokens = 0
    fake_final.usage.cache_read_input_tokens = 0
    fake_final.content = [MagicMock(text="test")]

    fake_stream_ctx = MagicMock()
    fake_stream_ctx.__aenter__ = AsyncMock(return_value=fake_stream_ctx)
    fake_stream_ctx.__aexit__ = AsyncMock(return_value=False)
    fake_stream_ctx.text_stream = fake_text_stream()
    fake_stream_ctx.get_final_message = AsyncMock(return_value=fake_final)

    with patch("fluxcompute.client.anthropic") as mock_anth:
        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=fake_stream_ctx)
        mock_anth.AsyncAnthropic.return_value = mock_client

        fc = FluxClient(anthropic_key="sk-ant-test")
        async with fc.messages.stream(
            model="auto",
            messages=[{"role": "user", "content": "hi"}],
        ) as s:
            async for _ in s:
                pass

    assert isinstance(s.fluxcompute, FluxMetadata)
    assert s.fluxcompute.savings_usd >= 0.0


class TestVerifyErrorHandling:
    """verify() is a diagnostic helper, so its failures have to say what's
    actually wrong. It previously let httpx.HTTPStatusError escape, so a user
    pointed at an undeployed or misconfigured endpoint got a raw 404 traceback
    implying nothing about whether their key was valid.
    """

    def _client(self):
        from fluxcompute.client import FluxClient
        with patch("fluxcompute.client.anthropic") as mock_anth:
            mock_anth.AsyncAnthropic.return_value = _fake_anthropic_client()
            return FluxClient(anthropic_key="sk-ant-test", fluxcompute_key="flx_test")

    async def _verify_with_status(self, status):
        import httpx
        fc = self._client()

        class _Stub:
            async def get(self, url, headers=None):
                return httpx.Response(status, json={"ok": True}, request=httpx.Request("GET", url))

            async def aclose(self):
                pass

        fc._graph_emitter._client = _Stub()
        return await fc.verify()

    async def test_401_is_reported_as_an_invalid_key(self):
        with pytest.raises(ValueError, match="Invalid FluxCompute key"):
            await self._verify_with_status(401)

    async def test_404_is_reported_as_an_endpoint_problem_not_a_bad_key(self):
        with pytest.raises(ConnectionError, match="endpoint or"):
            await self._verify_with_status(404)

    async def test_500_is_reported_as_a_service_problem(self):
        with pytest.raises(ConnectionError, match="HTTP 500"):
            await self._verify_with_status(500)

    async def test_network_failure_is_reported_as_unreachable(self):
        import httpx
        fc = self._client()

        class _Down:
            async def get(self, url, headers=None):
                raise httpx.ConnectError("no route to host")

            async def aclose(self):
                pass

        fc._graph_emitter._client = _Down()
        with pytest.raises(ConnectionError, match="Could not reach"):
            await fc.verify()

    async def test_success_returns_the_payload(self):
        assert await self._verify_with_status(200) == {"ok": True}
