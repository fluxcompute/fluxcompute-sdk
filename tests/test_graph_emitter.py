"""Tests for the graph event emitter."""

import asyncio
import contextlib

from fluxcompute.graph.emitter import GraphEmitter


class _StubHTTP:
    def __init__(self, status_code=200):
        self.calls = []
        self._status = status_code

    async def post(self, url, json=None, headers=None):
        self.calls.append({"url": url, "json": json, "headers": headers})

        class R:
            status_code = self._status
        R.status_code = self._status
        return R()

    async def aclose(self):
        pass


def _event(i=0):
    return {"node_id": f"n{i}", "task_id": "t1", "status": "succeeded"}


class TestGraphEmitter:
    def test_disabled_emitter_buffers_nothing(self):
        em = GraphEmitter(fluxcompute_key=None)
        em.record(_event())
        assert em._buffer == []

    def test_whoami_url_derived_from_events_url(self):
        em = GraphEmitter(fluxcompute_key="flx_test", events_url="http://localhost:8000/v1/graph/events")
        assert em.whoami_url() == "http://localhost:8000/v1/whoami"

    def test_whoami_url_derived_from_default_events_url(self, monkeypatch):
        # conftest points the env at a local sink for every other test; this
        # one is specifically about the built-in production default.
        monkeypatch.delenv("FLUX_GRAPH_EVENTS_URL", raising=False)
        em = GraphEmitter(fluxcompute_key="flx_test")
        assert em.whoami_url() == "https://api.fluxcompute.dev/v1/whoami"

    async def test_flush_posts_batched_events_with_auth(self):
        em = GraphEmitter(fluxcompute_key="flx_test",
                          events_url="http://test/v1/graph/events")
        stub = _StubHTTP()
        em._client = stub
        em.record(_event(0))
        em.record(_event(1))
        await em.flush()
        assert len(stub.calls) == 1
        call = stub.calls[0]
        assert call["url"] == "http://test/v1/graph/events"
        assert call["json"] == {"events": [_event(0), _event(1)]}
        assert call["headers"]["Authorization"] == "Bearer flx_test"
        assert em._buffer == []

    async def test_failed_flush_rebuffers_events(self):
        em = GraphEmitter(fluxcompute_key="flx_test")

        class _Boom:
            async def post(self, *a, **k):
                raise ConnectionError("down")
        em._client = _Boom()
        em.record(_event())
        await em.flush()
        assert len(em._buffer) == 1

    async def test_periodic_flush_ships_events_below_batch_size(self):
        em = GraphEmitter(fluxcompute_key="flx_test",
                          events_url="http://test/v1/graph/events",
                          flush_interval_seconds=0.05)
        stub = _StubHTTP()
        em._client = stub
        em.record(_event())  # well under batch_size=20; only the periodic
                              # loop should be able to ship this
        await asyncio.sleep(0.15)
        assert len(stub.calls) == 1
        assert stub.calls[0]["json"] == {"events": [_event()]}
        await em.close()

    async def test_cancelled_flush_rebuffers_events_instead_of_losing_them(self):
        """Regression: cancelling flush() mid-flight (e.g. close() cancelling the
        periodic flush task while a POST is in-flight) must not silently drop
        the batch — CancelledError bypasses a bare `except Exception`, and by
        the time it's raised the buffer has already been cleared."""
        em = GraphEmitter(fluxcompute_key="flx_test")

        class _Hangs:
            async def post(self, *a, **k):
                await asyncio.sleep(3600)  # never resolves on its own

        em._client = _Hangs()
        em.record(_event())

        task = asyncio.ensure_future(em.flush())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert em._buffer == []  # cleared internally once flush() started

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        assert em._buffer == [_event()]  # re-buffered instead of lost

    async def test_flush_chunks_to_server_batch_cap(self):
        """One POST above the server's 200-event cap would be 422'd whole —
        flush must chunk instead."""
        em = GraphEmitter(fluxcompute_key="flx_test",
                          events_url="http://test/v1/graph/events")
        stub = _StubHTTP()
        em._client = stub
        em._buffer = [_event(i) for i in range(450)]
        await em.flush()
        assert [len(c["json"]["events"]) for c in stub.calls] == [200, 200, 50]

    def test_content_is_withheld_by_default(self):
        """Response content must not leave the process unless opted in."""
        em = GraphEmitter(fluxcompute_key="flx_test")
        em.record({
            **_event(),
            "output_preview": "the model said this",
            "prompt_full": "the full prompt",
            "output_full": "the full output",
            "model": "claude-3-5-haiku",
            "cost_usd": 0.001,
        })
        buffered = em._buffer[0]
        assert "output_preview" not in buffered
        assert "prompt_full" not in buffered
        assert "output_full" not in buffered
        # structure and metrics still ship — the dashboard stays useful
        assert buffered["node_id"] == "n0"
        assert buffered["status"] == "succeeded"
        assert buffered["model"] == "claude-3-5-haiku"
        assert buffered["cost_usd"] == 0.001

    def test_error_string_still_ships_by_default(self):
        """`error` is a diagnostic, not model output — withholding it would
        make failed nodes unreadable in the dashboard."""
        em = GraphEmitter(fluxcompute_key="flx_test")
        em.record({**_event(), "status": "failed", "error": "TimeoutError: upstream"})
        assert em._buffer[0]["error"] == "TimeoutError: upstream"

    def test_content_capture_true_preserves_content(self):
        em = GraphEmitter(fluxcompute_key="flx_test", content_capture=True)
        event = {
            **_event(),
            "output_preview": "the model said this",
            "prompt_full": "the full prompt",
            "output_full": "the full output",
        }
        em.record(event)
        assert em._buffer[0] == event

    async def test_content_withheld_on_the_wire_not_just_in_the_buffer(self):
        """End-to-end: what actually gets POSTed carries no content."""
        em = GraphEmitter(fluxcompute_key="flx_test")
        stub = _StubHTTP()
        em._client = stub
        em.record({**_event(), "output_preview": "secret", "output_full": "secret"})
        await em.flush()
        sent = stub.calls[0]["json"]["events"][0]
        assert "secret" not in str(sent)

    async def test_rebuffer_strips_full_payloads(self):
        """A failing network must not pin snapshot payloads in memory.

        content_capture=True so the payloads actually reach the buffer —
        otherwise record() strips them first and this asserts nothing.
        """
        em = GraphEmitter(fluxcompute_key="flx_test", content_capture=True)

        class _Boom:
            async def post(self, *a, **k):
                raise ConnectionError("down")

        em._client = _Boom()
        em.record({**_event(), "prompt_full": "x" * 1000, "output_full": "y" * 1000})
        await em.flush()
        assert em._buffer == [_event()]

    async def test_buffer_does_not_grow_without_bound_while_offline(self):
        """flush() clears the buffer before sending, so a cap tested against
        len(self._buffer) can never fire — every failed flush put the whole
        batch back and the buffer grew by one event per call, forever."""
        em = GraphEmitter(fluxcompute_key="flx_test", batch_size=20)

        class _Boom:
            async def post(self, *a, **k):
                raise ConnectionError("down")

        em._client = _Boom()
        for i in range(500):
            em.record({**_event(i), "status": "succeeded"})
            await em.flush()
        assert len(em._buffer) <= 40, f"buffer grew to {len(em._buffer)} on a dead network"
