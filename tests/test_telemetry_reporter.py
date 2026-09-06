"""Tests for TelemetryReporter, pinning the documented batching contract
(docs/telemetry-contract.md: "10 events or 30 seconds"), including the
interval-based flush for processes that never fill a batch.
"""

from __future__ import annotations

import asyncio

import pytest

from fluxcompute.models import TelemetryEvent
from fluxcompute.telemetry.reporter import TelemetryReporter


def _event(**over):
    base = dict(
        customer_key="flx_test", session_id=None, difficulty_score=0.1,
        difficulty_label="easy", model_selected="claude-haiku-4-5-20251001",
        baseline_model="claude-opus-4-8", input_tokens=10, output_tokens=5,
        cost_usd=0.001, baseline_cost_usd=0.01, savings_usd=0.009,
        classification_ms=1.0, overhead_ms=2.0,
    )
    base.update(over)
    return TelemetryEvent(**base)


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


class TestEnablement:
    def test_no_key_means_no_buffering(self):
        r = TelemetryReporter(fluxcompute_key=None)
        r.record(_event())
        assert r._buffer == []

    def test_disabled_explicitly_means_no_buffering(self):
        r = TelemetryReporter(fluxcompute_key="flx_test", enabled=False)
        r.record(_event())
        assert r._buffer == []


class TestBatching:
    async def test_flushes_at_batch_size(self):
        r = TelemetryReporter(fluxcompute_key="flx_test", batch_size=3)
        r._client = _StubHTTP()
        for _ in range(3):
            r.record(_event())
        await asyncio.sleep(0)  # let the fire-and-forget flush run
        assert len(r._client.calls) == 1
        assert len(r._client.calls[0]["json"]["events"]) == 3
        assert r._buffer == []

    async def test_does_not_flush_below_batch_size(self):
        r = TelemetryReporter(fluxcompute_key="flx_test", batch_size=10)
        r._client = _StubHTTP()
        r.record(_event())
        await asyncio.sleep(0)
        assert r._client.calls == []
        assert len(r._buffer) == 1

    async def test_periodic_flush_sends_a_partial_batch(self):
        """The contract's 30-second half. Without it, a long-running process
        that never fills a batch never reports anything."""
        r = TelemetryReporter(
            fluxcompute_key="flx_test", batch_size=100, flush_interval_seconds=0.01,
        )
        r._client = _StubHTTP()
        r.record(_event())
        assert r._client.calls == []  # nowhere near batch_size
        await asyncio.sleep(0.05)
        assert len(r._client.calls) == 1, "periodic flush never ran"
        await r.close()

    async def test_close_flushes_a_partial_batch(self):
        r = TelemetryReporter(fluxcompute_key="flx_test", batch_size=100)
        stub = _StubHTTP()          # held: close() drops the reporter's ref
        r._client = stub
        r.record(_event())
        await r.close()
        assert len(stub.calls) == 1
        assert r._client is None     # and the client is released


class TestPayload:
    # Every field this channel is allowed to send, per
    # docs/telemetry-contract.md. Pinned as an exact set so adding a field
    # without updating the published contract fails here.
    ALLOWED = {
        "customer_key", "session_id", "difficulty_score", "difficulty_label",
        "model_selected", "baseline_model", "input_tokens", "output_tokens",
        "cost_usd", "baseline_cost_usd", "savings_usd", "classification_ms",
        "overhead_ms",
    }

    async def test_sends_metrics_and_auth_header_but_no_content(self):
        r = TelemetryReporter(fluxcompute_key="flx_test", batch_size=1)
        r._client = _StubHTTP()
        r.record(_event(session_id="s1"))
        await asyncio.sleep(0)
        call = r._client.calls[0]
        assert call["headers"]["Authorization"] == "Bearer flx_test"
        sent = call["json"]["events"][0]
        assert sent["cost_usd"] == 0.001
        assert sent["savings_usd"] == 0.009
        assert sent["model_selected"] == "claude-haiku-4-5-20251001"
        # metrics-only by contract: token *counts* are fine, text is not
        assert set(sent) == self.ALLOWED, (
            f"payload drifted from the published contract: {set(sent) ^ self.ALLOWED}"
        )
        assert all(not isinstance(v, (list, dict)) for v in sent.values()), (
            "structured values are how prompt/response text would sneak in"
        )


class TestResilience:
    async def test_rebuffers_on_network_failure(self):
        r = TelemetryReporter(fluxcompute_key="flx_test", batch_size=100)

        class _Boom:
            async def post(self, *a, **k):
                raise ConnectionError("down")

        r._client = _Boom()
        r.record(_event())
        await r.flush()
        assert len(r._buffer) == 1, "events dropped instead of re-buffered"

    async def test_buffer_does_not_grow_without_bound_while_offline(self):
        """A network that stays down must not turn the buffer into a leak.
        Re-buffering stops once the buffer is already at 2x batch_size."""
        r = TelemetryReporter(fluxcompute_key="flx_test", batch_size=2)

        class _Boom:
            async def post(self, *a, **k):
                raise ConnectionError("down")

        r._client = _Boom()
        for _ in range(50):
            r.record(_event())
            await r.flush()
        assert len(r._buffer) <= 12, f"buffer grew to {len(r._buffer)} while offline"

    async def test_5xx_response_rebuffers_events(self):
        """A non-200 response is a real failure, not a raised exception --
        the old flush() logged this and still cleared the buffer, so a
        server-side outage silently dropped data on the client side too."""
        r = TelemetryReporter(fluxcompute_key="flx_test", batch_size=100)
        r._client = _StubHTTP(status_code=503)
        r.record(_event())
        await r.flush()
        assert len(r._buffer) == 1

    async def test_4xx_response_does_not_rebuffer_events(self):
        """A 4xx will fail identically on every retry -- only >=500 is
        treated as transient and retried."""
        r = TelemetryReporter(fluxcompute_key="flx_test", batch_size=100)
        r._client = _StubHTTP(status_code=422)
        r.record(_event())
        await r.flush()
        assert r._buffer == []

    async def test_cancellation_rebuffers_instead_of_dropping(self):
        """CancelledError is a BaseException, so `except Exception` never saw
        it — a cancelled flush silently lost its events."""
        r = TelemetryReporter(fluxcompute_key="flx_test", batch_size=100)

        class _Cancel:
            async def post(self, *a, **k):
                raise asyncio.CancelledError()

        r._client = _Cancel()
        r.record(_event())
        with pytest.raises(asyncio.CancelledError):
            await r.flush()
        assert len(r._buffer) == 1, "cancelled flush dropped events"

    def test_record_outside_event_loop_does_not_warn_or_raise(self):
        """Recording from sync code must not emit a RuntimeWarning from SDK
        internals (the coroutine-never-awaited trap)."""
        r = TelemetryReporter(fluxcompute_key="flx_test", batch_size=1)
        r._client = _StubHTTP()
        r.record(_event())  # no running loop here
        assert len(r._buffer) == 1  # retained for close() to flush
