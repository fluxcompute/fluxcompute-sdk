"""
Telemetry reporter — sends anonymised metrics to FluxCompute backend.

Runs async and non-blocking. Never delays the LLM response.
Sends: difficulty, model, tokens, cost, savings. Never sends query content.

Graph events inside a task() scope are sent by a separate emitter
(fluxcompute/graph/emitter.py); see FluxClient.__init__ for exactly what
each flag enables.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from typing import Optional

import httpx

from fluxcompute.models import TelemetryEvent

logger = logging.getLogger("fluxcompute.telemetry")

TELEMETRY_URL = "https://api.fluxcompute.dev/v1/telemetry/event"


def _default_telemetry_url() -> str:
    """Resolve the telemetry endpoint, in precedence order:
    FLUX_TELEMETRY_URL > derived from FLUX_GRAPH_EVENTS_URL's host (a dev
    pointing graph events at a local server almost certainly wants telemetry
    on the same host) > the public default.
    """
    explicit = os.environ.get("FLUX_TELEMETRY_URL")
    if explicit:
        return explicit
    graph_url = os.environ.get("FLUX_GRAPH_EVENTS_URL")
    if graph_url and "/v1/" in graph_url:
        return graph_url.rsplit("/v1/", 1)[0] + "/v1/telemetry/event"
    return TELEMETRY_URL


class TelemetryReporter:
    """
    Async telemetry reporter that batches and sends events
    to the FluxCompute backend.
    """

    def __init__(
        self,
        fluxcompute_key: Optional[str] = None,
        telemetry_url: Optional[str] = None,
        enabled: bool = True,
        batch_size: int = 10,
        flush_interval_seconds: float = 30.0,
    ):
        self._key = fluxcompute_key
        self._url = telemetry_url or _default_telemetry_url()
        self._enabled = enabled and fluxcompute_key is not None
        self._batch_size = batch_size
        self._flush_interval = flush_interval_seconds
        self._buffer: list[dict] = []
        self._client: Optional[httpx.AsyncClient] = None
        self._flush_task: Optional[asyncio.Task] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    def _ensure_flush_loop(self) -> None:
        """Start the periodic flush. Without it, a process that never calls
        close() — a long-running server, a notebook — loses every event that
        didn't happen to fill a batch."""
        if self._flush_task is not None and not self._flush_task.done():
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return  # no running event loop; batch-size flush still works
        # Check for the loop BEFORE building the coroutine, or it is left
        # un-awaited and Python prints a RuntimeWarning from SDK internals.
        self._flush_task = asyncio.ensure_future(self._flush_loop())

    async def _flush_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._flush_interval)
                await self.flush()
        except asyncio.CancelledError:
            pass

    def record(self, event: TelemetryEvent) -> None:
        """
        Record a telemetry event. Non-blocking.
        Buffers locally and flushes when batch_size is reached.
        """
        if not self._enabled:
            return

        self._buffer.append({
            "customer_key": event.customer_key,
            "session_id": event.session_id,
            "difficulty_score": event.difficulty_score,
            "difficulty_label": event.difficulty_label,
            "model_selected": event.model_selected,
            "baseline_model": event.baseline_model,
            "input_tokens": event.input_tokens,
            "output_tokens": event.output_tokens,
            "cost_usd": event.cost_usd,
            "baseline_cost_usd": event.baseline_cost_usd,
            "savings_usd": event.savings_usd,
            "classification_ms": event.classification_ms,
            "overhead_ms": event.overhead_ms,
        })

        self._ensure_flush_loop()
        if len(self._buffer) >= self._batch_size:
            # Fire and forget — don't await
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return  # flushed on close() instead
            asyncio.ensure_future(self.flush())

    async def flush(self) -> None:
        """Send buffered events to FluxCompute backend."""
        if not self._buffer or not self._enabled:
            return

        events = self._buffer.copy()
        self._buffer.clear()

        try:
            client = await self._get_client()
            response = await client.post(
                self._url,
                json={"events": events},
                headers={
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type": "application/json",
                },
            )
            if response.status_code >= 500:
                # Transient server-side failure -- re-buffer instead of
                # treating a non-200 as delivered (matches GraphEmitter).
                logger.warning(f"Telemetry flush failed: {response.status_code} (will retry)")
                self._rebuffer(events)
                return
            if response.status_code != 200:
                logger.warning(f"Telemetry flush failed: {response.status_code}")
        except asyncio.CancelledError:
            # CancelledError is a BaseException, so the handler below never saw
            # it: a cancelled flush silently dropped its events instead of
            # re-buffering them.
            self._rebuffer(events)
            raise
        except Exception as e:
            logger.debug(f"Telemetry flush error (non-fatal): {e}")
            self._rebuffer(events)

    def _rebuffer(self, events: list) -> None:
        """Re-buffer unsent events, bounded so a failing network can't grow the
        buffer without limit.

        The bound is computed against the room left rather than checked up
        front, because flush() clears the buffer before calling this.
        """
        cap = self._batch_size * 2
        room = cap - len(self._buffer)
        if room <= 0:
            return
        if len(events) > room:
            logger.debug("Telemetry buffer full; dropping %d events", len(events) - room)
        self._buffer.extend(events[:room])

    async def close(self) -> None:
        """Flush remaining events and close HTTP client."""
        if self._flush_task is not None:
            self._flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._flush_task
            self._flush_task = None
        await self.flush()
        if self._client:
            await self._client.aclose()
            self._client = None
