"""
Graph event emitter — ships task-graph node events to the FluxCompute backend.

Same contract as telemetry: async, batched, non-blocking, never fails or
delays an LLM response. Point FLUX_GRAPH_EVENTS_URL at a local server
(http://localhost:8000/v1/graph/events) to feed the dashboard during dev.

By default this ships graph STRUCTURE only — node names, types, parentage,
status, failure reason, model, tokens, cost, timings. Response content
(the output preview and the full prompt/output snapshots taken on failure)
is opt-in via `content_capture=True`, since it would otherwise leave the
customer's process. See the `content_capture` docstring on
`FluxClient.__init__` for details.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger("fluxcompute.graph")

GRAPH_EVENTS_URL = "https://api.fluxcompute.dev/v1/graph/events"
_MAX_EVENTS_PER_POST = 200      # batch cap of the /v1/graph/events ingest API — bigger POSTs get 422'd whole
_SNAPSHOT_FIELDS = ("prompt_full", "output_full")
# Everything the model produced or consumed. Withheld unless content_capture
# is on. `error` is deliberately NOT here: it's an exception string the
# dashboard needs to be useful at all, not model output.
_CONTENT_FIELDS = (*_SNAPSHOT_FIELDS, "output_preview")


class GraphEmitter:
    """Batches node events and POSTs them to /v1/graph/events."""

    def __init__(
        self,
        fluxcompute_key: Optional[str] = None,
        events_url: Optional[str] = None,
        enabled: bool = True,
        batch_size: int = 20,
        flush_interval_seconds: float = 5.0,
        content_capture: bool = False,
    ):
        self._key = fluxcompute_key
        self._url = events_url or os.environ.get("FLUX_GRAPH_EVENTS_URL", GRAPH_EVENTS_URL)
        self._enabled = enabled and fluxcompute_key is not None
        self._content_capture = content_capture
        self._batch_size = batch_size
        self._flush_interval = flush_interval_seconds
        self._buffer: list[dict] = []
        self._client: Optional[httpx.AsyncClient] = None
        self._flush_task: Optional[asyncio.Task] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    def whoami_url(self) -> str:
        """Derive /v1/whoami from the same host as /v1/graph/events, so
        FLUX_GRAPH_EVENTS_URL (the existing local-dev override) governs both."""
        base = self._url.rsplit("/v1/", 1)[0]
        return f"{base}/v1/whoami"

    def record(self, event: dict) -> None:
        """Buffer one node event. Non-blocking; flushes at batch_size or periodically.

        Content is filtered here, at the wire boundary, rather than upstream:
        the in-process graph keeps full output so resume context and refusal
        detection still work locally when content_capture is off.
        """
        if not self._enabled:
            return
        self._buffer.append(event if self._content_capture else self._strip_content(event))
        self._ensure_flush_loop()
        if len(self._buffer) >= self._batch_size:
            asyncio.ensure_future(self.flush())

    def _ensure_flush_loop(self) -> None:
        """Start the periodic background flush so events appear live in the
        dashboard even when a task never crosses batch_size (the common case)."""
        if self._flush_task is not None and not self._flush_task.done():
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return  # no running event loop; batch-size flush still works
        # Check for the loop BEFORE building the coroutine: constructing it and
        # then failing to schedule leaves it un-awaited, which prints a
        # RuntimeWarning from the SDK's own internals on the user's stderr.
        self._flush_task = asyncio.ensure_future(self._flush_loop())

    async def _flush_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._flush_interval)
                await self.flush()
        except asyncio.CancelledError:
            pass

    async def flush(self) -> None:
        if not self._buffer or not self._enabled:
            return
        events = self._buffer.copy()
        self._buffer.clear()
        sent = 0
        try:
            client = await self._get_client()
            # Chunk to the server's batch cap — one oversize POST would be
            # rejected whole (422) and silently dropped.
            for i in range(0, len(events), _MAX_EVENTS_PER_POST):
                chunk = events[i:i + _MAX_EVENTS_PER_POST]
                response = await client.post(
                    self._url,
                    json={"events": chunk},
                    headers={
                        "Authorization": f"Bearer {self._key}",
                        "Content-Type": "application/json",
                    },
                )
                if response.status_code >= 500:
                    # Transient server-side failure (e.g. the backend's DB
                    # pool isn't up) -- re-buffer this chunk and everything
                    # after it instead of treating a non-200 as delivered.
                    # A prior version logged this and moved on regardless,
                    # so a real server-side outage silently lost data on
                    # both ends at once.
                    logger.warning("Graph event flush failed: %s (will retry)", response.status_code)
                    self._rebuffer(events[i:])
                    return
                if response.status_code != 200:
                    # 4xx: this exact payload will fail identically forever
                    # (oversize batch, malformed event) -- not retryable.
                    logger.warning("Graph event flush failed: %s", response.status_code)
                sent = i + len(chunk)
        except asyncio.CancelledError:
            self._rebuffer(events[sent:])
            raise
        except Exception as exc:
            logger.debug("Graph event flush error (non-fatal): %s", exc)
            self._rebuffer(events[sent:])

    @staticmethod
    def _strip_content(event: dict) -> dict:
        """Drop model-produced text, keeping the node's structure and metrics."""
        return {k: v for k, v in event.items() if k not in _CONTENT_FIELDS}

    def _rebuffer(self, events: list) -> None:
        """Re-buffer unsent events after a failure, stripped of full payloads so
        a failing network can't pin megabytes of snapshots in memory.

        The count bound is computed against the room left rather than checked
        up front, because flush() clears the buffer before calling this.
        """
        cap = self._batch_size * 2
        room = cap - len(self._buffer)
        if room <= 0:
            return
        if len(events) > room:
            logger.debug("Graph buffer full; dropping %d events", len(events) - room)
        self._buffer.extend(
            {k: v for k, v in e.items() if k not in _SNAPSHOT_FIELDS}
            for e in events[:room]
        )

    async def close(self) -> None:
        if self._flush_task is not None:
            self._flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._flush_task
        await self.flush()
        if self._client:
            await self._client.aclose()
            self._client = None
