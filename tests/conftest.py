"""Test isolation.

Two guarantees:

1. **No test reaches a real backend.** Endpoint env vars are pointed at a
   loopback blackhole, so a test that builds a client with a
   `fluxcompute_key` cannot POST anywhere real — even on paths where send
   failures are swallowed by design.

2. **Results don't depend on the developer's environment.** `FluxClient`
   falls back to `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `FLUXCOMPUTE_KEY`,
   so those are cleared to keep local runs and CI identical.
"""

from __future__ import annotations

import pytest

# Loopback on a port nothing listens on: connections are refused immediately.
# A non-routable address (e.g. 192.0.2.1) would hang until timeout instead,
# adding ~20s to the suite.
_BLACKHOLE = "http://127.0.0.1:1/v1"

_CLEARED_ENV = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "FLUXCOMPUTE_KEY",
    "FLUX_TELEMETRY_URL",
    "FLUX_GRAPH_EVENTS_URL",
)


@pytest.fixture(autouse=True)
def isolate_from_network_and_environment(monkeypatch):
    for var in _CLEARED_ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("FLUX_TELEMETRY_URL", f"{_BLACKHOLE}/telemetry/event")
    monkeypatch.setenv("FLUX_GRAPH_EVENTS_URL", f"{_BLACKHOLE}/graph/events")


@pytest.fixture(autouse=True)
def fail_on_requests_to_the_real_backend(monkeypatch):
    """Turn an accidental production call into a test failure rather than a
    silently swallowed one."""
    import httpx

    real_send = httpx.AsyncClient.send

    async def guarded_send(self, request, *args, **kwargs):
        host = request.url.host or ""
        if "fluxcompute.dev" in host:
            raise AssertionError(
                f"test attempted a request to the real backend: {request.url}"
            )
        return await real_send(self, request, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "send", guarded_send)
