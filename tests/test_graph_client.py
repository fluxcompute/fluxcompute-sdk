"""Integration tests: FluxClient auto-records LLM calls into the task graph."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fluxcompute import FluxClient
from fluxcompute.graph.types import TaskGraph


def _fake_result(text="four"):
    class Block:
        def __init__(self, t):
            self.text = t

    class Response:
        content = [Block(text)]
        model = "claude-haiku-4-5"

    return {
        "response": Response(),
        "response_ms": 12.0,
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_write_tokens": 0,
        "cache_read_tokens": 0,
    }


async def test_llm_call_inside_task_scope_is_recorded():
    client = FluxClient(anthropic_key="sk-ant-test", telemetry=False)
    with patch("fluxcompute.client.dispatch_anthropic",
               new_callable=AsyncMock, return_value=_fake_result()):
        with client.task("research") as t:
            await client.messages.create(
                model="auto",
                messages=[{"role": "user", "content": "What is 2+2?"}],
            )
    graph = client.get_task_graph(t.task_id)
    nodes = graph.in_order()
    assert nodes[0].node_type == "task"
    llm = [n for n in nodes if n.node_type == "llm_call"]
    assert len(llm) == 1
    assert llm[0].parent_id == nodes[0].node_id
    assert llm[0].status == "succeeded"
    assert llm[0].output_preview == "four"
    assert llm[0].input_tokens == 10
    assert llm[0].cost_usd > 0


async def test_llm_call_outside_task_scope_is_not_recorded_and_still_works():
    client = FluxClient(anthropic_key="sk-ant-test", telemetry=False)
    with patch("fluxcompute.client.dispatch_anthropic",
               new_callable=AsyncMock, return_value=_fake_result()):
        response = await client.messages.create(
            model="auto", messages=[{"role": "user", "content": "hi"}],
        )
    assert response.text == "four"


async def test_failed_dispatch_records_failed_llm_node():
    client = FluxClient(anthropic_key="sk-ant-test", telemetry=False)
    boom = RuntimeError("prompt is too long: maximum context length is 200000")
    with patch("fluxcompute.client.dispatch_anthropic",
               new_callable=AsyncMock, side_effect=boom):
        with pytest.raises(RuntimeError):
            with client.task("research") as t:
                await client.messages.create(
                    model="auto", messages=[{"role": "user", "content": "hi"}],
                )
    graph = client.get_task_graph(t.task_id)
    llm = [n for n in graph.in_order() if n.node_type == "llm_call"]
    assert llm[0].status == "failed"
    assert llm[0].failure_reason == "context_overflow"
    assert graph.root().status == "failed"


async def test_failed_dispatch_preserves_original_exception_even_if_graph_recording_breaks():
    """Regression: record_llm_call must never mask the real provider exception,
    even if graph bookkeeping itself raises internally (e.g. a future bug in
    TaskGraph.add_node/classify_failure)."""
    client = FluxClient(anthropic_key="sk-ant-test", telemetry=False)
    boom = RuntimeError("original provider failure — rate limited")

    original_add_node = TaskGraph.add_node

    def flaky_add_node(self, *, name, node_type, **kwargs):
        if node_type == "llm_call":
            raise RuntimeError("bookkeeping boom")
        return original_add_node(self, name=name, node_type=node_type, **kwargs)

    with patch("fluxcompute.client.dispatch_anthropic",
               new_callable=AsyncMock, side_effect=boom):
        with patch.object(TaskGraph, "add_node", new=flaky_add_node):
            with pytest.raises(RuntimeError, match="original provider failure"):
                with client.task("research") as t:
                    await client.messages.create(
                        model="auto", messages=[{"role": "user", "content": "hi"}],
                    )
    # The root task node still exists (its add_node call isn't the llm_call one)
    # but no llm_call node was recorded, since bookkeeping for it failed internally
    # and was swallowed by record_llm_call's guard.
    graph = client.get_task_graph(t.task_id)
    assert all(n.node_type != "llm_call" for n in graph.in_order())


async def test_streaming_llm_call_inside_task_scope_is_recorded():
    """_stream_execute must also record an llm_call node when inside a task() scope."""
    async def fake_text_stream():
        for word in ["four", "", ""]:
            if word:
                yield word

    fake_final = MagicMock()
    fake_final.usage.input_tokens = 10
    fake_final.usage.output_tokens = 3
    fake_final.usage.cache_creation_input_tokens = 0
    fake_final.usage.cache_read_input_tokens = 0
    fake_final.content = [MagicMock(text="four")]

    fake_stream_ctx = MagicMock()
    fake_stream_ctx.__aenter__ = AsyncMock(return_value=fake_stream_ctx)
    fake_stream_ctx.__aexit__ = AsyncMock(return_value=False)
    fake_stream_ctx.text_stream = fake_text_stream()
    fake_stream_ctx.get_final_message = AsyncMock(return_value=fake_final)

    with patch("fluxcompute.client.anthropic") as mock_anth:
        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=fake_stream_ctx)
        mock_anth.AsyncAnthropic.return_value = mock_client

        client = FluxClient(anthropic_key="sk-ant-test", telemetry=False)
        with client.task("research") as t:
            async with client.messages.stream(
                model="auto",
                messages=[{"role": "user", "content": "What is 2+2?"}],
            ) as s:
                async for _ in s:
                    pass

    graph = client.get_task_graph(t.task_id)
    nodes = graph.in_order()
    assert nodes[0].node_type == "task"
    llm = [n for n in nodes if n.node_type == "llm_call"]
    assert len(llm) == 1
    assert llm[0].parent_id == nodes[0].node_id
    assert llm[0].status == "succeeded"
    assert llm[0].model == s.fluxcompute.model_selected
    assert llm[0].input_tokens == 10
    assert llm[0].output_tokens == 3
    assert llm[0].cost_usd > 0
    assert llm[0].output_preview == "four"


async def test_client_close_flushes_partial_graph_buffer_end_to_end():
    """FluxClient.close() must flush a graph-event buffer that never crossed
    batch_size — the property this task exists to guarantee end-to-end, not
    just via GraphEmitter's own unit tests."""

    class _StubHTTP:
        def __init__(self):
            self.calls = []

        async def post(self, url, json=None, headers=None):
            self.calls.append({"url": url, "json": json, "headers": headers})

            class R:
                status_code = 200
            return R()

        async def aclose(self):
            pass

    stub = _StubHTTP()
    # content_capture=True: this test asserts the snapshot fields reach the
    # wire, which is opt-in behaviour.
    client = FluxClient(
        anthropic_key="sk-ant-test", fluxcompute_key="flx_test", content_capture=True,
    )
    client._graph_emitter._client = stub

    with patch("fluxcompute.client.dispatch_anthropic",
               new_callable=AsyncMock, return_value=_fake_result()):
        with client.task("research") as t:
            await client.messages.create(
                model="auto",
                messages=[{"role": "user", "content": "What is 2+2?"}],
            )

    assert stub.calls == []  # one llm_call event, well under batch_size=20

    await client.close()

    assert len(stub.calls) == 1
    events = stub.calls[0]["json"]["events"]
    llm_events = [e for e in events if e.get("node_type") == "llm_call"]
    # The llm_call appears twice for the SAME node: the live emit, then the
    # task-exit replay-ring re-emit carrying full payloads. The server
    # dedupes by node_id.
    assert {e["node_id"] for e in llm_events} == {llm_events[0]["node_id"]}
    assert any(e.get("output_full") for e in llm_events)
    assert llm_events[0]["task_id"] == t.task_id
    assert stub.calls[0]["headers"]["Authorization"] == "Bearer flx_test"


async def test_default_client_sends_no_response_content_end_to_end():
    """The privacy default, asserted at the public API boundary: a client
    built the normal way ships graph structure but never model output."""

    class _StubHTTP:
        def __init__(self):
            self.calls = []

        async def post(self, url, json=None, headers=None):
            self.calls.append({"url": url, "json": json, "headers": headers})

            class R:
                status_code = 200
            return R()

        async def aclose(self):
            pass

    stub = _StubHTTP()
    client = FluxClient(anthropic_key="sk-ant-test", fluxcompute_key="flx_test")
    client._graph_emitter._client = stub

    with patch("fluxcompute.client.dispatch_anthropic",
               new_callable=AsyncMock, return_value=_fake_result()):
        with client.task("research") as t:
            await client.messages.create(
                model="auto",
                messages=[{"role": "user", "content": "What is 2+2?"}],
            )
    await client.close()

    events = stub.calls[0]["json"]["events"]
    # "four" is the model's output in _fake_result()
    assert "four" not in str(events)
    for e in events:
        assert "output_preview" not in e
        assert "prompt_full" not in e
        assert "output_full" not in e
    # ...but the graph itself still arrived, and locally retains the output
    assert {e["task_id"] for e in events} == {t.task_id}
    assert any(e.get("node_type") == "llm_call" for e in events)
    llm = [n for n in client.get_task_graph(t.task_id).in_order()
           if n.node_type == "llm_call"]
    assert llm[0].output_preview == "four"


async def test_client_step_scope_records_tool_node():
    client = FluxClient(anthropic_key="sk-ant-test", telemetry=False)
    with client.task("t") as t:
        with client.step("fetch-data") as s:
            s.set_output("rows: 12")
    graph = client.get_task_graph(t.task_id)
    step = graph.in_order()[1]
    assert step.name == "fetch-data"
    assert step.output_preview == "rows: 12"
