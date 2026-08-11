"""Tests for graph-aware resume."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from fluxcompute import FluxClient
from fluxcompute.graph.resume import build_resume_plan, summarize_completed_steps
from fluxcompute.graph.types import TaskGraph
from fluxcompute.plugins import FluxRecoveryNotInstalled


class TestBuildResumePlan:
    def _graph_with_failure(self):
        g = TaskGraph(name="market-research")
        root = g.add_node(name="market-research", node_type="task")
        ok = g.add_node(name="fetch-pricing", node_type="tool_call", parent_id=root.node_id)
        g.finish_node(ok.node_id, "succeeded", output_preview="competitor A: $10/mo")
        bad = g.add_node(name="crm-lookup", node_type="tool_call", parent_id=root.node_id)
        g.finish_node(bad.node_id, "failed", error="TimeoutError: CRM timed out")
        return g, bad

    def test_plan_targets_first_failed_node_by_default(self):
        g, bad = self._graph_with_failure()
        plan = build_resume_plan(g)
        assert plan.failed_node.node_id == bad.node_id
        assert len(plan.completed_steps) == 1
        assert "fetch-pricing" in plan.context_messages[0]["content"]
        assert "competitor A: $10/mo" in plan.context_messages[0]["content"]

    def test_plan_with_no_failures_raises(self):
        g = TaskGraph()
        n = g.add_node(name="ok", node_type="tool_call")
        g.finish_node(n.node_id, "succeeded")
        with pytest.raises(ValueError):
            build_resume_plan(g)

    def test_summary_mentions_task_and_all_steps(self):
        text = summarize_completed_steps(
            [("fetch", "rows: 12"), ("clean", "12 valid")], "etl-run",
        )
        assert "etl-run" in text
        assert "fetch: rows: 12" in text
        assert "clean: 12 valid" in text


def _fake_result(text="resumed answer"):
    class Block:
        def __init__(self, t):
            self.text = t

    class Response:
        content = [Block(text)]
        model = "claude-sonnet-4-5"

    return {
        "response": Response(),
        "response_ms": 12.0,
        "input_tokens": 50,
        "output_tokens": 20,
        "cache_write_tokens": 0,
        "cache_read_tokens": 0,
    }


async def test_client_resume_reruns_only_failed_step():
    client = FluxClient(anthropic_key="sk-ant-test", telemetry=False)
    with client.task("market-research") as t:
        with client.step("fetch-pricing") as s:
            s.set_output("competitor A: $10/mo")
        try:
            with client.step("crm-lookup"):
                raise TimeoutError("CRM timed out")
        except TimeoutError:
            pass

    with patch("fluxcompute.client.dispatch_anthropic",
               new_callable=AsyncMock, return_value=_fake_result()) as disp:
        response = await client.resume(t.task_id)

    assert response.text == "resumed answer"
    # the reconstructed context (not a full transcript) was sent
    # (anthropic messages are content-block lists post-CacheManager, not plain strings)
    sent = disp.call_args.kwargs["messages"]

    def _text(content):
        if isinstance(content, str):
            return content
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict))

    joined = " ".join(_text(m["content"]) for m in sent)
    assert "fetch-pricing" in joined
    assert "crm-lookup" in joined
    # the new llm node landed in the SAME graph, linked to the failed node
    graph = client.get_task_graph(t.task_id)
    new = graph.in_order()[-1]
    failed = [n for n in graph.in_order() if n.status == "failed" and n.node_type == "tool_call"]
    assert new.node_type == "llm_call"
    assert failed[0].node_id in new.depends_on


async def test_resume_unknown_task_raises_recovery_not_installed():
    """With no recovery plugin installed, an unknown task_id can't be
    distinguished from one that exists only in another process —
    FluxRecoveryNotInstalled says exactly that, rather than a bare KeyError
    that dead-ends the caller. A plugin that IS installed and still can't
    find the task raises KeyError instead."""
    client = FluxClient(anthropic_key="sk-ant-test", telemetry=False)
    with pytest.raises(FluxRecoveryNotInstalled):
        await client.resume("task_nope")


async def test_resume_that_itself_fails_still_links_depends_on():
    """Linking must happen even when the resumed attempt also fails, or the
    retry chain silently disconnects."""
    client = FluxClient(anthropic_key="sk-ant-test", telemetry=False)
    with client.task("market-research") as t:
        try:
            with client.step("crm-lookup"):
                raise TimeoutError("CRM timed out")
        except TimeoutError:
            pass

    with patch("fluxcompute.client.dispatch_anthropic",
               new_callable=AsyncMock, side_effect=RuntimeError("still broken")):
        with pytest.raises(RuntimeError):
            await client.resume(t.task_id)

    graph = client.get_task_graph(t.task_id)
    nodes = graph.in_order()
    new_node = nodes[-1]
    original_failed = [n for n in nodes if n.node_type == "tool_call" and n.status == "failed"][0]
    assert new_node.node_type == "llm_call"
    assert new_node.status == "failed"
    assert original_failed.node_id in new_node.depends_on


async def test_repeated_resume_targets_newest_unresolved_failure():
    """A second resume() must target the failure from the FIRST resume
    attempt, not the original stale failure."""
    client = FluxClient(anthropic_key="sk-ant-test", telemetry=False)
    with client.task("market-research") as t:
        try:
            with client.step("crm-lookup"):
                raise TimeoutError("CRM timed out")
        except TimeoutError:
            pass

    # First resume: the resumed attempt ALSO fails.
    with patch("fluxcompute.client.dispatch_anthropic",
               new_callable=AsyncMock, side_effect=RuntimeError("still broken")):
        with pytest.raises(RuntimeError):
            await client.resume(t.task_id)

    graph = client.get_task_graph(t.task_id)
    nodes_after_first = graph.in_order()
    original_failed = [
        n for n in nodes_after_first if n.node_type == "tool_call" and n.status == "failed"
    ][0]
    first_resume_failed = nodes_after_first[-1]
    assert first_resume_failed.status == "failed"

    # Second resume (no explicit node_id): must target the newest unresolved
    # failure — the failed resume attempt — not the original stale failure.
    with patch("fluxcompute.client.dispatch_anthropic",
               new_callable=AsyncMock, return_value=_fake_result("finally works")):
        response = await client.resume(t.task_id)

    assert response.text == "finally works"
    second_resume_node = graph.in_order()[-1]
    assert second_resume_node.node_type == "llm_call"
    assert first_resume_failed.node_id in second_resume_node.depends_on
    assert original_failed.node_id not in second_resume_node.depends_on


async def test_resume_of_llm_call_failure_does_not_leak_session_history():
    """resume() must not reintroduce full-transcript replay via a shared
    session_id — only the reconstructed minimal context and the retry prompt
    should reach the model."""
    client = FluxClient(anthropic_key="sk-ant-test", telemetry=False)
    session_id = "sess-leak-test"

    with patch("fluxcompute.client.dispatch_anthropic",
               new_callable=AsyncMock, return_value=_fake_result("first turn ok")):
        await client.messages.create(
            model="auto",
            messages=[{"role": "user", "content": "remember the codeword: PINEAPPLE"}],
            session_id=session_id,
        )

    with client.task("multi-turn-task") as t:
        try:
            with patch("fluxcompute.client.dispatch_anthropic",
                       new_callable=AsyncMock, side_effect=TimeoutError("downstream timeout")):
                await client.messages.create(
                    model="auto",
                    messages=[{"role": "user", "content": "what's next"}],
                    session_id=session_id,
                )
        except TimeoutError:
            pass

    with patch("fluxcompute.client.dispatch_anthropic",
               new_callable=AsyncMock, return_value=_fake_result("resumed")) as disp:
        await client.resume(t.task_id)

    sent = disp.call_args.kwargs["messages"]

    def _text(content):
        if isinstance(content, str):
            return content
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict))

    joined = " ".join(_text(m["content"]) for m in sent)
    assert "PINEAPPLE" not in joined


async def test_resume_cancelled_before_new_node_added_does_not_self_loop():
    """A BaseException like asyncio.CancelledError (or any pre-dispatch
    failure) never reaches record_llm_call, so no new node is added to the
    graph. resume() must not fall back to linking the failed node's own id
    into its own depends_on — that would be a self-loop that permanently
    (and incorrectly) marks the failure as "resolved"."""
    client = FluxClient(anthropic_key="sk-ant-test", telemetry=False)
    with client.task("market-research") as t:
        try:
            with client.step("crm-lookup"):
                raise TimeoutError("CRM timed out")
        except TimeoutError:
            pass

    graph = client.get_task_graph(t.task_id)
    nodes_before = len(graph.in_order())
    original_failed = [
        n for n in graph.in_order() if n.node_type == "tool_call" and n.status == "failed"
    ][0]

    with patch("fluxcompute.client.dispatch_anthropic",
               new_callable=AsyncMock, side_effect=asyncio.CancelledError()):
        with pytest.raises(asyncio.CancelledError):
            await client.resume(t.task_id)

    assert len(graph.in_order()) == nodes_before
    assert original_failed.depends_on == []
