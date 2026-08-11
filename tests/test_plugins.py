"""
Tests for the fluxcompute.plugins entry-point seam. Confirms the SDK raises
a clear error with no plugin installed, dispatches to one when registered,
and never consults a plugin when the task's graph is held locally — without
the SDK ever importing a plugin package directly.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from fluxcompute import FluxClient
from fluxcompute.plugins import FluxRecoveryNotInstalled, get_recovery_plugin


# ---------------------------------------------------------------------------
# Entry-point discovery (unit-level, unchanged mechanism)
# ---------------------------------------------------------------------------

def test_raises_when_no_plugin_registered(monkeypatch):
    monkeypatch.setattr("fluxcompute.plugins.entry_points", lambda group: [])
    with pytest.raises(FluxRecoveryNotInstalled):
        get_recovery_plugin()


def test_error_message_names_the_install_command(monkeypatch):
    monkeypatch.setattr("fluxcompute.plugins.entry_points", lambda group: [])
    with pytest.raises(FluxRecoveryNotInstalled, match="pip install fluxcompute-recovery"):
        get_recovery_plugin()


def test_dispatches_to_registered_recovery_plugin(monkeypatch):
    class _FakePlugin:
        async def fetch_graph(self, task_id):
            return ("fake-graph", task_id)

    fake_plugin = _FakePlugin()

    class _FakeEntryPoint:
        name = "recovery"

        def load(self):
            return fake_plugin

    monkeypatch.setattr(
        "fluxcompute.plugins.entry_points", lambda group: [_FakeEntryPoint()]
    )

    plugin = get_recovery_plugin()
    assert plugin is fake_plugin


def test_ignores_entry_points_with_a_different_name(monkeypatch):
    class _OtherEntryPoint:
        name = "not-recovery"

        def load(self):
            raise AssertionError("should never be loaded")

    monkeypatch.setattr(
        "fluxcompute.plugins.entry_points", lambda group: [_OtherEntryPoint()]
    )
    with pytest.raises(FluxRecoveryNotInstalled):
        get_recovery_plugin()


# ---------------------------------------------------------------------------
# End-to-end through FluxClient.resume() — the seam that actually matters
# ---------------------------------------------------------------------------

def _register_fake_plugin(monkeypatch, plugin):
    class _FakeEntryPoint:
        name = "recovery"

        def load(self):
            return plugin

    monkeypatch.setattr(
        "fluxcompute.plugins.entry_points", lambda group: [_FakeEntryPoint()]
    )


def _fake_result(text="resumed via plugin"):
    class Block:
        def __init__(self, t):
            self.text = t

    class Response:
        content = [Block(text)]
        model = "claude-sonnet-4-5"

    return {
        "response": Response(), "response_ms": 12.0, "input_tokens": 50,
        "output_tokens": 20, "cache_write_tokens": 0, "cache_read_tokens": 0,
    }


def _graph_with_a_failure(task_id="remote-task-1"):
    """Build a graph exactly as GraphRecorder would, but never attach it to
    any client — this stands in for what a plugin reconstructs from a
    durable store the current process never wrote to."""
    from fluxcompute.graph.types import TaskGraph

    g = TaskGraph(task_id=task_id, name="cross-process-task")
    root = g.add_node(name="cross-process-task", node_type="task")
    bad = g.add_node(name="fetch-data", node_type="tool_call", parent_id=root.node_id)
    g.finish_node(bad.node_id, "failed", error="TimeoutError: upstream timed out")
    return g


async def test_unknown_task_with_no_plugin_raises_durability_error():
    client = FluxClient(anthropic_key="sk-ant-test", telemetry=False)
    with pytest.raises(FluxRecoveryNotInstalled):
        await client.resume("never-existed")


async def test_plugin_returning_none_raises_key_error_not_durability_error(monkeypatch):
    """A plugin that IS installed and still can't find the task means it's
    genuinely unknown — the caller gets KeyError, not a nudge to install
    something they already have."""
    class _EmptyPlugin:
        async def fetch_graph(self, task_id):
            return None

    _register_fake_plugin(monkeypatch, _EmptyPlugin())
    client = FluxClient(anthropic_key="sk-ant-test", telemetry=False)
    with pytest.raises(KeyError):
        await client.resume("still-unknown")


async def test_resume_succeeds_via_a_fetched_graph(monkeypatch):
    """The actual durability path: this process never ran the task, a plugin
    hands back a reconstructed graph, and resume() completes exactly as it
    would for a locally-recorded one."""
    remote_graph = _graph_with_a_failure("remote-task-1")

    class _FetchingPlugin:
        async def fetch_graph(self, task_id):
            assert task_id == "remote-task-1"
            return remote_graph

    _register_fake_plugin(monkeypatch, _FetchingPlugin())
    client = FluxClient(anthropic_key="sk-ant-test", telemetry=False)

    with patch("fluxcompute.client.dispatch_anthropic",
               new_callable=AsyncMock, return_value=_fake_result()):
        response = await client.resume("remote-task-1")

    assert response.text == "resumed via plugin"
    # the fetched graph is now locally adopted — a second call must not
    # re-fetch, and get_task_graph must expose it like any local task
    adopted = client.get_task_graph("remote-task-1")
    assert adopted is remote_graph
    new_node = adopted.in_order()[-1]
    assert new_node.node_type == "llm_call"


async def test_local_graph_present_never_consults_the_plugin(monkeypatch):
    """The precedence that matters most: a plugin registered for durability
    must never be asked about a task this process already holds — local is
    always authoritative and always faster."""
    class _PluginThatMustNotBeCalled:
        async def fetch_graph(self, task_id):
            raise AssertionError("fetch_graph called despite a local graph existing")

    _register_fake_plugin(monkeypatch, _PluginThatMustNotBeCalled())
    client = FluxClient(anthropic_key="sk-ant-test", telemetry=False)

    with client.task("local-task") as t:
        try:
            with client.step("do-thing"):
                raise TimeoutError("boom")
        except TimeoutError:
            pass

    with patch("fluxcompute.client.dispatch_anthropic",
               new_callable=AsyncMock, return_value=_fake_result("local resume")):
        response = await client.resume(t.task_id)

    assert response.text == "local resume"
