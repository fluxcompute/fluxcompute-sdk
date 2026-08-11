"""Tests for the LangGraph zero-touch adapter."""

import pytest

pytest.importorskip("langgraph")

from typing import TypedDict

from langgraph.graph import END, StateGraph

from fluxcompute import FluxClient
from fluxcompute.integrations.langgraph import FluxCheckpointer


class State(TypedDict):
    value: int


def test_checkpointer_mirrors_steps_into_task_graph():
    client = FluxClient(anthropic_key="sk-ant-test", telemetry=False)
    saver = FluxCheckpointer(client, task_name="lg-test-run")

    def add_one(state: State) -> State:
        return {"value": state["value"] + 1}

    def double(state: State) -> State:
        return {"value": state["value"] * 2}

    g = StateGraph(State)
    g.add_node("add_one", add_one)
    g.add_node("double", double)
    g.set_entry_point("add_one")
    g.add_edge("add_one", "double")
    g.add_edge("double", END)
    app = g.compile(checkpointer=saver)

    result = app.invoke({"value": 1}, config={"configurable": {"thread_id": "t1"}})
    assert result["value"] == 4

    graph = client.get_task_graph("lg_t1")
    assert graph is not None
    names = " ".join(n.name for n in graph.in_order())
    assert "add_one" in names
    assert "double" in names
    # checkpoint semantics untouched — nodes all recorded as succeeded
    assert all(n.status == "succeeded" for n in graph.in_order())


def test_multi_turn_same_thread_records_each_invoke():
    """versions_seen is cumulative for a thread's lifetime — a node name,
    once seen, never disappears as a key. Regression test for a bug where
    diffing on key presence (instead of value change) recorded only the
    first of several invokes on the same thread_id."""
    client = FluxClient(anthropic_key="sk-ant-test", telemetry=False)
    saver = FluxCheckpointer(client, task_name="multi-turn")

    def chatbot(state: State) -> State:
        return {"value": state["value"] + 1}

    g = StateGraph(State)
    g.add_node("chatbot", chatbot)
    g.set_entry_point("chatbot")
    g.add_edge("chatbot", END)
    app = g.compile(checkpointer=saver)

    for turn in range(3):
        app.invoke({"value": turn}, config={"configurable": {"thread_id": "multi"}})

    graph = client.get_task_graph("lg_multi")
    assert graph is not None
    chatbot_nodes = [n for n in graph.in_order() if n.name == "chatbot"]
    assert len(chatbot_nodes) == 3
    assert all(n.status == "succeeded" for n in chatbot_nodes)


def test_cyclic_graph_records_each_loop_iteration():
    """A node that loops back to itself via a conditional edge must be
    recorded once per actual execution, not once total — same underlying
    versions_seen cumulative-keys bug as the multi-turn case, but within a
    single invoke() call instead of across several."""

    class LoopState(TypedDict):
        count: int

    client = FluxClient(anthropic_key="sk-ant-test", telemetry=False)
    saver = FluxCheckpointer(client, task_name="loop-run")

    def bump(state: LoopState) -> LoopState:
        return {"count": state["count"] + 1}

    def should_continue(state: LoopState) -> str:
        return "loop" if state["count"] < 3 else "stop"

    g = StateGraph(LoopState)
    g.add_node("bump", bump)
    g.set_entry_point("bump")
    g.add_conditional_edges("bump", should_continue, {"loop": "bump", "stop": END})
    app = g.compile(checkpointer=saver)

    result = app.invoke({"count": 0}, config={"configurable": {"thread_id": "loop1"}})
    assert result["count"] == 3

    graph = client.get_task_graph("lg_loop1")
    assert graph is not None
    bump_nodes = [n for n in graph.in_order() if n.name == "bump"]
    assert len(bump_nodes) == 3
    assert all(n.status == "succeeded" for n in bump_nodes)
