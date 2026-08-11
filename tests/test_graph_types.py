"""Tests for execution-graph types."""

import pytest

from fluxcompute.graph.types import TaskGraph, TaskNode, node_event


class TestTaskGraph:
    def test_add_node_assigns_ids_and_order(self):
        g = TaskGraph(name="research")
        root = g.add_node(name="research", node_type="task")
        child = g.add_node(name="fetch", node_type="tool_call", parent_id=root.node_id)
        assert root.task_id == g.task_id
        assert child.parent_id == root.node_id
        assert [n.node_id for n in g.in_order()] == [root.node_id, child.node_id]
        assert g.root() is root

    def test_add_node_rejects_unknown_parent_id(self):
        g = TaskGraph()
        with pytest.raises(ValueError):
            g.add_node(name="orphan", node_type="tool_call", parent_id="bogus")

    def test_finish_node_sets_status_and_fields(self):
        g = TaskGraph()
        n = g.add_node(name="step", node_type="tool_call")
        g.finish_node(n.node_id, "failed", error="TimeoutError: boom")
        assert n.status == "failed"
        assert n.error == "TimeoutError: boom"
        assert n.ended_at is not None

    def test_finish_node_rejects_unknown_field(self):
        g = TaskGraph()
        n = g.add_node(name="step", node_type="tool_call")
        with pytest.raises(ValueError):
            g.finish_node(n.node_id, "succeeded", outputpreview="typo")

    def test_restore_node_preserves_identity_and_order(self):
        """A durability plugin rebuilds graphs from persisted rows: node_ids
        must survive (depends_on links refer to them) and insertion order is
        execution order."""
        g = TaskGraph(task_id="task_persisted", name="restored")
        a = TaskNode(node_id="aaa", task_id="task_persisted", name="root", node_type="task")
        b = TaskNode(
            node_id="bbb", task_id="task_persisted", name="step", node_type="tool_call",
            parent_id="aaa", depends_on=["aaa"], status="failed", error="boom",
        )
        g.restore_node(a)
        g.restore_node(b)
        assert [n.node_id for n in g.in_order()] == ["aaa", "bbb"]
        assert g.root() is a
        assert g.nodes["bbb"].depends_on == ["aaa"]
        assert g.failed_nodes() == [b]

    def test_restore_node_tolerates_a_missing_parent(self):
        """Partial ingest is a normal state for a persisted graph — a child
        whose parent row never arrived must still be restorable, where
        add_node would raise."""
        g = TaskGraph(task_id="task_partial")
        orphan = TaskNode(
            node_id="ccc", task_id="task_partial", name="late", node_type="llm_call",
            parent_id="never-ingested",
        )
        g.restore_node(orphan)
        assert g.nodes["ccc"].parent_id == "never-ingested"

    def test_restore_node_same_id_twice_updates_without_duplicating_order(self):
        """Persisted rows can be re-fetched; the newer row wins and the node
        appears once in execution order."""
        g = TaskGraph(task_id="task_refetch")
        first = TaskNode(node_id="ddd", task_id="task_refetch", name="n", node_type="tool_call",
                         status="running")
        second = TaskNode(node_id="ddd", task_id="task_refetch", name="n", node_type="tool_call",
                          status="succeeded", output_preview="done")
        g.restore_node(first)
        g.restore_node(second)
        assert len(g.in_order()) == 1
        assert g.nodes["ddd"].status == "succeeded"

    def test_ancestors_walks_parent_chain_root_first(self):
        g = TaskGraph()
        a = g.add_node(name="a", node_type="task")
        b = g.add_node(name="b", node_type="tool_call", parent_id=a.node_id)
        c = g.add_node(name="c", node_type="llm_call", parent_id=b.node_id)
        assert [n.node_id for n in g.ancestors(c.node_id)] == [a.node_id, b.node_id]

    def test_subtree_includes_node_and_descendants(self):
        g = TaskGraph()
        a = g.add_node(name="a", node_type="task")
        b = g.add_node(name="b", node_type="tool_call", parent_id=a.node_id)
        c = g.add_node(name="c", node_type="llm_call", parent_id=b.node_id)
        d = g.add_node(name="d", node_type="tool_call", parent_id=a.node_id)
        ids = {n.node_id for n in g.subtree(b.node_id)}
        assert ids == {b.node_id, c.node_id}
        assert d.node_id not in ids

    def test_failed_nodes_and_context_nodes(self):
        g = TaskGraph()
        root = g.add_node(name="t", node_type="task")
        ok = g.add_node(name="fetch", node_type="tool_call", parent_id=root.node_id)
        g.finish_node(ok.node_id, "succeeded", output_preview="data: 42")
        bad = g.add_node(name="analyze", node_type="llm_call", parent_id=root.node_id)
        g.finish_node(bad.node_id, "failed", error="boom")
        assert g.failed_nodes() == [g.nodes[bad.node_id]]
        # context for resuming `bad` = succeeded non-task nodes finished before it started
        ctx = g.context_nodes(bad.node_id)
        assert [n.node_id for n in ctx] == [ok.node_id]

    def test_node_event_serializes_all_fields(self):
        g = TaskGraph(task_id="task_abc", name="research")
        n = g.add_node(name="llm:haiku", node_type="llm_call", model="claude-haiku-4-5")
        g.finish_node(n.node_id, "succeeded", cost_usd=0.001, output_preview="four")
        ev = node_event(g, n)
        assert ev["task_id"] == "task_abc"
        assert ev["task_name"] == "research"
        assert ev["node_type"] == "llm_call"
        assert ev["status"] == "succeeded"
        assert ev["cost_usd"] == 0.001
        assert ev["output_preview"] == "four"
        assert ev["ended_at"] is not None

    def test_node_event_includes_attributes_and_events(self):
        g = TaskGraph(name="t")
        n = g.add_node(
            name="llm:x", node_type="llm_call",
            attributes={"ttft_ms": 12.5},
            events=[{"type": "attempt", "ok": True}],
        )
        ev = node_event(g, n)
        assert ev["attributes"] == {"ttft_ms": 12.5}
        assert ev["events"] == [{"type": "attempt", "ok": True}]
