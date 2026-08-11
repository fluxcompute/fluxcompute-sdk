"""Tests for contextvar-based graph recording."""

import pytest

from fluxcompute.graph.recorder import GraphRecorder


class TestTaskScope:
    def test_task_creates_graph_with_root_node(self):
        rec = GraphRecorder()
        with rec.task("research") as t:
            pass
        graph = rec.get_graph(t.task_id)
        assert graph.name == "research"
        root = graph.root()
        assert root.node_type == "task"
        assert root.status == "succeeded"

    def test_step_auto_parents_to_enclosing_scope(self):
        rec = GraphRecorder()
        with rec.task("research") as t:
            with rec.step("fetch") as s:
                s.set_output("data: 42")
        graph = rec.get_graph(t.task_id)
        step = graph.in_order()[1]
        assert step.node_type == "tool_call"
        assert step.parent_id == graph.root().node_id
        assert step.status == "succeeded"
        assert step.output_preview == "data: 42"

    def test_nested_steps_chain_parents(self):
        rec = GraphRecorder()
        with rec.task("t") as t:
            with rec.step("outer"):
                with rec.step("inner"):
                    pass
        nodes = rec.get_graph(t.task_id).in_order()
        assert nodes[2].parent_id == nodes[1].node_id

    def test_exception_marks_node_failed_and_propagates(self):
        rec = GraphRecorder()
        with pytest.raises(TimeoutError):
            with rec.task("t") as t:
                with rec.step("crm-lookup"):
                    raise TimeoutError("CRM API timed out")
        graph = rec.get_graph(t.task_id)
        step = graph.in_order()[1]
        assert step.status == "failed"
        assert "TimeoutError" in step.error
        assert step.failure_reason == "tool_error"
        assert graph.root().status == "failed"

    def test_step_outside_task_raises(self):
        rec = GraphRecorder()
        with pytest.raises(RuntimeError):
            with rec.step("orphan"):
                pass

    def test_record_llm_call_attaches_to_current_scope(self):
        rec = GraphRecorder()
        with rec.task("t") as t:
            node = rec.record_llm_call(
                model="claude-haiku-4-5", input_tokens=10, output_tokens=5,
                cost_usd=0.001, output_preview="four", session_id="s1",
            )
        assert node.parent_id == rec.get_graph(t.task_id).root().node_id
        assert node.status == "succeeded"
        assert node.name == "llm:claude-haiku-4-5"

    def test_record_llm_call_is_noop_outside_task(self):
        rec = GraphRecorder()
        assert rec.record_llm_call(
            model="m", input_tokens=0, output_tokens=0,
            cost_usd=0.0, output_preview="", session_id=None,
        ) is None

    def test_finished_nodes_are_emitted(self):
        events = []
        rec = GraphRecorder(emit=events.append)
        with rec.task("t"):
            with rec.step("fetch"):
                pass
        # step emitted first (inner scope closes first), then root
        assert [e["node_type"] for e in events] == ["tool_call", "task"]

    def test_attach_reactivates_existing_graph(self):
        rec = GraphRecorder()
        with rec.task("t") as t:
            pass
        with rec.attach(t.task_id):
            node = rec.record_llm_call(
                model="m", input_tokens=1, output_tokens=1,
                cost_usd=0.0, output_preview="x", session_id=None,
            )
        graph = rec.get_graph(t.task_id)
        assert node.node_id in graph.nodes
        assert node.parent_id == graph.root().node_id

    def test_bookkeeping_failure_still_resets_contextvars(self):
        from fluxcompute.graph import recorder as recorder_module

        rec = GraphRecorder()
        with pytest.raises(RuntimeError, match="boom"):
            with rec.task("t") as t:
                t._graph.finish_node = lambda *a, **kw: (_ for _ in ()).throw(
                    RuntimeError("boom")
                )
        # The failing __exit__ must still have released the ambient contextvars.
        assert recorder_module._current_graph.get() is None
        assert recorder_module._current_node_id.get() is None
        # And the recorder must be usable afterward — proves state isn't wedged.
        with rec.task("t2") as t2:
            pass
        assert rec.get_graph(t2.task_id).name == "t2"

    def test_attach_raises_when_graph_already_active(self):
        rec = GraphRecorder()
        with rec.task("t") as t:
            with pytest.raises(RuntimeError):
                rec.attach(t.task_id)

    def test_task_raises_on_duplicate_task_id(self):
        rec = GraphRecorder()
        with rec.task("t", task_id="dup-id"):
            pass
        with pytest.raises(ValueError):
            rec.task("t2", task_id="dup-id")

    def test_record_llm_call_swallows_internal_bookkeeping_errors(self):
        rec = GraphRecorder()
        with rec.task("t") as t:
            t._graph.add_node = lambda *a, **kw: (_ for _ in ()).throw(
                RuntimeError("bookkeeping boom")
            )
            result = rec.record_llm_call(
                model="m", input_tokens=1, output_tokens=1,
                cost_usd=0.0, output_preview="x", session_id=None,
            )
        assert result is None


class TestSnapshotEmission:
    def test_failed_step_emits_output_full(self):
        events = []
        rec = GraphRecorder(emit=events.append)
        with pytest.raises(ValueError):
            with rec.task("t"):
                with rec.step("s") as s:
                    s.set_output("partial result")
                    raise ValueError("boom")
        failed = [e for e in events
                  if e["node_type"] == "tool_call" and e["status"] == "failed"]
        assert failed and failed[0]["output_full"] == "partial result"

    def test_task_exit_reemits_ring_with_payloads(self):
        events = []
        rec = GraphRecorder(emit=events.append)
        with rec.task("t") as t:
            with rec.step("s") as s:
                s.set_output("the answer")
        replays = [e for e in events if e.get("output_full")]
        assert replays and replays[-1]["output_full"] == "the answer"
        assert rec._recent.get(t.task_id) is None  # ring drained on task exit

    def test_idempotency_key_counts_attempts(self):
        rec = GraphRecorder()
        with rec.task("t") as t:
            with rec.step("send-email") as s1:
                k1 = s1.idempotency_key
            with rec.step("send-email") as s2:
                k2 = s2.idempotency_key
        assert k1 == f"{t.task_id}:send-email:1"
        assert k2 == f"{t.task_id}:send-email:2"
