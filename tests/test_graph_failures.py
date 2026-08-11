"""Tests for rules-based failure classification."""

from fluxcompute.graph.failures import classify_failure, detect_refusal
from fluxcompute.graph.types import TaskGraph


def _failed(g, name, node_type="tool_call", error="boom", parent=None):
    n = g.add_node(name=name, node_type=node_type, parent_id=parent)
    g.finish_node(n.node_id, "failed", error=error)
    return n


class TestClassifyFailure:
    def test_context_overflow(self):
        g = TaskGraph()
        n = _failed(g, "summarize", node_type="llm_call",
                    error="BadRequestError: prompt is too long: maximum context length is 200000")
        assert classify_failure(n, g) == "context_overflow"

    def test_budget(self):
        g = TaskGraph()
        n = _failed(g, "call", node_type="llm_call",
                    error="RateLimitError: 429 rate limit exceeded, check your quota")
        assert classify_failure(n, g) == "budget"

    def test_tool_error(self):
        g = TaskGraph()
        n = _failed(g, "crm-lookup", node_type="tool_call", error="TimeoutError: timed out")
        assert classify_failure(n, g) == "tool_error"

    def test_stall_after_three_identical_failed_siblings(self):
        g = TaskGraph()
        root = g.add_node(name="t", node_type="task")
        _failed(g, "search", parent=root.node_id, error="TimeoutError: x")
        _failed(g, "search", parent=root.node_id, error="TimeoutError: x")
        third = _failed(g, "search", parent=root.node_id, error="TimeoutError: x")
        assert classify_failure(third, g) == "stall"

    def test_two_failures_is_not_a_stall(self):
        g = TaskGraph()
        root = g.add_node(name="t", node_type="task")
        _failed(g, "search", parent=root.node_id)
        second = _failed(g, "search", parent=root.node_id)
        assert classify_failure(second, g) == "tool_error"

    def test_interleaved_success_breaks_stall_streak(self):
        g = TaskGraph()
        root = g.add_node(name="t", node_type="task")
        _failed(g, "search", parent=root.node_id, error="TimeoutError: x")
        cleanup = g.add_node(name="cleanup", node_type="tool_call", parent_id=root.node_id)
        g.finish_node(cleanup.node_id, "succeeded")
        _failed(g, "search", parent=root.node_id, error="TimeoutError: x")
        third = _failed(g, "search", parent=root.node_id, error="TimeoutError: x")
        assert classify_failure(third, g) == "tool_error"

    def test_port_number_429_is_not_budget(self):
        g = TaskGraph()
        n = _failed(g, "connect", node_type="tool_call",
                    error="TimeoutError: connection refused on localhost:429")
        assert classify_failure(n, g) == "tool_error"

    def test_unrelated_input_length_is_not_context_overflow(self):
        g = TaskGraph()
        n = _failed(g, "validate", node_type="tool_call",
                    error="ValidationError: invalid input length: expected 10, got 5")
        assert classify_failure(n, g) == "tool_error"

    def test_unknown_fallback(self):
        g = TaskGraph()
        n = _failed(g, "step", node_type="llm_call", error="SomeWeirdError: ???")
        assert classify_failure(n, g) == "unknown"

    def test_succeeded_llm_call_with_refusal_output_is_flagged(self):
        g = TaskGraph()
        n = g.add_node(name="llm:haiku", node_type="llm_call")
        g.finish_node(n.node_id, "succeeded",
                      output_preview="I can't help with that request.")
        assert classify_failure(n, g) == "refusal"

    def test_midsentence_cant_is_not_a_refusal(self):
        g = TaskGraph()
        n = g.add_node(name="llm:haiku", node_type="llm_call")
        g.finish_node(n.node_id, "succeeded",
                      output_preview="The plan is solid; I can't stress enough how important it is. The answer is 42.")
        assert classify_failure(n, g) is None

    def test_healthy_node_returns_none(self):
        g = TaskGraph()
        n = g.add_node(name="llm:haiku", node_type="llm_call")
        g.finish_node(n.node_id, "succeeded", output_preview="The answer is 4.")
        assert classify_failure(n, g) is None


class TestDetectRefusal:
    def test_long_outputs_never_count_as_refusals(self):
        assert detect_refusal("I can't do X, but here is a detailed plan: " + "x" * 400) is False

    def test_empty_output_is_not_a_refusal(self):
        assert detect_refusal("") is False
