"""Tests for the ContextBuilder."""

from fluxcompute.state.context_builder import ContextBuilder, _group_turns


class TestContextBuilderHard:
    def test_hard_sends_full_history(self):
        cb = ContextBuilder()
        history = [
            {"role": "user", "content": f"msg {i}"}
            for i in range(40)
        ]
        current = [{"role": "user", "content": "final question"}]
        result = cb.build(current, history, "hard")
        # All history + current message present
        assert len(result) >= len(history) + len(current)

    def test_hard_preserves_system_message(self):
        cb = ContextBuilder()
        system = {"role": "system", "content": "You are an expert."}
        history = [{"role": "user", "content": "hello"}]
        current = [system, {"role": "user", "content": "analyze this"}]
        result = cb.build(current, history, "hard")
        assert result[0]["role"] == "system"


class TestContextBuilderEasy:
    def test_easy_truncates_long_history(self):
        cb = ContextBuilder()
        history = []
        for i in range(30):
            history.append({"role": "user", "content": f"question {i}"})
            history.append({"role": "assistant", "content": f"answer {i}"})
        current = [{"role": "user", "content": "simple lookup"}]
        result = cb.build(current, history, "easy")
        # Should be much less than the full 60-message history
        assert len(result) < len(history)
        # Current message always present
        assert result[-1]["content"] == "simple lookup"

    def test_easy_always_includes_current(self):
        cb = ContextBuilder()
        history = [{"role": "user", "content": f"old {i}"} for i in range(20)]
        current = [{"role": "user", "content": "new question"}]
        result = cb.build(current, history, "easy")
        contents = [m["content"] for m in result]
        assert "new question" in contents

    def test_easy_system_always_included(self):
        cb = ContextBuilder()
        sys_msg = {"role": "system", "content": "You are helpful."}
        history = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        current = [sys_msg, {"role": "user", "content": "what time is it"}]
        result = cb.build(current, history, "easy")
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful."


class TestContextBuilderMedium:
    def test_medium_more_than_easy(self):
        cb = ContextBuilder()
        history = []
        for i in range(30):
            history.append({"role": "user", "content": f"q{i} " * 20})
            history.append({"role": "assistant", "content": f"a{i} " * 20})
        current = [{"role": "user", "content": "medium complexity"}]

        easy_result = cb.build(current, history, "easy")
        medium_result = cb.build(current, history, "medium")
        # Medium keeps more history than easy
        assert len(medium_result) >= len(easy_result)

    def test_medium_less_than_hard(self):
        cb = ContextBuilder()
        history = []
        for i in range(30):
            history.append({"role": "user", "content": f"q{i} " * 20})
            history.append({"role": "assistant", "content": f"a{i} " * 20})
        current = [{"role": "user", "content": "analyze this"}]

        medium_result = cb.build(current, history, "medium")
        hard_result = cb.build(current, history, "hard")
        assert len(medium_result) <= len(hard_result)


class TestContextBuilderEmpty:
    def test_empty_history(self):
        cb = ContextBuilder()
        current = [{"role": "user", "content": "hello"}]
        result = cb.build(current, [], "easy")
        assert result == current

    def test_empty_current(self):
        cb = ContextBuilder()
        history = [{"role": "user", "content": "old"}]
        result = cb.build([], history, "easy")
        # No current messages — just returns whatever fits from history
        assert isinstance(result, list)

    def test_both_empty(self):
        cb = ContextBuilder()
        result = cb.build([], [], "easy")
        assert result == []


class TestSavingsEstimate:
    def test_no_compression_returns_zero(self):
        cb = ContextBuilder()
        msgs = [{"role": "user", "content": "hello"}]
        assert cb.savings_estimate(msgs, msgs) == 0.0

    def test_compression_returns_positive(self):
        cb = ContextBuilder()
        original = [{"role": "user", "content": "hello " * 100}]
        compressed = [{"role": "user", "content": "hello"}]
        saving = cb.savings_estimate(original, compressed)
        assert saving > 0.0
        assert saving <= 1.0


class TestGroupTurns:
    def test_alternating_roles(self):
        msgs = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "q2"},
        ]
        turns = _group_turns(msgs)
        assert len(turns) == 3

    def test_empty(self):
        assert _group_turns([]) == []

    def test_single_message(self):
        msgs = [{"role": "user", "content": "hello"}]
        turns = _group_turns(msgs)
        assert turns == [msgs]
