"""Tests for session manager."""

from fluxcompute.state.session import SessionManager


class TestSessionManager:
    def test_create_session(self):
        mgr = SessionManager()
        session = mgr.get_or_create("test-1")
        assert session.id == "test-1"
        assert session.total_queries == 0
        assert session.conversation_history == []

    def test_get_existing_session(self):
        mgr = SessionManager()
        s1 = mgr.get_or_create("test-1")
        s2 = mgr.get_or_create("test-1")
        assert s1 is s2  # same object

    def test_update_session(self):
        mgr = SessionManager()
        mgr.get_or_create("test-1")
        mgr.update(
            session_id="test-1",
            user_message={"role": "user", "content": "Hello"},
            assistant_message={"role": "assistant", "content": "Hi there"},
            model_used="claude-3-5-haiku",
            cost_usd=0.001,
            savings_usd=0.005,
        )
        session = mgr.get_session("test-1")
        assert session.total_queries == 1
        assert len(session.conversation_history) == 2
        assert session.total_cost_usd == 0.001
        assert session.total_savings_usd == 0.005

    def test_multi_turn_history(self):
        mgr = SessionManager()
        mgr.get_or_create("test-1")
        for i in range(5):
            mgr.update(
                session_id="test-1",
                user_message={"role": "user", "content": f"Message {i}"},
                assistant_message={"role": "assistant", "content": f"Response {i}"},
                model_used="claude-3-5-haiku",
                cost_usd=0.001,
                savings_usd=0.005,
            )
        session = mgr.get_session("test-1")
        assert session.total_queries == 5
        assert len(session.conversation_history) == 10  # 5 user + 5 assistant

    def test_get_context(self):
        mgr = SessionManager()
        mgr.get_or_create("test-1")
        mgr.update("test-1",
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            "haiku", 0.001, 0.005)
        context = mgr.get_context("test-1")
        assert len(context) == 2
        assert context[0]["content"] == "Hello"

    def test_get_context_empty_session(self):
        mgr = SessionManager()
        context = mgr.get_context("nonexistent")
        assert context == []

    def test_delete_session(self):
        mgr = SessionManager()
        mgr.get_or_create("test-1")
        mgr.delete("test-1")
        assert mgr.get_session("test-1") is None

    def test_eviction_at_capacity(self):
        mgr = SessionManager(max_sessions=3)
        mgr.get_or_create("s1")
        mgr.get_or_create("s2")
        mgr.get_or_create("s3")
        # This should evict s1 (oldest)
        mgr.get_or_create("s4")
        assert mgr.active_sessions == 3
        assert mgr.get_session("s1") is None

    def test_routing_history_tracked(self):
        mgr = SessionManager()
        mgr.get_or_create("test-1")
        mgr.update("test-1",
            {"role": "user", "content": "Simple question"},
            {"role": "assistant", "content": "Answer"},
            "claude-3-5-haiku", 0.001, 0.005)
        mgr.update("test-1",
            {"role": "user", "content": "Complex analysis please"},
            {"role": "assistant", "content": "Deep analysis"},
            "claude-opus-4", 0.01, 0.0)
        session = mgr.get_session("test-1")
        assert len(session.routing_history) == 2
        assert session.routing_history[0]["model"] == "claude-3-5-haiku"
        assert session.routing_history[1]["model"] == "claude-opus-4"
