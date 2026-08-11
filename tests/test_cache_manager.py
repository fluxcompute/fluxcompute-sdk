"""Tests for CacheManager — context persistence layer."""

from fluxcompute.state.cache_manager import CacheManager, _to_content_block


class TestCacheManagerAnthropicPrepare:
    def _cm(self):
        return CacheManager()

    def test_system_message_gets_cache_control(self):
        cm = self._cm()
        msgs = [
            {"role": "system", "content": "You are an expert agent."},
            {"role": "user", "content": "Hello"},
        ]
        result_msgs, system_blocks = cm.prepare_for_anthropic(msgs, [])
        assert system_blocks is not None
        assert system_blocks[0]["cache_control"] == {"type": "ephemeral"}
        assert system_blocks[0]["text"] == "You are an expert agent."

    def test_system_extracted_from_messages(self):
        cm = self._cm()
        msgs = [{"role": "user", "content": "current question"}]
        history = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
        ]
        _, system_blocks = cm.prepare_for_anthropic(msgs, history)
        assert system_blocks is not None
        assert "helpful assistant" in system_blocks[0]["text"]

    def test_current_message_no_cache_control(self):
        cm = self._cm()
        msgs = [{"role": "user", "content": "brand new question"}]
        history = [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
        ]
        result_msgs, _ = cm.prepare_for_anthropic(msgs, history)
        current = result_msgs[-1]
        content = current["content"]
        if isinstance(content, list):
            for block in content:
                assert "cache_control" not in block
        # (plain string content also has no cache_control)

    def test_returns_content_block_format(self):
        cm = self._cm()
        msgs = [{"role": "user", "content": "hello"}]
        result_msgs, _ = cm.prepare_for_anthropic(msgs, [])
        assert isinstance(result_msgs, list)
        for msg in result_msgs:
            # role must survive
            assert "role" in msg

    def test_history_boundary_gets_cache_marker(self):
        cm = self._cm()
        history = [
            {"role": "user", "content": "step 1 question " * 20},
            {"role": "assistant", "content": "step 1 answer " * 20},
        ]
        current = [{"role": "user", "content": "step 2 question"}]
        # In production, ContextBuilder merges history + current before calling
        # prepare_for_anthropic, so we pass the merged list as first arg.
        merged = history + current
        result_msgs, _ = cm.prepare_for_anthropic(merged, history)
        # The second-to-last message (last history turn) should have cache_control.
        # Last message (current query) must NOT have it.
        history_msgs = result_msgs[:-1]
        found_cache = False
        for msg in history_msgs:
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if block.get("cache_control"):
                        found_cache = True
        assert found_cache, "Expected a cache_control marker on session history boundary"

    def test_empty_messages(self):
        cm = self._cm()
        result, system = cm.prepare_for_anthropic([], [])
        assert result == []
        assert system is None

    def test_no_system_message(self):
        cm = self._cm()
        msgs = [{"role": "user", "content": "hello"}]
        _, system_blocks = cm.prepare_for_anthropic(msgs, [])
        assert system_blocks is None

    def test_existing_content_blocks_pass_through(self):
        cm = self._cm()
        # Already in block format
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        result, _ = cm.prepare_for_anthropic(msgs, [])
        assert result[0]["content"][0]["text"] == "hello"


class TestToContentBlock:
    def test_string_content_becomes_block(self):
        result = _to_content_block({"role": "user", "content": "hello"})
        assert isinstance(result["content"], list)
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "hello"

    def test_cache_flag_adds_cache_control(self):
        result = _to_content_block({"role": "user", "content": "hello"}, cache=True)
        assert result["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_no_cache_flag_no_cache_control(self):
        result = _to_content_block({"role": "user", "content": "hello"}, cache=False)
        assert "cache_control" not in result["content"][0]

    def test_list_content_passes_through(self):
        blocks = [{"type": "text", "text": "hello"}]
        result = _to_content_block({"role": "user", "content": blocks})
        assert result["content"] == blocks

    def test_none_content_handled(self):
        result = _to_content_block({"role": "user", "content": None})
        assert result["content"][0]["text"] == ""


class TestCacheManagerSavingsEstimate:
    def test_no_hit_returns_zero(self):
        cm = CacheManager()
        assert cm.estimate_cache_savings(10_000, is_cache_hit=False) == 0.0

    def test_cache_hit_returns_positive(self):
        cm = CacheManager()
        saving = cm.estimate_cache_savings(1_000_000, is_cache_hit=True)
        assert saving > 0.0

    def test_zero_tokens_returns_zero(self):
        cm = CacheManager()
        assert cm.estimate_cache_savings(0, is_cache_hit=True) == 0.0
