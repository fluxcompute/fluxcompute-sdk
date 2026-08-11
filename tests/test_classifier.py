"""Tests for the heuristic classifier."""

import pytest
from fluxcompute.classifier.heuristic import classify


class TestEasyQueries:
    def test_simple_factual(self):
        msgs = [{"role": "user", "content": "What is the capital of France?"}]
        result = classify(msgs)
        assert result.label == "easy"

    def test_definition(self):
        msgs = [{"role": "user", "content": "Define photosynthesis"}]
        result = classify(msgs)
        assert result.label == "easy"

    def test_translation(self):
        msgs = [{"role": "user", "content": "Translate hello to Spanish"}]
        result = classify(msgs)
        assert result.label == "easy"

    def test_simple_list(self):
        msgs = [{"role": "user", "content": "List the planets in our solar system"}]
        result = classify(msgs)
        assert result.label == "easy"

    def test_yes_no(self):
        msgs = [{"role": "user", "content": "Is Python a programming language? Yes or no"}]
        result = classify(msgs)
        assert result.label == "easy"

    def test_short_greeting(self):
        msgs = [{"role": "user", "content": "Hello"}]
        result = classify(msgs)
        assert result.label == "easy"

    def test_who_is(self):
        msgs = [{"role": "user", "content": "Who is Albert Einstein?"}]
        result = classify(msgs)
        assert result.label == "easy"


class TestMediumQueries:
    def test_explanation(self):
        msgs = [{"role": "user", "content": "Explain how neural networks learn from data"}]
        result = classify(msgs)
        assert result.label in ["medium", "hard"]

    def test_comparison(self):
        msgs = [{"role": "user", "content": "Compare Python and JavaScript for web development"}]
        result = classify(msgs)
        assert result.label in ["medium", "hard"]

    def test_light_coding(self):
        msgs = [{"role": "user", "content": "Write code for a basic sorting algorithm"}]
        result = classify(msgs)
        assert result.label in ["medium", "hard"]

    def test_light_math(self):
        msgs = [{"role": "user", "content": "Calculate the area of a circle with radius 5"}]
        result = classify(msgs)
        assert result.label in ["medium", "hard"]

    def test_analysis(self):
        msgs = [{"role": "user", "content": "Analyze the implications of remote work on productivity"}]
        result = classify(msgs)
        assert result.label in ["medium", "hard"]


class TestHardQueries:
    def test_complex_reasoning(self):
        msgs = [{"role": "user", "content":
            "Analyze the trade-offs between microservices and monolithic architecture. "
            "Consider scalability, team structure, deployment complexity, and cost implications. "
            "Explain why a startup might choose one approach over the other and evaluate "
            "the long-term consequences of each decision."}]
        result = classify(msgs)
        assert result.label in ["medium", "hard"]
        assert result.score >= 0.3  # definitely not easy

    def test_heavy_math(self):
        msgs = [{"role": "user", "content":
            "Solve this optimization problem: minimize the function f(x,y) = x^2 + y^2 "
            "subject to the constraint x + y = 10. Calculate the derivative, find critical "
            "points, and prove this is a minimum using the second derivative test."}]
        result = classify(msgs)
        assert result.label in ["medium", "hard"]
        assert result.score >= 0.3

    def test_complex_code(self):
        msgs = [{"role": "user", "content":
            "Implement a distributed consensus algorithm similar to Raft. Write code for "
            "leader election, log replication, and safety guarantees. Debug any edge cases "
            "in network partitioning scenarios."}]
        result = classify(msgs)
        assert result.label in ["medium", "hard"]
        assert result.score >= 0.3

    def test_deep_multi_turn(self):
        msgs = [{"role": "user", "content": "Let's discuss economics"}]
        # Simulate 12 turns
        for i in range(11):
            msgs.append({"role": "assistant", "content": f"Response {i}"})
            msgs.append({"role": "user", "content": f"Follow up question {i}"})
        msgs.append({"role": "user", "content":
            "Now analyze the trade-offs and explain the implications of this economic policy"})
        result = classify(msgs)
        assert result.label == "hard"


class TestClassifierPerformance:
    def test_classification_under_15ms(self):
        msgs = [{"role": "user", "content": "What is 2+2?"}]
        result = classify(msgs)
        assert result.classification_ms < 15

    def test_long_query_under_15ms(self):
        msgs = [{"role": "user", "content": "word " * 500}]
        result = classify(msgs)
        assert result.classification_ms < 15

    def test_deep_conversation_under_15ms(self):
        msgs = []
        for i in range(20):
            msgs.append({"role": "user", "content": f"Message {i}"})
            msgs.append({"role": "assistant", "content": f"Response {i}"})
        result = classify(msgs)
        assert result.classification_ms < 15


class TestClassifierEdgeCases:
    def test_empty_messages(self):
        result = classify([])
        assert result.label == "medium"  # default

    def test_no_user_message(self):
        msgs = [{"role": "system", "content": "You are a helpful assistant"}]
        result = classify(msgs)
        assert result.label == "medium"

    def test_score_clamped_to_0_1(self):
        # Many simple signals should clamp to 0
        msgs = [{"role": "user", "content": "what is who is where is define list name"}]
        result = classify(msgs)
        assert 0.0 <= result.score <= 1.0

    def test_returns_correct_model_for_provider(self):
        msgs = [{"role": "user", "content": "What is 2+2?"}]
        anthropic_result = classify(msgs, provider="anthropic")
        openai_result = classify(msgs, provider="openai")
        assert "claude" in anthropic_result.model
        assert "gpt" in openai_result.model or "o1" in openai_result.model

    def test_reasoning_field_populated(self):
        msgs = [{"role": "user", "content": "Explain quantum computing"}]
        result = classify(msgs)
        assert len(result.reasoning) > 0

    def test_system_prompt_complexity(self):
        long_system = "You are an expert. " * 100  # ~500 words
        msgs = [
            {"role": "system", "content": long_system},
            {"role": "user", "content": "Hello"},
        ]
        result = classify(msgs)
        assert result.score > 0  # system prompt should add signal


class TestContentBlockMessages:
    """The Anthropic API accepts `content` as a string OR a list of blocks
    (images, tool results, cache-control). The SDK bills itself as a drop-in
    replacement, so anything the provider accepts must not crash the
    classifier — this crashed with AttributeError before the fix.
    """

    def test_text_content_blocks_are_read(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "What is 2+2?"},
        ]}]
        result = classify(msgs)
        # scored on the extracted text, same as the plain-string form
        assert result.label == classify(
            [{"role": "user", "content": "What is 2+2?"}]
        ).label

    def test_non_text_blocks_do_not_crash(self):
        msgs = [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "data": "iVBOR..."}},
            {"type": "text", "text": "What is in this image?"},
        ]}]
        result = classify(msgs)
        assert result.model  # classified rather than raised

    def test_tool_result_blocks_do_not_crash(self):
        msgs = [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "42"},
        ]}]
        result = classify(msgs)
        assert result.model

    def test_none_content_does_not_crash(self):
        result = classify([{"role": "user", "content": None}])
        assert result.model

    def test_missing_content_key_does_not_crash(self):
        result = classify([{"role": "user"}])
        assert result.model

    def test_multiple_text_blocks_are_joined(self):
        """A long prompt split across blocks must score like the joined text,
        or word-count-based signals silently under-read."""
        joined = "Analyze the trade-offs and justify the architecture " * 6
        blocks = [{"type": "text", "text": joined[:len(joined) // 2]},
                  {"type": "text", "text": joined[len(joined) // 2:]}]
        block_result = classify([{"role": "user", "content": blocks}])
        plain_result = classify([{"role": "user", "content": joined}])
        assert block_result.label == plain_result.label
