"""
ContextBuilder — context compression for model switching.

When routing between models mid-session, re-sending the full history negates
the cost saving and dropping it breaks the agent; this module compresses
between those extremes.

Per difficulty tier:
  hard   → full history
  medium → system + last 20 turns + current
  easy   → system + last 6 turns + current

Tool call pairs (call + result) are always kept together to prevent
the model from seeing dangling tool references.
"""

from __future__ import annotations

from typing import Dict, List, Optional


# Max message pairs (user+assistant) to include per tier.
# "pairs" because we always keep turns together.
_MAX_PAIRS: Dict[str, Optional[int]] = {
    "hard": None,   # no limit
    "medium": 20,
    "easy": 6,
}

# Rough chars-to-tokens estimate (conservative; actual is ~3.5 for English).
_CHARS_PER_TOKEN = 4

# Hard token cap per tier — even Opus calls shouldn't balloon on prefill.
_TOKEN_BUDGET: Dict[str, Optional[int]] = {
    "hard": None,
    "medium": 16_000,
    "easy": 3_000,
}


class ContextBuilder:
    """
    Builds the optimal message list to send to the selected model.

    Preserves:
    - The system message (always)
    - Tool call / result pairs (never split)
    - Most recent turns (recency bias)
    - Current user message(s)
    """

    def build(
        self,
        current_messages: List[Dict[str, str]],
        session_history: List[Dict[str, str]],
        difficulty_label: str,
    ) -> List[Dict[str, str]]:
        """
        Produce the message list to send to the routed model.

        Args:
            current_messages: The new messages for this turn (may include a
                              system message if this is the first turn).
            session_history:  Prior conversation history from the session store.
            difficulty_label: "easy", "medium", or "hard".

        Returns:
            Compressed list of messages ready to pass to the provider SDK.
        """
        # Extract system message — keep it always, don't count toward budget.
        system_msg: Optional[Dict[str, str]] = None
        non_system: List[Dict[str, str]] = []

        for msg in current_messages:
            if msg.get("role") == "system":
                system_msg = msg
            else:
                non_system.append(msg)

        # Also check session history for a system message if we don't have one.
        if system_msg is None:
            for msg in session_history:
                if msg.get("role") == "system":
                    system_msg = msg
                    break

        history_no_system = [m for m in session_history if m.get("role") != "system"]

        max_pairs = _MAX_PAIRS.get(difficulty_label)
        token_budget = _TOKEN_BUDGET.get(difficulty_label)

        if max_pairs is None and token_budget is None:
            # Hard: send everything.
            compressed_history = history_no_system
        else:
            compressed_history = self._compress(
                history=history_no_system,
                max_pairs=max_pairs,
                token_budget=token_budget,
                # Reserve tokens for current messages.
                reserved_chars=self._total_chars(non_system),
            )

        result: List[Dict[str, str]] = []
        if system_msg:
            result.append(system_msg)
        result.extend(compressed_history)
        result.extend(non_system)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compress(
        self,
        history: List[Dict[str, str]],
        max_pairs: Optional[int],
        token_budget: Optional[int],
        reserved_chars: int,
    ) -> List[Dict[str, str]]:
        """
        Trim history from the front, keeping the most recent turns.
        Tool call pairs are always kept together.
        """
        if not history:
            return []

        # Group consecutive messages into logical turns so we never split
        # a tool call from its result.
        turns = _group_turns(history)

        # Apply max_pairs limit first (count from the end).
        if max_pairs is not None:
            turns = turns[-max_pairs:]

        # Apply token budget (trim from the front until we fit).
        if token_budget is not None:
            budget_chars = (token_budget * _CHARS_PER_TOKEN) - reserved_chars
            while turns:
                total = sum(self._total_chars(t) for t in turns)
                if total <= budget_chars:
                    break
                turns.pop(0)

        return [msg for turn in turns for msg in turn]

    def _total_chars(self, messages: List[Dict[str, str]]) -> int:
        return sum(len(m.get("content", "") or "") for m in messages)

    def savings_estimate(
        self,
        original_messages: List[Dict[str, str]],
        compressed_messages: List[Dict[str, str]],
    ) -> float:
        """Estimated fraction of tokens saved by compression (0.0–1.0)."""
        orig = max(self._total_chars(original_messages), 1)
        comp = self._total_chars(compressed_messages)
        return max(0.0, 1.0 - (comp / orig))


# ---------------------------------------------------------------------------
# Turn grouping
# ---------------------------------------------------------------------------

def _group_turns(messages: List[Dict[str, str]]) -> List[List[Dict[str, str]]]:
    """
    Group messages into logical turns.

    A "turn" is one or more consecutive messages that belong together:
    - user message (possibly followed by tool results)
    - assistant message (possibly followed by tool calls)

    We group so that tool_use + tool_result pairs are never split.
    """
    if not messages:
        return []

    turns: List[List[Dict[str, str]]] = []
    current: List[Dict[str, str]] = [messages[0]]

    for msg in messages[1:]:
        role = msg.get("role", "")
        prev_role = current[-1].get("role", "")

        # Start a new turn on role switch (except tool results stay with user).
        if role == "tool" and prev_role in ("tool", "user"):
            current.append(msg)
        elif role == prev_role:
            current.append(msg)
        else:
            turns.append(current)
            current = [msg]

    turns.append(current)
    return turns
