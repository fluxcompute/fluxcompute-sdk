"""
CacheManager — context persistence across model switches via prompt caching.

History is marked with cache_control at the session boundary; Anthropic
caches those KV states server-side for 5 minutes, so a model resuming the
session reads the cache instead of re-processing the full history.

Anthropic pricing (as of May 2026):
  cache write : 1.25× base input price (one-time)
  cache read  : 0.10× base input price (90% discount on repeat)
  min cacheable: 1 024 tokens (markers on shorter prompts are no-ops)

OpenAI:
  Automatic prefix caching on ≥ 1 024 token prefixes, 50% discount.
  No explicit markers — prefix structure is optimized instead.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# Chars below this threshold won't reach Anthropic's 1 024-token minimum.
# We still mark them — the API silently ignores the marker if too short.
_MIN_CHARS_FOR_CACHE = 100


class CacheManager:
    """
    Injects Anthropic prompt-cache breakpoints at the right positions.

    Call once per request, after ContextBuilder has compressed history.
    Returns messages in Anthropic's content-block format so the cache
    markers survive the API call.
    """

    def prepare_for_anthropic(
        self,
        messages: List[Dict[str, Any]],
        session_history: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
        """
        Transforms messages into Anthropic's content-block format with
        cache_control markers placed at:

          1. System message (most stable — always cache it)
          2. Last turn of session history (semi-stable — new per session,
             but the same across the easy/medium steps that follow)
          3. Current user turn — NO marker (changes every request)

        Returns:
            (messages_for_api, system_blocks)
            system_blocks is None if no system message is present.
        """
        # Split system from conversation
        system_text: Optional[str] = None
        conv: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_text = str(msg.get("content") or "")
            else:
                conv.append(msg)

        # Also pull system from session history if not in current messages
        if system_text is None:
            for msg in session_history:
                if msg.get("role") == "system":
                    system_text = str(msg.get("content") or "")
                    break

        # ── System block (always cached) ──────────────────────────────────
        system_blocks: Optional[List[Dict[str, Any]]] = None
        if system_text:
            system_blocks = [
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        if not conv:
            return [], system_blocks

        # ── Mark the session history boundary ─────────────────────────────
        # Rule: cache everything up to (but not including) the last message.
        # The last message is always the fresh current query — it changes every
        # request, so marking it would cause a cache miss on every turn.
        # The second-to-last message is the last stable turn (prior assistant
        # response or end of session history) — mark it as the cache boundary.
        #
        # This matches Anthropic's recommended pattern:
        #   [system (cached)] [... history (cached at boundary)] [current query]
        cache_breakpoint_idx: Optional[int] = len(conv) - 2 if len(conv) >= 2 else None

        # ── Build content-block messages ──────────────────────────────────
        result: List[Dict[str, Any]] = []
        for i, msg in enumerate(conv):
            should_cache = (
                i == cache_breakpoint_idx
                and len(str(msg.get("content") or "")) >= _MIN_CHARS_FOR_CACHE
            )
            result.append(_to_content_block(msg, cache=should_cache))

        return result, system_blocks

    def estimate_cache_savings(
        self,
        session_history_tokens: int,
        is_cache_hit: bool,
    ) -> float:
        """
        Estimated USD saved by a cache read vs re-processing the same tokens.
        Uses Haiku pricing as the lower bound (cheapest model that benefits most).
        """
        # Haiku input: $0.80/M; cache read: $0.08/M → saving: $0.72/M
        savings_per_million = 0.72
        if not is_cache_hit or session_history_tokens == 0:
            return 0.0
        return round((session_history_tokens / 1_000_000) * savings_per_million, 8)


def _to_content_block(
    msg: Dict[str, Any],
    cache: bool = False,
) -> Dict[str, Any]:
    """
    Convert a plain {role, content} message to Anthropic content-block format.
    If content is already a list of blocks, pass it through untouched.
    """
    content = msg.get("content")
    role = msg.get("role", "user")

    if isinstance(content, list):
        # Already in block format — inject cache_control on last text block
        if cache:
            blocks = list(content)
            for j in reversed(range(len(blocks))):
                if blocks[j].get("type") == "text":
                    blocks[j] = {**blocks[j], "cache_control": {"type": "ephemeral"}}
                    break
            return {"role": role, "content": blocks}
        return {"role": role, "content": content}

    # Plain string content
    block: Dict[str, Any] = {"type": "text", "text": str(content or "")}
    if cache:
        block["cache_control"] = {"type": "ephemeral"}
    return {"role": role, "content": [block]}
