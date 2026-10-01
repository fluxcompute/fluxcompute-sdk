"""
Errors the SDK raises on its own, as opposed to the provider SDK errors it
lets through unchanged.
"""

from __future__ import annotations

from typing import Any


class FluxEmptyResponseError(RuntimeError):
    """The model hit its output-token limit before writing any answer text.

    Reasoning models (gpt-5.x, gpt-6-luna, ...) bill hidden reasoning against
    the same limit as the answer, so a limit that is plenty for a chat model
    can be spent entirely on reasoning. The provider then returns an empty
    answer with a "length" finish reason and no error. Returning that "" to
    the caller looks like a real answer, so the SDK raises instead.

    The tokens were still billed: `output_tokens` and the recorded graph node
    carry the real usage and cost. Raise `max_tokens` and call again.
    """

    def __init__(
        self,
        *,
        model: str,
        max_tokens: int,
        output_tokens: int,
        reasoning_tokens: int,
        finish_reason: str,
        raw: Any = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.output_tokens = output_tokens
        self.reasoning_tokens = reasoning_tokens
        self.finish_reason = finish_reason
        self.raw = raw
        spent = f" ({reasoning_tokens} on reasoning)" if reasoning_tokens else ""
        super().__init__(
            f"token budget exhausted: {model} used {output_tokens}/{max_tokens} output "
            f"tokens{spent} and returned no answer text (finish_reason={finish_reason!r}). "
            "Raise max_tokens and call again."
        )
