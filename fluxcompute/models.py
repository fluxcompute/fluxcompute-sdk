"""
Data models for FluxCompute SDK.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """Result of the heuristic difficulty classifier."""

    score: float  # 0.0 – 1.0
    label: str  # "easy", "medium", "hard"
    model: str  # selected model identifier
    reasoning: str  # human-readable explanation
    classification_ms: float  # time to classify


# ---------------------------------------------------------------------------
# FluxCompute metadata attached to every response
# ---------------------------------------------------------------------------

@dataclass
class CacheStats:
    """Prompt-cache metrics for a single request."""

    cache_write_tokens: int = 0   # tokens written to Anthropic's cache
    cache_read_tokens: int = 0    # tokens read from cache (90% cheaper)
    cache_hit: bool = False       # True if at least some tokens were cached


@dataclass
class FluxStreamChunk:
    """A single streamed text delta from the model."""
    text: str
    model: str


@dataclass
class FluxMetadata:
    """Metadata that FluxCompute attaches to every LLM response."""

    difficulty_score: float
    difficulty_label: str
    model_selected: str
    baseline_model: str
    cost_usd: float
    baseline_cost_usd: float
    savings_usd: float
    classification_ms: float
    overhead_ms: float
    session_id: str
    context_compression: float = 0.0   # fraction of session tokens saved by compression
    cache: CacheStats = None            # prompt-cache stats (None if not Anthropic)

    def __post_init__(self):
        if self.cache is None:
            self.cache = CacheStats()


# ---------------------------------------------------------------------------
# Unified response wrapper
# ---------------------------------------------------------------------------

@dataclass
class FluxResponse:
    """
    Wraps the raw provider response and adds FluxCompute metadata.

    Attributes:
        raw         – The original response object from Anthropic/OpenAI SDK.
        fluxcompute – FluxCompute routing + cost metadata.
        provider    – "anthropic" or "openai".
    """

    raw: Any
    fluxcompute: FluxMetadata
    provider: str

    # Convenience pass-throughs so callers can treat this like the raw response
    @property
    def content(self):
        if self.provider == "anthropic":
            return self.raw.content
        # OpenAI
        return self.raw.choices

    @property
    def usage(self) -> dict:
        if self.provider == "anthropic":
            return {
                "input_tokens": self.raw.usage.input_tokens,
                "output_tokens": self.raw.usage.output_tokens,
            }
        # OpenAI
        return {
            "input_tokens": self.raw.usage.prompt_tokens,
            "output_tokens": self.raw.usage.completion_tokens,
        }

    @property
    def model(self) -> str:
        return self.raw.model

    @property
    def text(self) -> str:
        """Shortcut: first text block content."""
        if self.provider == "anthropic":
            for block in self.raw.content:
                if hasattr(block, "text"):
                    return block.text
            return ""
        # OpenAI
        return self.raw.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Telemetry event (sent to FluxCompute backend)
# ---------------------------------------------------------------------------

@dataclass
class TelemetryEvent:
    """Anonymised event sent to FluxCompute telemetry backend."""

    customer_key: str
    session_id: str
    difficulty_score: float
    difficulty_label: str
    model_selected: str
    baseline_model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    baseline_cost_usd: float
    savings_usd: float
    classification_ms: float
    overhead_ms: float
    # NOTE: no query content — privacy safe
