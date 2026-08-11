"""
Cost calculator for LLM inference.

Pricing is per 1M tokens (input / output) as of May 2026.
Source: provider pricing pages.
"""

from __future__ import annotations

from typing import Dict, Tuple


# (input_price_per_1M, output_price_per_1M)
MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    # Anthropic
    "claude-3-haiku-20240307": (0.25, 1.25),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-3-opus-20240229": (15.00, 75.00),
    "claude-opus-4-20250918": (15.00, 75.00),
    # Anthropic — current generation (what the classifier routes to)
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-8": (5.00, 25.00),
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    "o3-mini": (1.10, 4.40),
    # Qwen via HuggingFace (serverless = free; fill in dedicated-endpoint price per 1M tokens)
    "Qwen/Qwen2.5-Coder-7B-Instruct": (0.0, 0.0),
    "Qwen/Qwen2.5-Coder-32B-Instruct": (0.0, 0.0),
}

# Default baseline models — must be the most expensive model the router can
# select for that provider: savings are measured against what the call would
# have cost without routing, so a baseline cheaper than the hard tier
# understates baseline_cost_usd. Pinned against the tier maps by
# test_cost.py::TestBaselineInvariant.
DEFAULT_BASELINES = {
    "anthropic": "claude-opus-4-8",
    "openai": "o1",
}


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """
    Calculate cost in USD for a single request.

    Anthropic prompt caching pricing:
      cache_write_tokens : 1.25× base input price (written to cache)
      cache_read_tokens  : 0.10× base input price (read from cache — 90% discount)
    """
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return 0.0

    input_price, output_price = pricing
    cost = (
        (input_tokens / 1_000_000) * input_price
        + (output_tokens / 1_000_000) * output_price
        + (cache_write_tokens / 1_000_000) * input_price * 1.25
        + (cache_read_tokens / 1_000_000) * input_price * 0.10
    )
    return round(cost, 8)


def calculate_savings(
    model_used: str,
    baseline_model: str,
    input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> Tuple[float, float, float]:
    """
    Calculate actual cost, baseline cost, and savings.

    Baseline cost is computed at the baseline model's price with no cache
    discounts applied, so caching shows up as savings rather than as a
    lower baseline.

    Returns:
        (actual_cost, baseline_cost, savings_usd)
    """
    actual = calculate_cost(
        model_used, input_tokens, output_tokens,
        cache_write_tokens, cache_read_tokens,
    )
    # Baseline: same token counts but routed to top-tier with no caching
    baseline = calculate_cost(baseline_model, input_tokens + cache_write_tokens + cache_read_tokens, output_tokens)
    savings = max(0.0, baseline - actual)
    return round(actual, 8), round(baseline, 8), round(savings, 8)


def get_baseline_model(provider: str) -> str:
    """Get the default baseline (most expensive) model for a provider."""
    return DEFAULT_BASELINES.get(provider, "claude-opus-4-8")
