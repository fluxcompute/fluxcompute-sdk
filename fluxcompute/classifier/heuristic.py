"""
Heuristic difficulty classifier for agentic queries.

Classifies queries on a 0.0–1.0 difficulty scale using eight rule-based
signal categories.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Set

from fluxcompute.models import ClassificationResult


# ---------------------------------------------------------------------------
# Keyword sets
# ---------------------------------------------------------------------------

REASONING_KEYWORDS: Set[str] = {
    "why",
    "explain",
    "analyze",
    "compare",
    "evaluate",
    "reason",
    "think through",
    "break down",
    "consider",
    "implications",
    "trade-offs",
    "trade offs",
    "pros and cons",
    "what would happen if",
    "how would you approach",
    "argue for",
    "argue against",
    "critique",
    "assess",
    "justify",
    "interpret",
}

MATH_KEYWORDS: Set[str] = {
    "calculate",
    "compute",
    "solve",
    "equation",
    "formula",
    "derivative",
    "integral",
    "probability",
    "statistics",
    "optimize",
    "maximize",
    "minimize",
    "regression",
    "matrix",
    "vector",
    "eigenvalue",
    "proof",
    "theorem",
}

CODE_KEYWORDS: Set[str] = {
    "write code",
    "implement",
    "debug",
    "refactor",
    "algorithm",
    "function that",
    "class that",
    "api endpoint",
    "sql query",
    "regex",
    "script",
    "write a program",
    "build a",
    "create a function",
    "fix this code",
    "code review",
    "unit test",
}

SIMPLE_KEYWORDS: Set[str] = {
    "what is",
    "define",
    "list",
    "name",
    "when did",
    "who is",
    "where is",
    "how many",
    "translate",
    "format",
    "convert",
    "summarize briefly",
    "yes or no",
    "true or false",
    "what does",
    "spell",
    "abbreviation",
}

CREATIVE_KEYWORDS: Set[str] = {
    "write a story",
    "write a poem",
    "creative writing",
    "brainstorm",
    "imagine",
    "fiction",
    "narrative",
    "dialogue",
    "screenplay",
    "compose",
}


# ---------------------------------------------------------------------------
# Model mapping
# ---------------------------------------------------------------------------

# Anthropic model tiers (current generation — must exist in MODEL_PRICING)
ANTHROPIC_MODELS = {
    "easy": "claude-haiku-4-5-20251001",
    "medium": "claude-sonnet-4-6",
    "hard": "claude-opus-4-8",
}

# OpenAI model tiers
OPENAI_MODELS = {
    "easy": "gpt-4o-mini",
    "medium": "gpt-4o",
    "hard": "o1",
}

# Qwen via HuggingFace (must exist in MODEL_PRICING)
QWEN_MODELS = {
    "easy": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "medium": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "hard": "Qwen/Qwen2.5-Coder-32B-Instruct",
}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def _count_keyword_hits(text: str, keywords: Set[str]) -> int:
    """Count how many keywords appear in the text."""
    return sum(1 for kw in keywords if kw in text)


def _message_text(content: Any) -> str:
    """Text of a message whose content may be a string or Anthropic content
    blocks. Non-text blocks (images, tool results) contribute nothing to the
    difficulty signal but must not crash the classifier — the API accepts
    them, so the drop-in promise requires we do too."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def classify(
    messages: List[Dict[str, str]],
    provider: str = "anthropic",
    easy_threshold: float = 0.18,
    hard_threshold: float = 0.45,
) -> ClassificationResult:
    """
    Classify query difficulty using heuristic rules.

    Args:
        messages: List of message dicts with "role" and "content" keys.
        provider: "anthropic" or "openai" — determines model selection.

    Returns:
        ClassificationResult with score, label, model, reasoning.
    """
    start = time.monotonic()

    # Extract last user message
    last_msg = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_msg = _message_text(msg.get("content")).lower()
            break

    if not last_msg:
        # No user message found — default to medium
        elapsed = (time.monotonic() - start) * 1000
        models = ANTHROPIC_MODELS if provider == "anthropic" else (QWEN_MODELS if provider == "huggingface" else OPENAI_MODELS)
        return ClassificationResult(
            score=0.5,
            label="medium",
            model=models["medium"],
            reasoning="no user message found, defaulting to medium",
            classification_ms=round(elapsed, 2),
        )

    score = 0.0
    reasons: list[str] = []

    # -----------------------------------------------------------------------
    # Signal 1: Query length (0 – 0.15)
    # -----------------------------------------------------------------------
    word_count = len(last_msg.split())
    if word_count > 200:
        score += 0.20
        reasons.append(f"long query ({word_count} words)")
    elif word_count > 80:
        score += 0.15
        reasons.append(f"medium-long query ({word_count} words)")
    elif word_count > 30:
        score += 0.08
        reasons.append(f"moderate query ({word_count} words)")

    # -----------------------------------------------------------------------
    # Signal 2: Reasoning keywords (0 – 0.35)
    # -----------------------------------------------------------------------
    reasoning_hits = _count_keyword_hits(last_msg, REASONING_KEYWORDS)
    if reasoning_hits >= 3:
        score += 0.35
        reasons.append(f"heavy reasoning ({reasoning_hits} signals)")
    elif reasoning_hits >= 2:
        score += 0.25
        reasons.append(f"moderate reasoning ({reasoning_hits} signals)")
    elif reasoning_hits >= 1:
        score += 0.18
        reasons.append(f"some reasoning ({reasoning_hits} signals)")

    # -----------------------------------------------------------------------
    # Signal 3: Math / computation keywords (0 – 0.30)
    # -----------------------------------------------------------------------
    math_hits = _count_keyword_hits(last_msg, MATH_KEYWORDS)
    if math_hits >= 3:
        score += 0.30
        reasons.append(f"heavy math ({math_hits} signals)")
    elif math_hits >= 2:
        score += 0.22
        reasons.append(f"moderate math ({math_hits} signals)")
    elif math_hits >= 1:
        score += 0.18
        reasons.append(f"light math ({math_hits} signals)")

    # -----------------------------------------------------------------------
    # Signal 4: Code generation keywords (0 – 0.25)
    # -----------------------------------------------------------------------
    code_hits = _count_keyword_hits(last_msg, CODE_KEYWORDS)
    if code_hits >= 2:
        score += 0.30
        reasons.append(f"code generation ({code_hits} signals)")
    elif code_hits >= 1:
        score += 0.15
        reasons.append(f"light coding ({code_hits} signals)")

    # -----------------------------------------------------------------------
    # Signal 5: Simple query indicators (negative: -0.20 – 0)
    # -----------------------------------------------------------------------
    simple_hits = _count_keyword_hits(last_msg, SIMPLE_KEYWORDS)
    if simple_hits >= 3:
        score -= 0.20
        reasons.append(f"very simple query ({simple_hits} signals)")
    elif simple_hits >= 2:
        score -= 0.15
        reasons.append(f"simple query ({simple_hits} signals)")
    elif simple_hits >= 1:
        score -= 0.08
        reasons.append(f"likely simple ({simple_hits} signals)")

    # -----------------------------------------------------------------------
    # Signal 6: Multi-turn depth (0 – 0.15)
    # -----------------------------------------------------------------------
    turn_count = len([m for m in messages if m.get("role") == "user"])
    if turn_count > 10:
        score += 0.15
        reasons.append(f"deep conversation ({turn_count} user turns)")
    elif turn_count > 5:
        score += 0.10
        reasons.append(f"multi-turn ({turn_count} user turns)")
    elif turn_count > 2:
        score += 0.05
        reasons.append(f"short multi-turn ({turn_count} user turns)")

    # -----------------------------------------------------------------------
    # Signal 7: System prompt complexity (0 – 0.10)
    # -----------------------------------------------------------------------
    system_msgs = [m for m in messages if m.get("role") == "system"]
    if system_msgs:
        sys_len = len(system_msgs[0].get("content", "").split())
        if sys_len > 500:
            score += 0.10
            reasons.append(f"complex system prompt ({sys_len} words)")
        elif sys_len > 200:
            score += 0.05
            reasons.append(f"moderate system prompt ({sys_len} words)")

    # -----------------------------------------------------------------------
    # Signal 8: Creative writing (0 – 0.10)
    # -----------------------------------------------------------------------
    creative_hits = _count_keyword_hits(last_msg, CREATIVE_KEYWORDS)
    if creative_hits >= 1:
        score += 0.10
        reasons.append(f"creative writing ({creative_hits} signals)")

    # -----------------------------------------------------------------------
    # Clamp and map to label + model
    # -----------------------------------------------------------------------
    score = max(0.0, min(1.0, score))

    if score < easy_threshold:
        label = "easy"
    elif score < hard_threshold:
        label = "medium"
    else:
        label = "hard"

    models = ANTHROPIC_MODELS if provider == "anthropic" else (QWEN_MODELS if provider == "huggingface" else OPENAI_MODELS)
    model = models[label]

    elapsed = (time.monotonic() - start) * 1000

    return ClassificationResult(
        score=round(score, 3),
        label=label,
        model=model,
        reasoning="; ".join(reasons) if reasons else "default classification",
        classification_ms=round(elapsed, 2),
    )
