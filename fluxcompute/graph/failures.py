"""
Rules-based failure classification for task-graph nodes.

Categories:
context_overflow | budget | tool_error | stall | refusal | unknown

Order matters: stall (needs sibling history) beats per-node rules, and
error-text rules beat the node_type fallback.
"""

from __future__ import annotations

import re
from typing import Optional

from fluxcompute.graph.types import TaskGraph, TaskNode

CONTEXT_OVERFLOW_RE = re.compile(
    r"context.length|maximum context|prompt is too long|too many tokens"
    r"|context_length_exceeded",
    re.IGNORECASE,
)
BUDGET_RE = re.compile(
    r"rate.limit|quota|billing|credit balance|insufficient.credit",
    re.IGNORECASE,
)
REFUSAL_RE = re.compile(
    r"\b(i can't|i cannot|i'm unable|i am unable|i won't|i'm not able)\b",
    re.IGNORECASE,
)
STALL_WINDOW = 3


def detect_refusal(output: str) -> bool:
    """Short output that opens with a refusal phrase."""
    if not output or len(output) > 400:
        return False
    return bool(REFUSAL_RE.match(output.strip()))


def classify_failure(node: TaskNode, graph: TaskGraph) -> Optional[str]:
    """Classify why a node failed. Returns None for healthy nodes."""
    if node.status != "failed":
        if node.node_type == "llm_call" and detect_refusal(node.output_preview):
            return "refusal"
        return None
    if _is_stall(node, graph):
        return "stall"
    error = node.error or ""
    if CONTEXT_OVERFLOW_RE.search(error):
        return "context_overflow"
    if BUDGET_RE.search(error):
        return "budget"
    if node.node_type == "tool_call":
        return "tool_error"
    return "unknown"


def _is_stall(node: TaskNode, graph: TaskGraph) -> bool:
    """STALL_WINDOW consecutive failed same-name siblings = a retry loop."""
    siblings = [n for n in graph.in_order() if n.parent_id == node.parent_id]
    tail = siblings[-STALL_WINDOW:]
    if len(tail) < STALL_WINDOW or tail[-1].node_id != node.node_id:
        return False
    return all(n.name == node.name and n.status == "failed" for n in tail)
