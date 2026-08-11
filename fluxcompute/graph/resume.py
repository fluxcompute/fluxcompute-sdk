"""
Graph-aware resume — rebuild minimal context from succeeded nodes and re-run
only the failed step (routing may pick a different/stronger model).

The resumed call gets a deliberate summary of what already succeeded plus
the failed step's error — not a replay of every message that led up to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from fluxcompute.graph.types import TaskGraph, TaskNode


@dataclass
class ResumePlan:
    """Everything needed to re-run one failed step with minimal context."""

    task_id: str
    failed_node: TaskNode
    context_messages: List[Dict[str, str]] = field(default_factory=list)
    completed_steps: List[TaskNode] = field(default_factory=list)


def summarize_completed_steps(steps: List[Tuple[str, str]], task_name: str) -> str:
    """Render (name, output_preview) pairs into a resume-context message."""
    lines = [
        f"You are resuming the task: {task_name}",
        "Steps already completed successfully:",
    ]
    for name, preview in steps:
        lines.append(f"- {name}: {preview or '(no recorded output)'}")
    lines.append("Do not redo completed steps. Continue from the failed step described next.")
    return "\n".join(lines)


def build_resume_plan(graph: TaskGraph, node_id: Optional[str] = None) -> ResumePlan:
    """Plan a resume: target node (most recent unresolved failure by default) + minimal context."""
    if node_id is not None:
        failed = graph.nodes[node_id]
    else:
        failures = graph.failed_nodes()
        # skip the task root — resume targets the step that actually broke
        candidates = [n for n in failures if n.node_type != "task"] or failures
        # a failed node is "resolved" once some other node's depends_on already
        # references it — i.e. it's already been retried, successfully or not
        resolved_ids = {dep for n in graph.in_order() for dep in n.depends_on}
        unresolved = [n for n in candidates if n.node_id not in resolved_ids]
        if not unresolved:
            raise ValueError(f"Task {graph.task_id} has no unresolved failed nodes to resume from")
        failed = unresolved[-1]

    completed = graph.context_nodes(failed.node_id)
    messages: List[Dict[str, str]] = []
    if completed:
        summary = summarize_completed_steps(
            [(n.name, n.output_preview) for n in completed], graph.name,
        )
        messages.append({"role": "user", "content": summary})
    return ResumePlan(
        task_id=graph.task_id,
        failed_node=failed,
        context_messages=messages,
        completed_steps=completed,
    )
