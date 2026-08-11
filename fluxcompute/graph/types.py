"""
Execution graph types — the DAG of atomic steps behind every agent task.

A TaskGraph is created by FluxClient.task(); nodes are added automatically
for LLM calls made inside the task scope and explicitly via FluxClient.step().
Insertion order is execution order (parents always precede children).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Any, Dict, List, Optional


@dataclass
class TaskNode:
    """One atomic execution step inside a task graph."""

    node_id: str
    task_id: str
    name: str
    node_type: str                        # "task" | "llm_call" | "tool_call"
    parent_id: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    status: str = "running"               # "running" | "succeeded" | "failed"
    failure_reason: Optional[str] = None  # see fluxcompute/graph/failures.py
    model: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: Optional[str] = None
    output_preview: str = ""              # first 500 chars; resume context source
    attributes: Dict[str, Any] = field(default_factory=dict)  # OTel-style span attributes
    events: List[Dict[str, Any]] = field(default_factory=list)  # OTel-style span events
    session_id: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None


class TaskGraph:
    """In-memory DAG for one agent task."""

    def __init__(self, task_id: Optional[str] = None, name: str = ""):
        self.task_id = task_id or f"task_{uuid.uuid4().hex[:12]}"
        self.name = name
        self.nodes: Dict[str, TaskNode] = {}
        self._order: List[str] = []

    def add_node(
        self,
        *,
        name: str,
        node_type: str,
        parent_id: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
        **fields: Any,
    ) -> TaskNode:
        if parent_id is not None and parent_id not in self.nodes:
            raise ValueError(f"Unknown parent_id: {parent_id}")
        node = TaskNode(
            node_id=uuid.uuid4().hex,
            task_id=self.task_id,
            name=name,
            node_type=node_type,
            parent_id=parent_id,
            depends_on=list(depends_on or []),
            **fields,
        )
        self.nodes[node.node_id] = node
        self._order.append(node.node_id)
        return node

    def finish_node(self, node_id: str, status: str, **fields: Any) -> TaskNode:
        node = self.nodes[node_id]
        node.status = status
        node.ended_at = time.time()
        valid_fields = {f.name for f in dataclass_fields(node)}
        for key, value in fields.items():
            if key not in valid_fields:
                raise ValueError(f"Unknown TaskNode field: {key}")
            setattr(node, key, value)
        return node

    def restore_node(self, node: TaskNode) -> TaskNode:
        """Insert a node reconstructed from persisted events, preserving its
        identity. Unlike add_node, nothing is generated or validated here:
        the node keeps the node_id its history refers to (depends_on links
        from other nodes must stay resolvable), and a missing parent is not
        an error — partial ingest is a normal state for a persisted graph.
        Call in execution order; insertion order is the graph's order."""
        if node.node_id not in self.nodes:
            self._order.append(node.node_id)
        self.nodes[node.node_id] = node
        return node

    def in_order(self) -> List[TaskNode]:
        return [self.nodes[nid] for nid in self._order]

    def root(self) -> Optional[TaskNode]:
        return self.nodes[self._order[0]] if self._order else None

    def children(self, node_id: str) -> List[TaskNode]:
        return [n for n in self.in_order() if n.parent_id == node_id]

    def ancestors(self, node_id: str) -> List[TaskNode]:
        """Parent chain from the root down to (excluding) the node."""
        chain: List[TaskNode] = []
        current = self.nodes[node_id].parent_id
        while current is not None:
            node = self.nodes[current]
            chain.append(node)
            current = node.parent_id
        chain.reverse()
        return chain

    def subtree(self, node_id: str) -> List[TaskNode]:
        """The node plus all descendants, in execution order."""
        keep = {node_id}
        result: List[TaskNode] = []
        for node in self.in_order():
            if node.node_id in keep:
                result.append(node)
            elif node.parent_id in keep:
                keep.add(node.node_id)
                result.append(node)
        return result

    def failed_nodes(self) -> List[TaskNode]:
        return [n for n in self.in_order() if n.status == "failed"]

    def context_nodes(self, node_id: str) -> List[TaskNode]:
        """
        Nodes whose outputs form the minimal resume context for node_id.

        Heuristic: every succeeded non-task node that finished before the
        target node started, in execution order. In a linear agent loop this
        equals "ancestors + earlier completed siblings".
        """
        target = self.nodes[node_id]
        out: List[TaskNode] = []
        for node in self.in_order():
            if node.node_id == node_id or node.node_type == "task":
                continue
            if node.status == "succeeded" and (node.ended_at or 0) <= target.started_at:
                out.append(node)
        return out


def node_event(graph: TaskGraph, node: TaskNode) -> Dict[str, Any]:
    """Serialize one node for emission to the FluxCompute backend."""
    return {
        "node_id": node.node_id,
        "task_id": graph.task_id,
        "task_name": graph.name,
        "name": node.name,
        "node_type": node.node_type,
        "parent_id": node.parent_id,
        "depends_on": node.depends_on,
        "status": node.status,
        "failure_reason": node.failure_reason,
        "model": node.model,
        "input_tokens": node.input_tokens,
        "output_tokens": node.output_tokens,
        "cost_usd": node.cost_usd,
        "error": node.error,
        "output_preview": node.output_preview,
        "attributes": node.attributes,
        "events": node.events,
        "session_id": node.session_id,
        "started_at": node.started_at,
        "ended_at": node.ended_at,
    }
