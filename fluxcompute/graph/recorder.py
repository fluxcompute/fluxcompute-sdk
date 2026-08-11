"""
Graph recorder — builds TaskGraphs automatically via execution-context
propagation (the same contextvars mechanism OpenTelemetry uses for span
parenting).

`with recorder.task("name"):` opens a graph and makes it ambient; any
FluxClient LLM call inside the scope auto-attaches as a child node, and
`with recorder.step("name"):` records tool/custom steps. Works identically
in any framework or a hand-rolled loop.
"""

from __future__ import annotations

import logging
from collections import deque
from contextvars import ContextVar
from typing import Any, Callable, Dict, List, Optional

from fluxcompute.graph.failures import classify_failure
from fluxcompute.graph.types import TaskGraph, TaskNode, node_event

logger = logging.getLogger("fluxcompute.graph")

SNAPSHOT_LAST_N = 10        # replay window: last-N payloads re-emitted on task exit
SNAPSHOT_MAX_BYTES = 65536  # per-field cap on snapshot payloads
MAX_NODE_EVENTS = 20        # bounds recorded events per node

_current_graph: ContextVar[Optional[TaskGraph]] = ContextVar("flux_graph", default=None)
_current_node_id: ContextVar[Optional[str]] = ContextVar("flux_node_id", default=None)


class NodeScope:
    """Context manager for one node's lifetime. An exception escaping the
    block marks the node failed (and propagates)."""

    def __init__(
        self,
        recorder: "GraphRecorder",
        graph: TaskGraph,
        node: TaskNode,
        *,
        activates_graph: bool = False,
    ):
        self._recorder = recorder
        self._graph = graph
        self.node = node
        self._activates_graph = activates_graph
        self._graph_token = None
        self._node_token = None
        self._output = ""

    @property
    def task_id(self) -> str:
        return self._graph.task_id

    def set_output(self, text: str) -> None:
        """Record this step's output (used for resume-context reconstruction)."""
        self._output = text or ""

    def set_attribute(self, key: str, value: Any) -> None:
        self.node.attributes[key] = value

    def add_event(self, event: Dict[str, Any]) -> None:
        if len(self.node.events) < MAX_NODE_EVENTS:
            self.node.events.append(event)

    @property
    def idempotency_key(self) -> str:
        """Stable `task:name:attempt` key — pass it to side-effecting tools so a
        retried step can deduplicate its external actions."""
        same = [
            n for n in self._graph.in_order()
            if n.name == self.node.name and n.node_type == self.node.node_type
        ]
        attempt = same.index(self.node) + 1 if self.node in same else len(same) + 1
        return f"{self._graph.task_id}:{self.node.name}:{attempt}"

    def __enter__(self) -> "NodeScope":
        if self._activates_graph:
            self._graph_token = _current_graph.set(self._graph)
        self._node_token = _current_node_id.set(self.node.node_id)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            status = "failed" if exc_type else "succeeded"
            error = f"{exc_type.__name__}: {exc}" if exc_type else None
            self._graph.finish_node(
                self.node.node_id, status, error=error, output_preview=self._output[:500],
            )
            self.node.failure_reason = classify_failure(self.node, self._graph)
        finally:
            # Always release the contextvars, even if bookkeeping above raised —
            # otherwise a bookkeeping failure both leaks ambient state forever and
            # (if a real exception is already propagating) replaces it with the
            # bookkeeping exception per Python's exception-chaining semantics.
            _current_node_id.reset(self._node_token)
            if self._graph_token is not None:
                _current_graph.reset(self._graph_token)
        self._recorder._remember(self._graph.task_id, self.node.node_id, None, self._output)
        extra = None
        if self.node.status == "failed" and self._output:
            extra = {"output_full": self._output[:SNAPSHOT_MAX_BYTES]}
        self._recorder.emit_node(self._graph, self.node, extra)
        if self._activates_graph:
            # Task-root exit: re-emit the last-N ring with full payloads (the
            # replay window). The terminal-status upsert guard makes the node
            # side a no-op; the server keeps the payload snapshots.
            self._recorder.emit_ring_snapshots(self._graph)
        return False  # never swallow exceptions


class AttachScope:
    """Re-activate an existing graph (used by resume) without adding a node."""

    def __init__(self, graph: TaskGraph, parent_id: Optional[str] = None):
        self._graph = graph
        self._parent_id = parent_id
        self._graph_token = None
        self._node_token = None

    def __enter__(self) -> "AttachScope":
        root = self._graph.root()
        node_id = self._parent_id or (root.node_id if root else None)
        self._graph_token = _current_graph.set(self._graph)
        self._node_token = _current_node_id.set(node_id)
        return self

    def __exit__(self, *args) -> bool:
        _current_node_id.reset(self._node_token)
        _current_graph.reset(self._graph_token)
        return False


class GraphRecorder:
    """Owns all task graphs for one FluxClient; emits node events on finish."""

    def __init__(self, emit: Optional[Callable[[dict], None]] = None):
        self._graphs: Dict[str, TaskGraph] = {}
        self._emit = emit
        self._recent: Dict[str, deque] = {}  # task_id → last-N (node_id, prompt, output)

    def task(self, name: str, task_id: Optional[str] = None) -> NodeScope:
        graph = TaskGraph(task_id=task_id, name=name)
        if graph.task_id in self._graphs:
            raise ValueError(f"task_id already in use: {graph.task_id}")
        self._graphs[graph.task_id] = graph
        root = graph.add_node(name=name, node_type="task")
        return NodeScope(self, graph, root, activates_graph=True)

    def step(self, name: str, depends_on: Optional[List[str]] = None) -> NodeScope:
        graph = _current_graph.get()
        if graph is None:
            raise RuntimeError("step() must be used inside a task() scope")
        node = graph.add_node(
            name=name,
            node_type="tool_call",
            parent_id=_current_node_id.get(),
            depends_on=depends_on,
        )
        return NodeScope(self, graph, node)

    def attach(self, task_id: str, parent_id: Optional[str] = None) -> AttachScope:
        graph = self._graphs.get(task_id)
        if graph is None:
            raise KeyError(f"Unknown task_id: {task_id}")
        if _current_graph.get() is graph:
            raise RuntimeError(
                f"attach({task_id!r}) called while that graph is already active — "
                "attach() is for reactivating a graph outside its original task() scope."
            )
        return AttachScope(graph, parent_id=parent_id)

    def record_llm_call(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        output_preview: str,
        session_id: Optional[str],
        error: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        events: Optional[List[Dict[str, Any]]] = None,
        prompt_full: Optional[str] = None,
        output_full: Optional[str] = None,
    ) -> Optional[TaskNode]:
        """Auto-record one routed LLM call. No-op when no task scope is active."""
        graph = _current_graph.get()
        if graph is None:
            return None
        try:
            node = graph.add_node(
                name=f"llm:{model}",
                node_type="llm_call",
                parent_id=_current_node_id.get(),
                model=model,
                session_id=session_id,
                attributes=dict(attributes or {}),
                events=list(events or [])[:MAX_NODE_EVENTS],
            )
            graph.finish_node(
                node.node_id,
                "failed" if error else "succeeded",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                output_preview=(output_preview or "")[:500],
                error=error,
            )
            node.failure_reason = classify_failure(node, graph)
            self._remember(graph.task_id, node.node_id, prompt_full, output_full)
            extra = None
            if error and (prompt_full or output_full):
                # Failed calls ship their payload immediately — replayable even
                # if the task scope never exits cleanly.
                extra = {
                    "prompt_full": (prompt_full or "")[:SNAPSHOT_MAX_BYTES] or None,
                    "output_full": (output_full or "")[:SNAPSHOT_MAX_BYTES] or None,
                }
            self.emit_node(graph, node, extra)
            return node
        except Exception as exc:
            # Graph bookkeeping is non-critical — it must never mask or replace
            # the real exception from the caller's failure path (see NodeScope.__exit__).
            logger.debug("record_llm_call failed (non-fatal): %s", exc)
            return None

    def _remember(
        self,
        task_id: str,
        node_id: str,
        prompt_full: Optional[str],
        output_full: Optional[str],
    ) -> None:
        """Track the last-N payloads per task — the replay window for task exit."""
        if not prompt_full and not output_full:
            return
        ring = self._recent.setdefault(task_id, deque(maxlen=SNAPSHOT_LAST_N))
        ring.append((
            node_id,
            (prompt_full or "")[:SNAPSHOT_MAX_BYTES] or None,
            (output_full or "")[:SNAPSHOT_MAX_BYTES] or None,
        ))

    def emit_ring_snapshots(self, graph: TaskGraph) -> None:
        """Re-emit the task's last-N nodes with their full payloads attached."""
        ring = self._recent.pop(graph.task_id, None)
        if not ring:
            return
        for node_id, prompt_full, output_full in ring:
            node = graph.nodes.get(node_id)
            if node is None:
                continue
            self.emit_node(graph, node, {
                "prompt_full": prompt_full,
                "output_full": output_full,
            })

    def get_graph(self, task_id: str) -> Optional[TaskGraph]:
        return self._graphs.get(task_id)

    def adopt_graph(self, graph: TaskGraph) -> None:
        """Register a TaskGraph this process didn't record itself — e.g. one
        a recovery plugin reconstructed from durable storage — so attach()
        can operate on it exactly as if task() had run here. If a local
        graph already exists for this task_id (a race during the fetch), the
        local one wins; it's the authoritative copy."""
        self._graphs.setdefault(graph.task_id, graph)

    def emit_node(
        self,
        graph: TaskGraph,
        node: TaskNode,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._emit is None:
            return
        try:
            event = node_event(graph, node)
            if extra:
                event.update({k: v for k, v in extra.items() if v is not None})
            self._emit(event)
        except Exception as exc:
            logger.debug("graph emit failed (non-fatal): %s", exc)
