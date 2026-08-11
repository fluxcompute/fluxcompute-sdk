"""
Zero-touch LangGraph adapter.

Replace MemorySaver() with FluxCheckpointer(client) and every LangGraph
super-step is mirrored into a FluxCompute task graph — no flux.task()/
flux.step() wrapper needed inside your node bodies.

This is a convenience layer: client.task()/client.step()'s contextvar-based
auto-parenting (see fluxcompute/graph/recorder.py) already works inside any
LangGraph node; this adapter only removes the manual wrapping step by
observing LangGraph's own checkpoint writes instead.

Checkpoint storage/consistency semantics are entirely delegated to
LangGraph's own MemorySaver (we subclass it and always call super() first) —
Flux only observes each write after it succeeds and never alters the
returned config/checkpoint.

Known limitation: this adapter only mirrors SUCCESSFUL checkpoint writes.
When a LangGraph node raises, LangGraph never calls put()/aput() for that
step, so this adapter has no hook to observe the failure — it is never
recorded in the task graph. If you need failure visibility for a
failure-prone node, wrap that node's body in client.step() directly (its
contextvar-based auto-parenting works fine alongside FluxCheckpointer); don't
rely on this adapter alone for failure observability.

    app = graph.compile(checkpointer=FluxCheckpointer(flux_client))

Verified against langgraph==1.2.7. Node names are derived by diffing the
values in checkpoint["versions_seen"] (cumulative for a thread's lifetime,
keyed by node name) against the previous checkpoint's snapshot for that
thread, with a metadata-based fallback for checkpoint layouts that lack
versions_seen.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

try:
    from langgraph.checkpoint.memory import MemorySaver
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "FluxCheckpointer requires langgraph: pip install langgraph"
    ) from exc

logger = logging.getLogger("fluxcompute.integrations.langgraph")


class FluxCheckpointer(MemorySaver):
    """MemorySaver that mirrors each checkpoint write into a Flux task graph."""

    def __init__(self, client: Any, task_name: str = "langgraph-run"):
        super().__init__()
        self._client = client
        self._task_name = task_name
        # thread_id -> last-seen {node_name: version} snapshot, so each put()
        # can diff checkpoint["versions_seen"] by VALUE change (not just key
        # presence — see module docstring) to find newly-finished nodes.
        self._last_versions: Dict[str, Dict[str, Any]] = {}

    def put(self, config, checkpoint, metadata, new_versions):
        result = super().put(config, checkpoint, metadata, new_versions)
        self._record(config, checkpoint, metadata)
        return result

    async def aput(self, config, checkpoint, metadata, new_versions):
        result = await super().aput(config, checkpoint, metadata, new_versions)
        self._record(config, checkpoint, metadata)
        return result

    def _record(self, config, checkpoint, metadata) -> None:
        try:
            thread_id = (config.get("configurable") or {}).get("thread_id", "default")
            recorder = self._client._graph
            task_id = f"lg_{thread_id}"
            graph = recorder.get_graph(task_id)
            if graph is None:
                # Create the graph + root; the root scope closes immediately —
                # LangGraph owns the run lifecycle end-to-end, we only ever
                # observe writes after the fact, never drive them.
                with recorder.task(self._task_name, task_id=task_id):
                    pass
                graph = recorder.get_graph(task_id)

            new_names = self._new_node_names(thread_id, checkpoint, metadata)
            if not new_names:
                return

            root = graph.root()
            preview = str((checkpoint or {}).get("channel_values", ""))[:500]
            for name in new_names:
                prev = graph.in_order()[-1]
                node = graph.add_node(
                    name=name[:80],
                    node_type="tool_call",
                    parent_id=root.node_id if root else None,
                    depends_on=[prev.node_id] if prev is not root else [],
                )
                graph.finish_node(node.node_id, "succeeded", output_preview=preview)
                recorder.emit_node(graph, node)
        except Exception as exc:
            # Observability must never break the customer's LangGraph run.
            logger.debug("FluxCheckpointer record failed (non-fatal): %s", exc)

    def _new_node_names(self, thread_id, checkpoint, metadata) -> List[str]:
        versions_seen = checkpoint.get("versions_seen") if isinstance(checkpoint, dict) else None
        if versions_seen is not None:
            return self._diff_versions_seen(thread_id, versions_seen)

        # Fallback for checkpoint layouts without versions_seen: derive a
        # name from checkpoint metadata directly. Version-sensitive — adjust
        # here if a different langgraph puts node names elsewhere.
        meta = dict(metadata or {})
        writes = meta.get("writes") or {}
        if writes:
            return list(writes.keys())
        return [str(meta.get("source", "checkpoint"))]

    def _diff_versions_seen(self, thread_id: str, versions_seen: dict) -> List[str]:
        """
        versions_seen is cumulative for the thread's lifetime — once a node
        name appears as a key it never disappears. What changes on
        re-execution (a loop, a new conversation turn) is the version value
        under that key, not key presence — so diff on changed VALUES.
        """
        prev_snapshot = self._last_versions.get(thread_id, {})
        current = {k: v for k, v in versions_seen.items() if not str(k).startswith("__")}
        new_names = sorted(
            name for name, version in current.items()
            if prev_snapshot.get(name) != version
        )
        self._last_versions[thread_id] = current
        return new_names
