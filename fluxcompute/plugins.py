"""
Plugin discovery for optional FluxCompute add-ons.

The SDK defines the interface; separate packages (e.g. fluxcompute-recovery)
implement it and register via a `fluxcompute.plugins` entry point. This
keeps the dependency edge one-directional: the SDK never imports a plugin
package directly, and works with zero plugins installed.

In-process resume (fluxcompute.graph.resume) works offline whenever the
calling process still holds the task's graph in memory. A recovery plugin
extends resume to graphs this process never recorded, reconstructed from
durable storage (which requires telemetry to have been enabled when the
task ran).
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from fluxcompute.graph.types import TaskGraph


class FluxRecoveryNotInstalled(RuntimeError):
    """Raised by FluxClient.resume() when it can't find a task's graph
    locally and no recovery plugin is registered to look elsewhere."""

    def __init__(self) -> None:
        super().__init__(
            "This process doesn't hold that task's graph, and no recovery "
            "plugin is installed to fetch it from durable storage. "
            "Install one (e.g. `pip install fluxcompute-recovery`)."
        )


@runtime_checkable
class RecoveryPlugin(Protocol):
    """Interface a recovery add-on implements to back cross-process resume."""

    async def fetch_graph(self, task_id: str) -> Optional["TaskGraph"]:
        """Reconstruct a task's graph from durable storage, or None if it
        genuinely isn't known there either. See fluxcompute.graph.types.TaskGraph
        for the shape a plugin must return."""
        ...


def get_recovery_plugin() -> RecoveryPlugin:
    """
    Look up the registered `fluxcompute.plugins` entry point named
    "recovery". Not cached — entry-point discovery is cheap and this
    keeps plugin installation/uninstallation picked up without a process
    restart, and keeps tests free of cross-test cache poisoning.
    """
    for ep in entry_points(group="fluxcompute.plugins"):
        if ep.name == "recovery":
            return ep.load()
    raise FluxRecoveryNotInstalled()
