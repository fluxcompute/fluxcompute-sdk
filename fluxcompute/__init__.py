"""
FluxCompute — an optimization layer for agentic workflows.

Route every query to the cheapest model that can answer it. Queries that
don't need the top tier are routed to cheaper models; how much you save
overall depends on how much of your traffic actually needs a frontier model.

Usage:
    from fluxcompute import FluxClient

    client = FluxClient(
        anthropic_key="sk-ant-xxx",
        fluxcompute_key="flx_xxx",  # optional, enables telemetry + dashboard
    )

    response = client.messages.create(
        model="auto",  # let FluxCompute decide
        messages=[{"role": "user", "content": "What is 2+2?"}],
    )

    print(response.fluxcompute.model_selected)  # claude-haiku-4-5-20251001
    print(response.fluxcompute.savings_usd)     # 0.0035
"""

__version__ = "0.3.0"

from fluxcompute.client import FluxClient
from fluxcompute.graph.types import TaskGraph, TaskNode
from fluxcompute.models import FluxResponse, FluxMetadata, FluxStreamChunk, ClassificationResult, CacheStats
from fluxcompute.plugins import FluxRecoveryNotInstalled

__all__ = [
    "FluxClient", "FluxResponse", "FluxMetadata", "FluxStreamChunk",
    "ClassificationResult", "CacheStats", "TaskGraph", "TaskNode",
    "FluxRecoveryNotInstalled",
]
