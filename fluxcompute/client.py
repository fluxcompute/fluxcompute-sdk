"""
FluxClient — drop-in replacement for Anthropic/OpenAI SDKs.

Usage:
    client = FluxClient(anthropic_key="sk-ant-xxx")
    response = await client.messages.create(
        model="auto",
        messages=[{"role": "user", "content": "Hello"}],
    )
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]

from fluxcompute.classifier.heuristic import classify
from fluxcompute.cost import calculate_savings, get_baseline_model
from fluxcompute.graph.emitter import GraphEmitter
from fluxcompute.graph.recorder import GraphRecorder, NodeScope
from fluxcompute.graph.resume import build_resume_plan
from fluxcompute.graph.types import TaskGraph
from fluxcompute.models import CacheStats, FluxMetadata, FluxResponse, FluxStreamChunk, TelemetryEvent
from fluxcompute.plugins import FluxRecoveryNotInstalled, get_recovery_plugin
from fluxcompute.router.dispatcher import (
    _openai_stream_text,
    _openai_token_kwargs,
    dispatch_anthropic,
    dispatch_openai,
)
from fluxcompute.state.cache_manager import CacheManager
from fluxcompute.state.context_builder import ContextBuilder
from fluxcompute.state.session import SessionManager
from fluxcompute.telemetry.reporter import TelemetryReporter


class _MessagesAPI:
    """
    Mimics the anthropic.messages / openai.chat.completions interface.
    """

    def __init__(self, client: "FluxClient"):
        self._client = client

    async def create(
        self,
        *,
        model: str = "auto",
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 1.0,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> FluxResponse:
        """
        Create a chat completion with automatic model routing.

        Args:
            model: "auto" to let FluxCompute decide, or a specific model name.
            messages: List of message dicts (OpenAI/Anthropic format).
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.
            session_id: Optional session ID for multi-turn state tracking.
            **kwargs: Additional provider-specific arguments.

        Returns:
            FluxResponse with raw provider response + FluxCompute metadata.
        """
        return await self._client._route_and_execute(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            session_id=session_id,
            **kwargs,
        )

    def stream(
        self,
        *,
        model: str = "auto",
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 1.0,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> "_FluxStreamContext":
        """
        Stream a chat completion with automatic model routing.

        Returns an async context manager. Iterate over it to get FluxStreamChunk
        objects. After the loop, stream.fluxcompute holds routing metadata.

        Example:
            async with client.messages.stream(model="auto", messages=[...]) as s:
                async for chunk in s:
                    print(chunk.text, end="")
            print(s.fluxcompute.model_selected)
        """
        return _FluxStreamContext(
            self._client,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            session_id=session_id,
            **kwargs,
        )


class _FluxStreamContext:
    """
    Async context manager returned by client.messages.stream().

    Usage:
        async with client.messages.stream(model="auto", messages=[...]) as stream:
            async for chunk in stream:
                print(chunk.text, end="")
        print(stream.fluxcompute)   # populated after iteration completes
    """

    def __init__(self, client: "FluxClient", **kwargs):
        self._client = client
        self._kwargs = kwargs
        self._gen = None
        self.fluxcompute: Optional[FluxMetadata] = None

    async def __aenter__(self) -> "_FluxStreamContext":
        self._gen = self._client._stream_execute(**self._kwargs)
        return self

    async def __aexit__(self, *args) -> None:
        if self._gen is not None:
            await self._gen.aclose()

    def __aiter__(self):
        return self

    async def __anext__(self) -> "FluxStreamChunk":
        item = await self._gen.__anext__()   # raises StopAsyncIteration naturally
        if isinstance(item, FluxMetadata):
            self.fluxcompute = item
            raise StopAsyncIteration
        return item


class FluxClient:
    """
    Drop-in replacement for Anthropic/OpenAI SDKs with automatic
    model routing, cost optimization, and session tracking.
    """

    def __init__(
        self,
        *,
        anthropic_key: Optional[str] = None,
        openai_key: Optional[str] = None,
        fluxcompute_key: Optional[str] = None,
        baseline_model: Optional[str] = None,
        telemetry: bool = True,
        content_capture: bool = False,
        provider: Optional[str] = None,
    ):
        """
        Initialise FluxClient.

        Args:
            anthropic_key: Anthropic API key. Required for Anthropic models.
            openai_key: OpenAI API key. Required for OpenAI models.
            fluxcompute_key: FluxCompute key for telemetry + dashboard (optional).
                Falls back to the FLUXCOMPUTE_KEY environment variable.
            baseline_model: The model to compare savings against (defaults to most expensive).
            telemetry: Whether to send anonymised metrics to the FluxCompute
                backend: routing decisions, token counts, cost, latency, and —
                inside a task() scope — the execution graph's structure (node
                names, parentage, status, failure reason). No prompt or
                response text. Only sent when fluxcompute_key is set.
            content_capture: Also ship response content — the first 500 chars of
                each LLM output, plus full prompt/output snapshots for failed
                nodes — so the dashboard can show what a step actually produced.
                Off by default: this sends model input/output off your machine.
                Requires telemetry and fluxcompute_key.
            provider: Force "anthropic" or "openai". Auto-detected if only one key given.
        """
        # Fall back to environment variables (same convention as Anthropic/OpenAI SDKs)
        anthropic_key = anthropic_key or os.environ.get("ANTHROPIC_API_KEY")
        openai_key = openai_key or os.environ.get("OPENAI_API_KEY")
        fluxcompute_key = fluxcompute_key or os.environ.get("FLUXCOMPUTE_KEY")

        # Determine provider
        if provider:
            self._provider = provider
        elif anthropic_key and not openai_key:
            self._provider = "anthropic"
        elif openai_key and not anthropic_key:
            self._provider = "openai"
        elif anthropic_key and openai_key:
            self._provider = "anthropic"  # default when both provided
        else:
            raise ValueError(
                "No API key found. Pass anthropic_key/openai_key or set "
                "ANTHROPIC_API_KEY / OPENAI_API_KEY in your environment."
            )

        # Initialise provider clients
        self._anthropic_client = None
        self._openai_client = None

        if anthropic_key:
            self._anthropic_client = anthropic.AsyncAnthropic(api_key=anthropic_key)

        if openai_key:
            import openai
            self._openai_client = openai.AsyncOpenAI(api_key=openai_key)

        # Baseline model for cost comparison
        self._baseline_model = baseline_model or get_baseline_model(self._provider)

        # Session manager
        self._sessions = SessionManager()

        # Context compression and prompt-cache managers
        self._context_builder = ContextBuilder()
        self._cache_manager = CacheManager()

        # Telemetry
        self._telemetry = TelemetryReporter(
            fluxcompute_key=fluxcompute_key,
            enabled=telemetry and fluxcompute_key is not None,
        )
        self._fluxcompute_key = fluxcompute_key or ""

        # Execution graph (observability + resume).
        self._graph_emitter = GraphEmitter(
            fluxcompute_key=fluxcompute_key,
            enabled=telemetry and fluxcompute_key is not None,
            content_capture=content_capture,
        )
        self._graph = GraphRecorder(emit=self._graph_emitter.record)

        # Public API
        self.messages = _MessagesAPI(self)

    async def verify(self) -> Dict[str, Any]:
        """
        Loud, synchronous handshake: confirms fluxcompute_key resolves to a
        valid account before any task()/step() traffic is sent.

        GraphEmitter is deliberately non-blocking and swallows failures (a bad
        key must never delay or break an LLM call) — which means a typo'd key
        otherwise shows up only as a silent gap in the dashboard. Call this
        once at startup instead: it raises immediately on an invalid key.
        """
        if not self._fluxcompute_key:
            raise ValueError(
                "verify() requires fluxcompute_key — pass it to FluxClient() "
                "or set the FLUXCOMPUTE_KEY environment variable."
            )
        url = self._graph_emitter.whoami_url()
        client = await self._graph_emitter._get_client()
        try:
            response = await client.get(
                url, headers={"Authorization": f"Bearer {self._fluxcompute_key}"},
            )
        except httpx.RequestError as exc:
            raise ConnectionError(
                f"Could not reach the FluxCompute API at {url}: {exc}. "
                "Check network access, or set FLUX_GRAPH_EVENTS_URL to point at "
                "your own deployment."
            ) from exc
        if response.status_code == 401:
            raise ValueError(
                "Invalid FluxCompute key — check fluxcompute_key / FLUXCOMPUTE_KEY."
            )
        if response.status_code >= 400:
            # Anything else (404 from a wrong/undeployed host, 5xx from an
            # outage) is an endpoint problem, not a key problem. Saying so
            # beats leaking a raw httpx error out of a diagnostic helper.
            raise ConnectionError(
                f"FluxCompute API at {url} returned HTTP {response.status_code}. "
                "The key was not validated; this looks like an endpoint or "
                "service problem rather than a bad key."
            )
        try:
            return response.json()
        except ValueError as exc:
            raise ConnectionError(
                f"FluxCompute API at {url} returned a non-JSON response; "
                "the endpoint may be misconfigured."
            ) from exc

    def task(self, name: str, task_id: Optional[str] = None) -> NodeScope:
        """
        Open a task scope. Every LLM call made inside (and every step())
        is recorded as a node in this task's execution graph.

        Usage:
            with client.task("market-research") as t:
                await client.messages.create(...)
            graph = client.get_task_graph(t.task_id)
        """
        return self._graph.task(name, task_id)

    def step(self, name: str, depends_on: Optional[List[str]] = None) -> NodeScope:
        """Record a tool/custom step inside the current task scope."""
        return self._graph.step(name, depends_on)

    def get_task_graph(self, task_id: str) -> Optional[TaskGraph]:
        """The in-memory execution graph for a task, or None if unknown."""
        return self._graph.get_graph(task_id)

    async def resume(
        self,
        task_id: str,
        node_id: Optional[str] = None,
        model: str = "auto",
        instruction: Optional[str] = None,
    ) -> FluxResponse:
        """
        Resume a failed task: rebuild minimal context from succeeded steps and
        re-run only the failed step. Routing may escalate the model ("auto")
        or you can force one. The new call is recorded in the same graph,
        linked to the node it replaces via depends_on.

        Works fully in-process and offline when this process still holds the
        task's graph — the common case, since task() ran here. If it doesn't
        (the process that ran the task has since exited), resume falls back
        to a registered recovery plugin to fetch the graph from wherever it
        was durably persisted. That needs telemetry to have been on when the
        task ran (so the server had a copy to persist) and a recovery plugin
        (see fluxcompute.plugins) installed to fetch it back —
        FluxRecoveryNotInstalled otherwise.

        Assumes the task's original task()/step() scope has already exited in
        this same execution context — calling resume() concurrently with a
        still-open scope for the same task_id is undefined behavior.
        """
        graph = self._graph.get_graph(task_id)
        if graph is None:
            graph = await self._fetch_durable_graph(task_id)
        plan = build_resume_plan(graph, node_id)
        failed = plan.failed_node
        retry_prompt = instruction or (
            f"The step '{failed.name}' failed with: {failed.error or 'unknown error'}. "
            "Complete this step now."
        )
        messages = plan.context_messages + [{"role": "user", "content": retry_prompt}]
        # session_id=None: a resume must never inherit the failed call's session
        # history — that would silently reintroduce full-transcript replay on
        # top of the deliberately minimal reconstructed context above.
        with self._graph.attach(task_id, parent_id=failed.parent_id):
            nodes_before = len(graph.in_order())
            try:
                response = await self._route_and_execute(
                    model=model, messages=messages, session_id=None,
                )
            finally:
                # Link the retry chain even if this attempt also fails — an
                # unlinked failed node would otherwise disconnect from the
                # failure it was retrying. Only if a node was actually added:
                # a pre-dispatch failure (classify/context-builder) or a
                # BaseException like CancelledError never reaches
                # record_llm_call, so graph.in_order()[-1] would just be the
                # failed node itself — appending would create a self-loop.
                if len(graph.in_order()) > nodes_before:
                    new_node = graph.in_order()[-1]
                    if new_node.node_type == "llm_call" and failed.node_id not in new_node.depends_on:
                        new_node.depends_on.append(failed.node_id)
        return response

    async def _fetch_durable_graph(self, task_id: str) -> TaskGraph:
        """Cross-process fallback for resume(): this process doesn't hold
        task_id, so ask a registered recovery plugin to reconstruct it from
        wherever it was durably persisted.

        Two distinct failure modes, deliberately not collapsed into one:
        no plugin installed means we have no way to even check whether the
        task exists elsewhere, so FluxRecoveryNotInstalled tells the caller
        what to do about it. A plugin that IS installed and still can't find
        the task means it's genuinely unknown — plain KeyError, same as the
        in-process case.
        """
        try:
            plugin = get_recovery_plugin()
        except FluxRecoveryNotInstalled:
            raise
        graph = await plugin.fetch_graph(task_id)
        if graph is None:
            raise KeyError(f"Unknown task_id: {task_id}")
        self._graph.adopt_graph(graph)
        return graph

    async def _route_and_execute(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 1.0,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> FluxResponse:
        """Core routing logic: classify → select → execute → log."""
        total_start = time.monotonic()

        # Generate session ID if not provided
        if session_id is None:
            session_id = f"fc_{uuid.uuid4().hex[:12]}"

        # Get session context for multi-turn
        session = self._sessions.get_or_create(session_id)
        session_history = session.conversation_history.copy()
        full_messages = session_history + messages if session_history else messages

        # Classify
        classification = classify(full_messages, provider=self._provider)
        selected_model = classification.model if model == "auto" else model

        # Context compression (L2)
        full_size = sum(len(m.get("content") or "") for m in full_messages)
        compressed = self._context_builder.build(
            current_messages=messages,
            session_history=session_history,
            difficulty_label=classification.label,
        )
        comp_size = sum(len(m.get("content") or "") for m in compressed)
        context_compression = max(0.0, 1.0 - comp_size / max(full_size, 1))

        # Prompt-cache markers (Anthropic only)
        cache_messages, cache_system = compressed, None
        if self._provider == "anthropic":
            cache_messages, cache_system = self._cache_manager.prepare_for_anthropic(
                messages=compressed,
                session_history=session_history,
            )

        # Execute
        dispatch_start = time.monotonic()
        try:
            if self._provider == "anthropic":
                if self._anthropic_client is None:
                    raise ValueError("Anthropic client not initialised. Provide anthropic_key.")
                result = await dispatch_anthropic(
                    client=self._anthropic_client,
                    model=selected_model,
                    messages=cache_messages,
                    system=cache_system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
            else:
                if self._openai_client is None:
                    raise ValueError("OpenAI client not initialised. Provide openai_key.")
                result = await dispatch_openai(
                    client=self._openai_client,
                    model=selected_model,
                    messages=compressed,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
        except Exception as exc:
            # Record the failed call in the task graph (no-op outside a task scope)
            self._graph.record_llm_call(
                model=selected_model, input_tokens=0, output_tokens=0, cost_usd=0.0,
                output_preview="", session_id=session_id,
                error=f"{type(exc).__name__}: {exc}",
                attributes={
                    "difficulty_score": classification.score,
                    "difficulty_label": classification.label,
                },
                events=[{
                    "type": "attempt",
                    "model": selected_model,
                    "latency_ms": int((time.monotonic() - dispatch_start) * 1000),
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                }],
                prompt_full=json.dumps(cache_messages),
            )
            raise

        # Calculate costs
        actual_cost, baseline_cost, savings = calculate_savings(
            model_used=selected_model,
            baseline_model=self._baseline_model,
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            cache_write_tokens=result.get("cache_write_tokens", 0),
            cache_read_tokens=result.get("cache_read_tokens", 0),
        )

        total_ms = (time.monotonic() - total_start) * 1000
        overhead_ms = total_ms - result["response_ms"]

        # Build metadata
        metadata = FluxMetadata(
            difficulty_score=classification.score,
            difficulty_label=classification.label,
            model_selected=selected_model,
            baseline_model=self._baseline_model,
            cost_usd=actual_cost,
            baseline_cost_usd=baseline_cost,
            savings_usd=savings,
            classification_ms=classification.classification_ms,
            overhead_ms=round(overhead_ms, 1),
            session_id=session_id,
            context_compression=round(context_compression, 3),
            cache=CacheStats(
                cache_write_tokens=result.get("cache_write_tokens", 0),
                cache_read_tokens=result.get("cache_read_tokens", 0),
                cache_hit=result.get("cache_read_tokens", 0) > 0,
            ),
        )

        # Update session state
        user_msg = messages[-1] if messages else {"role": "user", "content": ""}
        assistant_content = ""
        if self._provider == "anthropic":
            for block in result["response"].content:
                if hasattr(block, "text"):
                    assistant_content = block.text
                    break
        else:
            assistant_content = result["response"].choices[0].message.content or ""

        self._graph.record_llm_call(
            model=selected_model,
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            cost_usd=actual_cost,
            output_preview=assistant_content,
            session_id=session_id,
            attributes={
                "difficulty_score": classification.score,
                "difficulty_label": classification.label,
                "overhead_ms": round(overhead_ms, 1),
                "context_compression": round(context_compression, 3),
                "cache_read_tokens": result.get("cache_read_tokens", 0),
                "cache_write_tokens": result.get("cache_write_tokens", 0),
                "cache_hit": result.get("cache_read_tokens", 0) > 0,
            },
            events=[{
                "type": "attempt",
                "model": selected_model,
                "latency_ms": int(result["response_ms"]),
                "ok": True,
                "error": None,
            }],
            prompt_full=json.dumps(cache_messages),
            output_full=assistant_content,
        )

        self._sessions.update(
            session_id=session_id,
            user_message=user_msg,
            assistant_message={"role": "assistant", "content": assistant_content},
            model_used=selected_model,
            cost_usd=actual_cost,
            savings_usd=savings,
        )

        # Send telemetry (async, non-blocking)
        self._telemetry.record(TelemetryEvent(
            customer_key=self._fluxcompute_key,
            session_id=session_id,
            difficulty_score=classification.score,
            difficulty_label=classification.label,
            model_selected=selected_model,
            baseline_model=self._baseline_model,
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            cost_usd=actual_cost,
            baseline_cost_usd=baseline_cost,
            savings_usd=savings,
            classification_ms=classification.classification_ms,
            overhead_ms=round(overhead_ms, 1),
        ))

        return FluxResponse(
            raw=result["response"],
            fluxcompute=metadata,
            provider=self._provider,
        )

    async def _stream_execute(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 1.0,
        session_id: Optional[str] = None,
        **kwargs,
    ):
        """
        Async generator: yields FluxStreamChunk per text delta, then one FluxMetadata sentinel.
        _FluxStreamContext.__anext__ catches the FluxMetadata to stop iteration and expose metadata.
        """
        session_id = session_id or f"fc_{uuid.uuid4().hex[:12]}"
        session = self._sessions.get_or_create(session_id)
        session_history = session.conversation_history.copy()
        full_messages = session_history + messages if session_history else messages

        classification = classify(full_messages, provider=self._provider)
        selected_model = classification.model if model == "auto" else model

        compressed = self._context_builder.build(
            current_messages=messages,
            session_history=session_history,
            difficulty_label=classification.label,
        )
        full_size = sum(len(m.get("content") or "") for m in full_messages)
        comp_size = sum(len(m.get("content") or "") for m in compressed)
        context_compression = max(0.0, 1.0 - comp_size / max(full_size, 1))

        cache_messages, cache_system = compressed, None
        if self._provider == "anthropic":
            cache_messages, cache_system = self._cache_manager.prepare_for_anthropic(
                messages=compressed,
                session_history=session_history,
            )

        total_start = time.monotonic()
        assistant_content = ""
        input_tokens = output_tokens = cache_write = cache_read = 0
        ttft_ms: Optional[float] = None

        if self._provider == "anthropic":
            if self._anthropic_client is None:
                raise ValueError("Anthropic client not initialised. Provide anthropic_key.")
            create_kwargs = {
                "model": selected_model,
                "messages": cache_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs,
            }
            if cache_system:
                create_kwargs["system"] = cache_system
            async with self._anthropic_client.messages.stream(**create_kwargs) as stream:
                async for text in stream.text_stream:
                    if ttft_ms is None:
                        ttft_ms = round((time.monotonic() - total_start) * 1000, 1)
                    assistant_content += text
                    yield FluxStreamChunk(text=text, model=selected_model)
                final = await stream.get_final_message()
            usage = final.usage
            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens
            cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0

        else:
            if self._openai_client is None:
                raise ValueError("OpenAI client not initialised. Provide openai_key.")
            async with self._openai_client.chat.completions.stream(
                model=selected_model,
                messages=compressed,
                # Same reasoning-model adaptation as the non-streaming path:
                # o-series models take max_completion_tokens and reject a
                # non-default temperature, so o1 (the OpenAI hard tier) would
                # otherwise be rejected outright when streaming.
                **_openai_token_kwargs(selected_model, max_tokens, temperature),
                **kwargs,
            ) as stream:
                # chat.completions.stream() yields typed ChatCompletionStreamEvent
                # objects, not raw chunks — none of them has `.choices`.
                async for event in stream:
                    text = _openai_stream_text(event)
                    if not text:
                        continue
                    if ttft_ms is None:
                        ttft_ms = round((time.monotonic() - total_start) * 1000, 1)
                    assistant_content += text
                    yield FluxStreamChunk(text=text, model=selected_model)
                final = await stream.get_final_completion()
            usage = getattr(final, "usage", None)
            if usage:
                input_tokens = usage.prompt_tokens
                output_tokens = usage.completion_tokens

        actual_cost, baseline_cost, savings = calculate_savings(
            model_used=selected_model,
            baseline_model=self._baseline_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_write_tokens=cache_write,
            cache_read_tokens=cache_read,
        )
        overhead_ms = (time.monotonic() - total_start) * 1000

        self._graph.record_llm_call(
            model=selected_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=actual_cost,
            output_preview=assistant_content,
            session_id=session_id,
            attributes={
                "difficulty_score": classification.score,
                "difficulty_label": classification.label,
                "ttft_ms": ttft_ms,
                "context_compression": round(context_compression, 3),
                "cache_read_tokens": cache_read,
                "cache_write_tokens": cache_write,
                "cache_hit": cache_read > 0,
            },
            events=[{
                "type": "attempt",
                "model": selected_model,
                "latency_ms": int(overhead_ms),
                "ok": True,
                "error": None,
            }],
            prompt_full=json.dumps(cache_messages),
            output_full=assistant_content,
        )

        # Update session state
        user_msg = messages[-1] if messages else {"role": "user", "content": ""}
        self._sessions.update(
            session_id=session_id,
            user_message=user_msg,
            assistant_message={"role": "assistant", "content": assistant_content},
            model_used=selected_model,
            cost_usd=actual_cost,
            savings_usd=savings,
        )

        self._telemetry.record(TelemetryEvent(
            customer_key=self._fluxcompute_key,
            session_id=session_id,
            difficulty_score=classification.score,
            difficulty_label=classification.label,
            model_selected=selected_model,
            baseline_model=self._baseline_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=actual_cost,
            baseline_cost_usd=baseline_cost,
            savings_usd=savings,
            classification_ms=classification.classification_ms,
            overhead_ms=round(overhead_ms, 1),
        ))

        # Yield metadata sentinel — _FluxStreamContext catches this, stops iteration, exposes metadata
        yield FluxMetadata(
            difficulty_score=classification.score,
            difficulty_label=classification.label,
            model_selected=selected_model,
            baseline_model=self._baseline_model,
            cost_usd=actual_cost,
            baseline_cost_usd=baseline_cost,
            savings_usd=savings,
            classification_ms=classification.classification_ms,
            overhead_ms=round(overhead_ms, 1),
            session_id=session_id,
            context_compression=round(context_compression, 3),
            cache=CacheStats(
                cache_write_tokens=cache_write,
                cache_read_tokens=cache_read,
                cache_hit=cache_read > 0,
            ),
        )

    async def close(self) -> None:
        """Flush telemetry and clean up resources."""
        await self._telemetry.close()
        await self._graph_emitter.close()
        if self._anthropic_client:
            await self._anthropic_client.close()
        if self._openai_client:
            await self._openai_client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
