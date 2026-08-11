# FluxCompute SDK

[![CI](https://github.com/fluxcompute/fluxcompute-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/fluxcompute/fluxcompute-sdk/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/fluxcompute.svg)](https://pypi.org/project/fluxcompute/)
[![Python](https://img.shields.io/pypi/pyversions/fluxcompute.svg)](https://pypi.org/project/fluxcompute/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Route every query to the cheapest model that can answer it.** FluxCompute
classifies each request and dispatches it to the right tier — so simple
questions stop costing frontier-model prices — and reports exactly what you
saved on every call.

```bash
pip install fluxcompute
```

```python
import asyncio
from fluxcompute import FluxClient


async def main():
    client = FluxClient(anthropic_key="sk-ant-...")

    response = await client.messages.create(
        model="auto",                                    # let FluxCompute decide
        messages=[{"role": "user", "content": "What is 2+2?"}],
    )

    print(response.text)                          # "4"
    print(response.fluxcompute.model_selected)    # claude-haiku-4-5
    print(response.fluxcompute.savings_usd)       # 0.0035

    await client.close()


asyncio.run(main())
```

Already using the Anthropic or OpenAI SDK? Swap the client and pass
`model="auto"` — the response object keeps the fields you already use, with
routing and cost data added under `response.fluxcompute`.

## How routing works

Each request gets a difficulty score from a heuristic classifier (prompt
structure, reasoning markers, length, task type), which maps to a tier:

| Score | Tier | Anthropic | OpenAI |
| ------------- | ------ | ----------------- | ------------- |
| `< 0.18` | easy | `claude-haiku-4-5` | `gpt-4o-mini` |
| `0.18` – `0.45` | medium | `claude-sonnet-4-6` | `gpt-4o` |
| `>= 0.45` | hard | `claude-opus-4-8` | `o1` |

Savings come from the queries that get downgraded: against the default
baseline, an easy-tier call costs **80% less** and a medium-tier call **40%
less**. Hard queries route to the baseline model itself, so they save
nothing by design — your overall reduction is therefore a function of how
much of your traffic genuinely needs a frontier model. The examples measure
this on a mixed workload rather than asserting a number.

Every response carries the score, the label, the model chosen, and the
counterfactual cost against your baseline model, so the routing is auditable
rather than a black box. Pin a specific model any time by passing it instead
of `"auto"`.

Multi-turn sessions get tier-aware context compression: pass a `session_id`
and history is carried across turns and compressed when a session drops to a
cheaper tier, so switching models mid-conversation doesn't re-bill the full
transcript.

## Execution graphs

Wrap work in a `task()` scope and the SDK records a DAG of everything that
happened inside it — LLM calls and your own steps, auto-parented, with
status, timings, tokens, cost, and a rules-based failure classification
(context overflow, budget, tool error, stall, refusal).

```python
async def research(client):
    with client.task("market-research") as t:
        await client.messages.create(
            model="auto",
            messages=[{"role": "user", "content": "Who are the top 3 competitors?"}],
        )

        try:
            with client.step("fetch-pricing"):
                raise TimeoutError("pricing API timed out")
        except TimeoutError:
            pass  # a step can fail without killing the whole task

    # Re-runs only the failed step, with a rebuilt minimal context — not a
    # replay of the whole transcript. The retry lands in the same graph,
    # linked to the node it replaces via depends_on.
    response = await client.resume(t.task_id)

    return client.get_task_graph(t.task_id)
```

It works in any framework — no integration code — because parenting uses
context variables rather than a wrapper API. A zero-touch
[LangGraph](https://langchain-ai.github.io/langgraph/) adapter is included
(`FluxCheckpointer`), which mirrors super-steps into the same task graph.

Recording and resume are both **free and fully offline** — `client.resume()`
works whenever the process that ran the task still holds its graph, with no
network call beyond the provider request itself. What's paid is *durability*:
if that process has since exited, resuming means reconstructing the graph
from wherever it was persisted, which needs telemetry to have been on and
the hosted platform to have kept a copy. Without both,
`client.resume()` on a task from a dead process raises
`FluxRecoveryNotInstalled`. See [fluxcompute.dev](https://fluxcompute.dev)
for the durable-recovery and dashboard offering.

## Telemetry & privacy

> The hosted FluxCompute dashboard is in **invite-only early access** — the
> public telemetry endpoint is not yet generally available, and without a
> reachable endpoint the SDK's telemetry is a silent no-op (it never delays
> or fails your LLM calls). Self-hosted and dev deployments can point the SDK
> anywhere via `FLUX_TELEMETRY_URL` / `FLUX_GRAPH_EVENTS_URL`. For dashboard
> access, get in touch at [fluxcompute.dev](https://fluxcompute.dev).

Telemetry is **off unless you pass a `fluxcompute_key`**. With one set, the
SDK reports:

- routing decisions, difficulty scores, token counts, cost, latency
- execution-graph **structure**: node names, types, parentage, status,
  failure reason, model, timings — and error strings, which are diagnostics

**Prompt and response text are never sent by default.** To include model
output in the dashboard (a 500-character preview per node, plus full
prompt/output snapshots for failed nodes), opt in explicitly:

```python
client = FluxClient(anthropic_key="...", fluxcompute_key="flx_...",
                    content_capture=True)
```

Turn everything off with `telemetry=False`. Content filtering happens before
anything leaves the process; your local graph always keeps full output.

## Configuration

Keys are read from the environment when not passed explicitly, matching the
Anthropic and OpenAI SDK conventions:

| Variable | Purpose |
| ---------------------- | -------------------------------------------- |
| `ANTHROPIC_API_KEY` | Anthropic provider key |
| `OPENAI_API_KEY` | OpenAI provider key |
| `FLUXCOMPUTE_KEY` | FluxCompute key — enables telemetry |
| `FLUX_TELEMETRY_URL` | Override the telemetry endpoint |
| `FLUX_GRAPH_EVENTS_URL` | Override the graph-events endpoint |

The wire format the SDK sends is documented and versioned in
[docs/telemetry-contract.md](docs/telemetry-contract.md).

## Examples

- [`examples/quickstart.ipynb`](examples/quickstart.ipynb) — first routed
  call and measured savings, ~5 minutes
- [`examples/full_walkthrough.ipynb`](examples/full_walkthrough.ipynb) —
  routing tiers, sessions, streaming, migrating from the Anthropic SDK

## Development

```bash
pip install -e ".[dev]"
ruff check fluxcompute/
pytest tests/ -v
```

We don't take external code contributions, but bug reports are genuinely
useful — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0 — see [LICENSE](LICENSE). Versions 0.1.0–0.2.1 were released
under MIT and remain so.

We don't accept external code contributions — see
[CONTRIBUTING.md](CONTRIBUTING.md) for what is useful to us (bug reports,
and security issues to security@fluxcompute.dev).
