# Telemetry wire contract (v1)

This is the exact set of data the SDK sends when a `fluxcompute_key` is
configured, and the wire format it sends it in. The SDK owns this contract;
the hosted backend implements it.

Two independent channels, both batched, both fire-and-forget, neither able
to delay or fail an LLM call:

| Channel | Endpoint | Carries |
| ------- | -------- | ------- |
| Telemetry | `POST /v1/telemetry/event` | Per-request routing + cost metrics |
| Graph events | `POST /v1/graph/events` | Execution-graph node structure |
| Identity | `GET /v1/whoami` | Key validation, on demand only |

Both POST bodies are `{"events": [ ... ]}` with
`Authorization: Bearer <fluxcompute_key>`. A non-200 is logged and the
events are re-buffered; failures are never raised to the caller.

## Versioning

The `v1` in the path is the contract version. Fields may be **added** within
v1; existing fields will not change meaning or type, and removals require
`v2`. The server must be a tolerant reader: SDK versions in the wild lag
indefinitely, and an old SDK must keep reporting successfully against a
newer backend.

## Endpoint availability

This document specifies the wire protocol. It does not promise that the
default host is serving it: the hosted FluxCompute backend is in invite-only
early access, so `https://api.fluxcompute.dev` may not answer for you.

That is a safe state by design: both channels are fire-and-forget and
swallow transport failures, so an unreachable endpoint costs nothing but the
telemetry itself. The one place it surfaces is `FluxClient.verify()`, which
raises `ConnectionError` naming the endpoint (as opposed to `ValueError`,
which means the endpoint answered and rejected your key).

To send this traffic somewhere you control (a local dev server, or your own
implementation of this contract), set `FLUX_TELEMETRY_URL` and
`FLUX_GRAPH_EVENTS_URL` (see [Endpoint overrides](#endpoint-overrides)).

## Defaults, and what is never sent

- **Nothing is sent without a `fluxcompute_key`.** No key, no telemetry.
- **`telemetry=False`** disables both channels entirely.
- **Prompt and response text are not sent by default.** The fields marked
  *content* below are only populated when the client is constructed with
  `content_capture=True`; otherwise they are stripped before buffering, so
  they never leave the process.
- Provider API keys are never transmitted.

## `POST /v1/telemetry/event`

One event per routed request. No content, metrics only.

| Field | Type | Notes |
| ----- | ---- | ----- |
| `customer_key` | string | The FluxCompute key the client is using |
| `session_id` | string \| null | Set only if the caller passed one |
| `difficulty_score` | number | Classifier output, 0.0–1.0 |
| `difficulty_label` | string | `easy` \| `medium` \| `hard` |
| `model_selected` | string | Model actually dispatched to |
| `baseline_model` | string | Model savings are measured against |
| `input_tokens` | integer | |
| `output_tokens` | integer | |
| `cost_usd` | number | Actual cost of this call |
| `baseline_cost_usd` | number | Counterfactual cost at the baseline model |
| `savings_usd` | number | `baseline_cost_usd - cost_usd` |
| `classification_ms` | number | Time spent classifying |
| `overhead_ms` | number | Total SDK overhead added to the call |

Batching: 10 events or 30 seconds, whichever comes first.

## `POST /v1/graph/events`

One event per node transition inside a `task()` scope. A node may be sent
more than once (a live emit, then a replay on task exit); the server
deduplicates by `node_id` and treats later events as upserts.

| Field | Type | Notes |
| ----- | ---- | ----- |
| `node_id` | string | Stable per node; the dedup key |
| `task_id` | string | |
| `task_name` | string | |
| `name` | string | Node name as given to `task()` / `step()` |
| `node_type` | string | `task` \| `llm_call` \| `tool_call` |
| `parent_id` | string \| null | Auto-parented via context variables |
| `depends_on` | string[] | Links a retry to the node it replaces |
| `status` | string | `running` \| `succeeded` \| `failed` |
| `failure_reason` | string \| null | `context_overflow` \| `budget` \| `tool_error` \| `stall` \| `refusal` \| `unknown` |
| `model` | string \| null | |
| `input_tokens` | integer | |
| `output_tokens` | integer | |
| `cost_usd` | number | |
| `error` | string \| null | Exception text. A diagnostic, not model output. Sent regardless of `content_capture` |
| `session_id` | string \| null | |
| `started_at` | string \| null | ISO 8601 |
| `ended_at` | string \| null | ISO 8601 |
| `attributes` | object | Caller-supplied key/values |
| `events` | object[] | Timestamped node events |
| `output_preview` | string | **content**: first 500 chars of model output |
| `prompt_full` | string | **content**: full prompt, failed nodes only |
| `output_full` | string | **content**: full output, failed nodes only |

Batching: 20 events or 5 seconds. **A single POST carries at most 200
events**. The SDK chunks to this cap because an oversize batch would be
rejected whole. Servers implementing this contract must accept batches up
to that size.

If a flush fails, unsent events are re-buffered *without* the content
fields, so a failing network cannot pin large payloads in memory.

## `GET /v1/whoami`

Called only by `FluxClient.verify()`, a startup handshake for catching a
bad key early (the emitters deliberately swallow failures, so a typo would
otherwise surface only as a silent gap in the dashboard).

- `200` with a JSON body describing the key's account → valid
- `401` → the SDK raises `ValueError` naming the key as invalid

## Endpoint overrides

| Variable | Effect |
| -------- | ------ |
| `FLUX_TELEMETRY_URL` | Sets the telemetry endpoint. Highest precedence |
| `FLUX_GRAPH_EVENTS_URL` | Sets the graph-events endpoint. If `FLUX_TELEMETRY_URL` is unset, the telemetry endpoint's host is derived from it, as is `/v1/whoami` |

Point `FLUX_GRAPH_EVENTS_URL` at `http://localhost:8000/v1/graph/events` to
send everything to a local server during development.
