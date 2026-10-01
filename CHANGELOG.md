# Changelog

All notable changes to the FluxCompute SDK are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Before 1.0, a minor version bump may contain breaking changes.

## [0.3.3] - 2026-10-01

### Fixed

- **gpt-5.x and gpt-6-luna requests were rejected with a 400.** The
  dispatcher decided which OpenAI parameters to send by model name, and only
  `o1`/`o3`/`o4` counted as reasoning models. Every other model, including
  `gpt-6-luna`, the easy tier `auto` has routed to since 0.3.2, was sent
  `max_tokens` and `temperature`, and reasoning models reject both. Every
  OpenAI model is now sent `max_completion_tokens`, which chat and reasoning
  models both accept (checked live on gpt-4o-mini, gpt-5.6-luna and
  gpt-6-luna). A model released later needs no SDK change.
- **A model that rejects one parameter no longer fails the call.** If a model
  answers with a 400 naming a parameter it doesn't support (OpenAI's
  `unsupported_parameter`/`unsupported_value`, or Anthropic's "`temperature`
  is deprecated for this model"), the SDK resends once without it, logs a
  warning, and leaves it out of later calls to that model in the process. A
  rejected request is not billed, so the resend costs nothing. Every other
  error, including a 400 for a value out of range, surfaces unchanged. Applies
  to `messages.create()` and `messages.stream()`.
- **An empty answer from a spent token limit was returned as `""`.** A
  reasoning model can spend its whole output limit on hidden reasoning and
  return no text with `finish_reason="length"`; the SDK returned that `""` and
  recorded the call as a success. It now raises `FluxEmptyResponseError`,
  which names the model, the limit, and how many tokens went to reasoning. The
  graph node is recorded as failed, with failure reason `token_budget`, and
  keeps the real tokens and cost, since they were billed. The empty turn is
  not added to session history. A tool call with no text is still a complete
  answer and does not raise.

### Added

- `FluxEmptyResponseError`, exported from `fluxcompute`.
- `FluxMetadata.truncated` (the answer stopped at `max_tokens` but has text;
  also logged as a warning) and `FluxMetadata.reasoning_tokens` (hidden
  reasoning billed inside the output tokens, for OpenAI and Gemini).
- `token_budget` failure reason in the telemetry contract.

### Changed

- **`temperature` now defaults to `None`, which leaves it out of the request**
  so the provider's own default applies. Previously `1.0` was always sent.
  The result is the same for every provider; reasoning models only accept the
  default.

## [0.3.2] - 2026-09-26

### Added

- **Google Gemini support** — pass `google_key` (or set `GOOGLE_API_KEY`) to
  route to Gemini 3 alongside Anthropic and OpenAI. `client.messages.create()`
  works end to end (classify → dispatch → cost/savings → session history);
  `client.messages.stream()` isn't wired up for Gemini yet and raises
  `NotImplementedError` with a clear message if you try.
- **Two worked agents under `examples/agents/`**: a company brain that keeps a
  document in sync with its sources, and a CRM inbox that turns mail into one
  row per conversation. Complete programs with fixtures and a `$0` test
  suite, runnable with a provider key alone; a `FLUXCOMPUTE_KEY` is optional
  and adds the dashboard link. PyYAML comes in via a new `examples` extra.
  `ruff` and the docs checks now cover `examples/`.

### Changed

- **Model tiers updated to the current generation.** Anthropic: medium/hard
  now route to `claude-sonnet-5`/`claude-opus-5-5` (was `claude-sonnet-4-6`/
  `claude-opus-4-8`). OpenAI: easy/medium/hard now route to
  `gpt-6-luna`/`gpt-6-sol`/`gpt-6-astra` (was `gpt-4o-mini`/`gpt-4o`/`o1`).
  Pricing for the previous-generation model IDs is kept in
  `fluxcompute/cost.py` for cost continuity on in-flight sessions and anyone
  pinning a model explicitly — only the classifier's "auto" routing changed.
- Default baseline models (`get_baseline_model`) updated to match: `anthropic`
  → `claude-opus-5-5`, `openai` → `gpt-6-astra`, plus a new `google` →
  `gemini-3.1-pro-preview` entry.

### Fixed

- **The system-prompt-length classifier signal crashed on list-shaped
  content.** Every other content-block path in `classify()` already went
  through `_message_text()` (added for the same reason on user messages,
  see `TestContentBlockMessages`), but the system-prompt signal still called
  `.split()` directly on `system_msgs[0]["content"]`, so a system prompt
  passed as content blocks (rather than a plain string) raised
  `AttributeError: 'list' object has no attribute 'split'`. Now routed
  through `_message_text()` like every other signal.
- **`GraphEmitter.flush()` and `TelemetryReporter.flush()` treated a
  non-200 HTTP response the same as a delivered batch.** Both only
  logged a warning on a bad status code, then still advanced past
  the batch as "sent" -- only a raised exception or cancellation
  triggered re-buffering. A backend outage (see the companion fix in
  fluxcompute-observability for roadmap #53, where a transient DB
  pool failure made `/v1/graph/events` return a real error status)
  meant the client silently dropped the events it had already
  buffered instead of retrying them once the backend recovered. Both
  now re-buffer on `>= 500` responses and retry on the next flush
  cycle; a 4xx (payload the server will never accept, e.g. an
  oversized batch) still logs and drops, since retrying it changes
  nothing.

## [0.3.1] - 2026-09-06

### Fixed

- **Pinned `anthropic<1.0.0` and `openai<3.0.0`.** The unbounded
  `anthropic>=0.30.0` spec in 0.3.0 let a plain `pip install` resolve
  into `anthropic` 1.0.0+, which removed `temperature` from
  `AsyncMessages.create()`. The dispatcher always passes `temperature`
  with no `**kwargs` to absorb it, so every real (non-mocked) routed
  call raised `TypeError` on a fresh install. Added
  `tests/test_provider_sdk_compat.py`, which introspects the actually
  installed provider SDKs against the exact kwargs the dispatchers
  send, so a future breaking change like this fails CI instead of
  only a real install.

### Changed

- Em dashes removed from README, CONTRIBUTING, this changelog, the
  CLA, and the telemetry contract doc.

## [0.3.0] - 2026-08-11

First release from the public repository.

### Licence

- **The licence is now Apache-2.0** (previously MIT). Apache-2.0 adds an
  explicit patent grant, which matters now that the project has commercial
  add-ons. It remains an OSI-approved permissive licence: use, modification,
  redistribution, and commercial use are all still allowed.
- Versions **0.1.0–0.2.1 were released under MIT and stay MIT**. That grant
  is irrevocable and unaffected by this change.
- We no longer accept external code contributions. See `CONTRIBUTING.md`;
  bug reports and security disclosures remain welcome.

### Breaking

- **`fluxcompute.intelligence` removed.** It held server-side background
  components that ran against a database the SDK never talks to; nothing in
  the SDK imported them. They belong to the hosted service.
- **`fluxcompute.state.redis_session` removed.** `RedisSessionManager` was
  never exported from `fluxcompute.state` and required the `server` extra's
  Redis dependency. In-memory `SessionManager` is unchanged.
- **The `server` extra is gone.** This package ships the SDK only; the hosted
  service is a separate product.

### Added

- **`FluxClient.resume()` is free and built in.** Graph-aware resume,
  rebuilding minimal context from succeeded steps and re-running only the
  failed one, works fully in-process and offline whenever the process that
  ran the task still holds its graph. `build_resume_plan` and `ResumePlan`
  are now public on `fluxcompute.graph`.
- **`fluxcompute.plugins`**: an entry-point seam (`fluxcompute.plugins`
  group) for optional add-ons. Its role narrowed from "provide the resume
  algorithm" to "provide durability": a plugin's `fetch_graph(task_id)`
  reconstructs a graph this process didn't record, for resuming a task after
  the process that ran it has exited. `resume()` only reaches this path when
  the graph isn't held locally, and raises `FluxRecoveryNotInstalled` (still
  exported from the top-level package) if no plugin can provide it. The SDK
  never imports a plugin package directly.
- **`content_capture` flag** on `FluxClient` and `GraphEmitter`.

### Changed

- **Response content is no longer sent by default.** Previously, setting a
  `fluxcompute_key` and opening a `task()` scope also shipped a 500-character
  preview of every model response, plus full prompt/output snapshots for
  failed nodes, to the FluxCompute backend. That is now opt-in via
  `content_capture=True`. Telemetry still reports routing decisions, token
  counts, cost, latency, and execution-graph *structure* (node names, types,
  parentage, status, failure reason); error strings still ship, since they
  are diagnostics rather than model output. The local in-process graph is
  unaffected and retains full output.

### Fixed

- **The sdist no longer ships a partial test suite.** setuptools' defaults
  swept in `tests/test*.py` but not `tests/conftest.py` or `examples/`, so the
  shipped tests ran without the guard that stops them reaching the real
  backend, and the notebook checks globbed a directory that wasn't there and
  passed without testing anything. The sdist now ships the package; the tests
  live in the repository.
- The periodic graph-event flush no longer emits a spurious `RuntimeWarning`
  ("coroutine `_flush_loop` was never awaited") when the SDK records events
  outside a running event loop.

## Earlier releases

Versions before 0.3.0 were published from a private repository, under MIT.
All three were **yanked** on 2026-08-10: they shipped modules that belong to
the hosted service, not the SDK. Start at 0.3.0.

| Version | Released | Status |
| ------- | ---------- | ------ |
| 0.2.1   | 2026-07-05 | yanked |
| 0.2.0   | 2026-07-05 | yanked |
| 0.1.0   | 2026-06-21 | yanked |
