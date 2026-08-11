# Changelog

All notable changes to the FluxCompute SDK are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Before 1.0, a minor version bump may contain breaking changes.

## [0.3.0] — unreleased

First release from the public repository.

### Changed — licence

- **The licence is now Apache-2.0** (previously MIT). Apache-2.0 adds an
  explicit patent grant, which matters now that the project has commercial
  add-ons. It remains an OSI-approved permissive licence: use, modification,
  redistribution, and commercial use are all still allowed.
- Versions **0.1.0–0.2.1 were released under MIT and stay MIT** — that grant
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

- **`FluxClient.resume()` is free and built in.** Graph-aware resume —
  rebuilding minimal context from succeeded steps and re-running only the
  failed one — works fully in-process and offline whenever the process that
  ran the task still holds its graph. `build_resume_plan` and `ResumePlan`
  are now public on `fluxcompute.graph`.
- **`fluxcompute.plugins`** — an entry-point seam (`fluxcompute.plugins`
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
  preview of every model response — plus full prompt/output snapshots for
  failed nodes — to the FluxCompute backend. That is now opt-in via
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
