# Contributing

**We don't accept external code contributions to this repository.**

FluxCompute is developed by a small team, and the SDK is built alongside a
hosted service whose roadmap and internals aren't public. Reviewing and
maintaining outside patches against that isn't something we can do well right
now, and we'd rather say so plainly than leave pull requests sitting open.

Pull requests from outside the team will be closed unread. That isn't a
judgement on the contribution: it's a capacity decision. This is enforced
mechanically, not just by policy: PRs from anyone outside the maintainer
list fail a required check automatically.

If this policy ever changes, any external Contribution (through a pull
request or any other channel, including code pasted into an issue, email, or
elsewhere) requires a signed [Contributor License Agreement](.github/CLA.md)
first. Without one, incorporating outside code would mean FluxCompute could
never relicense that portion of the codebase without tracking down every
contributor's individual consent. Cheap to require up front, expensive to
retrofit years later. No such exception exists today.

## What is useful to us

**Bug reports.** If the SDK misroutes, miscalculates cost, crashes, or its
docs are wrong, we want to know. Open an issue with the SDK version
(`fluxcompute.__version__`), your Python version, a minimal reproduction, and
what you expected instead. Reports that come with a reproduction get fixed
much faster.

**Questions about behaviour.** If something is surprising, that's usually a
documentation bug on our side. Ask.

**Security issues.** Please don't open a public issue. Email
**security@fluxcompute.dev** and we'll respond as quickly as we can.

## Licence

This project is Apache-2.0 (see [LICENSE](LICENSE)). You're free to use, run,
modify, and redistribute it under those terms, including commercially: the
restriction above is about what lands in *this* repository, not about what you
may do with the code.

If you maintain a fork, you're welcome to; we just won't be merging from it.

## Building on it

Graph-aware resume (`FluxClient.resume()`) is free and built into the SDK:
it works fully offline whenever the process that ran the task still holds
its graph in memory. The SDK is otherwise deliberately extensible without
touching this repo: optional add-ons attach through the `fluxcompute.plugins`
entry-point group rather than through patches to the core. The
`fluxcompute-recovery` add-on uses that seam to add *durable* resume,
reconstructing a task's graph from wherever it was persisted, for when the
process that ran it is gone. The same seam is available to you; see
`fluxcompute/plugins.py` for the interface.

## Running it locally

```bash
pip install -e ".[dev]"
ruff check fluxcompute/ examples/
pytest tests/ -v
```

Tests use `asyncio_mode = "auto"`, so async tests need no decorator, and
nothing in the suite makes a network call: provider clients are mocked and
`tests/conftest.py` fails any test that reaches a real endpoint.
