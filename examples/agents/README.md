# Worked agents

Two complete agents, not snippets. Each runs against a real provider key exactly as your own
code would, keeps its own state in SQLite, and prints a scorecard with gates at the end. A
patched-provider test suite under `tests/` runs them for $0 in CI.

The pattern they share: **a model does one bounded job; code decides everything with a
consequence.** The model turns a paragraph into typed facts, or classifies a message into a
fixed set. Code owns change detection, deduplication, precedence, gates, priority, thresholds,
idempotency and every write.

| Agent | What it does |
|---|---|
| `company_brain/` | Keeps one versioned document in sync with a repository snapshot, an inbox and a brand file, with a verbatim quote behind every fact and facts removed when their source stops asserting them. An unchanged run makes zero model calls. Includes a deliberate provider failure and its in-process `resume()`. |
| `crm_inbox/` | Turns a mailbox into one classified, deduplicated CRM row per conversation. Headers and deterministic rules decide what never reaches a model; a re-run is a zero-row diff; `--forget` erases one sender without a model call. |

The company in the fixtures is fictional.

## Run them

Only a provider key is required.

```bash
pip install -e ".[examples]"
export ANTHROPIC_API_KEY="sk-ant-..."

python examples/agents/company_brain/run.py            # first sync
python examples/agents/company_brain/run.py --version 2 # sources changed
python examples/agents/company_brain/run.py --version 2 # nothing changed: 0 calls

python examples/agents/crm_inbox/run.py                 # run 1
python examples/agents/crm_inbox/run.py                 # again: zero-row diff, 0 calls
```

Each run costs a few cents. Every option is listed at the top of each `run.py`.

`FLUXCOMPUTE_KEY` is optional. Without it, routing, the in-process execution graph and
`resume()` all work and nothing leaves your machine. With it, each run also appears in the
hosted dashboard, and the key is checked before any spend.

## The walkthroughs

Recorded runs, the numbers, and the one snippet from each agent worth copying are on the
docs site: [docs.fluxcompute.dev/examples.html](https://docs.fluxcompute.dev/examples.html).
Numbers live there and nowhere else, so this file cannot drift from them.
