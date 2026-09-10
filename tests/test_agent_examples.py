"""The worked agents under examples/agents/ run on every commit, so they cannot go stale.

The agents run with a real provider key and have no mock mode of their own. The provider
seam is patched here instead, from the outside, the same way the SDK's own tests do it
(`patch("fluxcompute.client.anthropic")`): real classifier, real graph recorder, fake provider
response. What that buys is a test that fails when the SDK's behaviour changes under the
agents, not one that passes because everything interesting was stubbed.

The scripted provider answers each extraction from the document it was given, so the verbatim
quote check is genuinely exercised rather than fed pre-agreed strings.

No FluxCompute key is set unless a test sets one: the default path is the SDK-only one.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
AGENTS = ROOT / "examples" / "agents"


def _load(name: str, path: Path, monkeypatch):
    """Execute a script as a module under `name`, registered only for the test's lifetime.

    The agents import `_common` by bare name, so it has to be in sys.modules while run.py
    executes; going through monkeypatch means it is gone again at teardown instead of leaking
    into every test collected after this file.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


# ─────────────────────────────────────────────────────────────────────────────
# A provider that answers from the document, so quote validation is real
# ─────────────────────────────────────────────────────────────────────────────

FACT_RULES: list[tuple[str, str, str | None]] = [
    (r"trading name|Northwind Logistics", "company.name", "Northwind Logistics Cloud"),
    (
        r"Route freight the way",
        "company.tagline",
        "Route freight the way a dispatcher would, without the dispatcher.",
    ),
    (
        r"Freight planning that shows its work",
        "company.tagline",
        "Freight planning that shows its work.",
    ),
    (r"Apache-2\.0", "company.licence", "Apache-2.0"),
    (r"pip install northwind", "company.install_command", "pip install northwind"),
    (r"Harbour Blue", "company.brand_colour", "Harbour Blue, #12405F"),
    (r"Rotterdam", "company.office", "Rotterdam"),
    (r"Starter \| \$(\d+)", "pricing.starter_monthly_usd", None),
    (r"Starter plan is \$(\d+) per month", "pricing.starter_monthly_usd", None),
    (r"Growth \| \$(\d+)", "pricing.growth_monthly_usd", None),
    (r"Kestrel TMS", "product.integration", "Kestrel TMS"),
    (r"Loadsmart", "product.integration", "Loadsmart"),
    (r"project44", "product.integration", "project44"),
    (r"SDK version (\d+\.\d+\.\d+)", "product.sdk_version", None),
    (r"p95 latency of (\d+) ms", "product.latency_p95_ms", None),
    (r"refreshed every ((?:fifteen|five) minutes)", "product.capacity_refresh", None),
    (r"SOC 2 Type II", "security.certification", "SOC 2 Type II"),
    (r"retained for (\d+) days", "security.data_retention_days", None),
    (r"AES-256", "security.encryption", "AES-256"),
    (r"(Priya Raman) is the Chief Executive", "team.ceo", None),
    (r"(Tomas Lindqvist) is the Chief Technology", "team.cto", None),
    (r"(Jordan Bell) is the Head of Customer Success", "team.head_of_customer_success", None),
    (r"(Wei Chen) is the Staff Engineer", "team.staff_engineer", None),
    (r"(Sofia Marchetti) joins", "team.sre", None),
    (r"(Multi-leg planning for intermodal shipments)", "roadmap.planned", None),
]

# A fact whose quote is not in the document. validate_facts must drop it, and the brain must
# never render it: this is the fabricated-citation case, present in every extraction.
FABRICATED = {
    "key": "company.description",
    "value": "the world's most advanced freight platform",
    "quote": "Northwind is the world's most advanced freight platform.",
}


def _document_from(kwargs: dict) -> str:
    # After the SDK's cache manager, message content is a list of blocks, not a string.
    raw = kwargs["messages"][-1]["content"]
    if isinstance(raw, str):
        return raw
    return " ".join(b.get("text", "") for b in raw if isinstance(b, dict))


def _answer(document: str) -> str:
    facts = [FABRICATED]
    for pattern, key, fixed in FACT_RULES:
        m = re.search(pattern, document)
        if not m:
            continue
        value = fixed if fixed is not None else m.group(1)
        quote = next(
            (line.strip() for line in document.splitlines() if m.group(0) in line), m.group(0)
        )
        facts.append({"key": key, "value": value, "quote": quote})
    return json.dumps({"facts": facts})


class _ScriptedMessages:
    def __init__(self) -> None:
        self.models: list[str] = []

    async def create(self, **kwargs):
        self.models.append(kwargs["model"])
        if kwargs["model"] == "claude-3-5-haiku-20241022":
            raise RuntimeError("model_not_found: this model was retired")
        return SimpleNamespace(
            model=kwargs["model"],
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text=_answer(_document_from(kwargs)))],
            usage=SimpleNamespace(
                input_tokens=400,
                output_tokens=120,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
        )


class _ScriptedAnthropic:
    def __init__(self, **_kwargs) -> None:
        self.messages = _ScriptedMessages()

    async def close(self) -> None:
        pass


async def _noop():
    return {}


@pytest.fixture
def brain(tmp_path, monkeypatch):
    """The company brain, wired to a scripted provider and a temporary output directory.

    No FluxCompute key: this is the SDK-only path, so telemetry is off and no emitter exists.
    syspath_prepend is undone at teardown, which also undoes the agent's own sys.path insert.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("FLUXCOMPUTE_KEY", raising=False)
    monkeypatch.delenv("FLUX_CONTENT_CAPTURE", raising=False)
    monkeypatch.syspath_prepend(str(AGENTS))

    common = _load("_common", AGENTS / "_common.py", monkeypatch)
    module = _load("brain_run", AGENTS / "company_brain" / "run.py", monkeypatch)
    monkeypatch.setattr(module, "OUT", tmp_path)

    fake = _ScriptedAnthropic()
    with patch("fluxcompute.client.anthropic") as anthropic_module:
        anthropic_module.AsyncAnthropic.return_value = fake
        yield SimpleNamespace(
            module=module, common=common, fake=fake, out=tmp_path, monkeypatch=monkeypatch
        )


async def _run(brain, *argv: str) -> int:
    brain.monkeypatch.setattr(sys, "argv", ["run.py", *argv])
    brain.module.OUT = brain.out
    return await brain.module.main()


# ─────────────────────────────────────────────────────────────────────────────
# What the walkthrough promises
# ─────────────────────────────────────────────────────────────────────────────


async def test_first_sync_publishes_a_document_with_sources(brain, capsys):
    assert await _run(brain, "--version", "1") == 0
    doc = (brain.out / "brain.md").read_text()
    assert "Northwind Logistics Cloud" in doc
    assert "`[repo]`" in doc and "`[email]`" in doc and "`[brand]`" in doc
    out = capsys.readouterr().out
    # SDK-only: the graph is recorded in-process and no dashboard link is printed.
    assert "nodes recorded in-process" in out
    assert "/ui/#task=" not in out


async def test_with_a_key_the_run_also_prints_its_dashboard_link(brain, capsys):
    """The one line a FluxCompute key adds. verify() is patched: the guard in conftest blackholes
    the endpoint, and the emitter's POSTs go there and are swallowed, which is the guard's job."""
    brain.monkeypatch.setenv("FLUXCOMPUTE_KEY", "flx_test")
    with patch("fluxcompute.client.FluxClient.verify", new=lambda self: _noop()):
        assert await _run(brain, "--version", "1") == 0
    assert "/ui/#task=" in capsys.readouterr().out


async def test_fabricated_quote_never_reaches_the_document(brain):
    """The model asserts an unquotable claim on every call. None of them survive."""
    await _run(brain, "--version", "1")
    assert "most advanced freight platform" not in (brain.out / "brain.md").read_text()


async def test_precedence_is_data_not_a_prompt(brain):
    """The repo says the Starter plan is $49 and an email says $59; the schema picks."""
    await _run(brain, "--version", "1")
    doc = (brain.out / "brain.md").read_text()
    assert "starter_monthly_usd**: 59" in doc
    assert "Conflicts resolved by precedence" in (brain.out / "brain.changelog.md").read_text()


async def test_sender_allowlist_runs_before_the_model(brain):
    """A prompt injection from an unlisted sender must not reach a model at all."""
    await _run(brain, "--version", "1")
    doc = (brain.out / "brain.md").read_text().lower()
    assert "bob vance" not in doc
    prompts = " ".join(brain.fake.messages.models)
    assert "ignore your previous instructions" not in prompts


async def test_personal_data_is_scrubbed_before_the_model_sees_it(brain):
    await _run(brain, "--version", "1")
    assert "900412" not in (brain.out / "brain.md").read_text()


async def test_unchanged_run_makes_no_llm_calls(brain):
    await _run(brain, "--version", "1")
    before = len(brain.fake.messages.models)
    assert await _run(brain, "--version", "1", "--attempt", "2") == 0
    assert len(brain.fake.messages.models) == before


async def test_a_rename_is_not_a_deletion(brain):
    """Identical bytes under a new path must carry their facts, not tombstone them."""
    await _run(brain, "--version", "1")
    await _run(brain, "--version", "2", "--attempt", "2")
    assert "Multi-leg planning" in (brain.out / "brain.md").read_text()


async def test_a_source_that_stops_asserting_removes_the_fact_that_run(brain):
    """integrations.md still exists and dropped Kestrel: no grace period applies."""
    await _run(brain, "--version", "1")
    await _run(brain, "--version", "2", "--attempt", "2")
    assert "Kestrel" not in (brain.out / "brain.md").read_text()


async def test_a_vanished_source_waits_out_the_grace_period(brain):
    """docs/security.md is deleted. An outage looks the same for one run, so it waits."""
    await _run(brain, "--version", "1")
    await _run(brain, "--version", "2", "--attempt", "2")
    assert "SOC 2" in (brain.out / "brain.md").read_text()
    await _run(brain, "--version", "2", "--attempt", "3")
    assert "SOC 2" not in (brain.out / "brain.md").read_text()


async def test_graph_carries_no_content_only_identifiers(brain):
    """Names and attributes are part of the graph whatever content_capture says, so they hold
    hashes and counts, never text from a source."""
    await _run(brain, "--version", "1")
    graph = brain.module.LAST_GRAPH
    leaks = ["@", "Priya", "Northwind", "sk-ant", "Starter"]
    for node in graph.in_order():
        blob = json.dumps({"name": node.name, "attributes": node.attributes})
        assert not any(term in blob for term in leaks), f"content leaked in node {node.name}"


async def test_injected_failure_is_retried_by_resume_in_the_same_graph(brain):
    """The walkthrough's recovery step depends on this shape, so pin it.

    A provider failure stops the run with the task root marked failed and one failed llm_call
    node. The handler then retries exactly that step with client.resume(): the retry is a
    second llm_call in the same graph, linked to the node it replaces, and the run reports the
    extractions it never reached rather than pretending to have finished.
    """
    from fluxcompute.graph.resume import build_resume_plan

    assert await _run(brain, "--version", "1", "--fail-at", "2") == 1
    graph = brain.module.LAST_GRAPH
    failed = [n for n in graph.failed_nodes() if n.node_type == "llm_call"]
    assert len(failed) == 1
    retries = [n for n in graph.in_order() if failed[0].node_id in n.depends_on]
    assert len(retries) == 1 and retries[0].status == "succeeded"
    assert "claude-3-5-haiku-20241022" in brain.fake.messages.models
    # The retry resolved the LLM failure. What the planner still sees is the step that wrapped
    # it, which stays failed so a reader can find where the run broke.
    assert build_resume_plan(graph).failed_node.node_type == "tool_call"
    db = sqlite3.connect(brain.out / "brain.db")
    assert db.execute("SELECT status FROM run").fetchone()[0] == "failed"
    assert db.execute("SELECT COUNT(*) FROM extraction").fetchone()[0] == 2  # 1 ok + 1 resumed
    db.close()


async def test_resume_from_a_fresh_process_finishes_the_run(brain):
    """The cursor must not be the marker of done work, or a failed run reports zero changes
    forever afterwards. The marker is the extraction row, so a second process picks up every
    item the first one never reached and nothing that it did."""
    assert await _run(brain, "--version", "1", "--fail-at", "2") == 1
    task_id = brain.module.LAST_GRAPH.task_id
    calls_before = len(brain.fake.messages.models)

    assert await _run(brain, "--resume", task_id) == 0
    db = sqlite3.connect(brain.out / "brain.db")
    assert db.execute("SELECT COUNT(*) FROM extraction").fetchone()[0] == 17
    assert db.execute("SELECT status FROM run WHERE task_id=?", (task_id,)).fetchone()[0] != (
        "failed"
    )
    db.close()
    # Fifteen items were never reached; the two already extracted are not paid for again.
    # QA is answered from the fact store, so it costs no model calls.
    assert len(brain.fake.messages.models) - calls_before == 15


# ─────────────────────────────────────────────────────────────────────────────
# The shared scrubber and error vocabulary, which every agent depends on
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def common(monkeypatch):
    monkeypatch.syspath_prepend(str(AGENTS))
    return _load("_common", AGENTS / "_common.py", monkeypatch)


def test_scrubber_labels_each_category_correctly(common):
    """Order in _SECRETS is load-bearing, not cosmetic.

    The phone pattern matches any long run of digits, so it is a superset of a card number.
    With PHONE first, cards were still redacted but were reported as phone numbers, and the
    counts are what a reviewer reads when asking what left the machine.
    """
    text, counts = common.scrub(
        "card 4111 1111 1111 1111, mobile +44 7700 900412, mail a@b.com, key sk-ant-abcdefgh12345"
    )
    assert counts == {"CARD": 1, "PHONE": 1, "EMAIL": 1, "ANTHROPIC_KEY": 1}
    for secret in ("4111", "900412", "a@b.com", "sk-ant-"):
        assert secret not in text


def test_error_codes_avoid_the_spend_vocabulary(common):
    """The SDK classifies a failed node as a budget failure by matching the error text.

    An agent that raised "quota exceeded" for its own reasons would be classified as a spend
    problem and, three in a row, as a stall.
    """
    banned = ("rate limit", "quota", "billing", "credit")
    for code in common.ERROR_CODES:
        assert not any(word in code.lower() for word in banned), code


def test_runs_without_a_fluxcompute_key(common, monkeypatch):
    """The SDK-only contract: no key means telemetry off and no verify() handshake."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("FLUXCOMPUTE_KEY", raising=False)
    import asyncio

    with patch("fluxcompute.client.anthropic"), patch(
        "fluxcompute.client.FluxClient.verify", side_effect=AssertionError("must not verify")
    ):
        clients = asyncio.run(common.make_clients())
    assert clients.anthropic._graph_emitter._enabled is False


# ─────────────────────────────────────────────────────────────────────────────
# The CRM inbox
#
# What these tests can and cannot prove. The scripted classifier maps a subject to a class,
# so it agrees with crm_labels.json by construction. That makes the precision figure a test
# of the pipeline -- that a decision reaches the CRM intact -- and NOT evidence that the
# model classifies well. Only a real run measures that. This suite asserts the properties
# that hold whatever the model says: what never reaches a model at all, what is idempotent,
# and what is decided in code.
# ─────────────────────────────────────────────────────────────────────────────

CRM_RULES = [
    (r"Carrier ranking API", "sales_inquiry", 0.93, False),
    (r"quick chat", "other", 0.41, False),
    (r"Demo request", "demo_request", 0.95, False),
    (r"502s", "support", 0.96, False),
    (r"Webhook retries", "support", 0.92, False),
    (r"joint integration listing", "partnership", 0.90, False),
    (r"observability bill", "vendor", 0.93, False),
    (r"three times your outbound", "vendor", 0.91, False),
    (r"backend engineers", "recruiting", 0.96, False),
    (r"Comment for a piece", "press", 0.90, False),
    (r"Invoice 4471", "billing", 0.95, False),
    (r"Pricing for 8,000 loads", "sales_inquiry", 0.94, False),
    (r"last-mile grocery", "sales_inquiry", 0.88, False),
    (r"demo of the new lane view", "demo_request", 0.94, False),
    (r"supplier banking details", "billing", 0.55, True),
]


def _classify_answer(user: str) -> str:
    subject_match = re.search(r"^Subject: (.*)$", user, re.M)
    subject = subject_match.group(1) if subject_match else ""
    body = user.split("\n\n", 1)[1] if "\n\n" in user else user
    kind, confidence, suspicious = "other", 0.35, False
    for pattern, k, c, s in CRM_RULES:
        if re.search(pattern, subject, re.I):
            kind, confidence, suspicious = k, c, s
            break
    lines = [ln.strip() for ln in body.splitlines() if len(ln.strip()) >= 30]
    return json.dumps(
        {
            "request_type": kind,
            "confidence": confidence,
            "quote": max(lines, key=len) if lines else body.strip()[:80],
            "summary": f"about: {subject[:50]}",
            "suspicious": suspicious,
            "language": "en",
        }
    )


class _ScriptedCrmMessages(_ScriptedMessages):
    async def create(self, **kwargs):
        self.models.append(kwargs["model"])
        return SimpleNamespace(
            model=kwargs["model"],
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text=_classify_answer(_document_from(kwargs)))],
            usage=SimpleNamespace(
                input_tokens=300,
                output_tokens=80,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
        )


class _ScriptedCrmAnthropic:
    def __init__(self, **_kwargs) -> None:
        self.messages = _ScriptedCrmMessages()

    async def close(self) -> None:
        pass


@pytest.fixture
def inbox(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("FLUXCOMPUTE_KEY", raising=False)
    monkeypatch.syspath_prepend(str(AGENTS))

    common = _load("_common", AGENTS / "_common.py", monkeypatch)
    module = _load("crm_run", AGENTS / "crm_inbox" / "run.py", monkeypatch)
    monkeypatch.setattr(module, "OUT", tmp_path)

    fake = _ScriptedCrmAnthropic()
    with patch("fluxcompute.client.anthropic") as anthropic_module:
        anthropic_module.AsyncAnthropic.return_value = fake
        yield SimpleNamespace(
            module=module, common=common, fake=fake, out=tmp_path, monkeypatch=monkeypatch
        )


async def _run_inbox(inbox, *argv: str) -> int:
    inbox.monkeypatch.setattr(sys, "argv", ["run.py", *argv])
    inbox.module.OUT = inbox.out
    return await inbox.module.main()


async def test_machine_mail_never_reaches_a_model(inbox, capsys):
    """An out-of-office and a newsletter are decided by their headers, before any spend."""
    await _run_inbox(inbox, "--run", "1")
    prompts = len(inbox.fake.messages.models)
    rows = (inbox.out / "crm.csv").read_text()
    assert "auto-replied" not in rows.lower()
    # 16 threads, minus two machine mails and one quarantined phishing message.
    assert prompts == 14
    out = capsys.readouterr().out
    assert "nodes recorded in-process" in out and "/ui/#task=" not in out


async def test_with_a_key_the_inbox_run_prints_its_dashboard_link(inbox, capsys):
    inbox.monkeypatch.setenv("FLUXCOMPUTE_KEY", "flx_test")
    with patch("fluxcompute.client.FluxClient.verify", new=lambda self: _noop()):
        await _run_inbox(inbox, "--run", "1")
    assert "/ui/#task=" in capsys.readouterr().out


async def test_phishing_is_quarantined_without_a_model_call(inbox):
    """The rules that catch a look-alike domain are code. A model is never asked."""
    await _run_inbox(inbox, "--run", "1")
    db = sqlite3.connect(inbox.out / "crm.db")
    quarantined = db.execute("SELECT COUNT(*) FROM message WHERE gate = 'quarantined'").fetchone()[
        0
    ]
    classified = db.execute(
        "SELECT COUNT(*) FROM classification WHERE message_id IN"
        " (SELECT message_id FROM message WHERE gate = 'quarantined')"
    ).fetchone()[0]
    db.close()
    assert quarantined >= 1
    assert classified == 0


async def test_priority_is_computed_in_code(inbox):
    """The model returns a class; the number a human sorts by is a pure function.

    A model that could set priority could be talked into setting it, which is exactly what
    the injection-shaped mail in the fixture tries.
    """
    priority_of = inbox.module.priority_of
    base = priority_of("sales_inquiry", "prospect", "Pricing", 1)
    assert priority_of("sales_inquiry", "customer", "Pricing", 1) == base + 1
    assert priority_of("sales_inquiry", "prospect", "URGENT: outage", 1) == base + 1
    assert priority_of("support", "customer", "we are down", 4) <= 5


async def test_a_reply_updates_its_thread_instead_of_inserting(inbox):
    await _run_inbox(inbox, "--run", "1")
    first = len((inbox.out / "crm.csv").read_text().strip().splitlines())
    await _run_inbox(inbox, "--run", "2", "--attempt", "2")
    rows = (inbox.out / "crm.csv").read_text().strip().splitlines()
    # Run 2 brings three messages, one of which continues an existing thread.
    assert len(rows) == first + 2


async def test_rerunning_the_same_inbox_changes_nothing(inbox):
    await _run_inbox(inbox, "--run", "1")
    before_csv = (inbox.out / "crm.csv").read_text()
    calls = len(inbox.fake.messages.models)
    await _run_inbox(inbox, "--run", "1", "--attempt", "2")
    assert (inbox.out / "crm.csv").read_text() == before_csv
    assert len(inbox.fake.messages.models) == calls


async def test_forget_erases_a_sender_without_needing_a_provider(inbox):
    """An erasure request must not fail because a key is bad. It touches no model."""
    await _run_inbox(inbox, "--run", "1")
    inbox.monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    calls = len(inbox.fake.messages.models)
    db = sqlite3.connect(inbox.out / "crm.db")
    before = db.execute(
        "SELECT COUNT(*) FROM message WHERE sender LIKE '%cascade-carriers%'"
    ).fetchone()[0]
    db.close()
    assert before >= 1, "the fixture must contain the sender this test erases"
    assert await _run_inbox(inbox, "--forget", "nadia@cascade-carriers.example") == 0
    assert len(inbox.fake.messages.models) == calls
    db = sqlite3.connect(inbox.out / "crm.db")
    after = db.execute(
        "SELECT COUNT(*) FROM message WHERE sender LIKE '%cascade-carriers%'"
    ).fetchone()[0]
    db.close()
    assert after == 0
