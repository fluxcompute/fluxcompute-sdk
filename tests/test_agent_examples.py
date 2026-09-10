"""The provider is patched at fluxcompute.client.anthropic, so the classifier and the graph
recorder are real. No FluxCompute key is set unless a test sets one."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
AGENTS = ROOT / "examples" / "agents"


def _load(name: str, path: Path, monkeypatch):
    """monkeypatch.setitem, so _common is gone again at teardown."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _load_common(monkeypatch):
    """The .env loader would put a real FLUXCOMPUTE_KEY back after the fixtures delete it."""
    common = _load("_common", AGENTS / "_common.py", monkeypatch)
    monkeypatch.setattr(common, "load_dotenv", lambda *args, **kwargs: None)
    return common


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

# a quote not in the document; validate_facts must drop it
FABRICATED = {
    "key": "company.description",
    "value": "the world's most advanced freight platform",
    "quote": "Northwind is the world's most advanced freight platform.",
}


def _text_of(content) -> str:
    # after the cache manager, content is a list of blocks
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") for b in content if isinstance(b, dict))


def _document_from(kwargs: dict) -> str:
    return _text_of(kwargs["messages"][-1]["content"])


def _everything_sent(kwargs: dict) -> str:
    parts = [_text_of(kwargs["system"])] if isinstance(kwargs.get("system"), (str, list)) else []
    parts += [_text_of(m["content"]) for m in kwargs["messages"]]
    return "\n".join(parts)


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
        self.prompts: list[str] = []  # everything the provider was shown, per call
        self.prose_for: str = ""  # a document containing this gets prose back, never JSON

    async def create(self, **kwargs):
        self.models.append(kwargs["model"])
        self.prompts.append(_everything_sent(kwargs))
        if kwargs["model"] == "claude-3-5-haiku-20241022":
            raise RuntimeError("model_not_found: this model was retired")
        if self.prose_for and self.prose_for in self.prompts[-1]:
            text = "Sorry, I cannot turn that document into facts."
        else:
            text = _answer(_document_from(kwargs))
        return SimpleNamespace(
            model=kwargs["model"],
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text=text)],
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
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("FLUXCOMPUTE_KEY", raising=False)
    monkeypatch.delenv("FLUX_CONTENT_CAPTURE", raising=False)
    monkeypatch.syspath_prepend(str(AGENTS))

    common = _load_common(monkeypatch)
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


async def test_first_sync_publishes_a_document_with_sources(brain, capsys):
    assert await _run(brain, "--version", "1") == 0
    doc = (brain.out / "brain.md").read_text()
    assert "Northwind Logistics Cloud" in doc
    assert "`[repo]`" in doc and "`[email]`" in doc and "`[brand]`" in doc
    out = capsys.readouterr().out
    assert "nodes recorded in-process" in out
    assert "/ui/#task=" not in out


async def test_with_a_key_the_run_also_prints_its_dashboard_link(brain, capsys):
    brain.monkeypatch.setenv("FLUXCOMPUTE_KEY", "flx_test")
    with patch("fluxcompute.client.FluxClient.verify", new=lambda self: _noop()):
        assert await _run(brain, "--version", "1") == 0
    assert "/ui/#task=" in capsys.readouterr().out


async def test_fabricated_quote_never_reaches_the_document(brain):
    await _run(brain, "--version", "1")
    assert "most advanced freight platform" not in (brain.out / "brain.md").read_text()


async def test_precedence_is_data_not_a_prompt(brain):
    await _run(brain, "--version", "1")
    doc = (brain.out / "brain.md").read_text()
    assert "starter_monthly_usd**: 59" in doc
    assert "Conflicts resolved by precedence" in (brain.out / "brain.changelog.md").read_text()


async def test_sender_allowlist_runs_before_the_model(brain):
    assert await _run(brain, "--version", "1") == 0
    doc = (brain.out / "brain.md").read_text().lower()
    assert "bob vance" not in doc
    prompts = "\n".join(brain.fake.messages.prompts).lower()
    assert prompts, "the provider was called"
    assert "ignore your previous instructions" not in prompts
    assert "bob vance" not in prompts
    assert "harbour blue" in prompts  # the brand file did reach the model


async def test_allowlist_reads_the_address_not_the_display_name(brain):
    common = brain.common
    _, addr, _ = common.split_address("Priya Raman <PRIYA@northwind.example.com>")
    assert addr == "priya@northwind.example.com"


async def test_personal_data_is_scrubbed_before_the_model_sees_it(brain):
    await _run(brain, "--version", "1")
    assert "900412" not in (brain.out / "brain.md").read_text()


async def test_unchanged_run_makes_no_llm_calls(brain, capsys):
    await _run(brain, "--version", "1")
    before = len(brain.fake.messages.models)
    assert await _run(brain, "--version", "1", "--attempt", "2") == 0
    assert len(brain.fake.messages.models) == before
    assert "diff: 0 changed, 17 reused" in capsys.readouterr().out


async def test_back_to_back_runs_get_their_own_task_ids(brain):
    assert await _run(brain, "--version", "1") == 0
    assert "over `repo` (49)" in (brain.out / "brain.changelog.md").read_text()
    assert await _run(brain, "--version", "2") == 0
    db = sqlite3.connect(brain.out / "brain.db")
    ids = [r[0] for r in db.execute("SELECT task_id FROM run ORDER BY task_id")]
    db.close()
    assert len(ids) == 2 and ids[0] != ids[1]
    assert "over `repo` (49)" not in (brain.out / "brain.changelog.md").read_text()


async def test_a_rename_is_not_a_deletion(brain):
    await _run(brain, "--version", "1")
    await _run(brain, "--version", "2", "--attempt", "2")
    assert "Multi-leg planning" in (brain.out / "brain.md").read_text()


async def test_a_source_that_stops_asserting_removes_the_fact_that_run(brain):
    await _run(brain, "--version", "1")
    await _run(brain, "--version", "2", "--attempt", "2")
    assert "Kestrel" not in (brain.out / "brain.md").read_text()


async def test_a_vanished_source_waits_out_the_grace_period(brain):
    await _run(brain, "--version", "1")
    await _run(brain, "--version", "2", "--attempt", "2")
    assert "SOC 2" in (brain.out / "brain.md").read_text()
    await _run(brain, "--version", "2", "--attempt", "3")
    assert "SOC 2" not in (brain.out / "brain.md").read_text()


async def test_graph_carries_no_content_only_identifiers(brain):
    await _run(brain, "--version", "1")
    graph = brain.module.LAST_GRAPH
    leaks = ["@", "Priya", "Northwind", "sk-ant", "Starter"]
    for node in graph.in_order():
        blob = json.dumps({"name": node.name, "attributes": node.attributes})
        assert not any(term in blob for term in leaks), f"content leaked in node {node.name}"


async def test_injected_failure_is_retried_by_resume_in_the_same_graph(brain):
    from fluxcompute.graph.resume import build_resume_plan

    assert await _run(brain, "--version", "1", "--fail-at", "2") == 1
    graph = brain.module.LAST_GRAPH
    failed = [n for n in graph.failed_nodes() if n.node_type == "llm_call"]
    assert len(failed) == 1
    retries = [n for n in graph.in_order() if failed[0].node_id in n.depends_on]
    assert len(retries) == 1 and retries[0].status == "succeeded"
    assert "claude-3-5-haiku-20241022" in brain.fake.messages.models
    # the retry resolved the llm_call; the wrapping step stays failed
    assert build_resume_plan(graph).failed_node.node_type == "tool_call"
    db = sqlite3.connect(brain.out / "brain.db")
    assert db.execute("SELECT status FROM run").fetchone()[0] == "failed"
    assert db.execute("SELECT COUNT(*) FROM extraction").fetchone()[0] == 2  # 1 ok + 1 resumed
    db.close()


async def test_resume_from_a_fresh_process_finishes_the_run(brain):
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
    # 15 never reached; the two already extracted are not paid for again
    assert len(brain.fake.messages.models) - calls_before == 15


async def test_resume_syncs_the_version_the_failed_run_was_syncing(brain):
    assert await _run(brain, "--version", "1") == 0
    assert await _run(brain, "--version", "2", "--fail-at", "2") == 1
    task_id = brain.module.LAST_GRAPH.task_id
    assert await _run(brain, "--resume", task_id) == 0
    db = sqlite3.connect(brain.out / "brain.db")
    version, status = db.execute(
        "SELECT version, status FROM run WHERE task_id=?", (task_id,)
    ).fetchone()
    db.close()
    assert (version, status) == (2, "ok")
    assert "Kestrel" not in (brain.out / "brain.md").read_text()  # v2 dropped it


async def test_a_reply_that_is_never_json_stops_cleanly(brain, capsys):
    brain.fake.messages.prose_for = "Harbour Blue"
    assert await _run(brain, "--version", "1") == 1
    out = capsys.readouterr().out
    assert "failed: parse_failed" in out
    assert "nothing to resume" in out
    assert "--resume" in out
    db = sqlite3.connect(brain.out / "brain.db")
    assert db.execute("SELECT status FROM run").fetchone()[0] == "failed"
    db.close()


async def test_a_held_run_publishes_nothing(brain, tmp_path, capsys):
    fixtures = tmp_path / "fixtures"
    shutil.copytree(AGENTS / "fixtures", fixtures)
    brain.monkeypatch.setattr(brain.module, "COMPANY", fixtures)
    assert await _run(brain, "--version", "1") == 0
    published = (brain.out / "brain.md").read_text()
    db = sqlite3.connect(brain.out / "brain.db")
    active_before = db.execute("SELECT COUNT(*) FROM fact WHERE status='active'").fetchone()[0]
    db.close()

    for path in (fixtures / "repo_v2").rglob("*.md"):
        path.write_text("# Under construction\n\nNothing here yet.\n")
    assert await _run(brain, "--version", "2") == 1
    out = capsys.readouterr().out
    assert "HELD" in out and "render: held" in out
    assert (brain.out / "brain.md").read_text() == published
    db = sqlite3.connect(brain.out / "brain.db")
    assert db.execute("SELECT COUNT(*) FROM fact WHERE status='removed'").fetchone()[0] == 0
    assert (
        db.execute("SELECT COUNT(*) FROM fact WHERE status='active'").fetchone()[0]
        >= active_before
    )
    db.close()


@pytest.fixture
def common(monkeypatch):
    monkeypatch.syspath_prepend(str(AGENTS))
    return _load_common(monkeypatch)


def test_scrubber_labels_each_category_correctly(common):
    text, counts = common.scrub(
        "card 4111 1111 1111 1111, mobile +44 7700 900412, mail a@b.com, key sk-ant-abcdefgh12345"
    )
    assert counts == {"CARD": 1, "PHONE": 1, "EMAIL": 1, "ANTHROPIC_KEY": 1}
    for secret in ("4111", "900412", "a@b.com", "sk-ant-"):
        assert secret not in text
    assert "[REDACTED:CARD], mobile" in text  # the number goes, the separator after it stays


def test_dates_and_version_numbers_are_not_phone_numbers(common):
    text, counts = common.scrub(
        "Released 2026-09-10 at 12:30, pinned to 0.3.0 (2024); support ends 2026-12-31T23:59:00Z."
        " Call +44 7700 900412 with questions about ticket 123456789012."
    )
    assert counts == {"PHONE": 2}  # the mobile and the twelve-digit ticket; nothing else
    for kept in ("2026-09-10 at 12:30", "0.3.0 (2024)", "2026-12-31T23:59:00Z"):
        assert kept in text
    assert common.quote_is_verbatim("support ends 2026-12-31T23:59:00Z", text)


def test_a_quote_is_verbatim_whatever_its_case(common):
    source = "Subject: New office\n\nFrom 1 November our European office is in Rotterdam."
    assert common.quote_is_verbatim("from 1 November our European office is in Rotterdam", source)
    assert common.quote_is_verbatim("FROM 1 NOVEMBER OUR EUROPEAN OFFICE", source)
    assert not common.quote_is_verbatim("our European office is in Antwerp", source)
    assert not common.quote_is_verbatim("", source)


def test_mailbox_dates_are_read_in_every_form_a_client_writes(common):
    for raw in (
        "2026-09-09T09:14:00+00:00",
        "2026-09-09T09:14:00Z",
        "Tue, 9 Sep 2026 09:14:00 +0000",
        "2026-09-09T09:14:00",
    ):
        assert common.parse_date(raw) == "2026-09-09T09:14:00+00:00", raw
    assert common.parse_date("2026-09-09T18:14:00+09:00") == "2026-09-09T09:14:00+00:00"


def test_error_codes_avoid_the_spend_vocabulary(common):
    banned = ("rate limit", "quota", "billing", "credit")
    for code in common.ERROR_CODES:
        assert not any(word in code.lower() for word in banned), code


def test_runs_without_a_fluxcompute_key(common, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("FLUXCOMPUTE_KEY", raising=False)
    import asyncio

    with patch("fluxcompute.client.anthropic"), patch(
        "fluxcompute.client.FluxClient.verify", side_effect=AssertionError("must not verify")
    ):
        clients = asyncio.run(common.make_clients())
    assert clients.anthropic._graph_emitter._enabled is False


# the scripted classifier agrees with crm_labels.json by construction: precision here tests the
# pipeline, not the model
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
        self.prompts.append(_everything_sent(kwargs))
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

    common = _load_common(monkeypatch)
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


def _own_inbox(inbox, messages: list[dict], labels: list[dict] | None = None) -> None:
    """Fixtures copy whose run-1 mailbox is `messages`; the shipped inboxes lack these cases."""
    fixtures = inbox.out / "fixtures"
    shutil.copytree(AGENTS / "fixtures", fixtures)
    (fixtures / "inbox_crm" / "run1.json").write_text(json.dumps(messages))
    spec = json.loads((fixtures / "crm_labels.json").read_text())
    spec["run1"] = labels or []
    (fixtures / "crm_labels.json").write_text(json.dumps(spec))
    inbox.monkeypatch.setattr(inbox.module, "COMPANY", fixtures)


def _mail(sender: str, subject: str, body: str, date: str, **extra) -> dict:
    return {
        "from": sender,
        "to": "hello@northwind.example.com",
        "subject": subject,
        "body": body,
        "date": date,
        **extra,
    }


def _crm_rows(inbox) -> list[dict]:
    db = sqlite3.connect(inbox.out / "crm.db")
    db.row_factory = sqlite3.Row
    rows = [dict(r) for r in db.execute("SELECT * FROM crm ORDER BY thread_key")]
    db.close()
    return rows


async def test_machine_mail_never_reaches_a_model(inbox, capsys):
    assert await _run_inbox(inbox, "--run", "1") == 0
    db = sqlite3.connect(inbox.out / "crm.db")
    ignored = db.execute("SELECT COUNT(*) FROM message WHERE gate='ignored'").fetchone()[0]
    classified_anyway = db.execute(
        "SELECT COUNT(*) FROM classification WHERE message_id IN"
        " (SELECT message_id FROM message WHERE gate='ignored')"
    ).fetchone()[0]
    db.close()
    assert ignored == 2 and classified_anyway == 0
    # 18 messages, 1 dropped as a forward: 17 kept, minus 2 machine mails and 1 quarantined
    assert len(inbox.fake.messages.models) == 14
    out = capsys.readouterr().out
    assert "nodes recorded in-process" in out and "/ui/#task=" not in out


async def test_with_a_key_the_inbox_run_prints_its_dashboard_link(inbox, capsys):
    inbox.monkeypatch.setenv("FLUXCOMPUTE_KEY", "flx_test")
    with patch("fluxcompute.client.FluxClient.verify", new=lambda self: _noop()):
        await _run_inbox(inbox, "--run", "1")
    assert "/ui/#task=" in capsys.readouterr().out


async def test_phishing_is_quarantined_without_a_model_call(inbox):
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
    priority_of = inbox.module.priority_of
    base = priority_of("sales_inquiry", "prospect", "Pricing", 1)
    assert priority_of("sales_inquiry", "customer", "Pricing", 1) == base + 1
    assert priority_of("sales_inquiry", "prospect", "URGENT: outage", 1) == base + 1
    assert priority_of("support", "customer", "we are down", 4) <= 5


async def test_a_reply_updates_its_thread_instead_of_inserting(inbox):
    assert await _run_inbox(inbox, "--run", "1") == 0
    first = len((inbox.out / "crm.csv").read_text().strip().splitlines())
    assert await _run_inbox(inbox, "--run", "2", "--attempt", "2") == 0
    rows = (inbox.out / "crm.csv").read_text().strip().splitlines()
    # run 2 brings three messages, one of which continues an existing thread
    assert len(rows) == first + 2


async def test_rerunning_the_same_inbox_changes_nothing(inbox):
    assert await _run_inbox(inbox, "--run", "1") == 0
    before_csv = (inbox.out / "crm.csv").read_text()
    calls = len(inbox.fake.messages.models)
    assert await _run_inbox(inbox, "--run", "1", "--attempt", "2") == 0
    assert (inbox.out / "crm.csv").read_text() == before_csv
    assert len(inbox.fake.messages.models) == calls


async def test_resume_reads_the_inbox_the_failed_run_was_reading(inbox):
    assert await _run_inbox(inbox, "--run", "1") == 0
    assert await _run_inbox(inbox, "--run", "2") == 0
    second = inbox.common.run_id("crm", 2)  # no --attempt passed: the second run today is 02
    assert await _run_inbox(inbox, "--resume", second) == 0
    db = sqlite3.connect(inbox.out / "crm.db")
    assert db.execute("SELECT inbox FROM run WHERE task_id=?", (second,)).fetchone()[0] == 2
    assert db.execute("SELECT COUNT(*) FROM run").fetchone()[0] == 2
    db.close()


async def test_a_machine_reply_on_an_open_thread_does_not_erase_it(inbox):
    _own_inbox(
        inbox,
        [
            _mail(
                "Sam Ortega <sam.ortega@globex-freight.example>",
                "Planning API returning 502s since 06:00",
                "Every call to the planning API has returned a 502 since six this morning.",
                "2026-09-09T06:10:00+00:00",
                message_id="<t1@globex-freight.example>",
            ),
            _mail(
                "Sam Ortega <sam.ortega@globex-freight.example>",
                "Automatic reply: Planning API returning 502s since 06:00",
                "I am out of the office until Monday and will reply then.",
                "2026-09-09T06:10:05+00:00",
                message_id="<t2@globex-freight.example>",
                in_reply_to="<t1@globex-freight.example>",
                auto_submitted="auto-replied",
            ),
        ],
    )
    assert await _run_inbox(inbox, "--run", "1") == 0
    (row,) = _crm_rows(inbox)
    assert row["request_type"] == "support" and row["status"] != "ignored"
    assert row["msg_count"] == 2 and row["last_msg_at"] == "2026-09-09T06:10:05+00:00"
    assert len(inbox.fake.messages.models) == 1  # the auto-reply never reached a model


async def test_messages_without_a_message_id_each_get_a_row(inbox):
    _own_inbox(
        inbox,
        [
            _mail(
                "Ana Ruiz <ana@cascade-carriers.example>",
                "Webhook retries not firing",
                "Our webhook retries stopped firing after the weekend deploy, can you check?",
                "2026-09-09T09:00:00+00:00",
            ),
            _mail(
                "Ben Achebe <ben@breadbasket-bakeries.example>",
                "Do you do last-mile grocery delivery routing?",
                "We run forty vans out of two depots and want to know if you cover last mile.",
                "2026-09-09T09:30:00+00:00",
            ),
        ],
    )
    assert await _run_inbox(inbox, "--run", "1") == 0
    rows = _crm_rows(inbox)
    assert len(rows) == 2
    assert {r["request_type"] for r in rows} == {"support", "sales_inquiry"}


async def test_forget_rebuilds_a_thread_that_keeps_other_people(inbox):
    _own_inbox(
        inbox,
        [
            _mail(
                "Lena Bright <lena@cascade-carriers.example>",
                "Webhook retries not firing",
                "Our webhook retries stopped firing after the weekend deploy, can you check?",
                "2026-09-09T09:00:00+00:00",
                message_id="<w1@cascade-carriers.example>",
            ),
            _mail(
                "Nadia Farr <nadia@cascade-carriers.example>",
                "Re: Webhook retries not firing",
                "Adding to Lena's note: the retry queue shows zero attempts since Saturday night.",
                "2026-09-09T11:00:00+00:00",
                message_id="<w2@cascade-carriers.example>",
                in_reply_to="<w1@cascade-carriers.example>",
            ),
        ],
    )
    assert await _run_inbox(inbox, "--run", "1") == 0
    (row,) = _crm_rows(inbox)
    assert "Saturday night" in row["quote"] and row["msg_count"] == 2

    assert await _run_inbox(inbox, "--forget", "nadia@cascade-carriers.example") == 0
    (row,) = _crm_rows(inbox)
    assert row["msg_count"] == 1 and row["last_msg_at"] == "2026-09-09T09:00:00+00:00"
    assert "Saturday night" not in row["quote"] and "weekend deploy" in row["quote"]
    assert "Saturday night" not in (inbox.out / "crm.csv").read_text()


async def test_a_reply_that_is_all_quoted_history_is_not_a_duplicate(inbox):
    quoted = "> thanks, will do\n> -- sent from my phone"
    _own_inbox(
        inbox,
        [
            _mail(
                "Ana Ruiz <ana@cascade-carriers.example>",
                "Re: Webhook retries not firing",
                quoted,
                "2026-09-09T09:00:00+00:00",
                message_id="<q1@cascade-carriers.example>",
            ),
            _mail(
                "Ben Achebe <ben@breadbasket-bakeries.example>",
                "Re: Do you do last-mile grocery delivery routing?",
                quoted,
                "2026-09-09T09:30:00+00:00",
                message_id="<q2@breadbasket-bakeries.example>",
            ),
        ],
    )
    assert await _run_inbox(inbox, "--run", "1") == 0
    assert len(_crm_rows(inbox)) == 2


async def test_forget_erases_a_sender_without_needing_a_provider(inbox):
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
