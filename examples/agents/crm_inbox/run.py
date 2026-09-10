#!/usr/bin/env python3
"""CRM inbox: every inbound email becomes one CRM record, and nothing is ever sent.

    python run.py                      run 1  (a full morning: 18 messages, 16 threads)
    python run.py --run 2              run 2  (three more, one of them on an open thread)
    python run.py --run 1              again  -> zero row changes and zero LLM calls
    python run.py --forget dana.okafor@umbrella-logistics.example
    python run.py --resume <task_id>   re-run only what did not finish

This makes real calls to Anthropic and costs a few cents. Set ANTHROPIC_API_KEY first.
FLUXCOMPUTE_KEY is optional: with it, the run also appears in the hosted dashboard.

The shape worth copying is the division of labour. The model is asked for one judgement per
message -- what kind of request this is, quoted from the message itself -- and code decides
everything with a consequence: what never reaches a model at all, which thread a message
belongs to, how urgent it is, which account it lands on, and what the row becomes. There is
no send path in this file, so the worst a bad classification can do is put a row in the wrong
column of a CSV a human reads.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import (  # noqa: E402
    COMPANY,
    ItemError,
    Scorecard,
    call,
    connect,
    content_hash,
    display,
    graph_line,
    make_clients,
    quote_is_verbatim,
    read_json,
    run_id,
    scrub,
    short_hash,
    utc_iso,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

# The mailbox these messages arrived in. Mail from ourselves is never a stranger, so it is
# exempt from the impersonation rules below -- a forward of a customer's complaint by our own
# operations team must not be quarantined as a look-alike of the customer.
OWN_DOMAIN = "northwind.example.com"

# Below this the model is telling us it does not know. A thread under the floor is routed to a
# human rather than filed, which is cheaper than filing it wrongly and hearing about it later.
CONFIDENCE_FLOOR = 0.60

# Bump on any edit to the rubric or to REQUEST_TYPES. Classifications are cached per
# (message, prompt_version), so a bump is what re-reads the whole inbox; without it a taxonomy
# fix would apply only to mail that happens to arrive next.
PROMPT_VERSION = 2

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS run (
  task_id TEXT PRIMARY KEY, started_at TEXT, finished_at TEXT,
  status TEXT, inbox INTEGER, stats_json TEXT);

CREATE TABLE IF NOT EXISTS message (
  message_id TEXT PRIMARY KEY, thread_key TEXT NOT NULL,
  sender TEXT NOT NULL, sender_domain TEXT NOT NULL, display_name TEXT,
  subject TEXT, body TEXT, body_hash TEXT NOT NULL, sent_at TEXT NOT NULL,
  gate TEXT NOT NULL DEFAULT 'pass', gate_flags TEXT NOT NULL DEFAULT '',
  redactions INTEGER NOT NULL DEFAULT 0, first_seen_task TEXT);
CREATE INDEX IF NOT EXISTS message_thread ON message(thread_key);
CREATE INDEX IF NOT EXISTS message_body ON message(body_hash);

CREATE TABLE IF NOT EXISTS classification (
  message_id TEXT NOT NULL, prompt_version INTEGER NOT NULL,
  request_type TEXT, confidence REAL, quote TEXT, summary TEXT,
  suspicious INTEGER, language TEXT, review_reason TEXT,
  meta_json TEXT, created_at TEXT,
  PRIMARY KEY (message_id, prompt_version));

CREATE TABLE IF NOT EXISTS crm (
  thread_key TEXT PRIMARY KEY,
  account_id TEXT, account_name TEXT, account_kind TEXT, sender_domain TEXT,
  request_type TEXT, status TEXT, priority INTEGER, confidence REAL,
  next_action TEXT, summary TEXT, quote TEXT, language TEXT,
  gate_flags TEXT, msg_count INTEGER, first_msg_at TEXT, last_msg_at TEXT,
  first_seen_task TEXT, last_seen_task TEXT);
"""

# The columns that make up the record. Everything outside this list is provenance -- which run
# last touched the row -- and provenance changing is not the record changing, which is what
# lets "re-running the same inbox changes nothing" be a checkable claim rather than a hope.
RECORD_COLUMNS = (
    "account_id",
    "account_name",
    "account_kind",
    "sender_domain",
    "request_type",
    "status",
    "priority",
    "confidence",
    "next_action",
    "summary",
    "quote",
    "language",
    "gate_flags",
    "msg_count",
    "first_msg_at",
    "last_msg_at",
)

# last_msg_at is an ISO UTC string of fixed width, so the string comparison in the WHERE clause
# is a real time comparison. `>=` rather than `>`: a message arriving out of order must never
# overwrite a newer thread state, but re-processing the message we already have must still be
# able to correct the row after a taxonomy change.
UPSERT_SQL = (
    "INSERT INTO crm(thread_key, "
    + ", ".join(RECORD_COLUMNS)
    + ", first_seen_task, last_seen_task) VALUES ("
    + ",".join("?" * (len(RECORD_COLUMNS) + 3))
    + ") ON CONFLICT(thread_key) DO UPDATE SET "
    + ", ".join(f"{c}=excluded.{c}" for c in RECORD_COLUMNS)
    + ", last_seen_task=excluded.last_seen_task"
    " WHERE excluded.last_msg_at >= crm.last_msg_at"
)

# The taxonomy is one table because the two things a request type decides -- how it is described
# to the model, and how urgent it starts out -- have to stay consistent with each other. Kept in
# a prompt and a lookup table separately, a class ends up described one way and scored another,
# and the drift is invisible until someone reads both.
REQUEST_TYPES: dict[str, dict[str, Any]] = {
    # The boundary between these two is where the first real run missed: "we want to run a
    # pilot, who do we sign with" is a buying question, not a request to be shown the product.
    # A demo_request asks to be scheduled a demonstration; everything about price, fit, pilots,
    # contracts and onboarding is a sales_inquiry.
    "sales_inquiry": {
        "base_priority": 3,
        "desc": (
            "asking what we do, what it costs, whether it fits their operation, or how to"
            " buy, pilot, sign or onboard"
        ),
    },
    "demo_request": {
        "base_priority": 3,
        "desc": "an explicit ask to schedule a demo or a walkthrough of the product",
    },
    "support": {
        "base_priority": 3,
        "desc": "something is broken, slow or surprising for someone already using the product",
    },
    "billing": {
        "base_priority": 3,
        "desc": "an invoice, a charge, a refund, a payment method or a plan change",
    },
    "partnership": {
        "base_priority": 2,
        "desc": "a proposal to integrate, resell or market together",
    },
    "press": {
        "base_priority": 2,
        "desc": "a journalist or analyst asking for comment, data or an interview",
    },
    "vendor": {
        "base_priority": 1,
        "desc": "an unsolicited pitch selling something to us",
    },
    "recruiting": {
        "base_priority": 1,
        "desc": "a recruiter or a candidate, about hiring, in either direction",
    },
    "other": {
        "base_priority": 2,
        "desc": "anything else, including a message too vague to place",
    },
}

# Two labels the model never produces. A message the gate stopped still has to appear in the
# same confusion matrix as one the model classified, or a phishing mail that slipped through to
# the classifier would be scored as a merely mediocre `billing` prediction instead of the miss
# it is.
GATE_LABELS = ("ignored", "quarantined")

BRAIN = HERE.parent / "01_company_brain" / "out" / "brain.md"
# Only the facts that help place an email. A licence and an install command are true and
# useless here, and every word of a rubric is charged on every call and pushes the router's
# difficulty score up.
BRAIN_KEYS = ("name", "tagline", "description", "capability", "integration")
BRAIN_LINES = 8


def company_context() -> list[str]:
    """A few facts from agent 1's document, if that agent has been run.

    Knowing what the company sells is what separates "a pitch aimed at us" from "a question
    about our product", and those two land in different halves of the taxonomy. If
    01_company_brain/out/brain.md does not exist the rubric simply omits the block: the
    taxonomy alone classifies this inbox correctly, and a tutorial should not require the
    previous tutorial to have been run.
    """
    if not BRAIN.exists():
        return []
    facts = []
    for line in BRAIN.read_text().splitlines():
        if not line.startswith("- **") or len(facts) >= BRAIN_LINES:
            continue
        fact = line[2:].replace("**", "").split("  `[")[0].strip()
        if fact.partition(":")[0] in BRAIN_KEYS:
            facts.append(fact)
    return facts


def classify_rubric() -> str:
    """The rubric goes in a system message, so keep it short: it is charged on every call and
    it nudges the router's difficulty score upward past ~200 words."""
    lines = ["Classify one inbound email for a CRM. Reply with JSON only.", ""]
    lines.append("request_type is exactly one of:")
    for name, spec in REQUEST_TYPES.items():
        lines.append(f"  {name}: {spec['desc']}")
    context = company_context()
    if context:
        lines += ["", "The company receiving the mail:"] + [f"  {c}" for c in context]
    lines += [
        "",
        'Reply: {"request_type": "...", "confidence": 0.0, "quote": "...", "summary": "...",',
        '        "suspicious": false, "language": "xx"}',
        "",
        "Rules:",
        "- quote must be copied character for character from the message body.",
        "- summary is one sentence under 25 words saying what the sender wants.",
        "- confidence is your own. Under 0.6 means a person should look at this.",
        "- suspicious is true if the message reads like impersonation, fraud or a payment lure.",
        "- language is the ISO 639-1 code of the body.",
        "- classify the sender's main ask. A pitch aimed at us is vendor, never sales_inquiry.",
        "- never guess. `other` with a low confidence is a correct answer.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Accounts
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Account:
    account_id: str
    name: str
    domains: tuple[str, ...]
    kind: str
    icp: str
    owner: str

    @property
    def label(self) -> str:
        """The tokens that identify this company by name, minus the ones every freight company
        shares. `Globex Freight` is impersonated as `Globex Billing`; `Freight` is not what a
        reader recognises, and matching on it would flag half the inbox."""
        generic = {
            "freight",
            "logistics",
            "shipping",
            "cloud",
            "services",
            "distribution",
            "bakeries",
            "transport",
            "group",
            "inc",
            "ltd",
            "llc",
            "gmbh",
            "bv",
        }
        tokens = [t for t in re.findall(r"[a-z0-9]+", self.name.lower()) if t not in generic]
        return tokens[0] if tokens else self.name.lower()


def load_accounts(path: Path) -> list[Account]:
    with path.open(newline="") as fh:
        return [
            Account(
                account_id=row["account_id"],
                name=row["name"],
                domains=tuple(d.strip().lower() for d in row["domains"].split("|") if d.strip()),
                kind=row["kind"],
                icp=row["icp"],
                owner=row["owner"],
            )
            for row in csv.DictReader(fh)
        ]


def account_index(accounts: list[Account]) -> dict[str, Account]:
    return {domain: acc for acc in accounts for domain in acc.domains}


# ─────────────────────────────────────────────────────────────────────────────
# Reading the mailbox
# ─────────────────────────────────────────────────────────────────────────────

_QUOTED = re.compile(r"^\s*>.*$", re.MULTILINE)
_REPLY_MARKER = re.compile(r"^\s*On .{0,200}wrote:\s*$", re.MULTILINE)
_FWD_MARKER = re.compile(r"^-{2,}\s*Forwarded message\s*-{2,}\s*$", re.MULTILINE | re.IGNORECASE)
_SUBJECT_PREFIX = re.compile(r"^\s*(?:(?:re|fw|fwd|aw|sv)\s*(?:\[\d+\])?\s*:\s*)+", re.IGNORECASE)


def decode_qp(text: str) -> str:
    import quopri

    return quopri.decodestring(text.encode()).decode("utf-8", errors="replace")


def decode_header(value: str) -> str:
    from email.header import decode_header as _dh

    out = []
    for raw, enc in _dh(value or ""):
        out.append(raw.decode(enc or "utf-8", errors="replace") if isinstance(raw, bytes) else raw)
    return "".join(out)


def split_address(value: str) -> tuple[str, str, str]:
    """`Display Name <user@host>` -> (display name, address, domain), all lowercased but the name.

    A display name and an address are two different claims about who sent a message. Mail
    clients show the first and check nothing, which is the entire mechanism behind the
    look-alike rules further down; keeping them apart here is what makes those rules possible.
    """
    from email.utils import parseaddr

    name, addr = parseaddr(value or "")
    addr = addr.lower()
    return decode_header(name).strip(), addr, addr.rpartition("@")[2]


def unwrap_forward(body: str) -> tuple[str, bool]:
    """Return the forwarded message, not the note wrapped around it.

    A colleague forwarding a customer's complaint has not written a new inbound request. The
    text that matters is below the marker, and taking it means the forward hashes to the same
    body as the original and is dropped instead of opening a second thread about one problem.
    """
    match = _FWD_MARKER.search(body)
    if not match:
        return body, False
    rest = body[match.end() :]
    head, _, tail = rest.partition("\n\n")  # the quoted From/Date/Subject/To block
    return (tail or head).strip(), True


def strip_quoted(body: str) -> str:
    """Drop quoted history, so a five-message thread is not re-read five times."""
    cut = _REPLY_MARKER.search(body)
    if cut:
        body = body[: cut.start()]
    return _QUOTED.sub("", body).strip()


def norm_id(value: str | None) -> str:
    return (value or "").strip().strip("<>").strip().lower()


def thread_key(msg: dict[str, Any], domain: str, sent_at: str) -> str:
    """Which conversation this message belongs to.

    References carries the whole ancestry and its first entry is the root, so it survives a
    client that rewrites In-Reply-To and a reply sent from a phone. The last resort is a hash
    of who wrote, about what, on which day: gateways and web forms do strip Message-ID, and a
    thread with no key at all would insert a fresh CRM row for every message.
    """
    refs = msg.get("references") or []
    for candidate in (refs[0] if refs else None, msg.get("in_reply_to"), msg.get("message_id")):
        key = norm_id(candidate)
        if key:
            return key
    subject = _SUBJECT_PREFIX.sub("", decode_header(msg.get("subject", ""))).strip().lower()
    return "synth:" + content_hash(f"{domain}|{subject}|{sent_at[:10]}")[:32]


def to_utc(value: str) -> str:
    """Every date in the inbox carries an offset; every date in the database is UTC.

    Sorting or comparing local wall-clock strings puts an 18:20 message from Osaka after a
    09:14 message from Ohio, which is backwards, and it is the thread's newest message that
    decides what the CRM row says.
    """
    return utc_iso(datetime.fromisoformat(value))


@dataclass
class Message:
    message_id: str
    thread_key: str
    sender: str
    sender_domain: str
    display_name: str
    subject: str
    body: str
    body_hash: str
    sent_at: str
    reply_to_domain: str
    auto_submitted: str
    bulk: bool
    redactions: int


def read_inbox(path: Path) -> list[Message]:
    out: list[Message] = []
    for raw in read_json(path):
        display, sender, domain = split_address(raw["from"])
        body = raw["body"]
        if raw.get("content_transfer_encoding") == "quoted-printable":
            body = decode_qp(body)
        body, _ = unwrap_forward(body)
        body = strip_quoted(body)
        # Scrubbed here, at the edge, rather than in front of the model: the local database is
        # a copy of the inbox that outlives it, and a card number in a support thread is not
        # something a CRM has any reason to keep.
        body, body_counts = scrub(body)
        subject, subject_counts = scrub(decode_header(raw.get("subject", "")))
        sent_at = to_utc(raw["date"])
        out.append(
            Message(
                message_id=norm_id(raw.get("message_id")),
                thread_key=thread_key(raw, domain, sent_at),
                sender=sender,
                sender_domain=domain,
                display_name=display,
                subject=subject,
                body=body,
                body_hash=content_hash(body),
                sent_at=sent_at,
                reply_to_domain=split_address(raw.get("reply_to", ""))[2],
                auto_submitted=str(raw.get("auto_submitted", "")).strip().lower(),
                bulk=bool(raw.get("list_unsubscribe") or raw.get("list_id")),
                redactions=sum(body_counts.values()) + sum(subject_counts.values()),
            )
        )
    out.sort(key=lambda m: (m.sent_at, m.message_id))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# The gate: what never reaches a model
# ─────────────────────────────────────────────────────────────────────────────

_BOUNCE_SENDER = re.compile(r"^(mailer-daemon|postmaster|no-?reply)@")
_BOUNCE_SUBJECT = re.compile(
    r"^\s*(undeliverable|delivery status notification|mail delivery (failed|subsystem))",
    re.IGNORECASE,
)
_LURE = re.compile(r"(verify|update|confirm).{0,20}(payment|account|password)", re.IGNORECASE)
_LINK = re.compile(r"https?://([^/\s>\"']+)", re.IGNORECASE)
_CONFUSABLE_SUFFIXES = ("-secure", "-billing", "-support")


def is_machine_mail(msg: Message) -> str:
    """Auto-replies, bulk mail and bounces, recognised from headers a sender sets themselves.

    Checked before anything else and answered with a row, not an exception: an out-of-office is
    an ordinary thing for an inbox to receive, and three of them in a row raising would be
    classified as a stall, which is an alert.
    """
    if msg.auto_submitted and msg.auto_submitted != "no":
        return "auto_reply"
    if msg.bulk:
        return "bulk"
    if _BOUNCE_SENDER.match(msg.sender) or _BOUNCE_SUBJECT.match(msg.subject):
        return "bounce"
    return ""


def registrable(host: str) -> str:
    """The last two labels of a host. Good enough for the single-label TLDs in this inbox; a
    real deployment wants the public suffix list, or `mail.example.co.uk` reads as `co.uk`."""
    parts = host.lower().strip(".").split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host.lower()


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def phishing_flags(msg: Message, accounts: list[Account], known: set[str]) -> list[str]:
    """Four cheap checks for a message pretending to be from someone we do business with.

    None of them reads the message's argument, which is the point: a lure is written to be
    convincing, and a model asked whether it is convincing will often agree. Every rule here
    compares two facts the sender cannot make agree -- the name they display against the domain
    they sent from, the domain they sent from against a domain we already know, the link
    against the sender, the reply address against the from address.

    Returns rule names, never message content: these end up as a count on a node.
    """
    flags: list[str] = []
    base = msg.sender_domain.partition(".")[0]

    if msg.sender_domain not in known and msg.sender_domain != OWN_DOMAIN:
        name_tokens = set(re.findall(r"[a-z0-9]+", msg.display_name.lower()))
        for acc in accounts:
            if acc.label in name_tokens and msg.sender_domain not in acc.domains:
                flags.append("display_name_mismatch")
                break
        for acc in accounts:
            for domain in acc.domains:
                crm_base = domain.partition(".")[0]
                if base == crm_base:
                    continue
                near = len(crm_base) >= 5 and levenshtein(base, crm_base) <= 2
                if near or base in {crm_base + s for s in _CONFUSABLE_SUFFIXES}:
                    flags.append("lookalike_domain")
                    break
            if "lookalike_domain" in flags:
                break

    if _LURE.search(msg.body):
        sender_site = registrable(msg.sender_domain)
        for host in _LINK.findall(msg.body):
            if registrable(host.split("@")[-1]) != sender_site:
                flags.append("link_domain_mismatch")
                break

    if msg.reply_to_domain and msg.reply_to_domain != msg.sender_domain:
        flags.append("reply_to_mismatch")
    return flags


# ─────────────────────────────────────────────────────────────────────────────
# What the record says
# ─────────────────────────────────────────────────────────────────────────────

_URGENT = re.compile(r"\b(urgent|down|outage|asap)\b", re.IGNORECASE)


def priority_of(request_type: str, kind: str, subject: str, msg_count: int) -> int:
    """Priority is computed, not asked for.

    A model that returns a number returns a slightly different one next week, from a different
    model, for the same email, and nobody can say why one thread is a 4. Every term here is
    visible in the row it scores, so a sales lead can be told exactly why their thread sits
    below an outage, and the same inputs always give the same number -- which is also what
    makes re-running the inbox a zero-row diff instead of a churn of priorities.
    """
    score = REQUEST_TYPES[request_type]["base_priority"]
    score += kind == "customer"  # someone who already pays is not a lead
    score += bool(_URGENT.search(subject))  # their word for it, not ours
    score += msg_count >= 3  # a thread nobody closed is a thread going wrong
    return max(1, min(5, score))


def next_action(request_type: str, status: str, account: Account | None) -> str:
    """One sentence a human can act on. Suggested only: this agent has no send path."""
    if status == "quarantined":
        return "hold for security review, do not reply"
    if status == "ignored":
        return "no action"
    if status == "needs_review":
        return "triage by hand before any reply"
    if request_type in ("sales_inquiry", "demo_request"):
        if account is None:
            return "qualify in the inbound queue"
        if account.icp == "no":
            return "send the self-serve docs, do not route to sales"
        if account.kind == "churned":
            return f"route to win-back, owner {account.owner}"
        return f"route to sales, owner {account.owner}"
    if request_type == "support":
        if account is None:
            return "ask which account this relates to before opening a ticket"
        return f"open a support ticket against {account.account_id}"
    return {
        "billing": "route to finance",
        "partnership": "route to partnerships",
        "press": "route to comms, do not comment directly",
        "vendor": "acknowledge, no owner",
        "recruiting": "acknowledge, no owner",
    }.get(request_type, "triage by hand before any reply")


# ─────────────────────────────────────────────────────────────────────────────
# The run
# ─────────────────────────────────────────────────────────────────────────────


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, default=1, choices=(1, 2))
    ap.add_argument("--resume", metavar="TASK_ID")
    ap.add_argument("--forget", metavar="SENDER", help="erase one sender from the database")
    ap.add_argument("--attempt", type=int, default=1)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    db = connect(OUT / "crm.db", SCHEMA_SQL)
    accounts = load_accounts(COMPANY / "crm_seed.csv")
    by_domain = account_index(accounts)

    if args.forget:
        # No model, no network, no task scope. An erasure request is answered from the local
        # store or it is not answered, and making it depend on a provider key would mean the
        # one operation a person is entitled to can fail for a reason that is none of their
        # business.
        try:
            forget(db, args.forget)
            return 0
        finally:
            db.close()

    rubric = classify_rubric()
    card = Scorecard("CRM inbox")
    inbox = COMPANY / "inbox_crm" / f"run{args.run}.json"
    print(f"  rubric {len(rubric.split())} words")

    clients = await make_clients()
    task_id = args.resume or run_id("crm", args.attempt)
    print(f"  task {task_id}")

    try:
        with clients.anthropic.task("crm-inbox", task_id=task_id):
            db.execute(
                "INSERT OR REPLACE INTO run(task_id, started_at, status, inbox) VALUES (?,?,?,?)",
                (task_id, utc_iso(), "running", args.run),
            )

            # ── fetch ────────────────────────────────────────────────────────
            with clients.anthropic.step("fetch") as s:
                messages = read_inbox(inbox)
                kept, duplicates, redactions = [], 0, 0
                for msg in messages:
                    twin = db.execute(
                        "SELECT message_id FROM message WHERE body_hash=? AND message_id<>?",
                        (msg.body_hash, msg.message_id),
                    ).fetchone()
                    if twin is not None:
                        # The same words under a new Message-ID: a forward, or the same
                        # complaint sent to two of our addresses. Threading alone would open a
                        # second record for one problem, and a person would answer it twice.
                        duplicates += 1
                        continue
                    kept.append(msg)
                    redactions += msg.redactions
                    db.execute(
                        "INSERT INTO message(message_id, thread_key, sender, sender_domain,"
                        " display_name, subject, body, body_hash, sent_at, redactions,"
                        " first_seen_task) VALUES (?,?,?,?,?,?,?,?,?,?,?)"
                        " ON CONFLICT(message_id) DO UPDATE SET thread_key=excluded.thread_key,"
                        " subject=excluded.subject, body=excluded.body,"
                        " body_hash=excluded.body_hash, sent_at=excluded.sent_at",
                        (
                            msg.message_id,
                            msg.thread_key,
                            msg.sender,
                            msg.sender_domain,
                            msg.display_name,
                            msg.subject,
                            msg.body,
                            msg.body_hash,
                            msg.sent_at,
                            msg.redactions,
                            task_id,
                        ),
                    )
                threads = {m.thread_key for m in kept}
                s.set_attribute("messages", len(kept))
                s.set_attribute("threads", len(threads))
                s.set_attribute("duplicates", duplicates)
                s.set_attribute("redactions", redactions)
                s.set_output(
                    json.dumps(
                        {
                            "messages": len(kept),
                            "threads": len(threads),
                            "duplicates": duplicates,
                            "redactions": redactions,
                        }
                    )
                )
                print(
                    f"  fetch: {len(kept)} messages, {len(threads)} threads, "
                    f"{duplicates} dropped as duplicate, {redactions} redactions"
                )

            # ── gate ─────────────────────────────────────────────────────────
            with clients.anthropic.step("gate") as s:
                known_domains = set(by_domain)
                to_classify: list[Message] = []
                ignored = quarantined = flagged_once = 0
                for msg in kept:
                    reason = is_machine_mail(msg)
                    if reason:
                        ignored += 1
                        gate, flags = "ignored", [reason]
                    else:
                        flags = phishing_flags(msg, accounts, known_domains)
                        if len(flags) >= 2:
                            # Two independent rules agreeing is the threshold because any one
                            # of them alone has an innocent explanation: a marketing platform
                            # really does set Reply-To elsewhere. Quarantine costs zero LLM
                            # calls, which is the other reason the gate runs first.
                            quarantined += 1
                            gate = "quarantined"
                        else:
                            gate = "pass"
                            if flags:
                                flagged_once += 1
                            to_classify.append(msg)
                    db.execute(
                        "UPDATE message SET gate=?, gate_flags=? WHERE message_id=?",
                        (gate, ",".join(sorted(flags)), msg.message_id),
                    )
                s.set_attribute("to_classify", len(to_classify))
                s.set_attribute("ignored", ignored)
                s.set_attribute("quarantined", quarantined)
                s.set_attribute("flagged_once", flagged_once)
                s.set_output(
                    json.dumps(
                        {
                            "to_classify": len(to_classify),
                            "ignored": ignored,
                            "quarantined": quarantined,
                            "flagged_once": flagged_once,
                        }
                    )
                )
                print(
                    f"  gate: {len(to_classify)} to classify, {ignored} ignored, "
                    f"{quarantined} quarantined, {flagged_once} flagged once"
                )

            # ── classify ─────────────────────────────────────────────────────
            classified = cached = failed = 0
            for msg in to_classify:
                done = db.execute(
                    "SELECT 1 FROM classification WHERE message_id=? AND prompt_version=?",
                    (msg.message_id, PROMPT_VERSION),
                ).fetchone()
                if done:
                    cached += 1
                    continue
                thread_size = db.execute(
                    "SELECT COUNT(*) c FROM message WHERE thread_key=?", (msg.thread_key,)
                ).fetchone()["c"]
                user = (
                    f"Subject: {msg.subject}\n"
                    f"From domain: {msg.sender_domain}\n"
                    f"Messages in this thread: {thread_size}\n\n"
                    f"{msg.body}"
                )
                # One name for every message, the id in an attribute: a hundred fan-out steps
                # under one name read as one row in a graph view rather than a hundred. The id
                # is a hash, because a node name is part of the graph -- and of the telemetry
                # when a key is set -- whether or not content capture is on, and a subject line
                # is content.
                try:
                    with clients.anthropic.step("classify") as s:
                        s.set_attribute("message", short_hash(msg.message_id))
                        s.set_attribute("thread", short_hash(msg.thread_key))
                        if msg.redactions:
                            s.set_attribute("redactions", msg.redactions)
                        result = card.record(
                            await call(clients.anthropic, rubric=rubric, user=user)
                        )
                        verdict = validate(result.data, msg.body)
                        db.execute(
                            "INSERT OR REPLACE INTO classification VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                            (
                                msg.message_id,
                                PROMPT_VERSION,
                                verdict["request_type"],
                                verdict["confidence"],
                                verdict["quote"],
                                verdict["summary"],
                                int(verdict["suspicious"]),
                                verdict["language"],
                                verdict["review_reason"],
                                json.dumps(
                                    {
                                        "model": result.model_selected,
                                        "tier": result.difficulty_label,
                                        "cost": result.cost_usd,
                                    }
                                ),
                                utc_iso(),
                            ),
                        )
                        classified += 1
                        s.set_attribute("request_type", verdict["request_type"])
                        s.set_attribute("review", bool(verdict["review_reason"]))
                        s.set_attribute("tier", result.difficulty_label)
                        s.set_output(
                            json.dumps(
                                {
                                    "request_type": verdict["request_type"],
                                    "review_reason": verdict["review_reason"],
                                    "tier": result.difficulty_label,
                                }
                            )
                        )
                except ItemError:
                    # One message failing is not the inbox failing, so the loop goes on and the
                    # thread lands in review. The node is still marked failed, so three in a
                    # row -- a provider outage rather than a bad reply -- raises the stall
                    # alert it should.
                    failed += 1
            print(f"  classify: {classified} new, {cached} cached, {failed} failed")

            # ── resolve and upsert ───────────────────────────────────────────
            with clients.anthropic.step("resolve-and-upsert") as s:
                touched = sorted({m.thread_key for m in kept})
                before = snapshot(db)
                rows = [build_row(db, key, by_domain) for key in touched]
                apply_rows(db, rows, task_id)
                after = snapshot(db)
                inserted = sorted(set(after) - set(before))
                updated = sorted(k for k in before if k in after and before[k] != after[k])
                # The upsert is applied a second time to the same inputs, and the second pass
                # has to change nothing. That is the idempotence claim stated as a test rather
                # than a comment, and it holds on the first run too, so the property is checked
                # every time instead of only when somebody remembers to re-run.
                apply_rows(db, rows, task_id)
                repeat = snapshot(db)
                churn = sorted(k for k in after if after[k] != repeat.get(k))
                export_csv(db, OUT / "crm.csv")
                s.set_attribute("threads", len(rows))
                s.set_attribute("inserted", len(inserted))
                s.set_attribute("updated", len(updated))
                s.set_attribute("idempotent", not churn)
                s.set_output(
                    json.dumps(
                        {
                            "threads": len(rows),
                            "inserted": len(inserted),
                            "updated": len(updated),
                            "second_pass_changed": len(churn),
                        }
                    )
                )
                print(
                    f"  upsert: {len(inserted)} inserted, {len(updated)} updated, "
                    f"{len(rows) - len(inserted) - len(updated)} unchanged "
                    f"(second pass changed {len(churn)})"
                )

            # ── report ───────────────────────────────────────────────────────
            score = report(db, args.run, touched, clients.anthropic)

            stats = {
                "messages": len(kept),
                "threads": len(touched),
                "duplicates": duplicates,
                "ignored": ignored,
                "quarantined": quarantined,
                "classified": classified,
                "inserted": len(inserted),
                "updated": len(updated),
                **{k: v for k, v in score.items() if isinstance(v, (int, float))},
            }
            db.execute(
                "UPDATE run SET finished_at=?, status=?, stats_json=? WHERE task_id=?",
                (utc_iso(), "ok", json.dumps(stats), task_id),
            )

        rows_total = db.execute("SELECT COUNT(*) c FROM crm").fetchone()["c"]
        threads_total = db.execute("SELECT COUNT(DISTINCT thread_key) c FROM message").fetchone()[
            "c"
        ]
        placeholders = ",".join("?" * len(touched))
        reviewable = db.execute(
            f"SELECT COUNT(*) c FROM crm WHERE thread_key IN ({placeholders})"
            " AND status NOT IN ('ignored','quarantined')",
            touched,
        ).fetchone()["c"]

        card.outcomes.update(
            {
                "emails / threads": f"{len(kept)} / {len(touched)}",
                "ignored / quarantined": f"{ignored} / {quarantined}",
                "classified (new / cached)": f"{classified} / {cached}",
                "needs review": f"{score['needs_review']} of {reviewable} "
                f"({score['needs_review'] / max(reviewable, 1):.0%})",
                "rows inserted / updated": f"{len(inserted)} / {len(updated)}",
                "redactions": redactions,
                "per class (P / R)": f"macro {score['macro_precision']:.2f}"
                f" / {score['macro_recall']:.2f}",
                **{
                    f"  {name:<14}": f"{p:.2f} / {r:.2f}  n={n}"
                    for name, p, r, n in score["per_class"]
                },
                "export": display(OUT / "crm.csv"),
            }
        )
        # A precision over three threads is one miss away from any value. Below ten labelled
        # threads the figure is printed but not gated; alerting on it would be alerting on noise.
        graded = sum(n for _, _, _, n in score["per_class"])
        card.gate(
            "macro precision >= 0.85",
            graded < 10 or score["macro_precision"] >= 0.85,
            f"{score['macro_precision']:.2f} over {graded} threads"
            + (" (fewer than 10, not gated)" if graded < 10 else ""),
        )
        card.gate(
            "phishing recall == 1.0",
            score["phishing_recall"] == 1.0,
            f"{score['phishing_recall']:.2f}",
        )
        card.gate(
            "exactly one row per thread",
            rows_total == threads_total,
            f"{rows_total} rows / {threads_total} threads",
        )
        card.gate("re-run is a zero-row diff", not churn, f"{len(churn)} rows would change")
        card.gate("no message left unclassified", failed == 0, f"{failed} failed")
        card.print()
        if score["wrong"]:
            print("  threads whose label did not match: " + ", ".join(score["wrong"]))
        print(f"\n{graph_line(clients.anthropic, task_id)}\n")
        return 0 if all(passed for _, passed, _ in card.gates) else 1
    finally:
        await clients.close()
        db.close()


def validate(data: Any, body: str) -> dict[str, Any]:
    """Turn the model's reply into a record, or into a request for a human.

    Every failure here is an ordinary outcome with a row at the end of it, never an exception.
    A fabricated quote, a class outside the taxonomy and a model that says it is unsure are all
    the same event from the inbox's point of view: this one is not safe to file automatically.
    """
    data = data if isinstance(data, dict) else {}
    request_type = str(data.get("request_type", "")).strip().lower()
    quote = str(data.get("quote", "")).strip()
    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0

    reason = ""
    if request_type not in REQUEST_TYPES:
        request_type, reason = "other", "schema_invalid"
    elif not quote_is_verbatim(quote, body):
        # The one guard worth its cost. A model that has to copy a sentence out of the message
        # cannot invent the reason for its answer without inventing a string that is checkably
        # not there, and checking it is one function call rather than a second model.
        reason = "quote_not_verbatim"
    elif confidence < CONFIDENCE_FLOOR:
        reason = "low_confidence"
    elif bool(data.get("suspicious")):
        reason = "model_flagged"
    # A quote that failed the check is not evidence and is not kept. One that passed is kept
    # even when the thread goes to review: the person triaging it wants the sentence the model
    # answered from more than anyone.
    return {
        "request_type": request_type,
        "confidence": confidence,
        "quote": "" if reason in ("quote_not_verbatim", "schema_invalid") else quote,
        "summary": str(data.get("summary", "")).strip()[:300],
        "suspicious": bool(data.get("suspicious")),
        "language": str(data.get("language", "")).strip().lower()[:8] or "und",
        "review_reason": reason,
    }


def build_row(db: sqlite3.Connection, key: str, by_domain: dict[str, Account]) -> tuple:
    """One CRM record from every message in a thread. Code, start to finish.

    The thread's newest message decides what the record says, because a conversation that has
    moved from a question to a pilot is about the pilot. The account comes from the domain, not
    from the model: a customer writing from their second domain is still that customer, and no
    amount of prompt is a substitute for the lookup.
    """
    msgs = db.execute(
        "SELECT * FROM message WHERE thread_key=? ORDER BY sent_at, message_id", (key,)
    ).fetchall()
    newest = msgs[-1]
    account = by_domain.get(newest["sender_domain"])
    verdict = db.execute(
        "SELECT * FROM classification WHERE message_id=? AND prompt_version=?",
        (newest["message_id"], PROMPT_VERSION),
    ).fetchone()

    gate = newest["gate"]
    flags = newest["gate_flags"]
    if gate in ("ignored", "quarantined"):
        status, request_type, confidence, summary, quote, language = gate, None, None, "", "", ""
    elif verdict is None:
        status, request_type, confidence, summary, quote, language = (
            "needs_review",
            None,
            None,
            "",
            "",
            "",
        )
    else:
        request_type = verdict["request_type"]
        confidence = verdict["confidence"]
        summary, quote, language = verdict["summary"], verdict["quote"], verdict["language"]
        # A single phishing rule is not enough to refuse the message, and too much to file it
        # unread. The model still runs, and a person still looks.
        status = "needs_review" if (verdict["review_reason"] or flags) else "open"

    kind = account.kind if account else "unknown"
    priority = (
        0 if request_type is None else priority_of(request_type, kind, newest["subject"], len(msgs))
    )
    return (
        key,
        account.account_id if account else "",
        account.name if account else "",
        kind,
        newest["sender_domain"],
        request_type,
        status,
        priority,
        confidence,
        next_action(request_type or "other", status, account),
        summary,
        quote,
        language,
        flags,
        len(msgs),
        msgs[0]["sent_at"],
        newest["sent_at"],
    )


def apply_rows(db: sqlite3.Connection, rows: list[tuple], task_id: str) -> None:
    for row in rows:
        db.execute(UPSERT_SQL, (*row, task_id, task_id))


def snapshot(db: sqlite3.Connection) -> dict[str, tuple]:
    """Every record as it stands, so a diff is a comparison rather than a row count."""
    columns = ", ".join(RECORD_COLUMNS)
    return {
        r["thread_key"]: tuple(r[c] for c in RECORD_COLUMNS)
        for r in db.execute(f"SELECT thread_key, {columns} FROM crm")
    }


def export_csv(db: sqlite3.Connection, path: Path) -> None:
    """Write to a temporary file and rename, so nothing ever reads a half-written CRM.

    os.replace is atomic on the same filesystem. Writing in place means a reader who opens the
    file during the export sees a file that is valid CSV and wrong, which is worse than an
    error. The export carries the record only, not which run last touched it, so an unchanged
    inbox produces a byte-identical file.
    """
    columns = ("thread_key", *RECORD_COLUMNS)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        writer.writerows(
            db.execute(f"SELECT {', '.join(columns)} FROM crm ORDER BY priority DESC, thread_key")
        )
    os.replace(tmp, path)


def report(db: sqlite3.Connection, run: int, touched: list[str], client) -> dict[str, Any]:
    """Score this run's threads against what a person said they were.

    Precision and recall per class, not accuracy: an inbox is mostly a handful of common
    classes, and a classifier that answered `support` to everything would score well on
    accuracy and be useless. Undefined values -- a class nothing was assigned to, a class
    nothing expected -- count as 1.0, because there is no error to charge for.
    """
    with client.step("report") as s:
        spec = read_json(COMPANY / "crm_labels.json")
        expected: dict[str, dict[str, Any]] = {}
        unmatched = 0
        for label in spec[f"run{run}"]:
            row = db.execute(
                "SELECT thread_key FROM message WHERE message_id=?", (norm_id(label["match"]),)
            ).fetchone()
            if row is None:
                unmatched += 1
                continue
            expected[row["thread_key"]] = label

        # Only labelled threads are scored. A thread nobody has an opinion about is not
        # evidence either way, and counting it as a false positive would make the score fall
        # every time the inbox grows.
        predicted: dict[str, dict[str, Any]] = {}
        for key in touched:
            row = db.execute("SELECT * FROM crm WHERE thread_key=?", (key,)).fetchone()
            if row is None or key not in expected:
                continue
            label = (
                row["status"]
                if row["status"] in GATE_LABELS
                else (row["request_type"] or "unclassified")
            )
            predicted[key] = {"label": label, "status": row["status"], "priority": row["priority"]}

        classes = {lb["request_type"] for lb in expected.values()} | {
            p["label"] for p in predicted.values()
        }
        per_class, precisions, recalls = [], [], []
        for name in sorted(classes):
            tp = sum(
                1
                for k, lb in expected.items()
                if lb["request_type"] == name and predicted.get(k, {}).get("label") == name
            )
            fp = sum(
                1
                for k, p in predicted.items()
                if p["label"] == name and expected.get(k, {}).get("request_type") != name
            )
            fn = sum(
                1
                for k, lb in expected.items()
                if lb["request_type"] == name and predicted.get(k, {}).get("label") != name
            )
            precision = tp / (tp + fp) if (tp + fp) else 1.0
            recall = tp / (tp + fn) if (tp + fn) else 1.0
            per_class.append((name, precision, recall, tp + fn))
            precisions.append(precision)
            recalls.append(recall)

        wrong = sorted(
            short_hash(k)
            for k, lb in expected.items()
            if predicted.get(k, {}).get("label") != lb["request_type"]
            or predicted.get(k, {}).get("status") != lb["status"]
            or predicted.get(k, {}).get("priority") != lb["priority"]
        )
        phish = next((r for n, _, r, _ in per_class if n == "quarantined"), 1.0)
        macro_p = sum(precisions) / len(precisions) if precisions else 1.0
        macro_r = sum(recalls) / len(recalls) if recalls else 1.0
        needs_review = sum(1 for p in predicted.values() if p["status"] == "needs_review")

        out = {
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "phishing_recall": phish,
            "needs_review": needs_review,
            "labelled": len(expected),
            "unmatched_labels": unmatched,
            "wrong": wrong,
            "per_class": per_class,
        }
        s.set_attribute("labelled", len(expected))
        s.set_attribute("macro_precision_pct", round(macro_p * 100))
        s.set_attribute("macro_recall_pct", round(macro_r * 100))
        s.set_attribute("phishing_recall_pct", round(phish * 100))
        s.set_attribute("needs_review", needs_review)
        s.set_attribute("wrong", len(wrong))
        s.set_output(
            json.dumps(
                {
                    "labelled": len(expected),
                    "macro_precision": round(macro_p, 3),
                    "macro_recall": round(macro_r, 3),
                    "wrong": len(wrong),
                }
            )
        )
        print(
            f"  report: macro precision {macro_p:.2f}, macro recall {macro_r:.2f}, "
            f"{len(expected) - len(wrong)}/{len(expected)} threads exactly right"
            + (f", {unmatched} labels matched nothing" if unmatched else "")
        )
        return out


def forget(db: sqlite3.Connection, sender: str) -> None:
    """Erase one person from the store: their messages, what the model said about them, and any
    record left with nothing behind it.

    A CRM row that outlives its last message is the failure mode worth naming. It looks like an
    ordinary record, so nobody deletes it, and it is the copy that gets exported to the next
    system. Threads that still hold other people's messages keep their row and are rebuilt from
    what remains on the next run, which is why this prints what it kept as well as what it
    removed.
    """
    sender = sender.strip().lower()
    ids = [
        r["message_id"]
        for r in db.execute("SELECT message_id FROM message WHERE lower(sender)=?", (sender,))
    ]
    if not ids:
        print(f"  forget: nothing stored for that sender ({short_hash(sender)})")
        return
    threads = {
        r["thread_key"]
        for r in db.execute(
            f"SELECT thread_key FROM message WHERE message_id IN ({','.join('?' * len(ids))})", ids
        )
    }
    placeholders = ",".join("?" * len(ids))
    classifications = db.execute(
        f"DELETE FROM classification WHERE message_id IN ({placeholders})", ids
    ).rowcount
    db.execute(f"DELETE FROM message WHERE message_id IN ({placeholders})", ids)

    emptied = kept = 0
    for key in sorted(threads):
        remaining = db.execute(
            "SELECT COUNT(*) c FROM message WHERE thread_key=?", (key,)
        ).fetchone()["c"]
        if remaining:
            kept += 1
            continue
        db.execute("DELETE FROM crm WHERE thread_key=?", (key,))
        emptied += 1
    export_csv(db, OUT / "crm.csv")
    print(
        f"  forget: messages {len(ids)}, classifications {classifications}, "
        f"rows deleted {emptied}, rows kept {kept} (threads with other people still in them)"
    )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
