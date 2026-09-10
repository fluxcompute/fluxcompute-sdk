#!/usr/bin/env python3
"""Company brain: keep one versioned document in sync with the sources that feed it.

    python run.py                      first sync   (repo v1, inbox run1, brand v1)
    python run.py --version 2          second sync  (repo v2, inbox run2, brand v2)
    python run.py --version 2          third time   -> no changes, and no LLM calls at all
    python run.py --fail-at 2          inject a provider failure at the 2nd extraction
    python run.py --resume <task_id>   re-run only what did not finish

This makes real calls to Anthropic and costs a few cents. Set ANTHROPIC_API_KEY first.
FLUXCOMPUTE_KEY is optional: with it, the run also appears in the hosted dashboard.

The shape worth copying is not the prompt. It is that the model does one job here -- turning
a paragraph into typed facts with a quote -- and code does everything else: what changed,
which source wins, what to remove, and what to publish.
"""

from __future__ import annotations

import argparse
import asyncio
import fnmatch
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

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
    normalize,
    parse_json,
    quote_is_verbatim,
    run_id,
    scrub,
    short_hash,
    utc_iso,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

# The most recent run's execution graph; tests/test_agent_examples.py asserts its shape (a
# failure leaves a resumable llm_call node, and no node name or attribute carries content).
LAST_GRAPH = None
MAX_EXTRACTIONS_PER_RUN = 40

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS run (
  task_id TEXT PRIMARY KEY, started_at TEXT, finished_at TEXT,
  status TEXT, version INTEGER, stats_json TEXT);

CREATE TABLE IF NOT EXISTS source_item (
  item_id TEXT PRIMARY KEY, source TEXT NOT NULL, external_id TEXT NOT NULL,
  content_hash TEXT NOT NULL, text TEXT NOT NULL, url TEXT,
  first_seen_run TEXT, last_seen_run TEXT,
  status TEXT NOT NULL DEFAULT 'present', missing_runs INTEGER NOT NULL DEFAULT 0,
  UNIQUE(source, external_id));

CREATE TABLE IF NOT EXISTS extraction (
  item_id TEXT NOT NULL, content_hash TEXT NOT NULL, prompt_version INTEGER NOT NULL,
  facts_json TEXT NOT NULL, meta_json TEXT, created_at TEXT,
  PRIMARY KEY (item_id, content_hash, prompt_version));

CREATE TABLE IF NOT EXISTS fact (
  fact_key TEXT PRIMARY KEY, entity TEXT, attribute TEXT, value TEXT, value_norm TEXT,
  quote TEXT, source TEXT, item_id TEXT,
  status TEXT NOT NULL DEFAULT 'active',
  missing_runs INTEGER NOT NULL DEFAULT 0,
  first_version INTEGER, last_version INTEGER, removed_version INTEGER, removed_reason TEXT);

CREATE TABLE IF NOT EXISTS conflict (
  version INTEGER, entity TEXT, attribute TEXT,
  winner_source TEXT, winner_value TEXT, loser_source TEXT, loser_value TEXT);

CREATE TABLE IF NOT EXISTS doc_version (
  version INTEGER PRIMARY KEY, task_id TEXT, created_at TEXT, facts_hash TEXT,
  added INTEGER, updated INTEGER, removed INTEGER, status TEXT, held_reason TEXT);
"""

# Bump on any edit to EXTRACT_RUBRIC. Extractions are cached per (item, hash, prompt_version),
# so a bump is what re-reads every source; without it a prompt fix would apply only to
# whatever happened to change next.
PROMPT_VERSION = 1


def extract_rubric(schema: dict[str, Any]) -> str:
    """The rubric goes in a system message, so keep it short: it is charged on every call and
    it nudges the router's difficulty score upward past ~200 words."""
    lines = ["Extract facts about a company from one document. Reply with JSON only.", ""]
    lines.append("Use ONLY these entity.attribute keys:")
    for entity, spec in schema["entities"].items():
        for attr, meta in spec["attributes"].items():
            desc = meta["desc"] if isinstance(meta, dict) else meta
            multi = " (several allowed)" if isinstance(meta, dict) and meta.get("multi") else ""
            lines.append(f"  {entity}.{attr}: {desc}{multi}")
    lines += [
        "",
        'Reply: {"facts": [{"key": "entity.attribute", "value": "...", "quote": "..."}]}',
        "",
        "Rules:",
        "- quote must be copied character for character from the document.",
        "- value is the fact itself, short and self-contained, no sentence framing.",
        "- state only what the document asserts as true now. Skip history and future plans",
        "  unless the key is roadmap.planned.",
        "- omit anything that does not fit a key above. An empty list is a valid answer.",
        "- never invent a quote. If you cannot quote it, do not report it.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Sources
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Item:
    source: str
    external_id: str
    text: str
    url: str = ""

    @property
    def item_id(self) -> str:
        return f"{self.source}:{self.external_id}"

    @property
    def hash(self) -> str:
        return content_hash(self.text)


_HEADING = re.compile(r"^##\s+(.+)$", re.MULTILINE)


def chunk_markdown(path: str, text: str) -> list[tuple[str, str]]:
    """Split a document at its second-level headings.

    A whole-file chunk means a one-line edit to a long FAQ re-extracts the entire FAQ. Section
    chunks mean one section is re-read and the rest cost nothing, which is most of why a
    scheduled sync stays cheap.
    """
    matches = list(_HEADING.finditer(text))
    if len(matches) < 2:
        return [(path, text)]
    chunks: list[tuple[str, str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        chunks.append((f"{path}#_intro", preamble))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        slug = re.sub(r"[^a-z0-9]+", "-", m.group(1).lower()).strip("-")
        chunks.append((f"{path}#{slug}", text[m.start() : end].strip()))
    return chunks


def collect_repo(root: Path, exclude: list[str]) -> list[Item]:
    items: list[Item] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(path.name, pat) for pat in exclude):
            continue
        if path.suffix.lower() not in {".md", ".txt", ".rst"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="cp1252", errors="replace")
        for external_id, chunk in chunk_markdown(rel, text):
            items.append(Item("repo", external_id, chunk, url=f"repo://{external_id}"))
    return items


_QUOTED = re.compile(r"^\s*>.*$", re.MULTILINE)
_REPLY_MARKER = re.compile(r"^\s*On .{0,80}wrote:\s*$", re.MULTILINE)


def strip_quoted(body: str) -> str:
    """Drop quoted history so a five-message thread is not re-extracted five times."""
    cut = _REPLY_MARKER.search(body)
    if cut:
        body = body[: cut.start()]
    return _QUOTED.sub("", body).strip()


def decode_qp(text: str) -> str:
    import quopri

    return quopri.decodestring(text.encode()).decode("utf-8", errors="replace")


def decode_header(value: str) -> str:
    from email.header import decode_header as _dh

    out = []
    for raw, enc in _dh(value):
        out.append(raw.decode(enc or "utf-8", errors="replace") if isinstance(raw, bytes) else raw)
    return "".join(out)


def collect_email(path: Path, allowlist: list[str]) -> tuple[list[Item], int]:
    """Read the inbox, keeping only senders a human put on the allowlist.

    The allowlist is checked here, in code, before a model ever sees the text. An inbound
    email is a stranger's writing: anything that reaches the extractor can try to talk it into
    changing the company's own description of itself.
    """
    items, rejected = [], 0
    seen_bodies: set[str] = set()
    for msg in json.loads(path.read_text()):
        sender = msg["from"].lower()
        if sender not in {a.lower() for a in allowlist}:
            rejected += 1
            continue
        body = msg["body"]
        if msg.get("content_transfer_encoding") == "quoted-printable":
            body = decode_qp(body)
        body = strip_quoted(body)
        body_hash = content_hash(body)
        if body_hash in seen_bodies:
            continue  # a forward of a message we already have: same words, new id
        seen_bodies.add(body_hash)
        subject = decode_header(msg["subject"])
        items.append(
            Item(
                "email",
                msg["message_id"],
                f"Subject: {subject}\nFrom: {sender}\n\n{body}",
                url=f"mailto:{msg['message_id']}",
            )
        )
    return items, rejected


def collect_brand(path: Path) -> list[Item]:
    return [Item("brand", path.name, path.read_text(), url=f"brand://{path.name}")]


# ─────────────────────────────────────────────────────────────────────────────
# Facts
# ─────────────────────────────────────────────────────────────────────────────


def is_multi(schema: dict[str, Any], entity: str, attribute: str) -> bool:
    meta = schema["entities"].get(entity, {}).get("attributes", {}).get(attribute)
    return isinstance(meta, dict) and bool(meta.get("multi"))


def fact_key(schema: dict[str, Any], entity: str, attribute: str, value: str) -> str:
    """One key per fact. For an attribute that holds several values at once, the value is part
    of the identity, so dropping one integration removes one fact. For a single-valued
    attribute it is not, so a new price updates the fact instead of adding a second one."""
    base = f"{entity}|{attribute}"
    if is_multi(schema, entity, attribute):
        base += f"|{normalize(value).lower()}"
    return content_hash(base)[:16]


def precedence_rank(schema: dict[str, Any], entity: str, attribute: str, source: str) -> int:
    for pattern, order in schema["precedence"].items():
        if fnmatch.fnmatch(f"{entity}.{attribute}", pattern):
            return order.index(source) if source in order else -1
    return -1


# ─────────────────────────────────────────────────────────────────────────────
# The run
# ─────────────────────────────────────────────────────────────────────────────


async def main() -> int:
    global LAST_GRAPH
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, default=1, choices=(1, 2))
    ap.add_argument("--resume", metavar="TASK_ID")
    ap.add_argument("--fail-at", type=int, metavar="N", help="fail the Nth extraction on purpose")
    ap.add_argument("--attempt", type=int, default=1)
    args = ap.parse_args()

    schema = yaml.safe_load((COMPANY / "brain_schema.yaml").read_text())
    rubric = extract_rubric(schema)
    OUT.mkdir(parents=True, exist_ok=True)
    db = connect(OUT / "brain.db", SCHEMA_SQL)
    card = Scorecard("Company brain")

    clients = await make_clients()
    task_id = args.resume or run_id("brain", args.attempt)
    print(f"  task {task_id}")
    failure: tuple[int, Item, str, str] | None = None
    changed: list[Item] = []

    try:
        with clients.anthropic.task("company-brain", task_id=task_id):
            db.execute(
                "INSERT OR REPLACE INTO run(task_id, started_at, status, version) VALUES (?,?,?,?)",
                (task_id, utc_iso(), "running", args.version),
            )

            # ── collect ──────────────────────────────────────────────────────
            repo_dir = COMPANY / f"repo_v{args.version}"
            brand_file = COMPANY / f"brand_v{args.version}.md"
            inbox = COMPANY / "inbox" / f"run{args.version}.json"
            items: list[Item] = []

            with clients.anthropic.step("collect:repo") as s:
                repo_items = collect_repo(repo_dir, schema.get("exclude", []))
                items += repo_items
                s.set_attribute("items", len(repo_items))
                s.set_output(json.dumps({"items": len(repo_items)}))

            with clients.anthropic.step("collect:email") as s:
                mail_items, rejected = collect_email(inbox, schema["email_allowlist"])
                items += mail_items
                s.set_attribute("items", len(mail_items))
                s.set_attribute("rejected_sender", rejected)
                s.set_output(json.dumps({"items": len(mail_items), "rejected": rejected}))

            with clients.anthropic.step("collect:brand") as s:
                brand_items = collect_brand(brand_file)
                items += brand_items
                s.set_attribute("items", len(brand_items))
                s.set_output(json.dumps({"items": len(brand_items)}))

            # ── diff ─────────────────────────────────────────────────────────
            with clients.anthropic.step("diff") as s:
                known = {
                    r["item_id"]: r
                    for r in db.execute("SELECT * FROM source_item WHERE status != 'removed'")
                }
                seen_now = {i.item_id: i for i in items}
                hashes_now = {i.hash: i for i in items}
                # "Processed" is an extraction row, not the hash on source_item. The hash is
                # written in this step, before any extraction runs; if it were the marker, a
                # run that failed halfway would report "0 changed" forever afterwards, which
                # is the quiet way a sync agent dies. Keyed on prompt_version too, so bumping
                # the rubric re-reads everything.
                extracted = {
                    (r["item_id"], r["content_hash"])
                    for r in db.execute(
                        "SELECT item_id, content_hash FROM extraction WHERE prompt_version=?",
                        (PROMPT_VERSION,),
                    )
                }

                changed: list[Item] = []
                renamed = 0
                for item in items:
                    prior = known.get(item.item_id)
                    if prior is None:
                        # A rename is the same bytes under a new id. Treating it as a delete
                        # plus an add would tombstone facts that never went anywhere.
                        old = next(
                            (
                                r
                                for k, r in known.items()
                                if r["content_hash"] == item.hash and k not in seen_now
                            ),
                            None,
                        )
                        if old is not None:
                            # Carry every table that keys on item_id. Missing one of them
                            # silently drops the facts: the extraction no longer joins to its
                            # source item, so reconcile sees the fact as unasserted and
                            # removes it. A rename would then read as a deletion.
                            for table in ("source_item", "extraction", "fact"):
                                db.execute(
                                    f"UPDATE {table} SET item_id=? WHERE item_id=?",
                                    (item.item_id, old["item_id"]),
                                )
                            db.execute(
                                "UPDATE source_item SET external_id=?, url=?, last_seen_run=?"
                                " WHERE item_id=?",
                                (item.external_id, item.url, task_id, item.item_id),
                            )
                            renamed += 1
                            continue
                    if (item.item_id, item.hash) not in extracted:
                        changed.append(item)
                    db.execute(
                        "INSERT INTO source_item(item_id, source, external_id, content_hash, text,"
                        " url, first_seen_run, last_seen_run, status, missing_runs)"
                        " VALUES (?,?,?,?,?,?,?,?, 'present', 0)"
                        " ON CONFLICT(item_id) DO UPDATE SET content_hash=excluded.content_hash,"
                        " text=excluded.text, last_seen_run=excluded.last_seen_run,"
                        " status='present', missing_runs=0",
                        (
                            item.item_id,
                            item.source,
                            item.external_id,
                            item.hash,
                            item.text,
                            item.url,
                            task_id,
                            task_id,
                        ),
                    )

                gone = [
                    r
                    for k, r in known.items()
                    if k not in seen_now and r["content_hash"] not in hashes_now
                ]
                for row in gone:
                    db.execute(
                        "UPDATE source_item SET missing_runs = missing_runs + 1,"
                        " status = CASE WHEN missing_runs + 1 >= ? THEN 'removed' ELSE 'missing' END"
                        " WHERE item_id = ?",
                        (schema["removal"]["grace_runs"], row["item_id"]),
                    )
                deferred = max(0, len(changed) - MAX_EXTRACTIONS_PER_RUN)
                changed = changed[:MAX_EXTRACTIONS_PER_RUN]
                s.set_attribute("changed", len(changed))
                s.set_attribute("renamed", renamed)
                s.set_attribute("missing", len(gone))
                s.set_attribute("deferred", deferred)
                s.set_output(
                    json.dumps(
                        {
                            "changed": len(changed),
                            "renamed": renamed,
                            "missing": len(gone),
                            "deferred": deferred,
                        }
                    )
                )
                print(
                    f"  diff: {len(changed)} changed, {renamed} renamed, {len(gone)} missing"
                    + (f", {deferred} deferred" if deferred else "")
                )

            # ── extract ──────────────────────────────────────────────────────
            extracted = cached = failed = redacted_items = 0
            for n, item in enumerate(changed, start=1):
                done = db.execute(
                    "SELECT 1 FROM extraction WHERE item_id=? AND content_hash=?"
                    " AND prompt_version=?",
                    (item.item_id, item.hash, PROMPT_VERSION),
                ).fetchone()
                if done:
                    cached += 1
                    continue
                clean, redactions = scrub(item.text)
                if redactions:
                    redacted_items += 1
                # Same name for every extraction, item id in an attribute: forty fan-out
                # steps under one name read as one row in a graph view rather than forty.
                try:
                    with clients.anthropic.step("extract") as s:
                        s.set_attribute("item", short_hash(item.item_id))
                        s.set_attribute("source", item.source)
                        if redactions:
                            s.set_attribute("redactions", sum(redactions.values()))
                        try:
                            if args.fail_at == n:
                                # A model id that the provider retired. This is the failure
                                # that actually happens in production, it is free, and
                                # resume() fixes it by routing the same step to a live model.
                                result = await call(
                                    clients.anthropic,
                                    rubric=rubric,
                                    user=clean,
                                    model="claude-3-5-haiku-20241022",
                                )
                            else:
                                result = await call(clients.anthropic, rubric=rubric, user=clean)
                        except ItemError as exc:
                            s.set_attribute("error_code", exc.code)
                            s.set_output(json.dumps({"error": exc.code}))
                            raise  # the step node must show as failed, not as done
                        card.record(result)
                        facts = validate_facts(result.data, clean, schema)
                        db.execute(
                            "INSERT OR REPLACE INTO extraction VALUES (?,?,?,?,?,?)",
                            (
                                item.item_id,
                                item.hash,
                                PROMPT_VERSION,
                                json.dumps(facts),
                                json.dumps(
                                    {"model": result.model_selected, "cost": result.cost_usd}
                                ),
                                utc_iso(),
                            ),
                        )
                        extracted += 1
                        s.set_attribute("facts", len(facts))
                        s.set_attribute("tier", result.difficulty_label)
                        s.set_output(
                            json.dumps({"facts": len(facts), "tier": result.difficulty_label})
                        )
                except ItemError as exc:
                    # A provider error is not a property of the item, so it is not skipped and
                    # recorded like a bad document would be. The run stops here, the task's
                    # root node is marked failed in the graph, and the handler below shows
                    # what recovery looks like.
                    failed += 1
                    failure = (n, item, clean, exc.code)
                    break

            print(f"  extract: {extracted} new, {cached} cached, {failed} failed")
            if failure is not None:
                raise ItemError(failure[3])

            # ── reconcile, render, check ─────────────────────────────────────
            stats = reconcile(db, schema, task_id, args.version, clients.anthropic)
            doc_version = render(db, schema, task_id, args.version, stats, clients.anthropic)
            qa = run_qa(db, doc_version, clients.anthropic)

            LAST_GRAPH = clients.anthropic.get_task_graph(task_id)

            db.execute(
                "UPDATE run SET finished_at=?, status=?, stats_json=? WHERE task_id=?",
                (utc_iso(), stats["status"], json.dumps(stats), task_id),
            )

        card.outcomes.update(
            {
                "facts active": stats["active"],
                "added / updated / removed": f"{stats['added']} / {stats['updated']} / {stats['removed']}",
                "conflicts resolved": stats["conflicts"],
                "items redacted": redacted_items,
                "extractions (new / cached)": f"{extracted} / {cached}",
                "document": display(OUT / "brain.md"),
            }
        )
        card.gate("answers correct", qa["rate"] >= 0.9, f"{qa['passed']}/{qa['total']}")
        card.gate("every fact has a verbatim quote", stats["quote_failures"] == 0)
        card.gate(
            "removals within threshold",
            stats["status"] != "held",
            stats.get("held_reason", ""),
        )
        card.print()
        if qa["failures"]:
            print("  questions answered wrongly: " + ", ".join(qa["failures"]))
        print(f"\n{graph_line(clients.anthropic, task_id)}\n")
        return 0 if stats["status"] != "held" and qa["rate"] >= 0.9 else 1
    except ItemError:
        # The task scope has exited: the root node is marked failed in the graph and the
        # provider failure is a failed llm_call node inside it. While this process still holds
        # the graph, one SDK call retries exactly that step, with routing free to pick a live
        # model. The retry lands in the same graph, linked to the node it replaces.
        n, item, clean, code = failure
        db.execute(
            "UPDATE run SET finished_at=?, status='failed', stats_json=? WHERE task_id=?",
            (utc_iso(), json.dumps({"error": code, "item": short_hash(item.item_id)}), task_id),
        )
        print(f"\n  extraction {n} of {len(changed)} failed: {code}")
        print("  retrying that one step with client.resume() ...")
        # resume() sends no system prompt of its own, and with content_capture off it has no
        # recorded output to rebuild from, so the instruction carries everything the step needs.
        instruction = f"{rubric}\n\n<document>\n{clean}\n</document>"
        # Name the node. The step that wrapped the call is failed too, and resume() would
        # otherwise pick whichever unresolved failure is most recent.
        graph = clients.anthropic.get_task_graph(task_id)
        failed_call = [n for n in graph.failed_nodes() if n.node_type == "llm_call"][-1]
        try:
            resp = await clients.anthropic.resume(
                task_id, node_id=failed_call.node_id, instruction=instruction
            )
        except Exception as exc:
            # The retry is routed to a live model, so if it fails too the provider itself is
            # refusing us -- most often a bad or expired ANTHROPIC_API_KEY.
            print(f"  retry also failed ({type(exc).__name__}); check ANTHROPIC_API_KEY")
            return 1
        facts = validate_facts(parse_json(resp.text), clean, schema)
        db.execute(
            "INSERT OR REPLACE INTO extraction VALUES (?,?,?,?,?,?)",
            (
                item.item_id,
                item.hash,
                PROMPT_VERSION,
                json.dumps(facts),
                json.dumps(
                    {
                        "model": resp.fluxcompute.model_selected,
                        "cost": resp.fluxcompute.cost_usd,
                        "resumed": True,
                    }
                ),
                utc_iso(),
            ),
        )
        fc = resp.fluxcompute
        print(
            f"  resumed: {len(facts)} facts, routed to {fc.model_selected}"
            f" ({fc.difficulty_label}), ${fc.cost_usd:.4f}"
        )
        remaining = len(changed) - n
        print(f"  {remaining} extractions were never reached. Finish the run from a new process:")
        print(f"    python run.py --resume {task_id}")
        print(f"\n{graph_line(clients.anthropic, task_id)}\n")
        return 1
    finally:
        LAST_GRAPH = clients.anthropic.get_task_graph(task_id) or LAST_GRAPH
        await clients.close()
        db.close()


def validate_facts(data: Any, source_text: str, schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep only facts the document actually supports.

    Every claim must carry a quote that appears in the source, and a key from the vocabulary.
    This is the whole anti-fabrication mechanism, and it is four lines of code rather than a
    sentence in the prompt asking the model to be careful.
    """
    out: list[dict[str, Any]] = []
    for raw in (data or {}).get("facts", []):
        key = str(raw.get("key", ""))
        if "." not in key:
            continue
        entity, _, attribute = key.partition(".")
        attrs = schema["entities"].get(entity, {}).get("attributes", {})
        if attribute not in attrs:
            continue
        value, quote = str(raw.get("value", "")).strip(), str(raw.get("quote", "")).strip()
        if not value or not quote_is_verbatim(quote, source_text):
            continue
        out.append({"entity": entity, "attribute": attribute, "value": value, "quote": quote})
    return out


def reconcile(db, schema, task_id, version, client) -> dict[str, Any]:
    """Decide what the brain now says. No model is involved."""
    with client.step("reconcile") as s:
        rows = db.execute(
            "SELECT e.item_id, e.facts_json, i.source FROM extraction e"
            " JOIN source_item i ON i.item_id = e.item_id"
            " WHERE i.status = 'present' AND e.content_hash = i.content_hash"
            " AND e.prompt_version = ?",
            (PROMPT_VERSION,),
        ).fetchall()

        proposed: dict[str, dict[str, Any]] = {}
        conflicts = 0
        for row in rows:
            for fact in json.loads(row["facts_json"]):
                key = fact_key(schema, fact["entity"], fact["attribute"], fact["value"])
                rank = precedence_rank(schema, fact["entity"], fact["attribute"], row["source"])
                candidate = {
                    **fact,
                    "source": row["source"],
                    "item_id": row["item_id"],
                    "rank": rank,
                }
                held = proposed.get(key)
                if held is None:
                    proposed[key] = candidate
                    continue
                if rank > held["rank"]:
                    # Two sources disagree and one of them is authoritative here. Record the
                    # loser: an unexplained overwrite is indistinguishable from a bug.
                    if normalize(held["value"]) != normalize(candidate["value"]):
                        conflicts += 1
                        db.execute(
                            "INSERT INTO conflict VALUES (?,?,?,?,?,?,?)",
                            (
                                task_id,
                                fact["entity"],
                                fact["attribute"],
                                candidate["source"],
                                candidate["value"],
                                held["source"],
                                held["value"],
                            ),
                        )
                    proposed[key] = candidate

        # Single-valued attributes: a new value replaces the old fact rather than joining it.
        for key, fact in list(proposed.items()):
            if is_multi(schema, fact["entity"], fact["attribute"]):
                continue
            for other_key, other in list(proposed.items()):
                if other_key == key or (other["entity"], other["attribute"]) != (
                    fact["entity"],
                    fact["attribute"],
                ):
                    continue
                if other["rank"] < fact["rank"]:
                    proposed.pop(other_key, None)

        existing = {r["fact_key"]: r for r in db.execute("SELECT * FROM fact")}
        active_before = sum(1 for r in existing.values() if r["status"] == "active")
        added = updated = 0
        for key, fact in proposed.items():
            prior = existing.get(key)
            if prior is None:
                added += 1
            elif normalize(prior["value"]) != normalize(fact["value"]):
                updated += 1
            elif prior["status"] == "active":
                pass
            db.execute(
                "INSERT INTO fact(fact_key, entity, attribute, value, value_norm, quote, source,"
                " item_id, status, missing_runs, first_version, last_version)"
                " VALUES (?,?,?,?,?,?,?,?, 'active', 0, ?, ?)"
                " ON CONFLICT(fact_key) DO UPDATE SET value=excluded.value,"
                " value_norm=excluded.value_norm, quote=excluded.quote, source=excluded.source,"
                " item_id=excluded.item_id, status='active', missing_runs=0,"
                " last_version=excluded.last_version",
                (
                    key,
                    fact["entity"],
                    fact["attribute"],
                    fact["value"],
                    normalize(fact["value"]).lower(),
                    fact["quote"],
                    fact["source"],
                    fact["item_id"],
                    version,
                    version,
                ),
            )

        never_absence = set(schema["removal"].get("never_absence_remove", []))
        removed = 0
        for key, row in existing.items():
            if key in proposed or row["status"] != "active":
                continue
            if row["source"] in never_absence:
                continue  # an email that is not in this window has not been retracted
            item = db.execute(
                "SELECT status FROM source_item WHERE item_id=?", (row["item_id"],)
            ).fetchone()
            item_present = item is not None and item["status"] == "present"
            if item_present:
                # The document still exists and stopped saying it. That is a decision.
                reason = "no longer stated by its source"
                db.execute(
                    "UPDATE fact SET status='removed', removed_version=?, removed_reason=?"
                    " WHERE fact_key=?",
                    (task_id, reason, key),
                )
                removed += 1
            else:
                # The document is gone. A deleted file and a failed fetch look the same for
                # one run, so wait before believing it.
                missing = row["missing_runs"] + 1
                if missing >= schema["removal"]["grace_runs"]:
                    db.execute(
                        "UPDATE fact SET status='removed', missing_runs=?, removed_version=?,"
                        " removed_reason=? WHERE fact_key=?",
                        (missing, task_id, "source removed", key),
                    )
                    removed += 1
                else:
                    db.execute("UPDATE fact SET missing_runs=? WHERE fact_key=?", (missing, key))

        active = db.execute("SELECT COUNT(*) c FROM fact WHERE status='active'").fetchone()["c"]
        threshold = schema["removal"]["anomaly_threshold"]
        status, held_reason = "ok", ""
        if active_before and removed / max(active_before, 1) > threshold:
            status, held_reason = (
                "held",
                f"{removed} of {active_before} facts would be removed (> {threshold:.0%})",
            )
        quote_failures = db.execute(
            "SELECT COUNT(*) c FROM fact WHERE status='active' AND (quote IS NULL OR quote='')"
        ).fetchone()["c"]

        stats = {
            "active": active,
            "added": added,
            "updated": updated,
            "removed": removed,
            "conflicts": conflicts,
            "quote_failures": quote_failures,
            "status": status,
            "held_reason": held_reason,
        }
        for k, v in stats.items():
            if isinstance(v, int):
                s.set_attribute(k, v)
        s.set_output(json.dumps(stats))
        print(
            f"  reconcile: {active} active (+{added} ~{updated} -{removed}), "
            f"{conflicts} conflicts" + (f" -- HELD: {held_reason}" if status == "held" else "")
        )
        return stats


def render(db, schema, task_id, version, stats, client) -> int:
    """Write brain.md and a changelog. Deterministic, so a re-render is free and diffable."""
    with client.step("render") as s:
        facts = db.execute(
            "SELECT * FROM fact WHERE status='active' ORDER BY entity, attribute, value"
        ).fetchall()
        facts_hash = content_hash("|".join(f"{f['fact_key']}={f['value_norm']}" for f in facts))
        prior = db.execute("SELECT * FROM doc_version ORDER BY version DESC LIMIT 1").fetchone()
        if prior and prior["facts_hash"] == facts_hash and stats["status"] != "held":
            s.set_attribute("no_change", True)
            s.set_output(json.dumps({"no_change": True}))
            print("  render: nothing changed")
            return prior["version"]

        version_no = (prior["version"] + 1) if prior else 1
        lines = [
            "# Company brain",
            "",
            f"Version {version_no}. Generated {utc_iso()} from {len(facts)} facts.",
            "Every line carries the source that asserts it. Nothing here is written by hand.",
            "",
        ]
        current_entity = None
        for f in facts:
            if f["entity"] != current_entity:
                current_entity = f["entity"]
                lines += ["", f"## {current_entity}", ""]
            lines.append(f"- **{f['attribute']}**: {f['value']}  `[{f['source']}]`")
        (OUT / "brain.md").write_text("\n".join(lines) + "\n")
        (OUT / f"brain.v{version_no}.md").write_text("\n".join(lines) + "\n")

        removed = db.execute(
            "SELECT * FROM fact WHERE status='removed' AND removed_version=?", (task_id,)
        ).fetchall()
        conflicts = db.execute("SELECT * FROM conflict WHERE version=?", (task_id,)).fetchall()
        change = [
            f"# Changelog for version {version_no}",
            "",
            f"- added: {stats['added']}",
            f"- updated: {stats['updated']}",
            f"- removed: {stats['removed']}",
            "",
        ]
        if removed:
            change += ["## Removed", ""]
            change += [
                f"- **{r['entity']}.{r['attribute']}**: {r['value']} ({r['removed_reason']})"
                for r in removed
            ]
            change.append("")
        if conflicts:
            change += ["## Conflicts resolved by precedence", ""]
            change += [
                f"- **{c['entity']}.{c['attribute']}**: kept `{c['winner_source']}` "
                f"({c['winner_value']}) over `{c['loser_source']}` ({c['loser_value']})"
                for c in conflicts
            ]
        (OUT / "brain.changelog.md").write_text("\n".join(change) + "\n")

        db.execute(
            "INSERT OR REPLACE INTO doc_version VALUES (?,?,?,?,?,?,?,?,?)",
            (
                version_no,
                task_id,
                utc_iso(),
                facts_hash,
                stats["added"],
                stats["updated"],
                stats["removed"],
                stats["status"],
                stats["held_reason"],
            ),
        )
        s.set_attribute("version", version_no)
        s.set_attribute("facts", len(facts))
        s.set_output(json.dumps({"version": version_no, "facts": len(facts)}))
        print(f"  render: version {version_no}, {len(facts)} facts")
        return version_no


def run_qa(db, doc_version: int, client) -> dict[str, Any]:
    """Ask the questions a human would ask, and check the answers by substring.

    No model judges this. A model asked whether its own document is right will say yes.
    The questions that must NOT be answerable are the useful half: they catch a brain that
    invents, and a brain that keeps a fact after its source dropped it.
    """
    with client.step("qa") as s:
        spec = json.loads((COMPANY / "qa.json").read_text())
        doc = (OUT / "brain.md").read_text().lower()
        key = f"v{min(doc_version, 3)}"
        passed, failures = 0, []
        for q in spec["questions"]:
            expect = q["expect"][key]
            if isinstance(expect, dict):  # {"absent": [...]} -- must NOT be answerable
                ok = not any(n.lower() in doc for n in expect["absent"])
            else:
                ok = all(n.lower() in doc for n in expect)
            passed += ok
            if not ok:
                failures.append(q["id"])
        total = len(spec["questions"])
        rate = passed / total if total else 0.0
        s.set_attribute("passed", passed)
        s.set_attribute("total", total)
        s.set_output(json.dumps({"passed": passed, "total": total}))
        print(f"  qa: {passed}/{total} answers correct")
        return {"passed": passed, "total": total, "rate": rate, "failures": failures}


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
