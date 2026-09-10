#!/usr/bin/env python3
"""Company brain: keep one versioned document in sync with the sources that feed it.

    python run.py                      first sync   (repo v1, inbox run1, brand v1)
    python run.py --version 2          second sync  (repo v2, inbox run2, brand v2)
    python run.py --version 2          third time   -> no changes, and no LLM calls at all
    python run.py --fail-at 2          inject a provider failure at the 2nd extraction
    python run.py --resume <task_id>   re-run only what did not finish

This makes real calls to Anthropic and costs a few cents. Set ANTHROPIC_API_KEY first.
FLUXCOMPUTE_KEY is optional: with it, the run also appears in the hosted dashboard.
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
    decode_header,
    decode_qp,
    display,
    graph_line,
    make_clients,
    next_attempt,
    normalize,
    parse_json,
    quote_is_verbatim,
    run_id,
    scrub,
    short_hash,
    split_address,
    strip_quoted,
    utc_iso,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

LAST_GRAPH = None  # tests/test_agent_examples.py asserts the shape of the last run's graph
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

# bump on any edit to the rubric: extractions are cached per (item, hash, prompt_version)
PROMPT_VERSION = 1


def extract_rubric(schema: dict[str, Any]) -> str:
    """Kept short: charged on every call, and the router's score climbs past ~200 words."""
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


def collect_email(path: Path, allowlist: list[str]) -> tuple[list[Item], int]:
    """The allowlist is checked here, before a model sees the text."""
    items, rejected = [], 0
    seen_bodies: set[str] = set()
    allowed = {a.lower() for a in allowlist}
    for msg in json.loads(path.read_text()):
        _, sender, _ = split_address(msg["from"])
        if sender not in allowed:
            rejected += 1
            continue
        body = msg["body"]
        if msg.get("content_transfer_encoding") == "quoted-printable":
            body = decode_qp(body)
        body = strip_quoted(body)
        body_hash = content_hash(body)
        if body_hash in seen_bodies:
            continue
        seen_bodies.add(body_hash)
        subject = decode_header(msg["subject"])
        # no From: line: the scrubber would redact it and count it
        items.append(
            Item(
                "email",
                msg["message_id"],
                f"Subject: {subject}\n\n{body}",
                url=f"mailto:{msg['message_id']}",
            )
        )
    return items, rejected


def collect_brand(path: Path) -> list[Item]:
    return [Item("brand", path.name, path.read_text(), url=f"brand://{path.name}")]


def is_multi(schema: dict[str, Any], entity: str, attribute: str) -> bool:
    meta = schema["entities"].get(entity, {}).get("attributes", {}).get(attribute)
    return isinstance(meta, dict) and bool(meta.get("multi"))


def fact_key(schema: dict[str, Any], entity: str, attribute: str, value: str) -> str:
    """The value is part of the key only for multi-valued attributes."""
    base = f"{entity}|{attribute}"
    if is_multi(schema, entity, attribute):
        base += f"|{normalize(value).lower()}"
    return content_hash(base)[:16]


def precedence_rank(schema: dict[str, Any], entity: str, attribute: str, source: str) -> int:
    for pattern, order in schema["precedence"].items():
        if fnmatch.fnmatch(f"{entity}.{attribute}", pattern):
            return order.index(source) if source in order else -1
    return -1


async def main() -> int:
    global LAST_GRAPH
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--version",
        type=int,
        choices=(1, 2),
        help="which snapshot of the sources to sync (default 1, or the resumed run's)",
    )
    ap.add_argument("--resume", metavar="TASK_ID")
    ap.add_argument("--fail-at", type=int, metavar="N", help="fail the Nth extraction on purpose")
    ap.add_argument("--attempt", type=int, help="default: the next unused number today")
    args = ap.parse_args()

    schema = yaml.safe_load((COMPANY / "brain_schema.yaml").read_text())
    rubric = extract_rubric(schema)
    OUT.mkdir(parents=True, exist_ok=True)
    db = connect(OUT / "brain.db", SCHEMA_SQL)
    card = Scorecard("Company brain")

    if args.resume and args.version is None:
        row = db.execute("SELECT version FROM run WHERE task_id=?", (args.resume,)).fetchone()
        if row is None:
            print(f"  no run recorded under {args.resume}")
            db.close()
            return 1
        args.version = row["version"]
    version = args.version or 1

    try:
        clients = await make_clients()
    except Exception:
        db.close()
        raise
    task_id = args.resume or run_id("brain", args.attempt or next_attempt(db, "brain"))
    print(f"  task {task_id}")
    failure: tuple[int, Item, str, str] | None = None
    changed: list[Item] = []

    try:
        with clients.anthropic.task("company-brain", task_id=task_id):
            db.execute(
                "INSERT OR REPLACE INTO run(task_id, started_at, status, version) VALUES (?,?,?,?)",
                (task_id, utc_iso(), "running", version),
            )

            repo_dir = COMPANY / f"repo_v{version}"
            brand_file = COMPANY / f"brand_v{version}.md"
            inbox = COMPANY / "inbox" / f"run{version}.json"
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

            with clients.anthropic.step("diff") as s:
                known = {
                    r["item_id"]: r
                    for r in db.execute("SELECT * FROM source_item WHERE status != 'removed'")
                }
                seen_now = {i.item_id: i for i in items}
                hashes_now = {i.hash: i for i in items}
                # done work is an extraction row, not a hash; a failed run must not read "0 changed"
                extracted = {
                    (r["item_id"], r["content_hash"])
                    for r in db.execute(
                        "SELECT item_id, content_hash FROM extraction WHERE prompt_version=?",
                        (PROMPT_VERSION,),
                    )
                }

                changed: list[Item] = []
                renamed = reused = 0
                deferred_ids: set[str] = set()
                for item in items:
                    prior = known.get(item.item_id)
                    if prior is None:
                        old = next(
                            (
                                r
                                for k, r in known.items()
                                if r["content_hash"] == item.hash and k not in seen_now
                            ),
                            None,
                        )
                        if old is not None:
                            # a rename: carry every table keyed on item_id, or its facts read as removed
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
                    if (item.item_id, item.hash) in extracted:
                        reused += 1
                    else:
                        changed.append(item)
                        if len(changed) > MAX_EXTRACTIONS_PER_RUN:
                            deferred_ids.add(item.item_id)
                    if item.item_id in deferred_ids:
                        # a deferred item keeps its stored hash, or reconcile reads its facts as gone
                        db.execute(
                            "INSERT INTO source_item(item_id, source, external_id, content_hash,"
                            " text, url, first_seen_run, last_seen_run, status, missing_runs)"
                            " VALUES (?,?,?,?,?,?,?,?, 'present', 0)"
                            " ON CONFLICT(item_id) DO UPDATE SET"
                            " last_seen_run=excluded.last_seen_run, status='present',"
                            " missing_runs=0",
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
                        continue
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
                deferred = len(deferred_ids)
                changed = changed[:MAX_EXTRACTIONS_PER_RUN]
                s.set_attribute("changed", len(changed))
                s.set_attribute("reused", reused)
                s.set_attribute("renamed", renamed)
                s.set_attribute("missing", len(gone))
                s.set_attribute("deferred", deferred)
                s.set_output(
                    json.dumps(
                        {
                            "changed": len(changed),
                            "reused": reused,
                            "renamed": renamed,
                            "missing": len(gone),
                            "deferred": deferred,
                        }
                    )
                )
                print(
                    f"  diff: {len(changed)} changed, {reused} reused, {renamed} renamed,"
                    f" {len(gone)} missing" + (f", {deferred} deferred" if deferred else "")
                )

            extracted = failed = redacted_items = 0
            for n, item in enumerate(changed, start=1):
                clean, redactions = scrub(item.text)
                if redactions:
                    redacted_items += 1
                # one node name for the fan-out; the item id is an attribute
                try:
                    with clients.anthropic.step("extract") as s:
                        s.set_attribute("item", short_hash(item.item_id))
                        s.set_attribute("source", item.source)
                        if redactions:
                            s.set_attribute("redactions", sum(redactions.values()))
                        try:
                            if args.fail_at == n:
                                # a retired model id: the production failure resume() is for
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
                            raise  # the step node must show as failed
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
                    # a provider error is not a property of the item: stop the run and recover below
                    failed += 1
                    failure = (n, item, clean, exc.code)
                    break

            print(f"  extract: {extracted} new, {reused} reused, {failed} failed")
            if failure is not None:
                raise ItemError(failure[3])

            stats = reconcile(db, schema, task_id, version, clients.anthropic)
            doc_version = render(db, schema, task_id, version, stats, clients.anthropic)
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
                "extractions (new / reused)": f"{extracted} / {reused}",
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
        print(f"\n{graph_line(clients, task_id)}\n")
        return 0 if stats["status"] != "held" and qa["rate"] >= 0.9 else 1
    except ItemError:
        n, item, clean, code = failure
        db.execute(
            "UPDATE run SET finished_at=?, status='failed', stats_json=? WHERE task_id=?",
            (utc_iso(), json.dumps({"error": code, "item": short_hash(item.item_id)}), task_id),
        )
        print(f"\n  extraction {n} of {len(changed)} failed: {code}")
        graph = clients.anthropic.get_task_graph(task_id)
        failed_calls = [node for node in graph.failed_nodes() if node.node_type == "llm_call"]
        if not failed_calls:
            # parse_failed: both calls succeeded, so there is no failed node to resume
            print("  the model returned no JSON for that item twice; nothing to resume.")
            print(f"  Try again from a new process:\n    python run.py --resume {task_id}")
            print(f"\n{graph_line(clients, task_id)}\n")
            return 1
        print("  retrying that one step with client.resume() ...")
        # resume() prepends a summary of finished steps and samples at the SDK defaults;
        # the quote check keeps other documents' facts off this one
        instruction = f"{rubric}\n\n<document>\n{clean}\n</document>"
        # name the node: the wrapping step is failed too
        failed_call = failed_calls[-1]
        try:
            resp = await clients.anthropic.resume(
                task_id, node_id=failed_call.node_id, instruction=instruction
            )
            facts = validate_facts(parse_json(resp.text), clean, schema)
        except ItemError:
            print("  the retry did not return JSON either; nothing was recorded for that item.")
            print(f"  Try again from a new process:\n    python run.py --resume {task_id}")
            return 1
        except Exception as exc:
            print(f"  retry also failed ({type(exc).__name__}); check ANTHROPIC_API_KEY")
            return 1
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
        print(f"\n{graph_line(clients, task_id)}\n")
        return 1
    finally:
        LAST_GRAPH = clients.anthropic.get_task_graph(task_id) or LAST_GRAPH
        await clients.close()
        db.close()


def validate_facts(data: Any, source_text: str, schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Every fact needs a key from the vocabulary and a quote found in the source."""
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
                # record the loser whichever order the rows came back in
                winner, loser = (candidate, held) if rank > held["rank"] else (held, candidate)
                if rank != held["rank"] and normalize(winner["value"]) != normalize(loser["value"]):
                    conflicts += 1
                    db.execute(
                        "INSERT INTO conflict VALUES (?,?,?,?,?,?,?)",
                        (
                            task_id,
                            fact["entity"],
                            fact["attribute"],
                            winner["source"],
                            winner["value"],
                            loser["source"],
                            loser["value"],
                        ),
                    )
                proposed[key] = winner

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
        # removals are decided first and applied second, so the anomaly check sees the whole set
        removals: list[tuple[str, tuple]] = []
        for key, row in existing.items():
            if key in proposed or row["status"] != "active":
                continue
            if row["source"] in never_absence:
                continue  # an email outside this window has not been retracted
            item = db.execute(
                "SELECT status FROM source_item WHERE item_id=?", (row["item_id"],)
            ).fetchone()
            item_present = item is not None and item["status"] == "present"
            if item_present:
                # still present and no longer asserting: removed this run, no grace
                reason = "no longer stated by its source"
                removals.append(
                    (
                        "UPDATE fact SET status='removed', removed_version=?, removed_reason=?"
                        " WHERE fact_key=?",
                        (task_id, reason, key),
                    )
                )
            else:
                # gone: a deleted file and a failed fetch look alike for one run
                missing = row["missing_runs"] + 1
                if missing >= schema["removal"]["grace_runs"]:
                    removals.append(
                        (
                            "UPDATE fact SET status='removed', missing_runs=?, removed_version=?,"
                            " removed_reason=? WHERE fact_key=?",
                            (missing, task_id, "source removed", key),
                        )
                    )
                else:
                    db.execute("UPDATE fact SET missing_runs=? WHERE fact_key=?", (missing, key))

        removed = len(removals)
        threshold = schema["removal"]["anomaly_threshold"]
        status, held_reason = "ok", ""
        if active_before and removed / active_before > threshold:
            # held: none of the removals is applied and nothing is published
            status, held_reason = (
                "held",
                f"{removed} of {active_before} facts would be removed (> {threshold:.0%})",
            )
        else:
            for sql, params in removals:
                db.execute(sql, params)

        active = db.execute("SELECT COUNT(*) c FROM fact WHERE status='active'").fetchone()["c"]
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
    with client.step("render") as s:
        prior = db.execute("SELECT * FROM doc_version ORDER BY version DESC LIMIT 1").fetchone()
        if stats["status"] == "held":
            # nothing is published from a held run
            s.set_attribute("held", True)
            s.set_output(json.dumps({"held": True}))
            print("  render: held, brain.md left as it was")
            return prior["version"] if prior else 0
        facts = db.execute(
            "SELECT * FROM fact WHERE status='active' ORDER BY entity, attribute, value"
        ).fetchall()
        facts_hash = content_hash("|".join(f"{f['fact_key']}={f['value_norm']}" for f in facts))
        if prior and prior["facts_hash"] == facts_hash:
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
    """Substring checks against qa.json; no model judges the document."""
    with client.step("qa") as s:
        spec = json.loads((COMPANY / "qa.json").read_text())
        doc = (OUT / "brain.md").read_text().lower()
        key = f"v{max(1, min(doc_version, 3))}"
        passed, failures = 0, []
        for q in spec["questions"]:
            expect = q["expect"][key]
            if isinstance(expect, dict):
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
