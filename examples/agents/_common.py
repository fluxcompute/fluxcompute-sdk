"""Shared plumbing for the sample agents.

Every agent runs with a REAL provider key, exactly as your own code would; there is no mock
mode in this module. The $0 safety net is the test suite under tests/, which patches the
provider from outside so the classifier and the graph recorder stay real.

A FluxCompute key is optional. Without one, routing, the in-process execution graph and
in-process resume() all work and nothing leaves the machine. With one, each run also appears
in the hosted dashboard.

What this module owns, so the agents can stay about their own domain:

  env + keys      load_dotenv(), require_keys(), assert_sdk_version()
  client          make_clients()
  LLM calls       call() -- rubric as a system MESSAGE, JSON parsed, one repair retry
  safety          scrub() -- secrets and PII removed before anything reaches a model
  errors          ItemError -- a fixed vocabulary, so no content leaks into a node's error
  storage         connect() -- SQLite as the agent's own checkpoint
  reporting       Scorecard, graph_line()
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import sqlite3
import time
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UTC = timezone.utc  # datetime.UTC is 3.11+; the SDK supports 3.10

# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

HERE = Path(__file__).resolve().parent
COMPANY = HERE / "fixtures"

MIN_SDK = (0, 3, 1)
DEFAULT_EVENTS_URL = "https://api.fluxcompute.dev/v1/graph/events"


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader. Never overrides a variable already set in the environment."""
    path = path or (HERE / ".env")
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.split(" #", 1)[0].strip().strip("'\""))


def assert_sdk_version() -> str:
    """Fail at startup, in one line, if the installed SDK is older than these agents need."""
    import fluxcompute

    version = getattr(fluxcompute, "__version__", "0.0.0")
    parts = tuple(int(p) for p in re.findall(r"\d+", version)[:3])
    if parts < MIN_SDK:
        need = ".".join(map(str, MIN_SDK))
        raise RuntimeError(
            f"fluxcompute {version} is too old (need >= {need}). "
            f"Run: pip install --upgrade 'fluxcompute>={need}'"
        )
    return version


class MissingKey(RuntimeError):
    """A required key is absent. The message names the variable and where to get it."""


_KEY_HELP = {
    "ANTHROPIC_API_KEY": "https://console.anthropic.com/settings/keys",
}


def require_keys(*names: str) -> dict[str, str]:
    """Read keys from the environment, or fail with a message that says how to get them."""
    load_dotenv()
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        lines = [f"  {n}  ->  {_KEY_HELP.get(n, 'see the README')}" for n in missing]
        raise MissingKey(
            "These environment variables are not set:\n"
            + "\n".join(lines)
            + "\n\nSet them in your shell, or put them in examples/agents/.env "
            "(gitignored). These examples make real API calls."
        )
    return {n: os.environ[n] for n in names}


# ─────────────────────────────────────────────────────────────────────────────
# Errors: a fixed vocabulary
# ─────────────────────────────────────────────────────────────────────────────

ERROR_CODES = frozenset(
    {
        "parse_failed",
        "provider_error",
        "pii_detected",
        "encoding_lossy",
        "quote_not_verbatim",
        "schema_invalid",
        "source_unavailable",
    }
)


class ItemError(RuntimeError):
    """One item failed. Carries a code from ERROR_CODES and never any item content.

    A node's name, attributes and error string are part of the execution graph. When a
    FluxCompute key is set they are sent as telemetry, and content_capture does not change
    that: it governs model input and output, not error text. So an exception message must not
    interpolate an email subject, a sender, a file path or a fact. It carries a code; the
    detail stays in the local database.

    The codes also avoid the words the SDK's failure classifier reads as a spend problem
    (rate limit, quota, billing, credit), which would otherwise mislabel the node.
    """

    def __init__(self, code: str):
        if code not in ERROR_CODES:
            raise ValueError(f"unknown error code: {code}")
        self.code = code
        super().__init__(code)


# ─────────────────────────────────────────────────────────────────────────────
# Text normalisation, hashing, scrubbing
# ─────────────────────────────────────────────────────────────────────────────

_WS = re.compile(r"\s+")
_CURLY = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})

_SECRETS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ANTHROPIC_KEY", re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}")),
    ("OPENAI_KEY", re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{20,}")),
    ("GITHUB_TOKEN", re.compile(r"gh[pousr]_[A-Za-z0-9]{16,}")),
    ("FLUX_KEY", re.compile(r"flx_[A-Za-z0-9_\-]{8,}")),
    ("AWS_KEY", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("PRIVATE_KEY", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")),
    # CARD before PHONE, and the order is load-bearing: the phone pattern is a superset of
    # any card-shaped run of digits, so with PHONE first every card was redacted (safe) but
    # reported as a phone number (wrong). The counts are what a reviewer reads.
    ("CARD", re.compile(r"\b(?:\d[ \-]?){13,16}\b")),
    ("PHONE", re.compile(r"\+?\d[\d\s().\-]{8,}\d")),
)


def normalize(text: str) -> str:
    """Whitespace- and encoding-normalised text, for hashing and verbatim comparison.

    Sources disagree about quotes and entities: GitHub's API returns HTML entities, mail
    clients emit curly quotes and non-breaking spaces, and copy-paste introduces soft hyphens.
    Comparing raw strings makes a correct quote look fabricated.
    """
    text = html.unescape(text or "")
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_CURLY).replace("­", "").replace(" ", " ")
    return _WS.sub(" ", text).strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def short_hash(text: str) -> str:
    """8 hex chars, for node attributes: identifies an item without naming it."""
    return content_hash(text)[:8]


def scrub(text: str) -> tuple[str, dict[str, int]]:
    """Remove secrets and personal data before any text reaches a model.

    Returns the redacted text and a count per category, so a step can record that redaction
    happened without recording what was redacted.
    """
    counts: dict[str, int] = {}
    for label, pattern in _SECRETS:
        text, n = pattern.subn(f"[REDACTED:{label}]", text)
        if n:
            counts[label] = counts.get(label, 0) + n
    return text, counts


def quote_is_verbatim(quote: str, source: str) -> bool:
    """The single highest-value guard against a fabricated citation.

    A model that must quote its source verbatim cannot invent the claim without inventing a
    string that is checkably absent. Validated in code, never by asking the model nicely.
    """
    if not quote or not quote.strip():
        return False
    if "[REDACTED:" in quote:
        return False
    return normalize(quote) in normalize(source)


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_iso(dt: datetime | None = None) -> str:
    dt = dt or utc_now()
    if dt.tzinfo is None:
        raise ValueError("naive datetime: every timestamp in these agents is timezone-aware")
    return dt.astimezone(UTC).isoformat(timespec="seconds")


# ─────────────────────────────────────────────────────────────────────────────
# Client
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Clients:
    """The FluxClient an agent runs on, plus the SDK version it was checked against."""

    anthropic: Any
    sdk_version: str = ""
    _closed: bool = False

    async def close(self) -> None:
        """Flush the execution graph and close. Safe to call twice: the agents close inside a
        `finally`, and a step may have closed earlier to report."""
        if self._closed:
            return
        self._closed = True
        await self.anthropic.close()


async def make_clients() -> Clients:
    """Build the client an agent needs.

    Only a provider key is required. A FluxCompute key is optional: without one, telemetry is
    off and nothing leaves the machine, while routing, the in-process execution graph and
    in-process resume() all work as normal. With one, the run also appears in the hosted
    dashboard, and the key is checked before any spend so that a key that cannot authenticate
    fails here, loudly, instead of showing up later as an empty dashboard.
    """
    version = assert_sdk_version()
    keys = require_keys("ANTHROPIC_API_KEY")
    flux_key = os.environ.get("FLUXCOMPUTE_KEY") or None
    capture = flux_key is not None and os.environ.get("FLUX_CONTENT_CAPTURE") == "1"

    from fluxcompute import FluxClient

    client = FluxClient(
        anthropic_key=keys["ANTHROPIC_API_KEY"],
        fluxcompute_key=flux_key,
        telemetry=flux_key is not None,
        content_capture=capture,
        provider="anthropic",
    )
    if flux_key:
        await client.verify()
    if capture:
        print(
            "  content capture is ON: model inputs and outputs for this run leave this machine."
            " See the SDK documentation on telemetry and privacy for what is kept."
        )
    return Clients(anthropic=client, sdk_version=version)


# ─────────────────────────────────────────────────────────────────────────────
# LLM calls
# ─────────────────────────────────────────────────────────────────────────────

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_json(text: str) -> Any:
    """Take the JSON object out of a reply that may be fenced or prefaced."""
    cleaned = _FENCE.sub("", (text or "").strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ItemError("parse_failed")
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ItemError("parse_failed") from exc


@dataclass
class LLMResult:
    """A parsed reply plus the routing metadata that makes a run measurable."""

    data: Any
    model_selected: str
    model_served: str
    difficulty_score: float
    difficulty_label: str
    cost_usd: float
    baseline_cost_usd: float
    savings_usd: float
    input_tokens: int
    output_tokens: int
    latency_ms: int
    overhead_ms: float
    repaired: bool
    provider: str


async def call(
    client: Any,
    *,
    rubric: str,
    user: str,
    model: str = "auto",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    provider: str = "anthropic",
) -> LLMResult:
    """One JSON-returning LLM call, with one repair retry, inside the caller's step.

    The rubric goes in as a role="system" MESSAGE, never a system= keyword argument: the SDK
    passes its own system= to the dispatcher, so a second one raises "got multiple values for
    keyword argument 'system'". The message form is also the one Anthropic prompt caching
    applies to, because the SDK's cache manager lifts it into system blocks.

    Keep a rubric under ~500 words. The classifier scores the last user message, and a system
    message adds at most +0.10 to that score (+0.05 over 200 words, +0.10 over 500) -- enough
    to push a short classification out of the cheap tier.
    """
    messages = [{"role": "system", "content": rubric}, {"role": "user", "content": user}]
    started = time.monotonic()
    repaired = False

    async def _once(msgs: list[dict[str, Any]]) -> Any:
        try:
            return await client.messages.create(
                model=model, messages=msgs, max_tokens=max_tokens, temperature=temperature
            )
        except Exception as exc:  # provider failures are item failures, never run failures
            raise ItemError("provider_error") from exc

    resp = await _once(messages)
    try:
        data = parse_json(resp.text)
    except ItemError:
        repaired = True
        resp = await _once(
            messages
            + [
                {"role": "assistant", "content": resp.text or ""},
                {
                    "role": "user",
                    "content": "That was not valid JSON. Reply with only the JSON object.",
                },
            ]
        )
        data = parse_json(resp.text)

    meta = resp.fluxcompute
    usage = resp.usage
    return LLMResult(
        data=data,
        model_selected=meta.model_selected,
        model_served=getattr(resp.raw, "model", meta.model_selected),
        difficulty_score=meta.difficulty_score,
        difficulty_label=meta.difficulty_label,
        cost_usd=float(meta.cost_usd),
        baseline_cost_usd=float(meta.baseline_cost_usd),
        savings_usd=float(meta.savings_usd),
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
        latency_ms=int((time.monotonic() - started) * 1000),
        overhead_ms=float(meta.overhead_ms),
        repaired=repaired,
        provider=provider,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Storage
# ─────────────────────────────────────────────────────────────────────────────


def connect(path: Path, schema: str) -> sqlite3.Connection:
    """SQLite is each agent's own checkpoint, and the reason a re-run is cheap.

    The SDK's resume() re-runs one failed LLM call. Everything else that makes a re-run skip
    finished work -- which item is done, which is deferred -- is these tables.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(schema)
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Scorecard:
    """What the run cost, how it routed, and whether it did its job.

    Printed at the end of every agent. The routing and cost numbers come from each response's
    metadata, so they are the same with or without a FluxCompute key.
    """

    title: str
    baseline: str = "claude-opus-4-8"
    calls: list[LLMResult] = field(default_factory=list)
    outcomes: dict[str, Any] = field(default_factory=dict)
    gates: list[tuple[str, bool, str]] = field(default_factory=list)

    def record(self, result: LLMResult) -> LLMResult:
        self.calls.append(result)
        return result

    def gate(self, name: str, passed: bool, detail: str = "") -> None:
        self.gates.append((name, passed, detail))

    @property
    def cost(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    @property
    def savings(self) -> float:
        return sum(c.savings_usd for c in self.calls)

    def tiers(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.calls:
            out[c.difficulty_label] = out.get(c.difficulty_label, 0) + 1
        return out

    @staticmethod
    def _pct(values: Sequence[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(int(q * len(ordered)), len(ordered) - 1)
        return ordered[idx]

    def render(self) -> str:
        lat = [c.latency_ms for c in self.calls]
        lines = [
            "",
            "─" * 62,
            f"  {self.title}",
            "─" * 62,
            f"  LLM calls          {len(self.calls)}",
            f"  Tiers              {self.tiers() or '-'}",
            f"  Cost               ${self.cost:.4f}",
            f"  Savings            ${self.savings:.4f}  (vs always-{self.baseline})",
            f"  Latency p50 / p95  {self._pct(lat, 0.5):.0f} ms / {self._pct(lat, 0.95):.0f} ms",
        ]
        if self.outcomes:
            lines.append("  Outcomes")
            for key, value in self.outcomes.items():
                lines.append(f"    {key:<24} {value}")
        if self.gates:
            lines.append("  Gates")
            for name, passed, detail in self.gates:
                mark = "PASS" if passed else "FAIL"
                lines.append(f"    [{mark}] {name}{f'  {detail}' if detail else ''}")
        lines.append("─" * 62)
        return "\n".join(lines)

    def print(self) -> None:
        print(self.render())


def graph_line(client: Any, task_id: str) -> str:
    """One line on the execution graph this process recorded, plus its dashboard link when a
    FluxCompute key is set. The same environment check decides both, so they cannot disagree."""
    graph = client.get_task_graph(task_id)
    nodes = len(graph.in_order()) if graph else 0
    line = f"  graph: {nodes} nodes recorded in-process"
    if os.environ.get("FLUXCOMPUTE_KEY"):
        base = os.environ.get("FLUX_GRAPH_EVENTS_URL", DEFAULT_EVENTS_URL).rsplit("/v1/", 1)[0]
        line += f"\n  {base}/ui/#task={task_id}"
    return line


def display(path: Path) -> str:
    """A path as printed: relative to the examples directory, never absolute."""
    try:
        return str(path.resolve().relative_to(HERE))
    except ValueError:
        return path.name


def run_id(prefix: str, attempt: int = 1) -> str:
    """A deterministic task id, so re-running today attaches to the same graph."""
    return f"{prefix}-{utc_now():%Y%m%d}-{attempt:02d}"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())
