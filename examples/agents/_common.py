"""Shared plumbing for the two sample agents."""

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

HERE = Path(__file__).resolve().parent
COMPANY = HERE / "fixtures"

MIN_SDK = (0, 3, 1)
DEFAULT_EVENTS_URL = "https://api.fluxcompute.dev/v1/graph/events"


def load_dotenv(path: Path | None = None) -> None:
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
    """A required key is absent."""


_KEY_HELP = {
    "ANTHROPIC_API_KEY": "https://console.anthropic.com/settings/keys",
}


def require_keys(*names: str) -> dict[str, str]:
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
    """A code from ERROR_CODES and never item content: error text is telemetry, and the codes
    avoid the words the SDK's failure classifier reads as a spend problem."""

    def __init__(self, code: str):
        if code not in ERROR_CODES:
            raise ValueError(f"unknown error code: {code}")
        self.code = code
        super().__init__(code)


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
    # CARD before PHONE: the phone pattern also matches a card number
    ("CARD", re.compile(r"\b\d(?:[ \-]?\d){12,15}\b")),
    ("PHONE", re.compile(r"\+?\d(?:[\s().\-]{0,2}\d){9,14}")),
)

# ISO dates are parked before the patterns run; to a regex a date is a phone number
_DATE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?\b"
)
_PUA = 0xE000  # parked dates are stood in by private-use code points
_PARKED = re.compile("[\ue000-\uf8ff]")


def normalize(text: str) -> str:
    """Entities, curly quotes, soft hyphens and whitespace normalised so quotes compare equal."""
    text = html.unescape(text or "")
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_CURLY).replace("­", "").replace(" ", " ")
    return _WS.sub(" ", text).strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def short_hash(text: str) -> str:
    return content_hash(text)[:8]


def scrub(text: str) -> tuple[str, dict[str, int]]:
    parked: list[str] = []

    def park(m: re.Match[str]) -> str:
        if len(parked) >= 0xF8FF - _PUA:
            return m.group()
        parked.append(m.group())
        return chr(_PUA + len(parked) - 1)

    text = _DATE.sub(park, text)
    counts: dict[str, int] = {}
    for label, pattern in _SECRETS:
        text, n = pattern.subn(f"[REDACTED:{label}]", text)
        if n:
            counts[label] = counts.get(label, 0) + n
    if parked:
        text = _PARKED.sub(
            lambda m: parked[i] if (i := ord(m.group()) - _PUA) < len(parked) else m.group(),
            text,
        )
    return text, counts


def quote_is_verbatim(quote: str, source: str) -> bool:
    """Case-insensitive: models quote mid-sentence text lowercased."""
    if not quote or not quote.strip():
        return False
    if "[REDACTED:" in quote:
        return False
    return normalize(quote).lower() in normalize(source).lower()


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_iso(dt: datetime | None = None) -> str:
    dt = dt or utc_now()
    if dt.tzinfo is None:
        raise ValueError("naive datetime: every timestamp in these agents is timezone-aware")
    return dt.astimezone(UTC).isoformat(timespec="seconds")


_QUOTED = re.compile(r"^\s*>.*$", re.MULTILINE)
_REPLY_MARKER = re.compile(r"^\s*On .{0,200}wrote:\s*$", re.MULTILINE)


def strip_quoted(body: str) -> str:
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
    for raw, enc in _dh(value or ""):
        out.append(raw.decode(enc or "utf-8", errors="replace") if isinstance(raw, bytes) else raw)
    return "".join(out)


def split_address(value: str) -> tuple[str, str, str]:
    from email.utils import parseaddr

    name, addr = parseaddr(value or "")
    addr = addr.lower()
    return decode_header(name).strip(), addr, addr.rpartition("@")[2]


def parse_date(value: str) -> str:
    """ISO with an offset, RFC 2822, or a trailing Z; a naive date is taken as UTC."""
    raw = (value or "").strip()
    try:
        dt = datetime.fromisoformat(raw[:-1] + "+00:00" if raw.endswith("Z") else raw)
    except ValueError:
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return utc_iso(dt)


@dataclass
class Clients:
    """telemetry and dashboard_base are fixed here, so what is printed cannot disagree with
    what the client did."""

    anthropic: Any
    sdk_version: str = ""
    telemetry: bool = False
    dashboard_base: str = ""
    _closed: bool = False

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.anthropic.close()


async def make_clients() -> Clients:
    """A FluxCompute key is verified before any provider spend."""
    version = assert_sdk_version()
    keys = require_keys("ANTHROPIC_API_KEY")
    flux_key = os.environ.get("FLUXCOMPUTE_KEY") or None
    capture = flux_key is not None and os.environ.get("FLUX_CONTENT_CAPTURE") == "1"
    events_url = os.environ.get("FLUX_GRAPH_EVENTS_URL", DEFAULT_EVENTS_URL)

    from fluxcompute import FluxClient

    client = FluxClient(
        anthropic_key=keys["ANTHROPIC_API_KEY"],
        fluxcompute_key=flux_key,
        telemetry=flux_key is not None,
        content_capture=capture,
        provider="anthropic",
    )
    if flux_key:
        try:
            await client.verify()
        except Exception:
            await client.close()
            raise
    if capture:
        print(
            "  content capture is ON: model inputs and outputs for this run leave this machine."
            " See the SDK documentation on telemetry and privacy for what is kept."
        )
    return Clients(
        anthropic=client,
        sdk_version=version,
        telemetry=flux_key is not None,
        dashboard_base=events_url.rsplit("/v1/", 1)[0],
    )


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_json(text: str) -> Any:
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
    """On a repaired call the routing fields are the first call's; cost and tokens are summed
    over both."""

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
) -> LLMResult:
    # a system= kwarg collides with the SDK's own
    messages = [{"role": "system", "content": rubric}, {"role": "user", "content": user}]
    started = time.monotonic()
    responses: list[Any] = []

    async def _once(msgs: list[dict[str, Any]], use_model: str) -> Any:
        try:
            resp = await client.messages.create(
                model=use_model, messages=msgs, max_tokens=max_tokens, temperature=temperature
            )
        except Exception as exc:
            raise ItemError("provider_error") from exc
        responses.append(resp)
        return resp

    resp = await _once(messages, model)
    try:
        data = parse_json(resp.text)
    except ItemError:
        if not (resp.text or "").strip():
            raise  # an empty assistant turn is rejected by the API
        # not "auto": the classifier would score the repair prompt, not the document
        repair_model = model if model != "auto" else resp.fluxcompute.model_selected
        resp = await _once(
            messages
            + [
                {"role": "assistant", "content": resp.text},
                {
                    "role": "user",
                    "content": "That was not valid JSON. Reply with only the JSON object.",
                },
            ],
            repair_model,
        )
        data = parse_json(resp.text)

    first, last = responses[0].fluxcompute, responses[-1].fluxcompute
    return LLMResult(
        data=data,
        model_selected=first.model_selected,
        model_served=getattr(responses[-1].raw, "model", last.model_selected),
        difficulty_score=first.difficulty_score,
        difficulty_label=first.difficulty_label,
        cost_usd=sum(float(r.fluxcompute.cost_usd) for r in responses),
        baseline_cost_usd=sum(float(r.fluxcompute.baseline_cost_usd) for r in responses),
        savings_usd=sum(float(r.fluxcompute.savings_usd) for r in responses),
        input_tokens=sum(int(r.usage.get("input_tokens", 0)) for r in responses),
        output_tokens=sum(int(r.usage.get("output_tokens", 0)) for r in responses),
        latency_ms=int((time.monotonic() - started) * 1000),
        overhead_ms=sum(float(r.fluxcompute.overhead_ms) for r in responses),
        repaired=len(responses) > 1,
        provider=getattr(responses[-1], "provider", "anthropic"),
    )


def connect(path: Path, schema: str) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(schema)
    return conn


def run_id(prefix: str, attempt: int = 1) -> str:
    return f"{prefix}-{utc_now():%Y%m%d}-{attempt:02d}"


def next_attempt(db: sqlite3.Connection, prefix: str) -> int:
    """Back-to-back runs on one day must not share a task id."""
    today = f"{prefix}-{utc_now():%Y%m%d}-"
    row = db.execute(
        "SELECT MAX(task_id) AS latest FROM run WHERE task_id LIKE ?", (today + "%",)
    ).fetchone()
    latest = row["latest"] if row else None
    return int(latest.rsplit("-", 1)[1]) + 1 if latest else 1


@dataclass
class Scorecard:
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
        repaired = sum(1 for c in self.calls if c.repaired)
        lines = [
            "",
            "─" * 62,
            f"  {self.title}",
            "─" * 62,
            f"  LLM calls          {len(self.calls) + repaired}"
            + (
                f"  ({repaired} repair {'retry' if repaired == 1 else 'retries'})"
                if repaired
                else ""
            ),
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


def graph_line(clients: Clients, task_id: str) -> str:
    graph = clients.anthropic.get_task_graph(task_id)
    nodes = len(graph.in_order()) if graph else 0
    line = f"  graph: {nodes} nodes recorded in-process"
    if clients.telemetry:
        line += f"\n  {clients.dashboard_base}/ui/#task={task_id}"
    return line


def display(path: Path) -> str:
    """Never an absolute path: printed paths end up in recordings."""
    try:
        return str(path.resolve().relative_to(HERE))
    except ValueError:
        return path.name


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())
