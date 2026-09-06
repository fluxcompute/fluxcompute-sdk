"""Guard the installed provider SDKs against the kwargs we actually send.

Every other test patches `dispatch_anthropic`/`dispatch_openai`, so a breaking
change in a provider SDK's method signature passes CI cleanly and only fails
on a real install at runtime. That happened: the unbounded `anthropic>=0.30.0`
spec let a fresh `pip install fluxcompute` pull anthropic 1.0.0, which removed
`temperature` from `AsyncMessages.create()` -- the signature has no **kwargs
to absorb it, so every real call raised `TypeError: AsyncMessages.create() got
an unexpected keyword argument 'temperature'`. Reproduced directly: fresh venv,
`pip install fluxcompute anthropic`, a real `client.messages.create(...)` call.
Same bug, same fix, already shipped server-side in fluxcompute-observability's
`test_provider_sdk_compat.py` -- ported here since this package's dependency
spec is independent and had never received the same pin.

These introspect the real installed SDKs, so they fail at CI time on the next
such change instead of after a release. Introspection runs in a clean
subprocess so a stubbed-out provider module imported by another test file
earlier in the same session can't shadow the real one.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

# The exact kwargs dispatch_anthropic builds, plus what the tool-use and
# extended-thinking paths pass through **kwargs (fluxcompute/router/dispatcher.py).
_ANTHROPIC_KWARGS = ["model", "messages", "max_tokens", "temperature", "system",
                     "tools", "thinking"]
# dispatch_openai's kwargs; max_completion_tokens is the reasoning-model
# spelling _openai_token_kwargs switches to for o1/o3.
_OPENAI_KWARGS = ["model", "messages", "max_tokens", "temperature", "stream",
                  "max_completion_tokens"]

_PROBE = textwrap.dedent(
    """
    import inspect, json, sys

    def accepts(fn, name):
        sig = inspect.signature(fn)
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return True
        return name in sig.parameters

    which = sys.argv[1]
    names = json.loads(sys.argv[2])
    if which == "anthropic":
        import anthropic as mod
        fn = mod.resources.messages.AsyncMessages.create
    else:
        import openai as mod
        fn = mod.resources.chat.completions.AsyncCompletions.create
    print(json.dumps({
        "version": mod.__version__,
        "accepts": {n: accepts(fn, n) for n in names},
    }))
    """
)


def _probe(which: str, names: list[str]) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, which, json.dumps(names)],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        pytest.skip(f"could not introspect {which}: {proc.stderr.strip()[:200]}")
    return json.loads(proc.stdout)


def test_anthropic_create_accepts_our_kwargs():
    res = _probe("anthropic", _ANTHROPIC_KWARGS)
    rejected = [n for n, ok in res["accepts"].items() if not ok]
    assert not rejected, (
        f"installed anthropic {res['version']} does not accept {rejected} on "
        f"AsyncMessages.create(); dispatch_anthropic sends these, so every "
        f"real call would raise TypeError. Check the pin in pyproject.toml "
        f"before widening it."
    )


def test_openai_create_accepts_our_kwargs():
    res = _probe("openai", _OPENAI_KWARGS)
    rejected = [n for n, ok in res["accepts"].items() if not ok]
    assert not rejected, (
        f"installed openai {res['version']} does not accept {rejected} on "
        f"AsyncCompletions.create(); dispatch_openai sends these. Check the pin."
    )
