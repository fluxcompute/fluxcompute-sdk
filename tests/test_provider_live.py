"""Live guards for the provider tier maps and the dispatchers' call contract.

Every other test patches `dispatch_anthropic`, so an API-side rejection of a kwarg
the dispatcher always sends (it puts `temperature` in every request) passes CI and
only shows on a real call. The recorded example runs never reached the hard tier,
so a tier that rejects the request could be down for every user without any test
knowing. These tests send the dispatcher's exact request to each tier model, then
route one genuinely hard query end to end with `model="auto"`.

The OpenAI half sends the dispatcher's request to the models whose call contract
broke in 0.3.2 (gpt-5.x and gpt-6-luna reject `max_tokens` and any non-default
`temperature`), and forces a reasoning model to spend its whole output limit on
reasoning, which must raise FluxEmptyResponseError rather than return "".

Opt-in, because they spend real money (well under a cent per run):

    FLUX_LIVE_TESTS=1 ANTHROPIC_API_KEY=sk-ant-... OPENAI_API_KEY=sk-... \
        pytest tests/test_provider_live.py

Each provider's tests skip without its key, and everything skips without
FLUX_LIVE_TESTS=1, which is what CI does.
"""

from __future__ import annotations

import os

import anthropic
import openai
import pytest

from fluxcompute import FluxClient, FluxEmptyResponseError
from fluxcompute.classifier.heuristic import ANTHROPIC_MODELS, OPENAI_MODELS, classify
from fluxcompute.router.dispatcher import dispatch_anthropic, dispatch_openai

# Read at collection time: conftest.py clears the provider keys for every test.
_KEY = os.environ.get("ANTHROPIC_API_KEY")
_OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
_OPTED_IN = os.environ.get("FLUX_LIVE_TESTS") == "1"

needs_anthropic = pytest.mark.skipif(
    not (_KEY and _OPTED_IN),
    reason="live provider test: set FLUX_LIVE_TESTS=1 and ANTHROPIC_API_KEY",
)
needs_openai = pytest.mark.skipif(
    not (_OPENAI_KEY and _OPTED_IN),
    reason="live provider test: set FLUX_LIVE_TESTS=1 and OPENAI_API_KEY",
)

# >= 3 reasoning keywords (+0.35) and >= 3 math keywords (+0.30) with no simple-query
# keywords: scores 0.65, above the 0.45 hard threshold, whatever the model answers.
HARD_QUERY = (
    "Explain and analyze the trade-offs, then compare and evaluate the two approaches: "
    "calculate the derivative and the integral of x squared, and the probability that "
    "both stay positive."
)


@pytest.fixture
async def anthropic_client():
    client = anthropic.AsyncAnthropic(api_key=_KEY)
    try:
        yield client
    finally:
        await client.close()


@needs_anthropic
@pytest.mark.parametrize("temperature", [None, 1.0, 0.0], ids=["unset", "default", "explicit-0"])
@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
async def test_tier_model_accepts_the_dispatchers_request(anthropic_client, tier, temperature):
    model = ANTHROPIC_MODELS[tier]
    try:
        result = await dispatch_anthropic(
            client=anthropic_client,
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=8,
            temperature=temperature,
        )
    except anthropic.APIStatusError as exc:
        pytest.fail(
            f"{tier} tier ({model}) rejected the dispatcher's request with "
            f"temperature={temperature}: HTTP {exc.status_code}: {exc.message}"
        )
    assert result["output_tokens"] > 0


@needs_anthropic
async def test_a_hard_query_routes_end_to_end():
    messages = [{"role": "user", "content": HARD_QUERY}]
    # If the heuristic ever changes, fail here rather than silently test a cheaper tier.
    assert classify(messages).label == "hard"

    client = FluxClient(anthropic_key=_KEY, telemetry=False, provider="anthropic")
    try:
        response = await client.messages.create(model="auto", messages=messages, max_tokens=8)
    finally:
        await client.close()

    assert response.fluxcompute.model_selected == ANTHROPIC_MODELS["hard"]
    assert response.usage.get("output_tokens", 0) > 0


# --------------------------------------------------------------------------- OpenAI

# A prompt that makes a reasoning model think before it answers, so a small
# output limit is spent entirely on reasoning (64 and 256 both were, 2026-10-01).
REASONING_QUERY = (
    "A train leaves at 3:17pm going 47 mph; another leaves the same station at 4:02pm "
    "going 61 mph on the same track. At what exact time does the second catch the first? "
    "Then classify this email as primary, promotional or notification: "
    "'Your invoice #4471 is overdue, pay by Friday.' Answer with the time and the label."
)


@pytest.fixture
async def openai_client():
    client = openai.AsyncOpenAI(api_key=_OPENAI_KEY)
    try:
        yield client
    finally:
        await client.close()


@needs_openai
@pytest.mark.parametrize("temperature", [None, 1.0, 0.0], ids=["unset", "default", "explicit-0"])
@pytest.mark.parametrize("model", ["gpt-4o-mini", "gpt-5.6-luna", *OPENAI_MODELS.values()])
async def test_openai_model_accepts_the_dispatchers_request(openai_client, model, temperature):
    """gpt-4o-mini: a chat model. gpt-5.6-luna: an older reasoning model. The rest: the tier map.
    explicit-0 on a reasoning model is first rejected, then resent without it."""
    try:
        result = await dispatch_openai(
            client=openai_client,
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=2000,
            temperature=temperature,
        )
    except openai.APIStatusError as exc:
        pytest.fail(
            f"{model} rejected the dispatcher's request with temperature={temperature}: "
            f"HTTP {exc.status_code}: {exc.message}"
        )
    assert result["output_tokens"] > 0
    assert result["response"].choices[0].message.content


@needs_openai
async def test_auto_routes_to_the_openai_easy_tier_end_to_end():
    """The easy tier is gpt-6-luna, a reasoning model: `auto` must work on it unpatched."""
    messages = [{"role": "user", "content": "Classify this email as primary, promotional or "
                 "notification: 'Your invoice #4471 is overdue, pay by Friday.' One word."}]
    client = FluxClient(openai_key=_OPENAI_KEY, telemetry=False, provider="openai")
    try:
        response = await client.messages.create(model="auto", messages=messages, max_tokens=2000)
    finally:
        await client.close()

    assert response.fluxcompute.model_selected == OPENAI_MODELS["easy"]
    assert response.text.strip()
    assert response.fluxcompute.cost_usd > 0


@needs_openai
async def test_a_limit_spent_on_reasoning_raises():
    client = FluxClient(openai_key=_OPENAI_KEY, telemetry=False, provider="openai")
    try:
        with pytest.raises(FluxEmptyResponseError) as raised:
            await client.messages.create(
                model="gpt-6-luna",
                messages=[{"role": "user", "content": REASONING_QUERY}],
                max_tokens=64,
            )
    finally:
        await client.close()

    assert raised.value.output_tokens == 64
    assert raised.value.reasoning_tokens > 0
