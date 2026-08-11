"""Tests for cost calculator."""

import pytest

from fluxcompute.classifier.heuristic import ANTHROPIC_MODELS, OPENAI_MODELS
from fluxcompute.cost import MODEL_PRICING, calculate_cost, calculate_savings, get_baseline_model


class TestCalculateCost:
    def test_haiku_cheap(self):
        cost = calculate_cost("claude-3-5-haiku-20241022", 1000, 500)
        assert cost < 0.01  # very cheap

    def test_opus_expensive(self):
        cost = calculate_cost("claude-opus-4-20250918", 1000, 500)
        assert cost > calculate_cost("claude-3-5-haiku-20241022", 1000, 500)

    def test_zero_tokens(self):
        cost = calculate_cost("claude-3-5-haiku-20241022", 0, 0)
        assert cost == 0.0

    def test_unknown_model(self):
        cost = calculate_cost("unknown-model-xyz", 1000, 500)
        assert cost == 0.0

    def test_openai_pricing(self):
        cost_mini = calculate_cost("gpt-4o-mini", 1000, 500)
        cost_4o = calculate_cost("gpt-4o", 1000, 500)
        assert cost_mini < cost_4o


class TestCalculateSavings:
    def test_haiku_vs_opus(self):
        actual, baseline, savings = calculate_savings(
            "claude-3-5-haiku-20241022", "claude-opus-4-20250918", 1000, 500
        )
        assert savings > 0
        assert baseline > actual
        assert savings == baseline - actual

    def test_opus_vs_opus_no_savings(self):
        actual, baseline, savings = calculate_savings(
            "claude-opus-4-20250918", "claude-opus-4-20250918", 1000, 500
        )
        assert savings == 0.0

    def test_savings_never_negative(self):
        # If somehow selected model is more expensive than baseline
        _, _, savings = calculate_savings(
            "claude-opus-4-20250918", "claude-3-5-haiku-20241022", 1000, 500
        )
        assert savings >= 0.0


class TestModelPricingCompleteness:
    def test_all_classifier_anthropic_models_have_pricing(self):
        """Every model the classifier can route to must have a price — missing entries silently return 0.0."""
        for tier, model_id in ANTHROPIC_MODELS.items():
            assert model_id in MODEL_PRICING, (
                f"ANTHROPIC_MODELS['{tier}'] = '{model_id}' is not in MODEL_PRICING. "
                "Add it to fluxcompute/cost.py or cost calculations will be wrong."
            )

    def test_all_classifier_openai_models_have_pricing(self):
        for tier, model_id in OPENAI_MODELS.items():
            assert model_id in MODEL_PRICING, (
                f"OPENAI_MODELS['{tier}'] = '{model_id}' is not in MODEL_PRICING. "
                "Add it to fluxcompute/cost.py."
            )


class TestGetBaselineModel:
    def test_anthropic_baseline(self):
        assert get_baseline_model("anthropic") == ANTHROPIC_MODELS["hard"]

    def test_openai_baseline(self):
        assert get_baseline_model("openai") == OPENAI_MODELS["hard"]

    def test_unknown_provider_falls_back_to_a_priced_model(self):
        assert get_baseline_model("unknown") in MODEL_PRICING


_PROVIDERS = [("anthropic", ANTHROPIC_MODELS), ("openai", OPENAI_MODELS)]


class TestBaselineInvariant:
    """Savings are `baseline_cost - actual_cost`, so the baseline must be the
    priciest model the router can pick for that provider. When it isn't, the
    reported baseline understates what an unrouted call really costs and
    savings collapse to zero.
    """

    @staticmethod
    def _price(model):
        return calculate_cost(model, 10_000, 2_000)

    @pytest.mark.parametrize("provider,tiers", _PROVIDERS)
    def test_baseline_is_priciest_model_in_its_own_tier_map(self, provider, tiers):
        baseline = get_baseline_model(provider)
        priciest = max(set(tiers.values()), key=self._price)
        assert self._price(baseline) >= self._price(priciest), (
            f"{provider} baseline {baseline} is cheaper than tier model {priciest}"
        )

    @pytest.mark.parametrize("provider,tiers", _PROVIDERS)
    def test_no_tier_reports_negative_or_absent_savings_at_the_cheap_end(self, provider, tiers):
        baseline = get_baseline_model(provider)
        for tier, model in tiers.items():
            _, _, savings = calculate_savings(model, baseline, 10_000, 2_000)
            assert savings >= 0, f"{provider}/{tier} ({model}) reports negative savings"
        _, _, easy_savings = calculate_savings(tiers["easy"], baseline, 10_000, 2_000)
        assert easy_savings > 0, f"{provider} easy tier shows no savings at all"

    @pytest.mark.parametrize("provider,tiers", _PROVIDERS)
    def test_every_tier_model_is_priced(self, provider, tiers):
        """An unpriced model makes calculate_cost() return 0.0 silently, which
        reads as 'free' rather than 'unknown' and fabricates savings."""
        for tier, model in tiers.items():
            assert model in MODEL_PRICING, f"{provider}/{tier} ({model}) has no price"
            assert MODEL_PRICING[model] != (0.0, 0.0), f"{model} is priced at zero"

    @pytest.mark.parametrize("provider,tiers", _PROVIDERS)
    def test_reported_baseline_cost_is_never_below_actual_cost(self, provider, tiers):
        """Reported baseline cost must never be below the actual cost of the
        call it is the baseline for."""
        baseline = get_baseline_model(provider)
        for tier, model in tiers.items():
            cost, baseline_cost, _ = calculate_savings(model, baseline, 10_000, 2_000)
            assert baseline_cost >= cost, (
                f"{provider}/{tier}: baseline ${baseline_cost:.4f} < actual ${cost:.4f}"
            )
