"""Tests for the retry/fallback cost amplification calculator."""

from __future__ import annotations

import pytest

from deadlineproxy import (
    AmplificationCalculator,
    CostModel,
    FailureModel,
    PolicyDescriptor,
    RetryPolicy,
)


def make_policy(
    *,
    error_rate: float = 0.02,
    attempts: int = 6,
    chain: list[str] | None = None,
) -> PolicyDescriptor:
    """Build a policy descriptor for the analysis."""
    chain = chain or ["primary"]
    return PolicyDescriptor(
        fallback_chain=chain,
        max_attempts_per_provider=attempts,
        retry_policy=RetryPolicy(max_attempts=attempts),
        cost_models={name: CostModel() for name in chain},
        failure_models={name: FailureModel(transient_error_rate=error_rate) for name in chain},
    )


def test_single_call_cost_is_computed_from_the_tokens() -> None:
    cost = CostModel(
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        prompt_tokens=1000,
        completion_tokens=1000,
    )
    assert cost.cost_of_one_call() == pytest.approx(0.018)


def test_no_failures_means_no_amplification() -> None:
    calculator = AmplificationCalculator(make_policy(error_rate=0.0))
    result = calculator.analyse()
    assert result.amplification_factor == pytest.approx(1.0)
    assert result.expected_attempts == pytest.approx(1.0)


def test_amplification_grows_with_the_error_rate() -> None:
    low = AmplificationCalculator(make_policy(error_rate=0.02)).analyse()
    high = AmplificationCalculator(make_policy(error_rate=0.20)).analyse()
    assert high.amplification_factor > low.amplification_factor
    assert high.expected_attempts > low.expected_attempts


def test_the_error_rate_matters_more_than_the_absolute_amplification() -> None:
    """The headline finding.

    The attempt count grows by roughly 22% between a 2% and a 20% error
    rate.  The cost amplification grows by less, because a failed attempt
    costs a fraction of a successful one -- but the absolute factors are
    small only per request, and they compound with volume: at a million
    requests a month the difference is real money, and every extra attempt
    also consumes latency that the deadline has to absorb.
    """
    low = AmplificationCalculator(make_policy(error_rate=0.02)).analyse()
    high = AmplificationCalculator(make_policy(error_rate=0.20)).analyse()
    assert high.expected_attempts / low.expected_attempts > 1.2
    assert high.amplification_factor > low.amplification_factor
    # At a million requests a month, the extra spend is material.
    extra = (high.expected_cost_per_request - low.expected_cost_per_request) * 1_000_000
    assert extra > 100.0


def test_expected_attempts_matches_the_geometric_series() -> None:
    """With error rate p and A attempts the expectation is sum(p**k)."""
    error_rate = 0.2
    attempts = 6
    calculator = AmplificationCalculator(make_policy(error_rate=error_rate, attempts=attempts))
    expected = sum(error_rate**k for k in range(attempts))
    assert calculator.expected_attempts() == pytest.approx(expected)


def test_a_fallback_chain_multiplies_the_attempts() -> None:
    single = AmplificationCalculator(make_policy(chain=["primary"])).analyse()
    chained = AmplificationCalculator(
        make_policy(chain=["primary", "secondary", "tertiary"])
    ).analyse()
    assert chained.expected_attempts > single.expected_attempts
    assert chained.worst_case_cost_per_request > single.worst_case_cost_per_request


def test_worst_case_costs_every_attempt() -> None:
    policy = make_policy(attempts=3, chain=["primary", "secondary"])
    result = AmplificationCalculator(policy).analyse()
    single = CostModel().cost_of_one_call()
    # Three attempts against each of two providers.
    assert result.worst_case_cost_per_request == pytest.approx(6 * single)


def test_permanent_errors_stop_the_chain_immediately() -> None:
    policy = PolicyDescriptor(
        fallback_chain=["primary", "secondary"],
        max_attempts_per_provider=5,
        retry_policy=RetryPolicy(max_attempts=5),
        cost_models={"primary": CostModel(), "secondary": CostModel()},
        failure_models={
            "primary": FailureModel(transient_error_rate=0.0, permanent_error_rate=1.0),
            "secondary": FailureModel(transient_error_rate=0.0),
        },
    )
    result = AmplificationCalculator(policy).analyse()
    # A permanent error means exactly one attempt and no fallback.
    assert result.expected_attempts == pytest.approx(1.0)


def test_breach_probability_is_monotone_in_the_error_rate() -> None:
    low = AmplificationCalculator(make_policy(error_rate=0.01)).analyse(
        requests=100_000, monthly_budget=460.0
    )
    high = AmplificationCalculator(make_policy(error_rate=0.30)).analyse(
        requests=100_000, monthly_budget=460.0
    )
    assert high.probability_of_budget_breach > low.probability_of_budget_breach


def test_breach_probability_is_a_probability() -> None:
    result = AmplificationCalculator(make_policy()).analyse()
    assert 0.0 <= result.probability_of_budget_breach <= 1.0


def test_breach_probability_is_reproducible() -> None:
    calculator = AmplificationCalculator(make_policy())
    first = calculator.probability_of_budget_breach(requests=1000, monthly_budget=1.0, seed=5)
    second = calculator.probability_of_budget_breach(requests=1000, monthly_budget=1.0, seed=5)
    assert first == second


def test_p99_latency_grows_with_the_chain_and_attempts() -> None:
    short = AmplificationCalculator(make_policy(attempts=1, chain=["a"])).analyse()
    long = AmplificationCalculator(make_policy(attempts=5, chain=["a", "b", "c"])).analyse()
    assert long.expected_p99_latency_ms > short.expected_p99_latency_ms


def test_failure_model_validates_its_rates() -> None:
    with pytest.raises(ValueError, match="transient_error_rate"):
        FailureModel(transient_error_rate=1.5)
    with pytest.raises(ValueError, match="permanent_error_rate"):
        FailureModel(permanent_error_rate=-0.1)


def test_analysis_reports_the_request_volume_and_budget() -> None:
    result = AmplificationCalculator(make_policy()).analyse(requests=123_456, monthly_budget=999.0)
    assert result.requests == 123_456
    assert result.monthly_budget == 999.0


def test_single_call_cost_is_reported() -> None:
    result = AmplificationCalculator(make_policy()).analyse()
    assert result.single_call_cost == pytest.approx(CostModel().cost_of_one_call())
