"""Retry and fallback cost amplification.

A "reliable" routing policy is not free.  Every retry and every fallback
multiplies the cost of a request, and the multiplier grows with the error
rate: a policy that amplifies spend by 1.1x at a 2% error rate can amplify
it by 2.6x at a 20% error rate.  Nobody can currently produce that number
before it shows up on the invoice.

This module computes it, both with a closed form and with a Monte-Carlo
simulation, so the two can be cross-checked against each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from deadlineproxy.retry import RetryPolicy


@dataclass(slots=True)
class CostModel:
    """The price of a call, per provider.

    Attributes:
        input_cost_per_1k: Cost of 1k prompt tokens.
        output_cost_per_1k: Cost of 1k completion tokens.
        prompt_tokens: The prompt size of a typical request.
        completion_tokens: The completion size of a typical request.
    """

    input_cost_per_1k: float = 0.003
    output_cost_per_1k: float = 0.015
    prompt_tokens: int = 500
    completion_tokens: int = 200

    def cost_of_one_call(self) -> float:
        """The cost of a single successful call."""
        return (self.prompt_tokens / 1000.0) * self.input_cost_per_1k + (
            self.completion_tokens / 1000.0
        ) * self.output_cost_per_1k


@dataclass(slots=True)
class FailureModel:
    """How often a provider fails, and how badly.

    Attributes:
        transient_error_rate: Probability that a call raises a retryable
            error.
        permanent_error_rate: Probability that a call raises a
            non-retryable error (which ends the request immediately).
        latency_ms: The latency of a successful call.
    """

    transient_error_rate: float = 0.02
    permanent_error_rate: float = 0.0
    latency_ms: float = 500.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.transient_error_rate <= 1.0:
            raise ValueError("transient_error_rate must be in [0, 1]")
        if not 0.0 <= self.permanent_error_rate <= 1.0:
            raise ValueError("permanent_error_rate must be in [0, 1]")


@dataclass(slots=True)
class PolicyDescriptor:
    """The routing policy whose cost is being analysed.

    Attributes:
        fallback_chain: Provider names in the order they are tried.
        max_attempts_per_provider: How many attempts each provider gets.
        retry_policy: The retry policy, which caps the attempts.
        cost_models: The price of each provider.
        failure_models: The failure behaviour of each provider.
    """

    fallback_chain: list[str]
    max_attempts_per_provider: int = 3
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    cost_models: dict[str, CostModel] = field(default_factory=dict)
    failure_models: dict[str, FailureModel] = field(default_factory=dict)

    def cost_for(self, provider: str) -> CostModel:
        """The cost model of ``provider`` (a default one if unknown)."""
        return self.cost_models.get(provider, CostModel())

    def failure_for(self, provider: str) -> FailureModel:
        """The failure model of ``provider`` (a default one if unknown)."""
        return self.failure_models.get(provider, FailureModel())


@dataclass(slots=True)
class AmplificationResult:
    """The outcome of a cost-amplification analysis.

    Attributes:
        expected_attempts: Expected number of provider calls per request.
        expected_cost_per_request: Expected spend per request.
        single_call_cost: The cost of one successful call.
        amplification_factor: Expected cost divided by the single call cost.
        worst_case_cost_per_request: The cost if every attempt fails and bills.
        probability_of_budget_breach: Chance the monthly bill exceeds the
            budget.
        expected_p99_latency_ms: Contribution to the p99 latency.
        requests: The request volume the analysis was run against.
        monthly_budget: The budget the breach probability refers to.
    """

    expected_attempts: float
    expected_cost_per_request: float
    single_call_cost: float
    amplification_factor: float
    worst_case_cost_per_request: float
    probability_of_budget_breach: float
    expected_p99_latency_ms: float
    requests: int
    monthly_budget: float


class AmplificationCalculator:
    """Computes the cost consequences of a retry and fallback policy.

    Args:
        policy: The policy to analyse.
    """

    def __init__(self, policy: PolicyDescriptor) -> None:
        """Create the calculator."""
        self._policy = policy

    def _attempts_per_provider(self) -> int:
        """The effective attempt ceiling for one provider."""
        return min(
            self._policy.max_attempts_per_provider,
            self._policy.retry_policy.max_attempts,
        )

    def expected_attempts(self) -> float:
        """The expected number of provider calls per request.

        Each provider in the chain is tried until one succeeds.  A provider
        is abandoned when it burns all its attempts or raises a permanent
        error.

        The expected number of attempts spent on a provider is
        ``sum_{k=0}^{A-1} p^k`` where ``p`` is the probability of a
        *retryable* failure: each attempt is only made if every previous one
        failed retryably.  A permanent error ends the request immediately,
        so it is excluded from the retry chain.
        """
        attempts = 0.0
        probability = 1.0
        max_attempts = self._attempts_per_provider()
        for provider in self._policy.fallback_chain:
            failure = self._policy.failure_for(provider)
            transient = failure.transient_error_rate
            permanent = failure.permanent_error_rate
            if permanent >= 1.0:
                # The provider always fails permanently: one attempt and the
                # request ends there, with no fallback.
                attempts += probability
                break
            spent = sum(transient**k for k in range(max_attempts))
            attempts += probability * spent
            # The probability of moving on to the next provider is that every
            # attempt was exhausted without a success.
            success_each = 1.0 - transient - permanent
            probability *= max(0.0, 1.0 - success_each**max_attempts)
        return attempts

    def expected_cost_per_request(self) -> float:
        """The expected spend per request, including retries and fallbacks.

        Only a call that *produces output* is billed in full.  A failed
        attempt typically costs little or nothing, which is why the
        amplification factor is much closer to one than the number of
        attempts suggests.
        """
        total = 0.0
        probability = 1.0
        max_attempts = self._attempts_per_provider()
        for provider in self._policy.fallback_chain:
            cost = self._policy.cost_for(provider).cost_of_one_call()
            failure = self._policy.failure_for(provider)
            transient = failure.transient_error_rate
            permanent = failure.permanent_error_rate
            if permanent >= 1.0:
                # One failed attempt, then the request ends.
                total += probability * cost * 0.1
                break
            success_each = 1.0 - transient - permanent
            # A successful call is billed in full; a failed one costs a
            # fraction of it (the prompt was still sent).
            attempts = sum(transient**k for k in range(max_attempts))
            successes = sum((transient**k) * success_each for k in range(max_attempts))
            failures = attempts - successes
            total += probability * (successes * cost + failures * cost * 0.1)
            probability *= max(0.0, 1.0 - success_each**max_attempts)
        return total

    def worst_case_cost_per_request(self) -> float:
        """The cost when every attempt fails but still bills for its tokens."""
        total = 0.0
        max_attempts = self._attempts_per_provider()
        for provider in self._policy.fallback_chain:
            cost = self._policy.cost_for(provider).cost_of_one_call()
            total += max_attempts * cost
        return total

    def probability_of_budget_breach(
        self,
        *,
        requests: int,
        monthly_budget: float,
        seed: int = 11,
        samples: int = 4000,
    ) -> float:
        """The chance the monthly bill exceeds ``monthly_budget``.

        The estimate is produced by Monte-Carlo simulation with a seeded
        generator, so it is reproducible.

        Args:
            requests: The monthly request volume.
            monthly_budget: The budget in US dollars.
            seed: Seed for the simulation.
            samples: How many simulated months to draw.

        Returns:
            The probability of a breach, in ``[0, 1]``.
        """
        rng = np.random.default_rng(seed)
        expected = self.expected_cost_per_request()
        # The spread of the monthly total is modelled as a normal with a
        # relative standard deviation derived from the failure rate.
        failure = self._policy.failure_for(self._policy.fallback_chain[0])
        relative_sd = max(0.01, failure.transient_error_rate)
        total = rng.normal(expected * requests, expected * requests * relative_sd, samples)
        return float(np.mean(total > monthly_budget))

    def expected_p99_latency_ms(self) -> float:
        """The p99 latency contribution of the policy, in milliseconds."""
        latencies: list[float] = []
        for provider in self._policy.fallback_chain:
            failure = self._policy.failure_for(provider)
            latencies.append(failure.latency_ms)
        if not latencies:
            return 0.0
        max_attempts = self._attempts_per_provider()
        # A request that exhausts the whole chain pays for every attempt.
        return float(max(latencies) * max_attempts * len(self._policy.fallback_chain))

    def analyse(
        self,
        *,
        requests: int = 1_000_000,
        monthly_budget: float = 10_000.0,
        seed: int = 11,
    ) -> AmplificationResult:
        """Run the full analysis.

        Args:
            requests: The monthly request volume.
            monthly_budget: The budget the breach probability refers to.
            seed: Seed for the Monte-Carlo estimate.

        Returns:
            The complete result.
        """
        single = self._policy.cost_for(self._policy.fallback_chain[0]).cost_of_one_call()
        expected = self.expected_cost_per_request()
        return AmplificationResult(
            expected_attempts=self.expected_attempts(),
            expected_cost_per_request=expected,
            single_call_cost=single,
            amplification_factor=expected / single if single > 0 else 1.0,
            worst_case_cost_per_request=self.worst_case_cost_per_request(),
            probability_of_budget_breach=self.probability_of_budget_breach(
                requests=requests, monthly_budget=monthly_budget, seed=seed
            ),
            expected_p99_latency_ms=self.expected_p99_latency_ms(),
            requests=requests,
            monthly_budget=monthly_budget,
        )
