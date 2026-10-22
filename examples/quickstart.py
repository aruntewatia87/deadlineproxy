#!/usr/bin/env python3
"""End-to-end demo of deadlineproxy.

Run with::

    uv run python examples/quickstart.py
"""

from __future__ import annotations

import contextlib

from deadlineproxy import (
    AmplificationCalculator,
    BreakerRegistry,
    CostModel,
    Deadline,
    FailureModel,
    Gateway,
    GatewayError,
    PolicyDescriptor,
    ProviderRegistry,
    Request,
    RetryPolicy,
    SimulatedProvider,
)


def main() -> None:
    # Three providers: one healthy, one that ignores its timeout, one that
    # fails transiently.
    fast = SimulatedProvider("fast", latency_ms=40.0, seed=1)
    stubborn = SimulatedProvider("stubborn", latency_ms=4000.0, ignore_deadline=True, seed=2)
    flaky = SimulatedProvider("flaky", latency_ms=50.0, fail_rate=0.5, seed=3)

    gateway = Gateway(
        ProviderRegistry([fast, stubborn, flaky]),
        retry_policy=RetryPolicy(max_attempts=3, base_delay=0.0, max_delay=0.0),
        breakers=BreakerRegistry(failure_threshold=2, recovery_timeout=30.0),
    )

    print("=" * 70)
    print("1. A normal request (500 ms deadline)")
    print("=" * 70)
    response = gateway.send(Request(prompt="Summarise the incident report"), timeout=0.5)
    print(f"provider   : {response.provider}")
    print(f"latency    : {response.latency_ms:.0f} ms")
    print(f"tokens     : {response.input_tokens} in / {response.output_tokens} out")
    print(f"cost       : ${response.cost_usd:.6f}")

    print()
    print("=" * 70)
    print("2. A provider that IGNORES its timeout (LiteLLM #47822, #47481)")
    print("=" * 70)
    print("The upstream takes 4000 ms but the client only allowed 50 ms.")
    print("The gateway enforces the deadline itself instead of trusting it.")
    strict = Gateway(
        ProviderRegistry([stubborn]),
        retry_policy=RetryPolicy(max_attempts=2, base_delay=0.0),
    )
    try:
        strict.send(Request(prompt="hello"), deadline=Deadline.from_timeout(0.05))
        print("ERROR: the deadline was not enforced")
    except GatewayError as exc:
        print(f"rejected   : {exc}")
        for record in exc.history:
            print(
                f"  attempt {record.attempt} on {record.provider}: "
                f"timed_out={record.timed_out} latency={record.latency_ms:.0f}ms"
            )

    print()
    print("=" * 70)
    print("3. The client cannot raise the retry ceiling (LiteLLM #46911)")
    print("=" * 70)
    gateway.reset_stats()
    with contextlib.suppress(GatewayError):
        gateway.send(Request(prompt="hello", max_attempts=1000), timeout=0.5)
    print("client asked for 1000 attempts")
    print(f"operator ceiling allowed : {gateway.stats().attempts} attempts")

    print()
    print("=" * 70)
    print("4. Circuit breaker opens after repeated failures")
    print("=" * 70)
    gateway.reset_stats()
    breaker_gateway = Gateway(
        ProviderRegistry([flaky]),
        retry_policy=RetryPolicy(max_attempts=1, base_delay=0.0),
        breakers=BreakerRegistry(failure_threshold=2, recovery_timeout=30.0),
    )
    for _ in range(3):
        with contextlib.suppress(GatewayError):
            breaker_gateway.send(Request(prompt="hello"), timeout=0.5)
    print(f"breaker state after 3 failures: {breaker_gateway.breakers.get('flaky').state.name}")
    print(f"breaker trips recorded      : {breaker_gateway.stats().breaker_trips}")

    print()
    print("=" * 70)
    print("5. What is your 'reliable' policy really costing?")
    print("=" * 70)
    print(f"{'error rate':>12}{'attempts':>12}{'cost/req':>12}{'amplification':>15}")
    print("-" * 51)
    for rate in (0.02, 0.05, 0.20):
        policy = PolicyDescriptor(
            fallback_chain=["primary", "secondary"],
            max_attempts_per_provider=5,
            retry_policy=RetryPolicy(max_attempts=5),
            cost_models={"primary": CostModel(), "secondary": CostModel()},
            failure_models={
                "primary": FailureModel(transient_error_rate=rate),
                "secondary": FailureModel(transient_error_rate=rate),
            },
        )
        result = AmplificationCalculator(policy).analyse(
            requests=1_000_000, monthly_budget=10_000.0
        )
        print(
            f"{rate:>11.0%}{result.expected_attempts:>12.4f}"
            f"{result.expected_cost_per_request:>12.6f}"
            f"{result.amplification_factor:>14.4f}x"
        )
    print()
    print("The attempt count grows much faster than the cost, and every extra")
    print("attempt also eats into the deadline the client asked for.")

    print()
    print("=" * 70)
    print("6. Gateway statistics")
    print("=" * 70)
    for key, value in gateway.stats().as_dict().items():
        print(f"  {key:<16} {value}")


if __name__ == "__main__":
    main()
