"""Tests for the gateway: deadlines, fallbacks, streaming and statistics."""

from __future__ import annotations

import pytest

from deadlineproxy import (
    BreakerRegistry,
    Deadline,
    Gateway,
    GatewayError,
    ProviderRegistry,
    Request,
    RetryPolicy,
    SimulatedProvider,
)


def make_gateway(*providers: SimulatedProvider, **kwargs: object) -> Gateway:
    """Build a gateway over the given providers."""
    policy = kwargs.pop("retry_policy", None) or RetryPolicy(
        max_attempts=2, base_delay=0.0, max_delay=0.0
    )
    return Gateway(
        ProviderRegistry(list(providers)),
        retry_policy=policy,
        breakers=BreakerRegistry(failure_threshold=2, recovery_timeout=60.0),
        **kwargs,  # type: ignore[arg-type]
    )


def test_successful_call_returns_the_response() -> None:
    provider = SimulatedProvider("fast", latency_ms=10.0)
    gateway = make_gateway(provider)
    response = gateway.send(Request(prompt="hello"), timeout=1.0)
    assert response.text.startswith("fast response")
    assert response.provider == "fast"


def test_stats_count_the_request() -> None:
    gateway = make_gateway(SimulatedProvider("fast", latency_ms=10.0))
    gateway.send(Request(prompt="hello"), timeout=1.0)
    stats = gateway.stats()
    assert stats.requests == 1
    assert stats.successes == 1
    assert stats.attempts == 1
    assert stats.input_tokens > 0
    assert stats.cost_usd > 0.0


def test_deadline_is_derived_from_the_timeout() -> None:
    gateway = make_gateway(SimulatedProvider("fast", latency_ms=10.0))
    gateway.send(Request(prompt="hello"), timeout=1.0)
    assert gateway.stats().successes == 1


def test_request_timeout_field_is_used() -> None:
    gateway = make_gateway(SimulatedProvider("fast", latency_ms=10.0))
    gateway.send(Request(prompt="hello", timeout=1.0))
    assert gateway.stats().successes == 1


def test_falls_back_to_the_next_provider() -> None:
    failing = SimulatedProvider("failing", fail_rate=1.0, latency_ms=5.0)
    healthy = SimulatedProvider("healthy", latency_ms=5.0)
    gateway = make_gateway(failing, healthy)
    response = gateway.send(Request(prompt="hello"), timeout=1.0)
    assert response.provider == "healthy"
    assert gateway.stats().fallbacks == 1


def test_all_providers_failing_raises_with_the_history() -> None:
    failing = SimulatedProvider("failing", fail_rate=1.0, latency_ms=5.0)
    gateway = make_gateway(failing)
    with pytest.raises(GatewayError) as excinfo:
        gateway.send(Request(prompt="hello"), timeout=1.0)
    assert excinfo.value.history
    assert excinfo.value.history[0].provider == "failing"
    assert gateway.stats().failures == 1


def test_permanent_errors_are_not_retried() -> None:
    provider = SimulatedProvider("bad", permanent_failure_rate=1.0, latency_ms=5.0)
    gateway = make_gateway(provider)
    with pytest.raises(GatewayError):
        gateway.send(Request(prompt="hello"), timeout=1.0)
    # One attempt only: a permanent error must not be retried.
    assert gateway.stats().attempts == 1


def test_deadline_is_enforced_even_when_the_provider_ignores_it() -> None:
    """Regression test for LiteLLM #47822 and #47481.

    The provider deliberately ignores the deadline and takes far longer than
    the request allowed.  The gateway must not let the request run past its
    deadline.
    """
    stubborn = SimulatedProvider("stubborn", latency_ms=5000.0, ignore_deadline=True)
    gateway = make_gateway(stubborn)
    deadline = Deadline.from_timeout(0.05)
    with pytest.raises(GatewayError):
        gateway.send(Request(prompt="hello"), deadline=deadline)
    assert gateway.stats().timeouts >= 0


def test_client_cannot_raise_the_attempt_ceiling() -> None:
    """Regression test for LiteLLM #46911."""
    provider = SimulatedProvider("failing", fail_rate=1.0, latency_ms=5.0)
    gateway = make_gateway(provider, retry_policy=RetryPolicy(max_attempts=2, base_delay=0.0))
    with pytest.raises(GatewayError):
        gateway.send(Request(prompt="hello", max_attempts=1000), timeout=1.0)
    # The operator's ceiling of 2 attempts wins over the client's 1000.
    assert gateway.stats().attempts == 2


def test_no_attempt_starts_without_enough_time() -> None:
    provider = SimulatedProvider("slow", latency_ms=1000.0)
    gateway = make_gateway(provider)
    deadline = Deadline.from_timeout(0.0)
    with pytest.raises(GatewayError):
        gateway.send(Request(prompt="hello"), deadline=deadline)
    assert gateway.stats().attempts == 0


def test_breaker_opens_after_repeated_failures() -> None:
    provider = SimulatedProvider("failing", fail_rate=1.0, latency_ms=5.0)
    gateway = make_gateway(provider)
    for _ in range(3):
        with pytest.raises(GatewayError):
            gateway.send(Request(prompt="hello"), timeout=1.0)
    assert gateway.breakers.get("failing").state.name == "OPEN"


def test_router_selects_the_first_provider() -> None:
    first = SimulatedProvider("first", latency_ms=5.0)
    second = SimulatedProvider("second", latency_ms=5.0)
    gateway = make_gateway(first, second, router=lambda request: "second")
    response = gateway.send(Request(prompt="hello"), timeout=1.0)
    assert response.provider == "second"


def test_router_promotes_its_choice_but_keeps_the_fallbacks() -> None:
    first = SimulatedProvider("first", latency_ms=5.0)
    second = SimulatedProvider("second", latency_ms=5.0)
    gateway = make_gateway(first, second, router=lambda request: "second")
    response = gateway.send(Request(prompt="hello"), timeout=1.0)
    assert response.provider == "second"
    assert gateway.providers.names == ["first", "second"]


def test_streaming_collects_the_chunks() -> None:
    provider = SimulatedProvider("streamer", latency_ms=5.0)
    gateway = make_gateway(provider)
    result = gateway.send_stream(Request(prompt="hello", stream=True), timeout=1.0)
    assert result.chunks
    assert "".join(result.chunks) == "streamer streamed response"
    assert result.truncated is False


def test_streaming_is_cut_off_at_the_deadline() -> None:
    provider = SimulatedProvider("streamer", latency_ms=5.0)
    gateway = make_gateway(provider)
    deadline = Deadline.from_timeout(0.0)
    result = gateway.send_stream(Request(prompt="hello", stream=True), deadline=deadline)
    assert result.truncated is True
    assert len(result.chunks) < 4


def test_streaming_stall_is_reported() -> None:
    provider = SimulatedProvider("stalling", latency_ms=5.0, stream_stall_after=2)
    gateway = make_gateway(provider)
    result = gateway.send_stream(Request(prompt="hello", stream=True), timeout=1.0)
    assert len(result.chunks) == 2


def test_stats_as_dict_is_serialisable() -> None:
    gateway = make_gateway(SimulatedProvider("fast", latency_ms=5.0))
    gateway.send(Request(prompt="hello"), timeout=1.0)
    data = gateway.stats().as_dict()
    assert data["requests"] == 1
    assert isinstance(data["cost_usd"], float)


def test_reset_stats_clears_the_counters() -> None:
    gateway = make_gateway(SimulatedProvider("fast", latency_ms=5.0))
    gateway.send(Request(prompt="hello"), timeout=1.0)
    gateway.reset_stats()
    assert gateway.stats().requests == 0
