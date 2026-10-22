"""Tests for deadlines, retry policy and the circuit breaker."""

from __future__ import annotations

import numpy as np
import pytest

from deadlineproxy import (
    EQUAL_JITTER,
    FULL_JITTER,
    NO_JITTER,
    CircuitBreaker,
    CircuitOpen,
    CircuitState,
    Deadline,
    DeadlineBudget,
    RetryPolicy,
    compute_backoff,
    should_retry,
)


class FakeClock:
    """A monotonic clock the tests can advance by hand."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# --------------------------------------------------------------------------- #
# Deadline
# --------------------------------------------------------------------------- #


def test_remaining_counts_down() -> None:
    clock = FakeClock()
    deadline = Deadline.from_timeout(1.0, clock=clock)
    assert deadline.remaining(clock=clock) == pytest.approx(1.0)
    clock.advance(0.25)
    assert deadline.remaining(clock=clock) == pytest.approx(0.75)


def test_remaining_never_goes_negative() -> None:
    clock = FakeClock()
    deadline = Deadline.from_timeout(1.0, clock=clock)
    clock.advance(5.0)
    assert deadline.remaining(clock=clock) == 0.0
    assert deadline.expired(clock=clock) is True


def test_from_timeout_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Deadline.from_timeout(-1.0)


def test_child_deadline_is_smaller() -> None:
    clock = FakeClock()
    deadline = Deadline.from_timeout(1.0, clock=clock)
    child = deadline.child(0.25, clock=clock)
    assert child.remaining(clock=clock) < deadline.remaining(clock=clock)
    assert child.remaining(clock=clock) == pytest.approx(0.25, abs=0.01)


def test_child_rejects_invalid_share() -> None:
    deadline = Deadline.from_timeout(1.0)
    with pytest.raises(ValueError, match="share"):
        deadline.child(0.0)
    with pytest.raises(ValueError, match="share"):
        deadline.child(1.5)


def test_with_timeout_never_extends() -> None:
    clock = FakeClock()
    deadline = Deadline.from_timeout(0.5, clock=clock)
    narrowed = deadline.with_timeout(10.0)
    assert narrowed.remaining(clock=clock) == pytest.approx(0.5, abs=0.01)


def test_at_creates_an_absolute_deadline() -> None:
    clock = FakeClock(1000.0)
    deadline = Deadline.at(1002.0)
    assert deadline.remaining(clock=clock) == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# DeadlineBudget
# --------------------------------------------------------------------------- #


def test_budget_reserves_and_tracks() -> None:
    budget = DeadlineBudget(total=1.0)
    assert budget.reserve("routing", 0.2) == pytest.approx(0.2)
    assert budget.reserve("provider", 0.7) == pytest.approx(0.7)
    assert budget.used == pytest.approx(0.9)
    assert budget.available == pytest.approx(0.1)


def test_budget_clamps_over_reservation() -> None:
    budget = DeadlineBudget(total=1.0)
    budget.reserve("routing", 0.4)
    assert budget.reserve("provider", 5.0) == pytest.approx(0.6)


def test_budget_rejects_negative_total() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        DeadlineBudget(total=-1.0)


def test_budget_rejects_negative_reservation() -> None:
    budget = DeadlineBudget(total=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        budget.reserve("routing", -1.0)


def test_budget_deadline_for_unknown_stage() -> None:
    budget = DeadlineBudget(total=1.0)
    with pytest.raises(KeyError):
        budget.deadline_for("nope")


def test_budget_deadline_accounts_for_other_stages() -> None:
    clock = FakeClock()
    budget = DeadlineBudget(total=1.0)
    budget.reserve("routing", 0.2)
    budget.reserve("provider", 0.7)
    deadline = budget.deadline_for("provider", clock=clock)
    # The provider stage gets the whole budget minus the routing stage.
    assert deadline.remaining(clock=clock) == pytest.approx(0.8, abs=0.01)


# --------------------------------------------------------------------------- #
# Retry policy
# --------------------------------------------------------------------------- #


def test_backoff_is_within_bounds() -> None:
    policy = RetryPolicy(base_delay=0.1, multiplier=2.0, max_delay=5.0)
    rng = np.random.default_rng(0)
    for attempt in range(1, 10):
        delay = compute_backoff(attempt, policy, rng)
        assert 0.0 <= delay <= min(5.0, 0.1 * 2 ** (attempt - 1))


def test_full_jitter_is_random_within_the_ceiling() -> None:
    policy = RetryPolicy(base_delay=1.0, multiplier=1.0, max_delay=10.0, jitter=FULL_JITTER)
    rng = np.random.default_rng(1)
    delays = [compute_backoff(1, policy, rng) for _ in range(50)]
    assert len(set(delays)) > 10
    assert all(0.0 <= d <= 1.0 for d in delays)


def test_equal_jitter_keeps_half_the_delay() -> None:
    policy = RetryPolicy(base_delay=1.0, multiplier=1.0, max_delay=10.0, jitter=EQUAL_JITTER)
    rng = np.random.default_rng(1)
    delays = [compute_backoff(1, policy, rng) for _ in range(50)]
    assert all(0.5 <= d <= 1.0 for d in delays)


def test_no_jitter_is_deterministic() -> None:
    policy = RetryPolicy(base_delay=1.0, multiplier=2.0, max_delay=10.0, jitter=NO_JITTER)
    rng = np.random.default_rng(1)
    assert compute_backoff(1, policy, rng) == 1.0
    assert compute_backoff(3, policy, rng) == 4.0


def test_backoff_rejects_bad_attempt() -> None:
    policy = RetryPolicy()
    with pytest.raises(ValueError, match="attempt"):
        compute_backoff(0, policy, np.random.default_rng(0))


def test_policy_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError, match="base_delay"):
        RetryPolicy(base_delay=-1.0)
    with pytest.raises(ValueError, match="multiplier"):
        RetryPolicy(multiplier=0.5)
    with pytest.raises(ValueError, match="jitter"):
        RetryPolicy(jitter="wild")


def test_client_cannot_exceed_the_operator_ceiling() -> None:
    """Regression test for LiteLLM #46911."""
    policy = RetryPolicy(max_attempts=3)
    assert policy.clamp_attempts(1000) == 3
    assert policy.clamp_attempts(2) == 2
    assert policy.clamp_attempts(None) == 3
    assert policy.clamp_attempts(0) == 1


def test_should_retry_respects_max_attempts() -> None:
    policy = RetryPolicy(max_attempts=2)
    assert should_retry(TimeoutError("x"), 1, policy) is True
    assert should_retry(TimeoutError("x"), 2, policy) is False


def test_should_not_retry_a_permanent_error() -> None:
    policy = RetryPolicy(max_attempts=5)
    assert should_retry(ValueError("bad request"), 1, policy) is False


def test_should_not_start_an_attempt_without_time() -> None:
    """A retry that cannot finish in the remaining time is pointless."""
    clock = FakeClock()
    policy = RetryPolicy(max_attempts=5, min_attempt_budget=0.5)
    deadline = Deadline.from_timeout(0.2, clock=clock)
    assert should_retry(TimeoutError("x"), 1, policy, deadline, clock=clock) is False


def test_should_retry_when_there_is_time_left() -> None:
    clock = FakeClock()
    policy = RetryPolicy(max_attempts=5, min_attempt_budget=0.5)
    deadline = Deadline.from_timeout(5.0, clock=clock)
    assert should_retry(TimeoutError("x"), 1, policy, deadline, clock=clock) is True


# --------------------------------------------------------------------------- #
# Circuit breaker
# --------------------------------------------------------------------------- #


def test_breaker_starts_closed() -> None:
    breaker = CircuitBreaker()
    assert breaker.state is CircuitState.CLOSED
    assert breaker.allow_request() is True


def test_breaker_opens_after_the_failure_threshold() -> None:
    breaker = CircuitBreaker(failure_threshold=3)
    for _ in range(3):
        breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    assert breaker.allow_request() is False


def test_old_failures_leave_the_sliding_window() -> None:
    """Failures outside the window must not count towards the threshold."""
    breaker = CircuitBreaker(failure_threshold=3, window_size=3)
    breaker.record_failure()
    breaker.record_failure()
    # Three more outcomes push both failures out of the three-slot window.
    breaker.record_success()
    breaker.record_success()
    breaker.record_success()
    assert breaker.failure_count == 0
    breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED


def test_breaker_becomes_half_open_after_the_recovery_timeout() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=30.0, clock=clock)
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    clock.advance(31.0)
    assert breaker.state is CircuitState.HALF_OPEN


def test_half_open_success_closes_the_circuit() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=30.0, clock=clock)
    breaker.record_failure()
    clock.advance(31.0)
    assert breaker.allow_request() is True
    breaker.record_success()
    assert breaker.state is CircuitState.CLOSED


def test_half_open_failure_reopens_the_circuit() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=30.0, clock=clock)
    breaker.record_failure()
    clock.advance(31.0)
    breaker.allow_request()
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN


def test_half_open_admits_only_a_limited_number_of_probes() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(
        failure_threshold=1, recovery_timeout=30.0, half_open_max_calls=1, clock=clock
    )
    breaker.record_failure()
    clock.advance(31.0)
    assert breaker.allow_request() is True
    assert breaker.allow_request() is False


def test_breaker_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="failure_threshold"):
        CircuitBreaker(failure_threshold=0)
    with pytest.raises(ValueError, match="recovery_timeout"):
        CircuitBreaker(recovery_timeout=-1.0)
    with pytest.raises(ValueError, match="half_open_max_calls"):
        CircuitBreaker(half_open_max_calls=0)
    with pytest.raises(ValueError, match="success_threshold"):
        CircuitBreaker(success_threshold=0)
    with pytest.raises(ValueError, match="window_size"):
        CircuitBreaker(window_size=0)


def test_circuit_open_exception_carries_the_provider() -> None:
    error = CircuitOpen("openai", CircuitState.OPEN)
    assert error.provider == "openai"
    assert error.state is CircuitState.OPEN
