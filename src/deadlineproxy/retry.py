"""Retry policy with bounded, jittered backoff.

The two properties that matter for an LLM gateway are:

* **Bounded.** A client-supplied retry count must never be able to exceed the
  operator's ceiling.  LiteLLM issue #46911 documents clients asking for a
  thousand retries, which is an amplification attack rather than a
  configuration mistake.
* **Deadline-aware.** An attempt must not even be *started* unless there is
  enough time left for it to finish.  Starting an attempt that cannot
  complete is how a "30 second timeout" turns into a 137 second hang.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from deadlineproxy.deadline import Deadline

JitterStrategy = str
FULL_JITTER = "full"
EQUAL_JITTER = "equal"
NO_JITTER = "none"


class RetryExhausted(Exception):
    """Raised when every permitted attempt has failed."""

    def __init__(self, attempts: int, last_error: BaseException | None = None) -> None:
        """Create the exception.

        Args:
            attempts: How many attempts were made.
            last_error: The error from the final attempt.
        """
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"retries exhausted after {attempts} attempt(s)")


@dataclass(slots=True)
class RetryPolicy:
    """How many times to retry, how long to wait, and what to retry on.

    Attributes:
        max_attempts: The operator's ceiling on total attempts, including the
            first one.
        base_delay: The delay before the first retry, in seconds.
        max_delay: The upper bound on any single delay, in seconds.
        multiplier: The exponential growth factor of the delay.
        jitter: ``"full"``, ``"equal"`` or ``"none"``.
        retry_on: Exception types that are worth retrying.
        respect_deadline: Whether to skip attempts that cannot finish in time.
        min_attempt_budget: The minimum time an attempt needs to be worth
            starting.
    """

    max_attempts: int = 3
    base_delay: float = 0.1
    max_delay: float = 5.0
    multiplier: float = 2.0
    jitter: JitterStrategy = FULL_JITTER
    retry_on: tuple[type[BaseException], ...] = (TimeoutError, ConnectionError)
    respect_deadline: bool = True
    min_attempt_budget: float = 0.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {self.max_attempts}")
        if self.base_delay < 0:
            raise ValueError(f"base_delay must be non-negative, got {self.base_delay}")
        if self.max_delay < 0:
            raise ValueError(f"max_delay must be non-negative, got {self.max_delay}")
        if self.multiplier < 1.0:
            raise ValueError(f"multiplier must be at least 1, got {self.multiplier}")
        if self.jitter not in {FULL_JITTER, EQUAL_JITTER, NO_JITTER}:
            raise ValueError(f"unknown jitter strategy {self.jitter!r}")

    def clamp_attempts(self, requested: int | None) -> int:
        """Clamp a client-requested attempt count to the operator's ceiling.

        This is the guard for LiteLLM #46911: a client may ask for any
        number of attempts, but the operator's ceiling always wins.

        Args:
            requested: The client-requested number of attempts, or ``None``.

        Returns:
            The number of attempts that will actually be permitted.
        """
        if requested is None:
            return self.max_attempts
        if requested < 1:
            return 1
        return min(int(requested), self.max_attempts)


def compute_backoff(
    attempt: int,
    policy: RetryPolicy,
    rng: np.random.Generator,
) -> float:
    """Compute the delay before the next attempt.

    With ``"full"`` jitter the delay is drawn uniformly from
    ``[0, min(max_delay, base_delay * multiplier ** (attempt - 1))]``.  Full
    jitter is the variant that actually breaks up a thundering herd: every
    client that failed at the same moment waits a *different* amount.

    Args:
        attempt: The number of the attempt that just failed (1-based).
        policy: The retry policy.
        rng: A seeded random generator.

    Returns:
        The delay in seconds.
    """
    if attempt < 1:
        raise ValueError(f"attempt must be at least 1, got {attempt}")
    ceiling = min(policy.max_delay, policy.base_delay * (policy.multiplier ** (attempt - 1)))
    if ceiling <= 0.0:
        return 0.0
    if policy.jitter == NO_JITTER:
        return float(ceiling)
    if policy.jitter == EQUAL_JITTER:
        half = ceiling / 2.0
        return float(half + rng.uniform(0.0, half))
    return float(rng.uniform(0.0, ceiling))


def should_retry(
    error: BaseException,
    attempt: int,
    policy: RetryPolicy,
    deadline: Deadline | None = None,
    *,
    clock: object = time.monotonic,
) -> bool:
    """Decide whether another attempt is permitted.

    Args:
        error: The error raised by the attempt.
        attempt: The number of the attempt that just failed (1-based).
        policy: The retry policy.
        deadline: The request deadline, if the request has one.
        clock: Monotonic clock callable, injectable for tests.

    Returns:
        ``True`` when another attempt should be made.
    """
    if attempt >= policy.max_attempts:
        return False
    if not isinstance(error, policy.retry_on):
        return False
    # Only start an attempt when there is time for it to finish.
    time_left = deadline.remaining(clock=clock) if deadline is not None else None
    if policy.respect_deadline and time_left is not None:
        return time_left >= policy.min_attempt_budget
    return True
