"""Circuit breaker with a sliding failure window and half-open probing.

The state machine is the familiar one, with two details that matter in
practice:

* Failures are counted in a **sliding window**, not cumulatively.  A provider
  that failed fifty times last week and has been perfect since should not
  still be considered unhealthy.
* The **half-open** state admits a limited number of probe calls.  Without
  that limit, a recovering provider is immediately swamped by the traffic
  that queued up while it was open, and it never recovers.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

CircuitState = Enum("CircuitState", "CLOSED OPEN HALF_OPEN")


class CircuitOpen(Exception):
    """Raised when a call is rejected because the circuit is open."""

    def __init__(self, provider: str, state: CircuitState) -> None:
        """Create the exception.

        Args:
            provider: The provider whose circuit rejected the call.
            state: The state the circuit was in.
        """
        self.provider = provider
        self.state = state
        super().__init__(f"circuit for {provider!r} is {state.name}")


@dataclass(slots=True)
class _Outcome:
    """One recorded call outcome."""

    success: bool
    timestamp: float


class CircuitBreaker:
    """A circuit breaker for one provider.

    Args:
        failure_threshold: Failures within the window that open the circuit.
        recovery_timeout: Seconds before an open circuit becomes half-open.
        half_open_max_calls: Probe calls allowed while half-open.
        success_threshold: Successes in half-open that close the circuit.
        window_size: How many recent outcomes the window holds.
        clock: Monotonic clock callable, injectable for tests.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
        success_threshold: int = 1,
        window_size: int = 20,
        clock: object = time.monotonic,
    ) -> None:
        """Create the breaker.

        Raises:
            ValueError: If any threshold is out of range.
        """
        if failure_threshold < 1:
            raise ValueError(f"failure_threshold must be at least 1, got {failure_threshold}")
        if recovery_timeout < 0:
            raise ValueError(f"recovery_timeout must be non-negative, got {recovery_timeout}")
        if half_open_max_calls < 1:
            raise ValueError(f"half_open_max_calls must be at least 1, got {half_open_max_calls}")
        if success_threshold < 1:
            raise ValueError(f"success_threshold must be at least 1, got {success_threshold}")
        if window_size < 1:
            raise ValueError(f"window_size must be at least 1, got {window_size}")

        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_calls = half_open_max_calls
        self._success_threshold = success_threshold
        self._window_size = window_size
        self._clock = clock

        self._state = CircuitState.CLOSED
        self._outcomes: deque[_Outcome] = deque(maxlen=self._window_size)
        self._opened_at: float | None = None
        self._half_open_calls = 0
        self._half_open_successes = 0

    @property
    def state(self) -> CircuitState:
        """The current state, after applying the recovery timeout."""
        self._maybe_half_open()
        return self._state

    @property
    def failure_count(self) -> int:
        """Failures currently inside the sliding window."""
        return sum(1 for outcome in self._outcomes if not outcome.success)

    @property
    def success_count(self) -> int:
        """Successes currently inside the sliding window."""
        return sum(1 for outcome in self._outcomes if outcome.success)

    def _maybe_half_open(self) -> None:
        """Move an expired open circuit into the half-open state."""
        if self._state is not CircuitState.OPEN or self._opened_at is None:
            return
        if self._clock() - self._opened_at >= self._recovery_timeout:  # type: ignore[operator]
            self._state = CircuitState.HALF_OPEN
            self._half_open_calls = 0
            self._half_open_successes = 0

    def allow_request(self) -> bool:
        """Whether a call may proceed right now.

        Returns:
            ``True`` when the call is allowed.
        """
        self._maybe_half_open()
        if self._state is CircuitState.CLOSED:
            return True
        if self._state is CircuitState.OPEN:
            return False
        # Half-open: admit a limited number of probes.
        if self._half_open_calls < self._half_open_max_calls:
            self._half_open_calls += 1
            return True
        return False

    def record_success(self) -> None:
        """Record a successful call and update the state."""
        now = self._clock()  # type: ignore[operator]
        if self._state is CircuitState.HALF_OPEN:
            self._half_open_successes += 1
            if self._half_open_successes >= self._success_threshold:
                self._state = CircuitState.CLOSED
                self._outcomes.clear()
                self._half_open_calls = 0
                self._half_open_successes = 0
            return
        self._outcomes.append(_Outcome(success=True, timestamp=now))

    def record_failure(self) -> None:
        """Record a failed call and update the state."""
        now = self._clock()  # type: ignore[operator]
        if self._state is CircuitState.HALF_OPEN:
            # Any failure while probing sends the circuit straight back to open.
            self._state = CircuitState.OPEN
            self._opened_at = now
            self._half_open_calls = 0
            self._half_open_successes = 0
            return
        self._outcomes.append(_Outcome(success=False, timestamp=now))
        if self.failure_count >= self._failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = now

    def reset(self) -> None:
        """Force the circuit closed and forget the window."""
        self._state = CircuitState.CLOSED
        self._outcomes.clear()
        self._opened_at = None
        self._half_open_calls = 0
        self._half_open_successes = 0


class BreakerRegistry:
    """A circuit breaker per provider name."""

    def __init__(self, **breaker_kwargs: object) -> None:
        """Create the registry.

        Args:
            **breaker_kwargs: Keyword arguments forwarded to every
                :class:`CircuitBreaker` that is created.
        """
        self._kwargs = breaker_kwargs
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, provider: str) -> CircuitBreaker:
        """Return (creating if needed) the breaker for ``provider``."""
        if provider not in self._breakers:
            self._breakers[provider] = CircuitBreaker(**self._kwargs)  # type: ignore[arg-type]
        return self._breakers[provider]

    def states(self) -> dict[str, CircuitState]:
        """The state of every known breaker."""
        return {name: breaker.state for name, breaker in self._breakers.items()}

    def reset(self) -> None:
        """Reset every breaker."""
        for breaker in self._breakers.values():
            breaker.reset()
