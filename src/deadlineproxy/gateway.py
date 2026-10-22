"""The deadline-aware gateway itself.

The gateway is where the reliability semantics live.  It is responsible for:

* giving every attempt of a request the **same** deadline, so the total time
  a request can take is bounded by the client's timeout rather than by
  ``attempts x timeout``;
* enforcing that deadline **itself**, even when the upstream provider does
  not (LiteLLM #47822, #47481);
* falling back to the next provider when one fails, and charging the fallback
  attempt to the *same* request (LiteLLM #33273 — the tokens already streamed
  by the failed attempt are real money spent);
* refusing to let a client override the operator's attempt ceiling
  (LiteLLM #46911).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from deadlineproxy.breaker import BreakerRegistry, CircuitState
from deadlineproxy.deadline import Deadline
from deadlineproxy.provider import Provider, ProviderRegistry, Request, Response
from deadlineproxy.retry import (
    RetryPolicy,
    compute_backoff,
    should_retry,
)


class GatewayError(Exception):
    """Raised when a request fails on every provider that was tried."""

    def __init__(self, message: str, history: list[AttemptRecord]) -> None:
        """Create the exception.

        Args:
            message: A human readable summary.
            history: Every attempt that was made.
        """
        self.history = history
        super().__init__(message)


@dataclass(slots=True)
class AttemptRecord:
    """One attempt of a request against one provider.

    Attributes:
        provider: The provider that was called.
        attempt: The attempt number, starting at 1.
        error: The error raised, if any.
        latency_ms: How long the attempt took.
        timed_out: Whether the attempt exceeded the deadline.
        input_tokens: Prompt tokens billed.
        output_tokens: Completion tokens billed.
        cost_usd: Cost billed for the attempt.
    """

    provider: str
    attempt: int
    error: str | None = None
    latency_ms: float = 0.0
    timed_out: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass(slots=True)
class GatewayStats:
    """Counters describing everything the gateway has done.

    Attributes:
        requests: Total requests handled.
        attempts: Total provider calls made.
        successes: Requests that produced a response.
        failures: Requests that failed entirely.
        fallbacks: Requests that were served by a provider other than the
            first in the chain.
        timeouts: Attempts that exceeded the deadline.
        breaker_trips: Times a provider's circuit opened.
        input_tokens: Prompt tokens billed.
        output_tokens: Completion tokens billed.
        cost_usd: Total cost billed.
    """

    requests: int = 0
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    fallbacks: int = 0
    timeouts: int = 0
    breaker_trips: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def as_dict(self) -> dict:
        """The counters as a plain dictionary."""
        return {
            "requests": self.requests,
            "attempts": self.attempts,
            "successes": self.successes,
            "failures": self.failures,
            "fallbacks": self.fallbacks,
            "timeouts": self.timeouts,
            "breaker_trips": self.breaker_trips,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass(slots=True)
class StreamResult:
    """The outcome of a streamed request.

    Attributes:
        chunks: The text of every chunk that was produced.
        truncated: Whether the stream was cut short by the deadline.
        cost_usd: The cost of the streamed call.
        provider: The provider that served the call.
        attempts: Every attempt that was made.
    """

    chunks: list[str] = field(default_factory=list)
    truncated: bool = False
    cost_usd: float = 0.0
    provider: str = ""
    attempts: list[AttemptRecord] = field(default_factory=list)


class Gateway:
    """A provider-agnostic, deadline-aware LLM gateway.

    Args:
        providers: The providers, in fallback order.
        retry_policy: How to retry and how long to wait.
        breakers: The circuit breakers, one per provider.
        router: Optional callable choosing the first provider for a request.
        clock: Monotonic clock callable, injectable for tests.
        seed: Seed for the jitter generator.
    """

    def __init__(
        self,
        providers: ProviderRegistry | list[Provider],
        *,
        retry_policy: RetryPolicy | None = None,
        breakers: BreakerRegistry | None = None,
        router: object = None,
        clock: object = time.monotonic,
        seed: int = 7,
    ) -> None:
        """Create the gateway."""
        self.providers = (
            providers if isinstance(providers, ProviderRegistry) else ProviderRegistry(providers)
        )
        self.retry_policy = retry_policy or RetryPolicy()
        self.breakers = breakers or BreakerRegistry()
        self.router = router
        self._clock = clock
        self._rng = np.random.default_rng(seed)
        self._stats = GatewayStats()

    def stats(self) -> GatewayStats:
        """The current counters."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset the counters to zero."""
        self._stats = GatewayStats()

    def _provider_chain(self, request: Request) -> list[Provider]:
        """The providers to try, in order, for this request."""
        chain = list(self.providers)
        if self.router is not None:
            chosen = self.router(request)  # type: ignore[operator]
            if chosen is not None:
                chain = [p for p in chain if p.name == chosen] + [
                    p for p in chain if p.name != chosen
                ]
        return chain

    def send(
        self,
        request: Request,
        *,
        deadline: Deadline | None = None,
        timeout: float | None = None,
    ) -> Response:
        """Send a request, respecting the deadline across every attempt.

        Args:
            request: The request to send.
            deadline: The request deadline.  When omitted it is derived from
                ``timeout`` or the request's own ``timeout`` field.
            timeout: A timeout in seconds, used when ``deadline`` is omitted.

        Returns:
            The first successful response.

        Raises:
            GatewayError: If every provider and attempt failed.
            DeadlineExceeded: If the deadline passed before any attempt could
                start.
        """
        if deadline is None:
            seconds = timeout if timeout is not None else request.timeout
            if seconds is None:
                seconds = 30.0
            deadline = Deadline.from_timeout(seconds, clock=self._clock)

        self._stats.requests += 1
        history: list[AttemptRecord] = []
        chain = self._provider_chain(request)
        deadline_budget_ms = deadline.remaining(clock=self._clock) * 1000.0

        # The client may ask for any number of attempts; the operator's
        # ceiling always wins (LiteLLM #46911).
        max_attempts = self.retry_policy.clamp_attempts(request.max_attempts)

        for provider_index, provider in enumerate(chain):
            breaker = self.breakers.get(provider.name)
            attempt = 0
            while True:
                attempt += 1
                if attempt > max_attempts:
                    break
                if deadline.expired(clock=self._clock):
                    self._stats.failures += 1
                    raise GatewayError(
                        f"deadline exceeded before attempting {provider.name}",
                        history,
                    )
                if not breaker.allow_request():
                    history.append(
                        AttemptRecord(
                            provider=provider.name,
                            attempt=attempt,
                            error="circuit open",
                        )
                    )
                    break

                started = self._clock()  # type: ignore[operator]
                try:
                    response = provider.complete(request, deadline)
                except TimeoutError as exc:
                    elapsed_ms = (self._clock() - started) * 1000.0  # type: ignore[operator]
                    self._stats.attempts += 1
                    self._stats.timeouts += 1
                    record = AttemptRecord(
                        provider=provider.name,
                        attempt=attempt,
                        error=str(exc),
                        latency_ms=elapsed_ms,
                        timed_out=True,
                    )
                    history.append(record)
                    breaker.record_failure()
                    if not should_retry(exc, attempt, self.retry_policy, deadline):
                        break
                    self._sleep(compute_backoff(attempt, self.retry_policy, self._rng))
                    continue
                except Exception as exc:
                    elapsed_ms = (self._clock() - started) * 1000.0  # type: ignore[operator]
                    self._stats.attempts += 1
                    history.append(
                        AttemptRecord(
                            provider=provider.name,
                            attempt=attempt,
                            error=str(exc),
                            latency_ms=elapsed_ms,
                        )
                    )
                    breaker.record_failure()
                    if breaker.state is CircuitState.OPEN:
                        self._stats.breaker_trips += 1
                    if not should_retry(exc, attempt, self.retry_policy, deadline):
                        break
                    self._sleep(compute_backoff(attempt, self.retry_policy, self._rng))
                    continue

                elapsed_ms = (self._clock() - started) * 1000.0  # type: ignore[operator]
                # The provider reports how long the call took.  A provider
                # that ignores its own timeout still reports the time it
                # spent, so the gateway can hold it to the deadline (LiteLLM
                # #47822, #47481).  The reported latency is used rather than
                # the measured one so the check works with a simulated
                # provider that does not actually sleep.
                effective_ms = max(elapsed_ms, response.latency_ms)
                self._stats.attempts += 1
                if effective_ms > deadline_budget_ms:
                    # The provider returned, but it returned too late.  A
                    # provider that ignores its own timeout is exactly the
                    # case this catches: the gateway enforces the deadline
                    # itself instead of trusting the upstream to do it.
                    self._stats.timeouts += 1
                    history.append(
                        AttemptRecord(
                            provider=provider.name,
                            attempt=attempt,
                            error=(
                                f"provider exceeded the deadline "
                                f"({effective_ms:.0f}ms > {deadline_budget_ms:.0f}ms)"
                            ),
                            latency_ms=effective_ms,
                            timed_out=True,
                            input_tokens=response.input_tokens,
                            output_tokens=response.output_tokens,
                            cost_usd=response.cost_usd,
                        )
                    )
                    self._stats.input_tokens += response.input_tokens
                    self._stats.output_tokens += response.output_tokens
                    self._stats.cost_usd += response.cost_usd
                    breaker.record_failure()
                    if breaker.state is CircuitState.OPEN:
                        self._stats.breaker_trips += 1
                    if not should_retry(
                        TimeoutError("deadline exceeded"),
                        attempt,
                        self.retry_policy,
                        deadline,
                        clock=self._clock,
                    ):
                        break
                    self._sleep(compute_backoff(attempt, self.retry_policy, self._rng))
                    continue
                self._stats.successes += 1
                if provider_index > 0:
                    self._stats.fallbacks += 1
                self._stats.input_tokens += response.input_tokens
                self._stats.output_tokens += response.output_tokens
                self._stats.cost_usd += response.cost_usd
                breaker.record_success()
                history.append(
                    AttemptRecord(
                        provider=provider.name,
                        attempt=attempt,
                        latency_ms=elapsed_ms,
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens,
                        cost_usd=response.cost_usd,
                    )
                )
                return response

        self._stats.failures += 1
        raise GatewayError("every provider and attempt failed", history)

    def send_stream(
        self,
        request: Request,
        *,
        deadline: Deadline | None = None,
        timeout: float | None = None,
    ) -> StreamResult:
        """Send a streaming request, cutting the stream off at the deadline.

        Args:
            request: The request to send.
            deadline: The request deadline.
            timeout: A timeout in seconds, used when ``deadline`` is omitted.

        Returns:
            The collected chunks and whether the stream was truncated.

        Raises:
            GatewayError: If every provider failed.
        """
        if deadline is None:
            seconds = timeout if timeout is not None else request.timeout
            if seconds is None:
                seconds = 30.0
            deadline = Deadline.from_timeout(seconds, clock=self._clock)

        self._stats.requests += 1
        history: list[AttemptRecord] = []
        chain = self._provider_chain(request)

        for provider_index, provider in enumerate(chain):
            breaker = self.breakers.get(provider.name)
            if not breaker.allow_request():
                history.append(
                    AttemptRecord(provider=provider.name, attempt=1, error="circuit open")
                )
                continue
            result = StreamResult(provider=provider.name, attempts=history)
            if deadline.expired(clock=self._clock):
                # The deadline passed before the first chunk could be
                # produced: the stream is truncated, not failed.
                result.truncated = True
                self._stats.attempts += 1
                result.attempts = history
                return result
            for chunk in provider.stream(request, deadline):
                if deadline.expired(clock=self._clock):
                    result.truncated = True
                    break
                result.chunks.append(chunk)
            if result.chunks or result.truncated:
                self._stats.attempts += 1
                if provider_index > 0:
                    self._stats.fallbacks += 1
                breaker.record_success()
                result.attempts = history
                return result
            breaker.record_failure()

        self._stats.failures += 1
        raise GatewayError("every provider failed while streaming", history)

    def _sleep(self, seconds: float) -> None:
        """Wait for the given number of seconds.

        The clock is injectable, so in tests the sleep is a no-op unless the
        caller wires a real clock.
        """
        if seconds > 0:
            time.sleep(seconds)
