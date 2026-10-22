"""Provider abstraction and a deterministic simulated provider.

The simulated provider is not a toy: it reproduces the *specific* upstream
behaviour that makes real gateways unreliable, so the gateway's defences can
be demonstrated rather than asserted.

Most importantly, a provider can be configured to **ignore the deadline**.
LiteLLM issue #47822 ("httpx.Timeout connect timeout ignored for non-OpenAI
providers") and issue #47481 ("aiohttp transport hangs 40-137 s on a stopped
upstream") are both this bug.  A gateway that trusts the provider's own
timeout handling inherits the hang; a gateway that enforces the deadline
itself does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from deadlineproxy.deadline import Deadline

DEFAULT_INPUT_COST_PER_1K = 0.003
DEFAULT_OUTPUT_COST_PER_1K = 0.015


@dataclass(slots=True)
class ProviderCapabilities:
    """What a provider supports.

    Attributes:
        streaming: Whether the provider can stream tokens.
        tools: Whether the provider supports tool/function calling.
        json_mode: Whether the provider supports structured JSON output.
        max_context_tokens: The largest prompt the provider accepts.
    """

    streaming: bool = True
    tools: bool = True
    json_mode: bool = False
    max_context_tokens: int = 128_000


@dataclass(slots=True)
class Response:
    """A provider's answer.

    Attributes:
        text: The generated text.
        model: The model that produced it.
        input_tokens: Prompt tokens consumed.
        output_tokens: Completion tokens produced.
        cost_usd: The cost of the call.
        latency_ms: How long the call took.
        finish_reason: ``"stop"``, ``"length"`` or ``"error"``.
        streamed_chunks: How many chunks were streamed.
        provider: The provider that served the call.
    """

    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    streamed_chunks: int = 0
    provider: str = ""


@dataclass(slots=True)
class Request:
    """A request to send to a provider.

    Attributes:
        prompt: The prompt text.
        model: The model the caller asked for.
        max_output_tokens: Requested completion length.
        timeout: The client's own timeout hint, in seconds.
        max_attempts: The client's requested attempt count.  The gateway
            clamps this to the operator's ceiling.
        stream: Whether the caller wants a streamed response.
        metadata: Arbitrary caller metadata.
    """

    prompt: str
    model: str = "default"
    max_output_tokens: int = 128
    timeout: float | None = None
    max_attempts: int | None = None
    stream: bool = False
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class Provider(Protocol):
    """The contract every provider implements."""

    @property
    def name(self) -> str:
        """The provider's name."""
        ...

    @property
    def capabilities(self) -> ProviderCapabilities:
        """What the provider supports."""
        ...

    def complete(self, request: Request, deadline: Deadline) -> Response:
        """Produce a response, honouring ``deadline`` if the provider can."""
        ...

    def stream(self, request: Request, deadline: Deadline):
        """Yield response chunks, honouring ``deadline`` if the provider can."""
        ...


class SimulatedProvider:
    """A deterministic provider used for tests and demonstrations.

    Args:
        name: The provider name.
        latency_ms: How long a successful call takes.
        fail_rate: Probability that a call raises a transient error.
        permanent_failure_rate: Probability that a call raises a
            non-retryable error.
        ignore_deadline: When ``True`` the provider **ignores** the deadline
            and always takes ``latency_ms``, reproducing the upstream bugs
            this gateway defends against.
        stream_stall_after: Stop streaming after this many chunks, to
            simulate a provider that stalls mid-stream.
        input_cost_per_1k: Cost per 1k prompt tokens.
        output_cost_per_1k: Cost per 1k completion tokens.
        seed: Seed for the failure generator.
    """

    def __init__(
        self,
        name: str = "simulated",
        *,
        latency_ms: float = 50.0,
        fail_rate: float = 0.0,
        permanent_failure_rate: float = 0.0,
        ignore_deadline: bool = False,
        stream_stall_after: int | None = None,
        input_cost_per_1k: float = DEFAULT_INPUT_COST_PER_1K,
        output_cost_per_1k: float = DEFAULT_OUTPUT_COST_PER_1K,
        seed: int = 0,
        capabilities: ProviderCapabilities | None = None,
    ) -> None:
        """Create the simulated provider."""
        self._name = name
        self._latency_ms = latency_ms
        self._fail_rate = fail_rate
        self._permanent_failure_rate = permanent_failure_rate
        self._ignore_deadline = ignore_deadline
        self._stream_stall_after = stream_stall_after
        self._input_cost = input_cost_per_1k
        self._output_cost = output_cost_per_1k
        self._seed = seed
        self._capabilities = capabilities or ProviderCapabilities()
        self._calls = 0

    @property
    def name(self) -> str:
        """The provider name."""
        return self._name

    @property
    def capabilities(self) -> ProviderCapabilities:
        """What the provider supports."""
        return self._capabilities

    @property
    def calls(self) -> int:
        """How many times the provider was called."""
        return self._calls

    def _should_fail(self, counter: int) -> bool:
        """Deterministically decide whether this call fails."""
        permanent = int(self._permanent_failure_rate * 100)
        transient = int((self._permanent_failure_rate + self._fail_rate) * 100)
        position = counter % 100
        return position < permanent or position < transient

    def complete(self, request: Request, deadline: Deadline) -> Response:
        """Produce a response.

        Args:
            request: The request.
            deadline: The request deadline.

        Returns:
            The provider's response.

        Raises:
            TimeoutError: If the deadline passes before the call completes.
            ConnectionError: On a simulated transient failure.
            ValueError: On a simulated permanent failure.
        """
        self._calls += 1
        counter = self._calls
        if self._should_fail(counter):
            if counter % 100 < int(self._permanent_failure_rate * 100):
                raise ValueError(f"{self._name}: permanent failure")
            raise ConnectionError(f"{self._name}: transient failure")

        remaining_ms = deadline.remaining() * 1000.0
        if not self._ignore_deadline and remaining_ms < self._latency_ms:
            raise TimeoutError(
                f"{self._name}: deadline of {remaining_ms:.0f}ms is below "
                f"the {self._latency_ms:.0f}ms the call needs"
            )

        input_tokens = max(1, len(request.prompt) // 4)
        output_tokens = max(1, request.max_output_tokens)
        cost = (input_tokens / 1000.0) * self._input_cost + (
            output_tokens / 1000.0
        ) * self._output_cost
        return Response(
            text=f"{self._name} response to: {request.prompt[:40]}",
            model=request.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=self._latency_ms,
            finish_reason="stop",
            provider=self._name,
        )

    def stream(self, request: Request, deadline: Deadline):
        """Yield response chunks.

        Args:
            request: The request.
            deadline: The request deadline.

        Yields:
            The text of each chunk.  The chunks together always spell out the
            whole response, so a consumer that joins them loses nothing.
        """
        self._calls += 1
        text = f"{self._name} streamed response"
        chunk_size = 5
        chunk_count = max(1, -(-len(text) // chunk_size))
        for index in range(chunk_count):
            if self._stream_stall_after is not None and index >= self._stream_stall_after:
                return
            if not self._ignore_deadline and deadline.expired():
                return
            start = index * chunk_size
            yield text[start : start + chunk_size]


class ProviderRegistry:
    """An ordered collection of providers.

    The order is the **fallback chain**: the first provider is tried first and
    the rest are used as fallbacks.
    """

    def __init__(self, providers: list[Provider] | None = None) -> None:
        """Create the registry.

        Args:
            providers: The providers, in fallback order.
        """
        self._providers: list[Provider] = list(providers or [])

    def __len__(self) -> int:
        """Number of registered providers."""
        return len(self._providers)

    def __iter__(self):
        """Iterate over the providers in fallback order."""
        return iter(self._providers)

    @property
    def names(self) -> list[str]:
        """The provider names in fallback order."""
        return [provider.name for provider in self._providers]

    def get(self, name: str) -> Provider:
        """Return the provider called ``name``.

        Raises:
            KeyError: If no such provider is registered.
        """
        for provider in self._providers:
            if provider.name == name:
                return provider
        raise KeyError(f"unknown provider {name!r}")

    def add(self, provider: Provider) -> None:
        """Append a provider to the end of the fallback chain."""
        self._providers.append(provider)
