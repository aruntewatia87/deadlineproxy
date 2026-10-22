"""deadlineproxy - a deadline-aware, provider-agnostic LLM gateway.

The reliability semantics of a real LLM gateway are the hard part, and they
are where the open bugs live.  ``deadlineproxy`` gets them right:

* one end-to-end **deadline** shared by every attempt, so a request cannot
  outlive its timeout no matter how many retries happen;
* the deadline is enforced by the **gateway**, not trusted to the provider --
  upstream providers that ignore their own timeouts are handled rather than
  inherited;
* **bounded retries** with full-jitter backoff, and an operator ceiling that
  a client cannot raise (LiteLLM #46911);
* **circuit breakers** with a sliding failure window and half-open probing;
* **fallback** across providers that charges every attempt to the same
  request, so the tokens burned by a failed attempt are not lost from the
  ledger (LiteLLM #33273);
* a **cost amplification** calculator that tells you what your "reliable"
  policy is really costing before the invoice arrives.

Everything runs offline with deterministic simulated providers, so the whole
thing is testable without an API key.
"""

from deadlineproxy.amplification import (
    AmplificationCalculator,
    AmplificationResult,
    CostModel,
    FailureModel,
    PolicyDescriptor,
)
from deadlineproxy.breaker import (
    BreakerRegistry,
    CircuitBreaker,
    CircuitOpen,
    CircuitState,
)
from deadlineproxy.deadline import Deadline, DeadlineBudget, DeadlineExceeded
from deadlineproxy.gateway import (
    AttemptRecord,
    Gateway,
    GatewayError,
    GatewayStats,
    StreamResult,
)
from deadlineproxy.provider import (
    Provider,
    ProviderCapabilities,
    ProviderRegistry,
    Request,
    Response,
    SimulatedProvider,
)
from deadlineproxy.retry import (
    EQUAL_JITTER,
    FULL_JITTER,
    NO_JITTER,
    RetryExhausted,
    RetryPolicy,
    compute_backoff,
    should_retry,
)

__version__ = "0.1.0"

__all__ = [
    "EQUAL_JITTER",
    "FULL_JITTER",
    "NO_JITTER",
    "AmplificationCalculator",
    "AmplificationResult",
    "AttemptRecord",
    "BreakerRegistry",
    "CircuitBreaker",
    "CircuitOpen",
    "CircuitState",
    "CostModel",
    "Deadline",
    "DeadlineBudget",
    "DeadlineExceeded",
    "FailureModel",
    "Gateway",
    "GatewayError",
    "GatewayStats",
    "PolicyDescriptor",
    "Provider",
    "ProviderCapabilities",
    "ProviderRegistry",
    "Request",
    "Response",
    "RetryExhausted",
    "RetryPolicy",
    "SimulatedProvider",
    "StreamResult",
    "__version__",
    "compute_backoff",
    "should_retry",
]
