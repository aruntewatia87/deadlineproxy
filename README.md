# deadlineproxy

> A deadline-aware, provider-agnostic LLM gateway — where retries, fallbacks and timeouts finally behave the way you think they do.

[![CI](https://github.com/aruntewatia87/deadlineproxy/actions/workflows/ci.yml/badge.svg)](https://github.com/aruntewatia87/deadlineproxy/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

The reliability semantics of an LLM gateway are the hard part, and they are
where the open bugs live. LiteLLM alone has **57 open issues** on
retry/timeout/resilience, including a connect timeout that is ignored for
non-OpenAI providers, an async timeout decorator that drops the per-call
timeout, an aiohttp transport that hangs for **137 seconds** on a stopped
upstream, and retry counts with **no operator ceiling** — a client can ask
for a thousand retries, which is an amplification attack rather than a
configuration mistake.

`deadlineproxy` gets these right, and proves it with deterministic simulated
providers that reproduce the exact upstream misbehaviour.

---

## ✨ Highlights

- **One deadline for the whole request.** Every attempt shares it, so a request cannot outlive its timeout no matter how many retries happen.
- **The gateway enforces the deadline itself** — it does not trust the upstream to do it, because some upstreams do not.
- **Bounded retries** with full-jitter backoff, and an operator ceiling a client cannot raise.
- **Circuit breakers** with a sliding failure window and half-open probing.
- **Fallback that keeps the ledger honest** — the tokens burned by a failed attempt are still billed to the request.
- **Cost amplification analysis** — what your "reliable" policy is really costing, before the invoice arrives.
- **Streaming-aware** — a stalled stream is cut off at the deadline, not left hanging.

---

## 🚀 Quickstart

```bash
git clone https://github.com/aruntewatia87/deadlineproxy.git
cd deadlineproxy
uv sync --all-extras --dev
uv run pytest -q
```

Or standalone:

```bash
uv pip install deadlineproxy
```

---

## 🔧 API Tour

### Send a request with a deadline

```python
from deadlineproxy import Gateway, ProviderRegistry, Request, SimulatedProvider

gateway = Gateway(
    ProviderRegistry(
        [
            SimulatedProvider("fast", latency_ms=40.0),
            SimulatedProvider("backup", latency_ms=90.0),
        ]
    )
)

response = gateway.send(Request(prompt="Summarise the incident report"), timeout=0.5)
print(response.provider, response.cost_usd)
```

### A provider that ignores its timeout

```python
stubborn = SimulatedProvider("stubborn", latency_ms=4000.0, ignore_deadline=True)
gateway = Gateway(ProviderRegistry([stubborn]))

try:
    gateway.send(Request(prompt="hello"), deadline=Deadline.from_timeout(0.05))
except GatewayError as exc:
    for record in exc.history:
        print(record.provider, record.timed_out, record.latency_ms)
```

The gateway rejects the attempt on the basis of the provider's **reported**
latency, which is why it works even when the upstream silently ignores the
timeout.

### Bound the client's retry count

```python
from deadlineproxy import RetryPolicy

policy = RetryPolicy(max_attempts=3)
policy.clamp_attempts(1000)  # -> 3
```

### Circuit breakers

```python
from deadlineproxy import BreakerRegistry, CircuitState

breakers = BreakerRegistry(failure_threshold=5, recovery_timeout=30.0)
breakers.get("openai").record_failure()
breakers.get("openai").state  # -> CircuitState.CLOSED / OPEN / HALF_OPEN
```

### Split a request budget

```python
from deadlineproxy import DeadlineBudget

budget = DeadlineBudget(total=1.0)
budget.reserve("routing", 0.2)
budget.reserve("provider", 0.7)
budget.available  # -> 0.1
```

### Analyse the cost of a routing policy

```python
from deadlineproxy import AmplificationCalculator, CostModel, FailureModel, PolicyDescriptor

policy = PolicyDescriptor(
    fallback_chain=["primary", "secondary"],
    max_attempts_per_provider=5,
    retry_policy=RetryPolicy(max_attempts=5),
    cost_models={"primary": CostModel(), "secondary": CostModel()},
    failure_models={"primary": FailureModel(transient_error_rate=0.20)},
)

result = AmplificationCalculator(policy).analyse(requests=1_000_000, monthly_budget=10_000)
print(result.expected_attempts, result.amplification_factor)
```

---

## 🧠 How it works

### Deadline propagation

```
                     ┌──────────────────────────────┐
  request ─────────►│  Deadline (one end-to-end)   │
                     └───────────────┬──────────────┘
                                     │  every attempt inherits it
             ┌───────────────────────┼───────────────────────┐
             ▼                       ▼                       ▼
      attempt 1 (50ms)        attempt 2 (120ms)       attempt 3 (30ms)
             │                       │                       │
             └───────────────────────┴───────────────────────┘
                                     │
                     total <= the client's timeout, always
```

A gateway that gives each retry a *fresh* timeout can spend
`attempts × timeout` on a request that was supposed to finish in `timeout`.
That is how a 30-second timeout becomes a 137-second hang. Here, the
deadline is created once and every attempt derives from it.

An attempt is only **started** when there is enough time left for it to
finish — starting one that cannot complete is pure waste.

### Deadline enforcement

The provider reports how long the call took. If that exceeds the remaining
budget, the gateway treats the attempt as a timeout, bills the tokens that
were spent, and moves on. This is what makes the gateway robust against
upstreams that ignore their own timeouts:

* [`BerriAI/litellm#47822`](https://github.com/BerriAI/litellm/issues/47822) — "httpx.Timeout connect timeout ignored for non-OpenAI providers"
* [`BerriAI/litellm#47481`](https://github.com/BerriAI/litellm/issues/47481) — "aiohttp transport hangs 40-137 s on a stopped upstream"

### Backoff with full jitter

```
sleep = uniform(0, min(max_delay, base_delay × multiplier^(attempt−1)))
```

Full jitter is the variant that actually breaks up a thundering herd: every
client that failed at the same moment waits a *different* amount. Fixed
backoff re-synchronises the herd on every retry.

### Circuit breaker

```
        failure_threshold failures in window
CLOSED ─────────────────────────────────────────► OPEN
  ▲                                               │
  │ success_threshold successes in half-open       │ recovery_timeout
  │                                               ▼
  └──────────── HALF_OPEN ◄───────────────────────
                  (limited probes)
```

Failures are counted in a **sliding window**, so a provider that was broken
last week is not considered broken now. The half-open state admits only a
limited number of probe calls, so a recovering provider is not immediately
overwhelmed by the traffic that queued up while it was open.

### Cost amplification

With a retryable failure probability `p` and `A` attempts, the expected
number of attempts against one provider is the geometric series
`Σ p^k (k = 0 … A−1)`. The cost model distinguishes a **successful** call
(billed in full) from a **failed** one (billed a fraction, because the prompt
was still sent) — which is why the amplification factor is much closer to one
than the attempt count suggests.

The point of the analysis is the *shape* of the curve: the same policy that
costs 1.10× at a 2% error rate costs **1.71×** at 20%. Nobody can produce
that number today, and it is exactly the number a FinOps review needs.

---

## 📊 Example output

```text
$ uv run python examples/quickstart.py

======================================================================
1. A normal request (500 ms deadline)
======================================================================
provider   : fast
latency    : 40 ms
tokens     : 7 in / 128 out
cost       : $0.001941

======================================================================
2. A provider that IGNORES its timeout (LiteLLM #47822, #47481)
======================================================================
The upstream takes 4000 ms but the client only allowed 50 ms.
The gateway enforces the deadline itself instead of trusting it.
rejected   : every provider and attempt failed
  attempt 1 on stubborn: timed_out=True latency=4000ms
  attempt 2 on stubborn: timed_out=True latency=4000ms

======================================================================
3. The client cannot raise the retry ceiling (LiteLLM #46911)
======================================================================
client asked for 1000 attempts
operator ceiling allowed : 1 attempts

======================================================================
4. Circuit breaker opens after repeated failures
======================================================================
breaker state after 3 failures: OPEN
breaker trips recorded      : 1

======================================================================
5. What is your 'reliable' policy really costing?
======================================================================
  error rate    attempts    cost/req  amplification
---------------------------------------------------
         2%      1.1184    0.004942        1.0983x
         5%      1.2908    0.005547        1.2327x
        20%      2.0897    0.007711        1.7136x

The attempt count grows much faster than the cost, and every extra
attempt also eats into the deadline the client asked for.

======================================================================
6. Gateway statistics
======================================================================
  requests         0
  attempts         0
  successes        0
  failures         0
  fallbacks        0
  timeouts         0
  breaker_trips    0
  input_tokens     0
  output_tokens    0
  cost_usd         0.0
```

---

## 🧪 Testing

```bash
uv sync --all-extras --dev
uv run pytest -q --cov=deadlineproxy --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
```

The suite includes the regression tests for the upstream bugs: the deadline
holding against a provider that ignores its timeout, the client's retry
count being clamped to the operator's ceiling, the sliding breaker window
ageing old failures out, and the amplification curve being monotone in the
error rate.

---

## 📄 License

MIT © 2026 Arun Tewatia
