# redlynr-python

Python SDK for [Redlynr](https://redlynr.com) — the agent guardrail decision engine.

Detection isn't enforcement. Redlynr identifies when an agent should stop. This SDK makes sure it does.

## Installation

```bash
pip install redlynr
```

Zero external dependencies for trial token usage. For x402 payment support, install:

```bash
pip install redlynr requests "x402[evm]" eth_account
```

---

## Quickstart

### 1. Register your tenant (one-time)

```python
from redlynr import RedlynrClient

owner_token = RedlynrClient.register(
    base_url="https://redlynr.com",
    tenant_id="my-agent-prod",
)
# Store owner_token securely — it is never shown again.
```

### 2. Create a client

```python
client = RedlynrClient(
    base_url="https://redlynr.com",
    tenant_id="my-agent-prod",
    owner_token=owner_token,
    chain_id="chain-001",
)
```

### 3. Guard every tool call

```python
from redlynr import RedlynrBlocked, RedlynrSlowDown

try:
    result = client.guard(my_tool_fn, {"query": "..."}, step_cost=0.01)
except RedlynrBlocked as e:
    print(f"Stopped: {e.reason}")   # tool was NOT called
except RedlynrSlowDown as e:
    print(f"Slow down: wait {e.wait_seconds}s")  # only if raise_on_slow_down=True
```

When Redlynr says stop, `guard()` raises `RedlynrBlocked` and the tool function is never called. When Redlynr says proceed or slow_down (default), `guard()` calls the tool and returns its result.

---

## Authentication

Redlynr supports two authentication modes for `/run` calls.

### Trial token (free tier)

Obtain a PoW trial token for 30 free calls:

```python
import hashlib, json, urllib.request

def get_trial_token(base_url="https://redlynr.com"):
    with urllib.request.urlopen(f"{base_url}/trial/challenge") as r:
        ch = json.loads(r.read())
    nonce, difficulty = ch["nonce"], ch["difficulty"]
    prefix = "0" * difficulty
    counter = 0
    while not hashlib.sha256(f"{nonce}{counter}".encode()).hexdigest().startswith(prefix):
        counter += 1
    data = json.dumps({"nonce": nonce, "solution": str(counter)}).encode()
    req = urllib.request.Request(
        f"{base_url}/trial/claim", data=data,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())["trial_token"]

trial_token = get_trial_token()

client = RedlynrClient(
    base_url="https://redlynr.com",
    tenant_id="my-tenant",
    owner_token=owner_token,
    chain_id="chain-001",
    trial_token=trial_token,
)
```

### x402 payment (paid tier)

For production use, pass an x402-wrapped `requests.Session`. The session handles USDC payment on Base mainnet automatically on every `/run` call.

```python
import requests as req_lib
from eth_account import Account
from x402.client import x402ClientSync
from x402.mechanisms.evm.exact import ExactEvmScheme
from x402.http.clients.requests import wrapRequestsWithPayment

# Build x402 session once
account = Account.from_key(os.environ["BUYER_PRIVATE_KEY"])
x402_client = x402ClientSync()
x402_client.register("eip155:8453", ExactEvmScheme(signer=account))
session = wrapRequestsWithPayment(req_lib.Session(), x402_client)

client = RedlynrClient(
    base_url="https://redlynr.com",
    tenant_id="my-tenant",
    owner_token=owner_token,
    chain_id="chain-001",
    session=session,    # x402 payments handled automatically
)
```

When `session` is provided it is used for all `/run` calls. All other endpoints (`/register`, `/policy`, `/audit`, `/reset`, `/pause`) use plain urllib and are unpriced.

---

## RedlynrClient reference

```python
RedlynrClient(
    base_url,           # str — Redlynr service URL
    tenant_id,          # str — your registered tenant namespace
    owner_token,        # str — token from /register
    chain_id,           # str — unique ID for this agent chain
    agent_id="orchestrator",        # str — identifier for this agent
    depth=0,                        # int — nesting level (0=top, 1=sub-agent, ...)
    chain_type=None,                # str|None — "interactive", "batch", "scheduled", or None (auto)
    cost_fn=None,                   # callable(tool_name, args) -> float — auto cost resolution
    raise_on_slow_down=False,       # bool — raise RedlynrSlowDown instead of transparent wait
    trial_token=None,               # str|None — PoW trial token for free tier
    session=None,                   # requests.Session|None — x402-wrapped session for paid tier
)
```

**One client per chain.** Each `RedlynrClient` owns state for exactly one `chain_id`. For multi-agent systems, each agent instantiates its own client with the same `chain_id` and `tenant_id` but a distinct `agent_id` and appropriate `depth`.

### chain_type — declare explicitly

If `chain_type` is not declared, Redlynr auto-detects the chain type from call patterns and may promote to `"batch"` after 5 consecutive novel-target calls. Batch chains have a default step limit of 100. Always declare `chain_type="interactive"` explicitly when you want tight step caps to fire:

```python
client = RedlynrClient(
    ...
    chain_type="interactive",   # pins limits for the chain's lifetime
)
```

---

## guard()

```python
result = client.guard(fn, args, *, tool_name=None, step_cost=None)
```

The main enforcement primitive. Calls `/run` before executing `fn`. Enforces the decision:

| Decision | Behavior |
|---|---|
| `proceed` | `fn(**args)` is called, result returned |
| `slow_down` | Waits `wait_seconds` if set, then calls `fn`. If `raise_on_slow_down=True`, raises `RedlynrSlowDown` instead |
| `stop` | Raises `RedlynrBlocked`. `fn` is **never called** |

Chain state (step count, retry detection, tool fixation, novelty) is tracked automatically in-process with no external dependencies.

**args are passed to fn as keyword arguments:**

```python
def search(query: str, limit: int = 10):
    ...

result = client.guard(search, {"query": "...", "limit": 5}, step_cost=0.01)
# SDK calls: search(query="...", limit=5)
```

---

## Policy

Set guardrail limits for your tenant:

```python
client.set_policy(policy={
    "steps_retries": {
        "max_steps_per_chain": 20,
        "max_retries": 2,
        "warn_at_steps_pct": 75,
    },
    "cost": {
        "max_spend_per_period": 1.00,
        "period_seconds": 86400,
    },
})
```

Or apply a named template:

```python
client.set_policy(template="interactive_agent")
# Available: "interactive_agent", "batch_pipeline", "autonomous_researcher"
```

### Default policy limits

| Limit | Interactive | Batch |
|---|---|---|
| `max_steps_per_chain` | 20 | 100 |
| `max_retries` | 2 | 2 |
| `max_depth` | 5 | 5 |
| `max_spend_per_chain` | $0.30 | $2.00 |
| `warn_at_steps_pct` | 75% | 75% |

---

## Multi-agent usage

```python
# Orchestrator (depth=0)
orchestrator = RedlynrClient(
    base_url="https://redlynr.com",
    tenant_id="my-tenant",
    owner_token=owner_token,
    chain_id="chain-001",
    agent_id="orchestrator",
    depth=0,
    chain_type="interactive",
    session=session,
)

# Sub-agent (depth=1) — same chain_id and tenant_id
researcher = RedlynrClient(
    base_url="https://redlynr.com",
    tenant_id="my-tenant",
    owner_token=owner_token,
    chain_id="chain-001",
    agent_id="researcher",
    depth=1,
    chain_type="interactive",
    session=session,
)
```

Both agents call `guard()` before every tool execution. Step counters are shared server-side by `chain_id`. The chain stops when either agent exceeds the limit.

---

## Exception reference

```python
from redlynr import RedlynrBlocked, RedlynrSlowDown, RedlynrError
```

| Exception | When raised | Key attributes |
|---|---|---|
| `RedlynrBlocked` | Redlynr returns `stop` | `reason`, `response` |
| `RedlynrSlowDown` | Redlynr returns `slow_down` and `raise_on_slow_down=True` | `reason`, `wait_seconds`, `response` |
| `RedlynrError` | API or connection error | message |

### Stop reasons

`retries_exhausted`, `step_cap_exceeded`, `cost_budget_exceeded`, `chain_cost_exceeded`,
`chain_depth_exceeded`, `repetition_detected`, `tool_fixation_detected`,
`volume_pressure_blocked`, `agent_paused`, `lock_contention`

---

## Other methods

```python
client.reset()                    # clear this chain's step counter server-side
client.audit()                    # last 100 /run decisions for this tenant
client.audit_analyze(min_chains=3) # advisory threshold suggestions
client.pause(agent_id=None, paused=True)  # pause or unpause an agent
client.health()                   # service health check
client.run_raw(**kwargs)          # raw /run call, bypasses guard() state tracking
```

---

## Pricing

- `/run` calls: **$0.001 USDC** per call (Base mainnet)
- All other endpoints (`/register`, `/policy`, `/audit`, `/reset`, `/pause`, `/health`): free
- Free trial: 30 `/run` calls per PoW token

---

## Links

- Service: [redlynr.com](https://redlynr.com)
- PyPI: [pypi.org/project/redlynr](https://pypi.org/project/redlynr/)
- GitHub: [github.com/HDGForge-Labs/redlynr-python](https://github.com/HDGForge-Labs/redlynr-python)
- License: MIT
