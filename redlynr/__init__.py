"""
redlynr — Official Python SDK for the Redlynr agent guardrail service.

Detection isn't enforcement. Redlynr identifies when an agent should stop;
this SDK makes sure it does.

Quick start::

    from redlynr import RedlynrClient, RedlynrBlocked

    # One-time: register your tenant namespace
    owner_token = RedlynrClient.register(
        base_url="https://redlynr.com",
        tenant_id="my-agent-prod",
    )
    # Store owner_token securely — it is never shown again.

    # Per agent run: one client per chain
    client = RedlynrClient(
        base_url="https://redlynr.com",
        tenant_id="my-agent-prod",
        owner_token=owner_token,
        chain_id="chain-abc-001",
    )

    try:
        result = client.guard(my_tool_fn, {"arg": "value"}, step_cost=0.02)
    except RedlynrBlocked as e:
        print(f"Agent stopped: {e.reason}")
"""

from .client import RedlynrClient
from .exceptions import RedlynrBlocked, RedlynrError, RedlynrSlowDown

__all__ = [
    "RedlynrClient",
    "RedlynrBlocked",
    "RedlynrSlowDown",
    "RedlynrError",
]

__version__ = "0.1.0"
