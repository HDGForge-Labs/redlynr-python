"""
redlynr.client — RedlynrClient and guard() implementation.

Detection isn't enforcement. Redlynr identifies when an agent should stop;
this SDK makes sure it does.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

from .exceptions import RedlynrBlocked, RedlynrError, RedlynrSlowDown
from .state import ChainState


class RedlynrClient:
    """
    Client for the Redlynr agent guardrail API.

    One client instance owns state for exactly one agent chain. This is the
    correct model for the common case: one client per agent execution, bound
    to a single chain_id. If you need to manage multiple simultaneous chains,
    instantiate multiple clients.

    Basic usage::

        client = RedlynrClient(
            base_url="https://redlynr.com",
            tenant_id="my-agent-prod",
            owner_token="<your-owner-token>",
            chain_id="chain-abc-001",
        )

        @client.guard
        def call_llm(prompt: str) -> str:
            ...

        # Or with explicit step cost:
        result = client.guard(call_search, {"query": "..."}, step_cost=0.02)

    Registration (one-time, before you have an instance)::

        token = RedlynrClient.register(
            base_url="https://redlynr.com",
            tenant_id="my-agent-prod",
        )
        # Store token securely — it is never shown again.
    """

    def __init__(
        self,
        base_url: str,
        tenant_id: str,
        owner_token: str,
        chain_id: str,
        agent_id: str = "orchestrator",
        depth: int = 0,
        chain_type: Optional[str] = None,
        cost_fn: Optional[Callable[[str, Dict], float]] = None,
        raise_on_slow_down: bool = False,
        trial_token: Optional[str] = None,
    ) -> None:
        """
        Args:
            base_url: Redlynr service URL, e.g. "https://redlynr.com".
            tenant_id: Your registered tenant namespace.
            owner_token: The token returned by /register. Required for
                /policy, /reset, /audit, and /pause calls.
            chain_id: Unique identifier for this agent chain. The client
                owns all chain state for this chain_id.
            agent_id: Identifier for this agent within the tenant. Defaults
                to "orchestrator". Sub-agents should use distinct identifiers.
            depth: Nesting level of this agent (0 = top-level orchestrator,
                1 = first-generation sub-agent, etc.). Redlynr checks this
                directly against max_depth in policy.
            chain_type: Declare the chain type ("interactive", "batch",
                "scheduled"). None lets Redlynr auto-detect from call
                patterns. An explicit declaration pins the type for the
                chain's lifetime and prevents auto-detection.
            cost_fn: Optional hook for automatic step cost resolution.
                Signature: cost_fn(tool_name: str, args: dict) -> float.
                Used when step_cost is not passed explicitly to guard().
                Per-call step_cost always takes precedence over cost_fn.
            raise_on_slow_down: If True, slow_down decisions raise
                RedlynrSlowDown instead of being handled transparently
                (wait + proceed). Defaults to False.
            trial_token: PoW trial token for free-tier access. Obtained via
                GET /trial/challenge → solve SHA-256 → POST /trial/claim.
                Sent as X-Trial-Token header. 30 calls granted per token.
        """
        self.base_url = base_url.rstrip("/")
        self.tenant_id = tenant_id
        self.owner_token = owner_token
        self.chain_id = chain_id
        self.agent_id = agent_id
        self.depth = depth
        self.chain_type = chain_type
        self.cost_fn = cost_fn
        self.raise_on_slow_down = raise_on_slow_down
        self.trial_token = trial_token

        # In-process chain state tracker — no external dependencies.
        self._state = ChainState()

    # ------------------------------------------------------------------
    # Class method: register (called before you have an instance)
    # ------------------------------------------------------------------

    @classmethod
    def register(
        cls,
        base_url: str,
        tenant_id: str,
        trial_token: Optional[str] = None,
    ) -> str:
        """
        Register a tenant namespace and return the owner_token.

        This is a one-time operation. The returned token is never shown again
        after this call — store it securely (environment variable, secrets
        manager, etc.).

        If the tenant_id is already registered by someone else, raises
        RedlynrError. Registration does not confirm or deny ownership of an
        existing namespace.

        Args:
            base_url: Redlynr service URL.
            tenant_id: The namespace to claim.
            trial_token: Optional PoW trial token for free-tier access.

        Returns:
            The owner_token string.

        Raises:
            RedlynrError: If tenant_id is already registered, or on any
                API/connection error.
        """
        url = base_url.rstrip("/") + "/register"
        resp = _post(url, {"tenant_id": tenant_id}, trial_token=trial_token)

        status = resp.get("status")
        if status == "already_registered":
            raise RedlynrError(
                f"tenant_id '{tenant_id}' is already registered. "
                "If you own this namespace, retrieve your token from secure storage."
            )
        if status != "registered":
            raise RedlynrError(f"Unexpected registration response: {resp}")

        token = resp.get("owner_token")
        if not token:
            raise RedlynrError(f"Registration succeeded but no owner_token returned: {resp}")

        return token

    # ------------------------------------------------------------------
    # guard() — the main enforcement primitive
    # ------------------------------------------------------------------

    def guard(
        self,
        fn: Callable,
        args: Optional[Dict] = None,
        *,
        tool_name: Optional[str] = None,
        step_cost: Optional[float] = None,
    ) -> Any:
        """
        Wrap a tool call with full Redlynr enforcement.

        Calls /run before executing fn. Enforces the decision:
          - proceed  → execute fn, return result
          - slow_down → wait (if wait_seconds provided), execute fn, return
                        result. If raise_on_slow_down=True was set on the
                        client, raises RedlynrSlowDown instead.
          - stop     → raises RedlynrBlocked. fn is NOT called.

        Chain state (step count, retry detection, tool fixation, novelty) is
        tracked automatically in-process with no external dependencies.

        Args:
            fn: The tool function to call if Redlynr says proceed/slow_down.
            args: Keyword arguments to pass to fn. Also used for novelty
                hashing. Defaults to {}.
            tool_name: Override the tool name sent to Redlynr. Defaults to
                fn.__name__.
            step_cost: Estimated cost of this tool call in USD. Overrides
                cost_fn if both are set. If neither is set, defaults to 0.0.

        Returns:
            The return value of fn(**(args or {})).

        Raises:
            RedlynrBlocked: When Redlynr returns stop.
            RedlynrSlowDown: When Redlynr returns slow_down and
                raise_on_slow_down=True.
            RedlynrError: On API or connection errors.
        """
        if args is None:
            args = {}

        resolved_tool_name = tool_name or getattr(fn, "__name__", "unknown_tool")

        # Resolve step cost: per-call arg > cost_fn > 0.0
        if step_cost is None:
            if self.cost_fn is not None:
                try:
                    step_cost = float(self.cost_fn(resolved_tool_name, args))
                except Exception:
                    step_cost = 0.0
            else:
                step_cost = 0.0

        # Compute chain signals from in-process state
        is_retry, is_novel_target, tool_fixation_signal = self._state.compute_signals(
            resolved_tool_name, args
        )

        # Build /run payload
        payload: Dict[str, Any] = {
            "tenant_id": self.tenant_id,
            "agent_id": self.agent_id,
            "chain_id": self.chain_id,
            "depth": self.depth,
            "step_cost": step_cost,
            "is_retry": is_retry,
            "is_novel_target": is_novel_target,
            "tool_fixation_signal": tool_fixation_signal,
        }
        if self.chain_type is not None:
            payload["chain_type"] = self.chain_type

        # Call /run
        response = self._run(payload)

        decision = response.get("decision")
        reason = response.get("reason")
        wait_seconds = response.get("wait_seconds")

        if decision == "stop":
            # Record the failed attempt before raising so subsequent
            # is_retry signals are computed correctly.
            self._state.record_attempt(
                resolved_tool_name, args, success=False
            )
            raise RedlynrBlocked(
                reason=reason,
                decision=decision,
                response=response,
            )

        if decision == "slow_down":
            if self.raise_on_slow_down:
                self._state.record_attempt(
                    resolved_tool_name, args, success=False
                )
                raise RedlynrSlowDown(
                    reason=reason,
                    wait_seconds=wait_seconds,
                    response=response,
                )
            # Default: handle transparently — wait if asked, then proceed.
            if wait_seconds and wait_seconds > 0:
                time.sleep(wait_seconds)

        # proceed (or transparent slow_down): execute the tool
        result = fn(**(args or {}))
        self._state.record_attempt(
            resolved_tool_name, args, success=True
        )
        return result

    # ------------------------------------------------------------------
    # run_raw() — escape hatch for direct /run calls
    # ------------------------------------------------------------------

    def run_raw(self, **kwargs) -> Dict:
        """
        Send a raw /run request with full caller control over all fields.

        Does NOT update in-process chain state. Use this when you need to
        send a custom payload that guard() cannot express, or for debugging.

        All /run fields must be supplied by the caller. Required: tenant_id,
        agent_id, chain_id, depth. The client's tenant_id and chain_id are
        injected automatically but can be overridden via kwargs.

        Returns:
            The raw Redlynr /run response dict.
        """
        payload: Dict[str, Any] = {
            "tenant_id": self.tenant_id,
            "agent_id": self.agent_id,
            "chain_id": self.chain_id,
            "depth": self.depth,
        }
        payload.update(kwargs)
        return self._run(payload)

    # ------------------------------------------------------------------
    # Management endpoints
    # ------------------------------------------------------------------

    def set_policy(
        self,
        policy: Optional[Dict] = None,
        template: Optional[str] = None,
    ) -> Dict:
        """
        Set or update the tenant's guardrail policy.

        Args:
            policy: Partial or full policy overrides. Deep-merged onto the
                existing stored policy (or onto the template if supplied).
            template: Named policy preset to apply before merging any
                overrides. Replaces the stored policy entirely.
                Available templates: "interactive_agent", "batch_pipeline",
                "autonomous_researcher". See GET /templates for full specs.

        Returns:
            The API response dict including the merged policy that was saved.
        """
        payload: Dict[str, Any] = {
            "tenant_id": self.tenant_id,
            "owner_token": self.owner_token,
        }
        if policy is not None:
            payload["policy"] = policy
        if template is not None:
            payload["template"] = template
        return _post(
            self.base_url + "/policy", payload, trial_token=self.trial_token
        )

    def reset(self) -> Dict:
        """
        Clear this chain's step counter and chain type registration in Redlynr.

        Also resets the in-process chain state tracker so the client is fully
        synchronized with the server after reset.

        Returns:
            The API response dict.
        """
        payload = {
            "tenant_id": self.tenant_id,
            "owner_token": self.owner_token,
            "chain_id": self.chain_id,
        }
        response = _post(
            self.base_url + "/reset", payload, trial_token=self.trial_token
        )
        # Also reset in-process state so the client stays synchronized.
        self._state = ChainState()
        return response

    def audit(self) -> Dict:
        """
        Retrieve the last 100 /run decisions for this tenant.

        Returns:
            Dict with "count" and "entries" (list of audit records, newest first).
        """
        url = (
            f"{self.base_url}/audit/{self.tenant_id}"
            f"?owner_token={self.owner_token}"
        )
        return _get(url, trial_token=self.trial_token)

    def audit_analyze(self, min_chains: int = 3) -> Dict:
        """
        Analyze the tenant's audit log and get advisory threshold suggestions.

        Results are advisory only — Redlynr never auto-applies changes.
        Review suggestions and call set_policy() if desired.

        Args:
            min_chains: Minimum distinct chains required before suggestions
                are produced. Defaults to 3.

        Returns:
            Dict with chain classifications and threshold suggestions.
        """
        payload = {
            "tenant_id": self.tenant_id,
            "owner_token": self.owner_token,
            "min_chains": min_chains,
        }
        return _post(
            self.base_url + "/audit/analyze",
            payload,
            trial_token=self.trial_token,
        )

    def pause(
        self,
        agent_id: Optional[str] = None,
        paused: bool = True,
    ) -> Dict:
        """
        Pause or unpause an agent.

        Args:
            agent_id: The agent to pause. Defaults to this client's agent_id.
            paused: True to pause, False to unpause.

        Returns:
            The API response dict.
        """
        payload = {
            "tenant_id": self.tenant_id,
            "owner_token": self.owner_token,
            "agent_id": agent_id or self.agent_id,
            "paused": paused,
        }
        return _post(
            self.base_url + "/pause", payload, trial_token=self.trial_token
        )

    def health(self) -> Dict:
        """
        Check the Redlynr service health.

        Returns:
            Dict with status, core_loaded, and trial endpoint info.
        """
        return _get(self.base_url + "/health", trial_token=self.trial_token)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run(self, payload: Dict) -> Dict:
        return _post(
            self.base_url + "/run",
            payload,
            trial_token=self.trial_token,
        )


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _post(
    url: str,
    payload: Dict,
    trial_token: Optional[str] = None,
    timeout: int = 10,
) -> Dict:
    """POST JSON payload, return parsed response dict."""
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if trial_token:
        headers["X-Trial-Token"] = trial_token

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8")
        except Exception:
            pass
        raise RedlynrError(
            f"HTTP {e.code} from {url}: {body}"
        ) from e
    except urllib.error.URLError as e:
        raise RedlynrError(f"Connection error to {url}: {e.reason}") from e
    except json.JSONDecodeError as e:
        raise RedlynrError(f"Invalid JSON response from {url}: {e}") from e


def _get(
    url: str,
    trial_token: Optional[str] = None,
    timeout: int = 10,
) -> Dict:
    """GET request, return parsed response dict."""
    headers = {}
    if trial_token:
        headers["X-Trial-Token"] = trial_token

    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8")
        except Exception:
            pass
        raise RedlynrError(f"HTTP {e.code} from {url}: {body}") from e
    except urllib.error.URLError as e:
        raise RedlynrError(f"Connection error to {url}: {e.reason}") from e
    except json.JSONDecodeError as e:
        raise RedlynrError(f"Invalid JSON response from {url}: {e}") from e
