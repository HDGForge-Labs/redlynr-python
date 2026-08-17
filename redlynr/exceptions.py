"""
redlynr.exceptions — Exception hierarchy for the Redlynr SDK.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class RedlynrError(Exception):
    """
    API or connection error.

    Raised when the Redlynr service returns an HTTP error, is unreachable,
    or returns a response that cannot be parsed. This is an infrastructure
    error, not a guardrail decision.
    """


class RedlynrBlocked(Exception):
    """
    Raised when Redlynr returns a stop decision.

    The wrapped tool function is NOT called. Catch this to implement your
    agent's stop handling (graceful termination, user notification, etc.).

    Attributes:
        reason: The stop reason from Redlynr. One of:
            "retries_exhausted"     — retry budget consumed
            "step_cap_exceeded"     — chain step limit hit
            "cost_budget_exceeded"  — tenant period cost limit hit
            "chain_cost_exceeded"   — per-chain cost ceiling hit
            "chain_depth_exceeded"  — nesting depth limit exceeded
            "repetition_detected"   — agent looping on same inputs
            "tool_fixation_detected"— agent hammering one tool
            "volume_pressure_blocked" — request rate too high
            "agent_paused"          — agent manually paused by operator
            "lock_contention"       — transient; safe to retry once after
                                      a short delay (see response.wait_seconds)
        decision: Always "stop".
        response: The full Redlynr /run response dict. Contains "checks"
            with per-dimension state. Note: checks is empty ({}) for
            lock_contention stops.
    """

    def __init__(
        self,
        reason: Optional[str],
        decision: str,
        response: Dict[str, Any],
    ) -> None:
        self.reason = reason
        self.decision = decision
        self.response = response
        super().__init__(
            f"Redlynr blocked execution: {reason}"
        )

    @property
    def wait_seconds(self) -> Optional[int]:
        """Suggested wait before retry (only set for lock_contention)."""
        return self.response.get("wait_seconds")


class RedlynrSlowDown(Exception):
    """
    Raised when Redlynr returns a slow_down decision and raise_on_slow_down=True.

    By default, the SDK handles slow_down transparently: it waits
    wait_seconds (if provided) and then proceeds to execute the tool.
    Set raise_on_slow_down=True on RedlynrClient to catch this instead
    and decide for yourself.

    Attributes:
        reason: The slow_down reason from Redlynr. One of:
            "step_cap_warning"       — approaching step cap (soft warning)
            "cost_budget_warning"    — approaching period cost limit
            "volume_pressure_warning"— request rate elevated
        wait_seconds: Suggested wait in seconds before retrying. May be
            None for step_cap_warning and cost_budget_warning (no wait
            implied — just a signal to wind down gracefully).
        response: The full Redlynr /run response dict.
    """

    def __init__(
        self,
        reason: Optional[str],
        wait_seconds: Optional[int],
        response: Dict[str, Any],
    ) -> None:
        self.reason = reason
        self.wait_seconds = wait_seconds
        self.response = response
        super().__init__(
            f"Redlynr slow_down: {reason}"
            + (f" (wait {wait_seconds}s)" if wait_seconds else "")
        )
