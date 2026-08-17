"""
redlynr.state — In-process chain state tracking.

Computes is_retry, is_novel_target, and tool_fixation_signal for each
guard() call with no external dependencies. All state is per-client-instance
and lives only in memory for the duration of the agent chain.

Design decisions:
  - is_retry: True only when the immediately preceding call to the same tool
    actually failed AND this is a genuine retry of that failure. Never inferred
    from input-matching alone. Matches Redlynr server-side contract exactly.
  - is_novel_target: False when (tool_name, inputs_hash) matches any call in
    the last 5 calls on this chain. True otherwise (including the first call).
  - tool_fixation_signal: consecutive trailing calls to the same tool name,
    reset to 0 whenever a different tool name is called.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from typing import Any, Dict, Optional, Tuple


# Number of recent calls to check for novelty comparison
NOVELTY_WINDOW = 5


def _hash_inputs(args: Dict[str, Any]) -> str:
    """
    Stable hash of a dict of arguments for novelty comparison.

    Uses SHA-256 of the JSON-serialized args with sorted keys so that
    dict ordering differences don't produce false novelty signals.
    Falls back to repr() if JSON serialization fails (e.g. non-serializable
    objects) — still stable within a single process.
    """
    try:
        canonical = json.dumps(args, sort_keys=True, default=repr)
    except Exception:
        canonical = repr(sorted(args.items()))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class _CallRecord:
    """One historical call in the chain."""

    __slots__ = ("tool_name", "inputs_hash", "success")

    def __init__(self, tool_name: str, inputs_hash: str, success: bool) -> None:
        self.tool_name = tool_name
        self.inputs_hash = inputs_hash
        self.success = success


class ChainState:
    """
    Tracks per-chain state for signal computation.

    Instantiated once per RedlynrClient. All state is in-memory and lives
    for the lifetime of the client instance.

    Thread safety: not thread-safe. If you're running concurrent tool calls
    on the same chain from multiple threads, instantiate separate clients
    per thread (each with its own chain_id).
    """

    def __init__(self) -> None:
        # Recent call history for novelty and retry detection.
        # Bounded to NOVELTY_WINDOW entries — only the last N calls
        # are needed for is_novel_target; the full history is not retained.
        self._history: deque[_CallRecord] = deque(maxlen=NOVELTY_WINDOW)

        # The last failed call (tool_name, inputs_hash), or None.
        # Used for is_retry computation: a retry is the same tool as the
        # last failed call.
        self._last_failed: Optional[Tuple[str, str]] = None

        # Consecutive same-tool trailing call count for tool_fixation_signal.
        self._fixation_tool: Optional[str] = None
        self._fixation_count: int = 0

        # Total step count (informational — Redlynr tracks authoritatively).
        self.step_count: int = 0

    def compute_signals(
        self,
        tool_name: str,
        args: Dict[str, Any],
    ) -> Tuple[bool, bool, int]:
        """
        Compute the three signals for the upcoming /run call.

        Called BEFORE the attempt is recorded. The attempt is recorded
        separately via record_attempt() after the Redlynr decision is known.

        Args:
            tool_name: The name of the tool about to be called.
            args: The arguments that will be passed to the tool.

        Returns:
            (is_retry, is_novel_target, tool_fixation_signal)

            is_retry: True if the last failed call was to this same tool
                (indicating this is a genuine retry of a failure).

            is_novel_target: False if (tool_name, inputs_hash) appears in
                any of the last NOVELTY_WINDOW calls. True otherwise.

            tool_fixation_signal: Current count of consecutive trailing calls
                to this same tool name (before this call is added).
        """
        inputs_hash = _hash_inputs(args)

        # is_retry: True only when the last failed call was the same tool.
        is_retry = (
            self._last_failed is not None
            and self._last_failed[0] == tool_name
        )

        # is_novel_target: False if this (tool, inputs) pair appears in
        # the last NOVELTY_WINDOW calls.
        recent_pairs = {(r.tool_name, r.inputs_hash) for r in self._history}
        is_novel_target = (tool_name, inputs_hash) not in recent_pairs

        # tool_fixation_signal: consecutive trailing same-tool count.
        # We report the current count BEFORE this call is added.
        # If this call is the same tool as the current fixation streak,
        # the signal is the existing streak count. If it's a different
        # tool, the streak resets to 0.
        if self._fixation_tool == tool_name:
            tool_fixation_signal = self._fixation_count
        else:
            tool_fixation_signal = 0

        return is_retry, is_novel_target, tool_fixation_signal

    def record_attempt(
        self,
        tool_name: str,
        args: Dict[str, Any],
        success: bool,
    ) -> None:
        """
        Record the outcome of a tool call attempt.

        Called AFTER the Redlynr decision and tool execution (or block).
        Updates history, retry state, and fixation counter.

        Args:
            tool_name: The tool that was attempted.
            args: The arguments that were passed (or would have been passed).
            success: True if the tool executed (proceed/transparent slow_down).
                False if Redlynr blocked it or raise_on_slow_down fired.
        """
        inputs_hash = _hash_inputs(args)
        record = _CallRecord(tool_name, inputs_hash, success)
        self._history.append(record)

        # Update last_failed for next call's is_retry computation.
        if success:
            # Successful execution clears the failed state — a retry that
            # succeeds is no longer a pending failure.
            self._last_failed = None
        else:
            # Failed/blocked — record as the pending failure so the next
            # call to the same tool is correctly flagged as is_retry=True.
            self._last_failed = (tool_name, inputs_hash)

        # Update fixation counter.
        if self._fixation_tool == tool_name:
            self._fixation_count += 1
        else:
            self._fixation_tool = tool_name
            self._fixation_count = 1

        self.step_count += 1
