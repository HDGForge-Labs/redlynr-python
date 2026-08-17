"""
Tests for the redlynr SDK.

All tests are offline — no real network calls. The Redlynr API is mocked
at the HTTP layer (urllib) so tests run without a live service.

Coverage:
  - ChainState: signal computation, retry detection, fixation, novelty
  - RedlynrClient.guard(): all decision branches (proceed, slow_down, stop)
  - RedlynrClient.guard(): transparent slow_down handling vs raise_on_slow_down
  - RedlynrClient.guard(): lock_contention response shape (empty checks)
  - RedlynrClient.guard(): cost_fn resolution, per-call step_cost override
  - RedlynrClient.guard(): tool_name inference from fn.__name__
  - RedlynrClient.register(): success and already_registered paths
  - RedlynrClient.reset(): clears in-process state
  - RedlynrBlocked: carries reason, decision, response, wait_seconds
  - RedlynrSlowDown: carries reason, wait_seconds, response
  - RedlynrError: HTTP errors, connection errors
  - agent_paused: minimal checks shape (not full checks block)
  - deprecation_warning field on proceed response (ignored, not raised)
"""

import json
import unittest
from io import BytesIO
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

from redlynr import RedlynrClient, RedlynrBlocked, RedlynrError, RedlynrSlowDown
from redlynr.state import ChainState, _hash_inputs, NOVELTY_WINDOW


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(data: Dict) -> MagicMock:
    """Build a mock urllib response for a given dict payload."""
    raw = json.dumps(data).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = raw
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _make_client(
    raise_on_slow_down: bool = False,
    cost_fn=None,
    trial_token: Optional[str] = None,
    chain_type: Optional[str] = None,
) -> RedlynrClient:
    return RedlynrClient(
        base_url="https://redlynr.com",
        tenant_id="test-tenant",
        owner_token="test-token",
        chain_id="chain-001",
        agent_id="orchestrator",
        depth=0,
        chain_type=chain_type,
        cost_fn=cost_fn,
        raise_on_slow_down=raise_on_slow_down,
        trial_token=trial_token,
    )


def _proceed_response(**extra) -> Dict:
    resp = {
        "decision": "proceed",
        "reason": None,
        "wait_seconds": None,
        "checks": {
            "paused": False,
            "chain_type": {"declared": "interactive", "limits_applied": "default", "source": "default"},
            "cost": {"spent": 0.0, "limit": 50.0, "step_cost_recorded": 0.02},
            "chain_depth": {"reported_depth": 0, "limit": 5},
            "steps_retries": {"steps": 0, "retries": 0, "step_limit": 20, "retry_limit": 2, "retries_remaining": 2},
            "all_dimensions": {},
            "volume": {"pct_used": 0.0, "agent_pct_used": 0.0, "blocking": False},
        },
    }
    resp.update(extra)
    return resp


def _stop_response(reason: str, wait_seconds=None, checks=None) -> Dict:
    return {
        "decision": "stop",
        "reason": reason,
        "wait_seconds": wait_seconds,
        "checks": checks if checks is not None else {"paused": False},
    }


def _slow_down_response(reason: str, wait_seconds=None) -> Dict:
    return {
        "decision": "slow_down",
        "reason": reason,
        "wait_seconds": wait_seconds,
        "checks": {
            "paused": False,
            "cost": {"spent": 0.0, "limit": 50.0, "step_cost_recorded": 0.0},
        },
    }


def my_tool(value: str = "x") -> str:
    return f"result:{value}"


def other_tool(value: str = "y") -> str:
    return f"other:{value}"


# ---------------------------------------------------------------------------
# ChainState tests
# ---------------------------------------------------------------------------

class TestChainState(unittest.TestCase):

    def setUp(self):
        self.state = ChainState()

    def test_first_call_not_retry_is_novel(self):
        is_retry, is_novel, fixation = self.state.compute_signals("tool_a", {"x": 1})
        self.assertFalse(is_retry)
        self.assertTrue(is_novel)
        self.assertEqual(fixation, 0)

    def test_successful_call_clears_retry_state(self):
        # Simulate a failed call
        self.state.compute_signals("tool_a", {"x": 1})
        self.state.record_attempt("tool_a", {"x": 1}, success=False)

        # Retry of same tool should be flagged
        is_retry, _, _ = self.state.compute_signals("tool_a", {"x": 1})
        self.assertTrue(is_retry)

        # After success, next call is NOT a retry
        self.state.record_attempt("tool_a", {"x": 1}, success=True)
        is_retry2, _, _ = self.state.compute_signals("tool_a", {"x": 1})
        self.assertFalse(is_retry2)

    def test_retry_only_fires_for_same_tool(self):
        # tool_a fails
        self.state.compute_signals("tool_a", {"x": 1})
        self.state.record_attempt("tool_a", {"x": 1}, success=False)

        # Calling tool_b next is NOT a retry even though tool_a failed
        is_retry, _, _ = self.state.compute_signals("tool_b", {"x": 1})
        self.assertFalse(is_retry)

    def test_novelty_same_tool_same_args(self):
        args = {"query": "hello"}
        self.state.compute_signals("search", args)
        self.state.record_attempt("search", args, success=True)

        # Same tool + same args = not novel
        _, is_novel, _ = self.state.compute_signals("search", args)
        self.assertFalse(is_novel)

    def test_novelty_same_tool_different_args(self):
        self.state.compute_signals("search", {"query": "hello"})
        self.state.record_attempt("search", {"query": "hello"}, success=True)

        # Same tool but different args = novel
        _, is_novel, _ = self.state.compute_signals("search", {"query": "world"})
        self.assertTrue(is_novel)

    def test_novelty_window_eviction(self):
        # Fill the novelty window with NOVELTY_WINDOW unique calls after the
        # first, so the first entry is evicted from the deque (maxlen=NOVELTY_WINDOW).
        first_args = {"n": 0}
        self.state.compute_signals("tool", first_args)
        self.state.record_attempt("tool", first_args, success=True)

        # Add NOVELTY_WINDOW more unique calls to push the first one out
        for i in range(1, NOVELTY_WINDOW + 1):
            args = {"n": i}
            self.state.compute_signals("tool", args)
            self.state.record_attempt("tool", args, success=True)

        # First call should now be outside the window — novel again
        _, is_novel, _ = self.state.compute_signals("tool", first_args)
        self.assertTrue(is_novel)

    def test_tool_fixation_signal_increments(self):
        # First call: fixation = 0 (no prior calls)
        _, _, fix = self.state.compute_signals("tool_a", {})
        self.assertEqual(fix, 0)
        self.state.record_attempt("tool_a", {}, success=True)

        # Second call same tool: fixation = 1
        _, _, fix = self.state.compute_signals("tool_a", {"x": 2})
        self.assertEqual(fix, 1)
        self.state.record_attempt("tool_a", {"x": 2}, success=True)

        # Third call same tool: fixation = 2
        _, _, fix = self.state.compute_signals("tool_a", {"x": 3})
        self.assertEqual(fix, 2)

    def test_tool_fixation_resets_on_different_tool(self):
        # Build up fixation on tool_a
        for i in range(3):
            self.state.compute_signals("tool_a", {"i": i})
            self.state.record_attempt("tool_a", {"i": i}, success=True)

        # Call tool_b — fixation resets to 0
        _, _, fix = self.state.compute_signals("tool_b", {})
        self.assertEqual(fix, 0)

    def test_step_count_increments(self):
        self.assertEqual(self.state.step_count, 0)
        self.state.compute_signals("t", {})
        self.state.record_attempt("t", {}, success=True)
        self.assertEqual(self.state.step_count, 1)
        self.state.compute_signals("t", {"x": 2})
        self.state.record_attempt("t", {"x": 2}, success=False)
        self.assertEqual(self.state.step_count, 2)

    def test_hash_inputs_stable(self):
        # Dict ordering should not matter
        h1 = _hash_inputs({"b": 2, "a": 1})
        h2 = _hash_inputs({"a": 1, "b": 2})
        self.assertEqual(h1, h2)

    def test_hash_inputs_different_values(self):
        h1 = _hash_inputs({"a": 1})
        h2 = _hash_inputs({"a": 2})
        self.assertNotEqual(h1, h2)

    def test_hash_inputs_empty(self):
        # Should not raise
        h = _hash_inputs({})
        self.assertIsInstance(h, str)
        self.assertTrue(len(h) > 0)


# ---------------------------------------------------------------------------
# guard() — decision branch tests
# ---------------------------------------------------------------------------

class TestGuardProceed(unittest.TestCase):

    @patch("urllib.request.urlopen")
    def test_proceed_calls_fn_and_returns_result(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(_proceed_response())
        client = _make_client()

        result = client.guard(my_tool, {"value": "hello"})
        self.assertEqual(result, "result:hello")

    @patch("urllib.request.urlopen")
    def test_proceed_with_no_args(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(_proceed_response())
        client = _make_client()

        result = client.guard(my_tool)
        self.assertEqual(result, "result:x")  # default arg

    @patch("urllib.request.urlopen")
    def test_proceed_tool_name_from_fn_name(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(_proceed_response())
        client = _make_client()

        client.guard(my_tool, {"value": "v"})
        call_args = mock_urlopen.call_args
        payload = json.loads(call_args[0][0].data)
        # tool_name is not sent to /run directly — just verifying no error
        self.assertEqual(payload["tenant_id"], "test-tenant")

    @patch("urllib.request.urlopen")
    def test_proceed_explicit_tool_name_override(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(_proceed_response())
        client = _make_client()

        def unnamed_fn():
            return "ok"

        # Should not raise even with custom tool_name
        result = client.guard(unnamed_fn, {}, tool_name="my_custom_tool")
        self.assertEqual(result, "ok")

    @patch("urllib.request.urlopen")
    def test_proceed_with_deprecation_warning_field(self, mock_urlopen):
        # deprecation_warning field on proceed response should not raise
        resp = _proceed_response()
        resp["deprecation_warning"] = (
            "is_progressing is deprecated. Use is_novel_target instead."
        )
        mock_urlopen.return_value = _make_response(resp)
        client = _make_client()

        result = client.guard(my_tool, {"value": "v"})
        self.assertEqual(result, "result:v")


class TestGuardStop(unittest.TestCase):

    @patch("urllib.request.urlopen")
    def test_stop_raises_redlynr_blocked(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(
            _stop_response("step_cap_exceeded")
        )
        client = _make_client()

        with self.assertRaises(RedlynrBlocked) as ctx:
            client.guard(my_tool, {"value": "v"})

        self.assertEqual(ctx.exception.reason, "step_cap_exceeded")
        self.assertEqual(ctx.exception.decision, "stop")

    @patch("urllib.request.urlopen")
    def test_stop_fn_is_not_called(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(
            _stop_response("retries_exhausted")
        )
        client = _make_client()
        called = []

        def fn_that_records():
            called.append(True)

        with self.assertRaises(RedlynrBlocked):
            client.guard(fn_that_records, {})

        self.assertEqual(called, [])

    @patch("urllib.request.urlopen")
    def test_stop_carries_full_response(self, mock_urlopen):
        full_checks = {"paused": False, "chain_depth": {"reported_depth": 6, "limit": 5}}
        resp = _stop_response("chain_depth_exceeded", checks=full_checks)
        mock_urlopen.return_value = _make_response(resp)
        client = _make_client()

        with self.assertRaises(RedlynrBlocked) as ctx:
            client.guard(my_tool, {})

        self.assertEqual(ctx.exception.response["checks"], full_checks)

    @patch("urllib.request.urlopen")
    def test_lock_contention_empty_checks(self, mock_urlopen):
        # lock_contention: stop with empty checks dict and wait_seconds=1
        resp = {
            "decision": "stop",
            "reason": "lock_contention",
            "wait_seconds": 1,
            "checks": {},
        }
        mock_urlopen.return_value = _make_response(resp)
        client = _make_client()

        with self.assertRaises(RedlynrBlocked) as ctx:
            client.guard(my_tool, {})

        exc = ctx.exception
        self.assertEqual(exc.reason, "lock_contention")
        self.assertEqual(exc.response["checks"], {})
        self.assertEqual(exc.wait_seconds, 1)

    @patch("urllib.request.urlopen")
    def test_agent_paused_minimal_checks(self, mock_urlopen):
        # agent_paused: stop with minimal checks (only paused + chain_type)
        resp = {
            "decision": "stop",
            "reason": "agent_paused",
            "wait_seconds": None,
            "checks": {
                "paused": True,
                "chain_type": {
                    "declared": "interactive",
                    "limits_applied": "default",
                    "source": "default",
                },
            },
        }
        mock_urlopen.return_value = _make_response(resp)
        client = _make_client()

        with self.assertRaises(RedlynrBlocked) as ctx:
            client.guard(my_tool, {})

        exc = ctx.exception
        self.assertEqual(exc.reason, "agent_paused")
        self.assertTrue(exc.response["checks"]["paused"])
        # Must not assume full checks block is present
        self.assertNotIn("steps_retries", exc.response["checks"])

    @patch("urllib.request.urlopen")
    def test_all_stop_reasons_raise_blocked(self, mock_urlopen):
        stop_reasons = [
            "retries_exhausted",
            "step_cap_exceeded",
            "cost_budget_exceeded",
            "chain_cost_exceeded",
            "chain_depth_exceeded",
            "repetition_detected",
            "tool_fixation_detected",
            "volume_pressure_blocked",
            "agent_paused",
            "lock_contention",
        ]
        client = _make_client()

        for reason in stop_reasons:
            checks = {} if reason == "lock_contention" else {"paused": False}
            mock_urlopen.return_value = _make_response(
                _stop_response(reason, checks=checks)
            )
            with self.assertRaises(RedlynrBlocked) as ctx:
                client.guard(my_tool, {})
            self.assertEqual(ctx.exception.reason, reason, f"failed for reason: {reason}")

    @patch("urllib.request.urlopen")
    def test_stop_sets_retry_state_for_next_call(self, mock_urlopen):
        # After a stop, the next call to the same tool should have is_retry=True
        mock_urlopen.return_value = _make_response(
            _stop_response("step_cap_exceeded")
        )
        client = _make_client()

        with self.assertRaises(RedlynrBlocked):
            client.guard(my_tool, {"value": "v"})

        # Now simulate a proceed for the retry
        mock_urlopen.return_value = _make_response(_proceed_response())
        client.guard(my_tool, {"value": "v"})

        # Check that the second /run call had is_retry=True
        call_args = mock_urlopen.call_args
        payload = json.loads(call_args[0][0].data)
        self.assertTrue(payload["is_retry"])


class TestGuardSlowDown(unittest.TestCase):

    @patch("urllib.request.urlopen")
    def test_slow_down_transparent_by_default(self, mock_urlopen):
        # Default: slow_down is handled transparently (no raise, fn is called)
        mock_urlopen.return_value = _make_response(
            _slow_down_response("step_cap_warning")
        )
        client = _make_client()
        called = []

        def fn():
            called.append(True)
            return "done"

        result = client.guard(fn, {})
        self.assertEqual(result, "done")
        self.assertEqual(called, [True])

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_slow_down_waits_when_wait_seconds_provided(self, mock_urlopen, mock_sleep):
        mock_urlopen.return_value = _make_response(
            _slow_down_response("volume_pressure_warning", wait_seconds=6)
        )
        client = _make_client()
        client.guard(my_tool, {})

        mock_sleep.assert_called_once_with(6)

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_slow_down_no_wait_when_wait_seconds_none(self, mock_urlopen, mock_sleep):
        mock_urlopen.return_value = _make_response(
            _slow_down_response("step_cap_warning", wait_seconds=None)
        )
        client = _make_client()
        client.guard(my_tool, {})

        mock_sleep.assert_not_called()

    @patch("urllib.request.urlopen")
    def test_slow_down_raises_when_raise_on_slow_down_true(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(
            _slow_down_response("cost_budget_warning", wait_seconds=None)
        )
        client = _make_client(raise_on_slow_down=True)

        with self.assertRaises(RedlynrSlowDown) as ctx:
            client.guard(my_tool, {})

        exc = ctx.exception
        self.assertEqual(exc.reason, "cost_budget_warning")
        self.assertIsNone(exc.wait_seconds)

    @patch("urllib.request.urlopen")
    def test_slow_down_raise_carries_wait_seconds(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(
            _slow_down_response("volume_pressure_warning", wait_seconds=6)
        )
        client = _make_client(raise_on_slow_down=True)

        with self.assertRaises(RedlynrSlowDown) as ctx:
            client.guard(my_tool, {})

        self.assertEqual(ctx.exception.wait_seconds, 6)

    @patch("urllib.request.urlopen")
    def test_all_slow_down_reasons(self, mock_urlopen):
        slow_reasons = [
            ("step_cap_warning", None),
            ("cost_budget_warning", None),
            ("volume_pressure_warning", 6),
        ]
        for reason, wait in slow_reasons:
            mock_urlopen.return_value = _make_response(
                _slow_down_response(reason, wait_seconds=wait)
            )
            client = _make_client(raise_on_slow_down=True)
            with self.assertRaises(RedlynrSlowDown) as ctx:
                client.guard(my_tool, {})
            self.assertEqual(ctx.exception.reason, reason)


# ---------------------------------------------------------------------------
# Cost resolution tests
# ---------------------------------------------------------------------------

class TestCostResolution(unittest.TestCase):

    @patch("urllib.request.urlopen")
    def test_explicit_step_cost_sent(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(_proceed_response())
        client = _make_client()
        client.guard(my_tool, {}, step_cost=0.15)

        payload = json.loads(mock_urlopen.call_args[0][0].data)
        self.assertAlmostEqual(payload["step_cost"], 0.15)

    @patch("urllib.request.urlopen")
    def test_cost_fn_used_when_no_explicit_cost(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(_proceed_response())
        cost_fn = lambda tool_name, args: 0.07
        client = _make_client(cost_fn=cost_fn)
        client.guard(my_tool, {})

        payload = json.loads(mock_urlopen.call_args[0][0].data)
        self.assertAlmostEqual(payload["step_cost"], 0.07)

    @patch("urllib.request.urlopen")
    def test_explicit_cost_overrides_cost_fn(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(_proceed_response())
        cost_fn = lambda tool_name, args: 0.99
        client = _make_client(cost_fn=cost_fn)
        client.guard(my_tool, {}, step_cost=0.01)

        payload = json.loads(mock_urlopen.call_args[0][0].data)
        self.assertAlmostEqual(payload["step_cost"], 0.01)

    @patch("urllib.request.urlopen")
    def test_cost_fn_receives_tool_name(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(_proceed_response())
        costs = {"my_tool": 0.05, "other_tool": 0.10}
        received = []
        def cost_fn(tool_name, args):
            received.append(tool_name)
            return costs.get(tool_name, 0.0)

        client = _make_client(cost_fn=cost_fn)
        client.guard(my_tool, {})
        self.assertEqual(received, ["my_tool"])

    @patch("urllib.request.urlopen")
    def test_no_cost_fn_defaults_to_zero(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(_proceed_response())
        client = _make_client()
        client.guard(my_tool, {})

        payload = json.loads(mock_urlopen.call_args[0][0].data)
        self.assertAlmostEqual(payload["step_cost"], 0.0)

    @patch("urllib.request.urlopen")
    def test_cost_fn_exception_falls_back_to_zero(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(_proceed_response())
        def bad_cost_fn(tool_name, args):
            raise ValueError("cost lookup failed")
        client = _make_client(cost_fn=bad_cost_fn)
        # Should not raise; falls back to 0.0
        client.guard(my_tool, {})
        payload = json.loads(mock_urlopen.call_args[0][0].data)
        self.assertAlmostEqual(payload["step_cost"], 0.0)


# ---------------------------------------------------------------------------
# Payload construction tests
# ---------------------------------------------------------------------------

class TestPayloadConstruction(unittest.TestCase):

    @patch("urllib.request.urlopen")
    def test_required_fields_always_present(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(_proceed_response())
        client = _make_client()
        client.guard(my_tool, {})

        payload = json.loads(mock_urlopen.call_args[0][0].data)
        for field in ("tenant_id", "agent_id", "chain_id", "depth",
                      "step_cost", "is_retry", "is_novel_target",
                      "tool_fixation_signal"):
            self.assertIn(field, payload, f"missing field: {field}")

    @patch("urllib.request.urlopen")
    def test_chain_type_included_when_set(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(_proceed_response())
        client = _make_client(chain_type="batch")
        client.guard(my_tool, {})

        payload = json.loads(mock_urlopen.call_args[0][0].data)
        self.assertEqual(payload["chain_type"], "batch")

    @patch("urllib.request.urlopen")
    def test_chain_type_omitted_when_none(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(_proceed_response())
        client = _make_client()  # chain_type=None
        client.guard(my_tool, {})

        payload = json.loads(mock_urlopen.call_args[0][0].data)
        self.assertNotIn("chain_type", payload)

    @patch("urllib.request.urlopen")
    def test_trial_token_sent_as_header(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(_proceed_response())
        client = _make_client(trial_token="my-trial-token")
        client.guard(my_tool, {})

        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.get_header("X-trial-token"), "my-trial-token")

    @patch("urllib.request.urlopen")
    def test_depth_sent_correctly(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(_proceed_response())
        client = RedlynrClient(
            base_url="https://redlynr.com",
            tenant_id="t", owner_token="o", chain_id="c",
            agent_id="sub-agent", depth=2,
        )
        client.guard(my_tool, {})

        payload = json.loads(mock_urlopen.call_args[0][0].data)
        self.assertEqual(payload["depth"], 2)
        self.assertEqual(payload["agent_id"], "sub-agent")


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------

class TestRegister(unittest.TestCase):

    @patch("urllib.request.urlopen")
    def test_register_returns_owner_token(self, mock_urlopen):
        mock_urlopen.return_value = _make_response({
            "status": "registered",
            "tenant_id": "new-tenant",
            "owner_token": "abc123",
            "notice": "Store this token securely.",
        })
        token = RedlynrClient.register("https://redlynr.com", "new-tenant")
        self.assertEqual(token, "abc123")

    @patch("urllib.request.urlopen")
    def test_register_already_registered_raises(self, mock_urlopen):
        mock_urlopen.return_value = _make_response({
            "status": "already_registered",
            "tenant_id": "existing-tenant",
        })
        with self.assertRaises(RedlynrError) as ctx:
            RedlynrClient.register("https://redlynr.com", "existing-tenant")
        self.assertIn("already registered", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_register_sends_trial_token(self, mock_urlopen):
        mock_urlopen.return_value = _make_response({
            "status": "registered",
            "tenant_id": "t",
            "owner_token": "tok",
        })
        RedlynrClient.register("https://redlynr.com", "t", trial_token="trial-xyz")
        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.get_header("X-trial-token"), "trial-xyz")


# ---------------------------------------------------------------------------
# Management endpoint tests
# ---------------------------------------------------------------------------

class TestManagementEndpoints(unittest.TestCase):

    @patch("urllib.request.urlopen")
    def test_set_policy(self, mock_urlopen):
        mock_urlopen.return_value = _make_response({"status": "policy_saved"})
        client = _make_client()
        resp = client.set_policy({"steps_retries": {"max_steps_per_chain": 10}})
        self.assertEqual(resp["status"], "policy_saved")

        payload = json.loads(mock_urlopen.call_args[0][0].data)
        self.assertEqual(payload["tenant_id"], "test-tenant")
        self.assertEqual(payload["owner_token"], "test-token")

    @patch("urllib.request.urlopen")
    def test_set_policy_with_template(self, mock_urlopen):
        mock_urlopen.return_value = _make_response({
            "status": "policy_saved",
            "template_applied": "interactive_agent",
        })
        client = _make_client()
        resp = client.set_policy(template="interactive_agent")
        self.assertEqual(resp["template_applied"], "interactive_agent")

        payload = json.loads(mock_urlopen.call_args[0][0].data)
        self.assertEqual(payload["template"], "interactive_agent")

    @patch("urllib.request.urlopen")
    def test_reset_clears_in_process_state(self, mock_urlopen):
        mock_urlopen.return_value = _make_response({
            "status": "chain_reset",
            "tenant_id": "test-tenant",
            "chain_id": "chain-001",
            "steps_cleared": True,
        })
        client = _make_client()
        # Dirty up the state
        client._state.record_attempt("tool_a", {"x": 1}, success=False)
        self.assertEqual(client._state.step_count, 1)
        self.assertIsNotNone(client._state._last_failed)

        client.reset()

        # In-process state should be fresh
        self.assertEqual(client._state.step_count, 0)
        self.assertIsNone(client._state._last_failed)

    @patch("urllib.request.urlopen")
    def test_audit(self, mock_urlopen):
        mock_urlopen.return_value = _make_response({
            "tenant_id": "test-tenant",
            "count": 2,
            "entries": [{"decision": "proceed"}, {"decision": "stop"}],
        })
        client = _make_client()
        resp = client.audit()
        self.assertEqual(resp["count"], 2)

        # Check owner_token in URL query string
        req = mock_urlopen.call_args[0][0]
        self.assertIn("owner_token=test-token", req.full_url)

    @patch("urllib.request.urlopen")
    def test_pause(self, mock_urlopen):
        mock_urlopen.return_value = _make_response({
            "status": "pause_state_updated",
            "tenant_id": "test-tenant",
            "agent_id": "orchestrator",
            "paused": True,
        })
        client = _make_client()
        resp = client.pause(paused=True)
        self.assertTrue(resp["paused"])

        payload = json.loads(mock_urlopen.call_args[0][0].data)
        self.assertTrue(payload["paused"])
        self.assertEqual(payload["agent_id"], "orchestrator")

    @patch("urllib.request.urlopen")
    def test_health(self, mock_urlopen):
        mock_urlopen.return_value = _make_response({
            "status": "ok",
            "app": "redlynr",
            "core_loaded": True,
            "core_error": None,
        })
        client = _make_client()
        resp = client.health()
        self.assertTrue(resp["core_loaded"])

    @patch("urllib.request.urlopen")
    def test_run_raw(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(_proceed_response())
        client = _make_client()
        resp = client.run_raw(depth=1, step_cost=0.05, is_retry=False,
                              is_novel_target=True, tool_fixation_signal=0)
        self.assertEqual(resp["decision"], "proceed")

        payload = json.loads(mock_urlopen.call_args[0][0].data)
        self.assertEqual(payload["depth"], 1)
        self.assertEqual(payload["tenant_id"], "test-tenant")


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestErrorHandling(unittest.TestCase):

    @patch("urllib.request.urlopen")
    def test_http_error_raises_redlynr_error(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://redlynr.com/run",
            code=402,
            msg="Payment Required",
            hdrs=None,
            fp=BytesIO(b'{"error": "payment required"}'),
        )
        client = _make_client()
        with self.assertRaises(RedlynrError) as ctx:
            client.guard(my_tool, {})
        self.assertIn("402", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_connection_error_raises_redlynr_error(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("connection refused")
        client = _make_client()
        with self.assertRaises(RedlynrError) as ctx:
            client.guard(my_tool, {})
        self.assertIn("connection", str(ctx.exception).lower())


# ---------------------------------------------------------------------------
# Exception attribute tests
# ---------------------------------------------------------------------------

class TestExceptionAttributes(unittest.TestCase):

    def test_redlynr_blocked_attributes(self):
        response = {"decision": "stop", "reason": "step_cap_exceeded",
                    "wait_seconds": None, "checks": {}}
        exc = RedlynrBlocked(
            reason="step_cap_exceeded",
            decision="stop",
            response=response,
        )
        self.assertEqual(exc.reason, "step_cap_exceeded")
        self.assertEqual(exc.decision, "stop")
        self.assertEqual(exc.response, response)
        self.assertIsNone(exc.wait_seconds)
        self.assertIn("step_cap_exceeded", str(exc))

    def test_redlynr_blocked_lock_contention_wait_seconds(self):
        response = {"decision": "stop", "reason": "lock_contention",
                    "wait_seconds": 1, "checks": {}}
        exc = RedlynrBlocked(
            reason="lock_contention",
            decision="stop",
            response=response,
        )
        self.assertEqual(exc.wait_seconds, 1)

    def test_redlynr_slow_down_attributes(self):
        response = {"decision": "slow_down", "reason": "step_cap_warning",
                    "wait_seconds": None, "checks": {}}
        exc = RedlynrSlowDown(
            reason="step_cap_warning",
            wait_seconds=None,
            response=response,
        )
        self.assertEqual(exc.reason, "step_cap_warning")
        self.assertIsNone(exc.wait_seconds)
        self.assertIn("step_cap_warning", str(exc))

    def test_redlynr_slow_down_with_wait(self):
        response = {"decision": "slow_down", "reason": "volume_pressure_warning",
                    "wait_seconds": 6, "checks": {}}
        exc = RedlynrSlowDown(
            reason="volume_pressure_warning",
            wait_seconds=6,
            response=response,
        )
        self.assertEqual(exc.wait_seconds, 6)
        self.assertIn("6s", str(exc))


if __name__ == "__main__":
    unittest.main()
