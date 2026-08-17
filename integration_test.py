#!/usr/bin/env python3
"""
redlynr-python SDK — Live Integration Test
==========================================
Runs against the real Redlynr service (localhost:8406 on the server, or
https://redlynr.com externally). Uses PoW trial tokens — no USDC spent.

Usage:
    python3 integration_test.py
    python3 integration_test.py --url https://redlynr.com

Each run registers a fresh tenant_id. All state expires via Redis TTL.
"""

import argparse, hashlib, json, sys, uuid, urllib.request, urllib.error, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from redlynr import RedlynrClient, RedlynrBlocked, RedlynrError, RedlynrSlowDown

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results = []

def test(name, fn):
    print(f"  {name} ...", end=" ", flush=True)
    try:
        fn()
        print(PASS)
        results.append((name, PASS, None))
    except AssertionError as e:
        print(f"FAIL -- {e}")
        results.append((name, FAIL, str(e)))
    except Exception as e:
        print(f"FAIL -- {type(e).__name__}: {e}")
        results.append((name, FAIL, f"{type(e).__name__}: {e}"))

# ---------------------------------------------------------------------------
# PoW solver — difficulty is leading hex zeros
# ---------------------------------------------------------------------------
def solve_pow(nonce, difficulty):
    prefix = "0" * difficulty
    counter = 0
    while True:
        if hashlib.sha256(f"{nonce}{counter}".encode()).hexdigest().startswith(prefix):
            return str(counter)
        counter += 1

def fetch_token(base_url):
    """Fetch a fresh PoW trial token."""
    req = urllib.request.Request(f"{base_url}/trial/challenge", method="GET")
    with urllib.request.urlopen(req, timeout=10) as r:
        ch = json.loads(r.read())
    nonce, diff = ch["nonce"], ch["difficulty"]
    solution = solve_pow(nonce, diff)
    data = json.dumps({"nonce": nonce, "solution": solution}).encode()
    req2 = urllib.request.Request(
        f"{base_url}/trial/claim", data=data,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req2, timeout=10) as r:
        claim = json.loads(r.read())
    token = claim.get("trial_token") or claim.get("token")
    if not token:
        raise RuntimeError(f"No token in claim response: {claim}")
    return token, diff

# ---------------------------------------------------------------------------
# TokenBag: auto-refreshes when a call count threshold is reached
# ---------------------------------------------------------------------------
class TokenBag:
    def __init__(self, base_url, calls_per_token=30):
        self.base_url = base_url
        self.calls_per_token = calls_per_token
        self._token = None
        self._calls_used = 0

    def get(self, calls_needed=1):
        """Return current token, refreshing if we're too close to the limit."""
        if self._token is None or self._calls_used + calls_needed > self.calls_per_token - 2:
            print(f"\n  [token] Fetching new trial token ...", end=" ", flush=True)
            self._token, diff = fetch_token(self.base_url)
            self._calls_used = 0
            print(f"ok (difficulty={diff})")
        return self._token

    def used(self, n=1):
        """Record that n calls were made."""
        self._calls_used += n

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8406")
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    print(f"\nRedlynr SDK Integration Test")
    print(f"Target: {base_url}")
    print("=" * 60)

    run_id = uuid.uuid4().hex[:8]
    tenant_id = f"sdk-test-{run_id}"
    owner_token = None
    bag = TokenBag(base_url, calls_per_token=30)

    # ------------------------------------------------------------------
    # 1. PoW challenge/claim
    # ------------------------------------------------------------------
    print("\n[ Trial token ]")
    def t_pow():
        nonlocal owner_token
        token, diff = fetch_token(base_url)
        assert isinstance(token, str) and len(token) > 10
        bag._token = token
        bag._calls_used = 0
        print(f"    Token: {token[:20]}... (difficulty={diff})")
    test("1. PoW challenge/claim", t_pow)

    if not bag._token:
        print("Cannot continue without trial token.")
        _summary(); sys.exit(1)

    # ------------------------------------------------------------------
    # 2-3. Registration
    # ------------------------------------------------------------------
    print("\n[ Registration ]")
    def t_register():
        nonlocal owner_token
        tok = RedlynrClient.register(base_url, tenant_id, trial_token=bag.get())
        assert isinstance(tok, str) and len(tok) > 10, f"bad token: {tok!r}"
        owner_token = tok
    test("2. Register fresh tenant", t_register)

    if not owner_token:
        print("Cannot continue without owner_token.")
        _summary(); sys.exit(1)

    def t_register_dup():
        try:
            RedlynrClient.register(base_url, tenant_id, trial_token=bag.get())
            raise AssertionError("Expected RedlynrError, got none")
        except RedlynrError as e:
            assert "already registered" in str(e).lower(), f"wrong message: {e}"
    test("3. Register duplicate raises RedlynrError", t_register_dup)

    # ------------------------------------------------------------------
    # 4. Health
    # ------------------------------------------------------------------
    print("\n[ Health ]")
    def t_health():
        client = RedlynrClient(base_url=base_url, tenant_id=tenant_id,
            owner_token=owner_token, chain_id="hc", trial_token=bag.get())
        h = client.health()
        assert h.get("status") == "ok", f"status: {h.get('status')}"
    test("4. Health check", t_health)

    # ------------------------------------------------------------------
    # 5. Set policy — tight limits: cap=5, warn at 80%
    # ------------------------------------------------------------------
    print("\n[ Policy ]")
    def t_policy():
        client = RedlynrClient(base_url=base_url, tenant_id=tenant_id,
            owner_token=owner_token, chain_id="pc", trial_token=bag.get())
        resp = client.set_policy(policy={
            "steps_retries": {"max_steps_per_chain": 5, "max_retries": 1,
                              "warn_at_steps_pct": 80},
            "cost": {"max_spend_per_chain": 1.00, "warn_at_pct": None},
            "chain_depth": {"max_depth": 3},
            "volume": {"block_on_volume_pressure": False},
        })
        assert resp.get("status") == "policy_saved", f"resp: {resp}"
    test("5. Set policy", t_policy)

    # ------------------------------------------------------------------
    # 6. guard() proceed
    # ------------------------------------------------------------------
    print("\n[ guard() -- proceed ]")
    def t_proceed():
        client = RedlynrClient(base_url=base_url, tenant_id=tenant_id,
            owner_token=owner_token, chain_id=f"proceed-{run_id}",
            trial_token=bag.get(1))
        called = []
        def my_tool(value="x"):
            called.append(value); return f"result:{value}"
        result = client.guard(my_tool, {"value": "hello"}, step_cost=0.0)
        bag.used(1)
        assert result == "result:hello", f"wrong result: {result!r}"
        assert called == ["hello"], f"tool not called: {called}"
    test("6. guard() proceed -- tool executes, result returned", t_proceed)

    # ------------------------------------------------------------------
    # 7. cost_fn
    # ------------------------------------------------------------------
    print("\n[ guard() -- cost_fn ]")
    def t_cost_fn():
        received = []
        def cost_fn(tool_name, args):
            received.append(tool_name); return 0.001
        client = RedlynrClient(base_url=base_url, tenant_id=tenant_id,
            owner_token=owner_token, chain_id=f"costfn-{run_id}",
            trial_token=bag.get(1), cost_fn=cost_fn)
        def my_tool(value="x"): return "ok"
        client.guard(my_tool, {"value": "v"})
        bag.used(1)
        assert len(received) == 1, f"cost_fn called {len(received)}x"
        assert received[0] == "my_tool", f"wrong tool_name: {received[0]}"
    test("7. guard() cost_fn -- hook fires with correct tool_name", t_cost_fn)

    # ------------------------------------------------------------------
    # 8-9. slow_down
    # ------------------------------------------------------------------
    print("\n[ guard() -- slow_down ]")
    def t_slow_down_transparent():
        # 4 calls: steps 1-3 proceed, step 4 slow_downs (warn at 80% of 5 = 4.0)
        client = RedlynrClient(base_url=base_url, tenant_id=tenant_id,
            owner_token=owner_token, chain_id=f"sd-{run_id}",
            trial_token=bag.get(4))
        executed = []
        def my_tool(n=0): executed.append(n); return n
        for i in range(3):
            client.guard(my_tool, {"n": i}, step_cost=0.0)
        result = client.guard(my_tool, {"n": 99}, step_cost=0.0)
        bag.used(4)
        assert 99 in executed, f"tool not called on slow_down: {executed}"
        assert result == 99
    test("8. guard() slow_down transparent -- tool still executes", t_slow_down_transparent)

    def t_slow_down_raises():
        client = RedlynrClient(base_url=base_url, tenant_id=tenant_id,
            owner_token=owner_token, chain_id=f"sdr-{run_id}",
            trial_token=bag.get(4), raise_on_slow_down=True)
        def my_tool(n=0): return n
        for i in range(3):
            client.guard(my_tool, {"n": i}, step_cost=0.0)
        called = []
        def guarded(n=0): called.append(n); return n
        try:
            client.guard(guarded, {"n": 99}, step_cost=0.0)
            raise AssertionError("Expected RedlynrSlowDown")
        except RedlynrSlowDown as e:
            bag.used(4)
            assert e.reason == "step_cap_warning", f"wrong reason: {e.reason}"
            assert called == [], f"tool called despite raise_on_slow_down: {called}"
    test("9. guard() slow_down raises RedlynrSlowDown", t_slow_down_raises)

    # ------------------------------------------------------------------
    # 10-11. stop
    # ------------------------------------------------------------------
    print("\n[ guard() -- stop ]")
    def t_stop_blocked():
        # Need up to 8 calls: steps 4+5 slow_down (transparent), step 6 stops.
        # chain_type="interactive" pins the type and prevents auto-detection as
        # batch (which would raise max_steps to 100 after 5 novel calls).
        client = RedlynrClient(base_url=base_url, tenant_id=tenant_id,
            owner_token=owner_token, chain_id=f"stop-{run_id}",
            trial_token=bag.get(8), chain_type="interactive")
        def my_tool(n=0): return n
        hard_stopped = False
        calls = 0
        for i in range(8):
            try:
                client.guard(my_tool, {"n": i}, step_cost=0.0)
                calls += 1
            except RedlynrBlocked:
                calls += 1
                hard_stopped = True
                break
        bag.used(calls)
        assert hard_stopped, f"No stop in {calls} calls (cap=5)"

        # Confirm next call also stops and does NOT execute the tool
        called = []
        def sentinel(n=0): called.append(n); return n
        try:
            client.guard(sentinel, {"n": 999}, step_cost=0.0)
            bag.used(1)
            raise AssertionError("Expected RedlynrBlocked, tool returned instead")
        except RedlynrBlocked as e:
            bag.used(1)
            assert e.reason == "step_cap_exceeded", f"wrong reason: {e.reason}"
            assert called == [], f"tool called despite stop: {called}"
            assert e.decision == "stop"
    test("10. guard() stop -- RedlynrBlocked raised, tool NOT called", t_stop_blocked)

    def t_stop_carries_response():
        client = RedlynrClient(base_url=base_url, tenant_id=tenant_id,
            owner_token=owner_token, chain_id=f"stopr-{run_id}",
            trial_token=bag.get(8), chain_type="interactive")
        def my_tool(n=0): return n
        exc = None
        calls = 0
        for i in range(8):
            try:
                client.guard(my_tool, {"n": i}, step_cost=0.0)
                calls += 1
            except RedlynrBlocked as e:
                calls += 1
                exc = e
                break
        bag.used(calls)
        assert exc is not None, f"No stop in {calls} calls"
        assert isinstance(exc.response, dict), "response not dict"
        assert "decision" in exc.response
        assert "checks" in exc.response
    test("11. guard() stop carries full response dict", t_stop_carries_response)

    # ------------------------------------------------------------------
    # 12. reset()
    # ------------------------------------------------------------------
    print("\n[ reset() ]")
    def t_reset():
        client = RedlynrClient(base_url=base_url, tenant_id=tenant_id,
            owner_token=owner_token, chain_id=f"reset-{run_id}",
            trial_token=bag.get(10), chain_type="interactive")
        def my_tool(n=0): return n
        # Exhaust the chain
        calls = 0
        for i in range(8):
            try:
                client.guard(my_tool, {"n": i}, step_cost=0.0)
                calls += 1
            except RedlynrBlocked:
                calls += 1
                break
        # Confirm stopped
        try:
            client.guard(my_tool, {"n": 99}, step_cost=0.0)
            calls += 1
            raise AssertionError("Expected stop before reset")
        except RedlynrBlocked:
            calls += 1
        bag.used(calls)
        # Reset
        resp = client.reset()
        assert resp.get("status") == "chain_reset", f"reset: {resp}"
        # Should proceed again
        result = client.guard(my_tool, {"n": 0}, step_cost=0.0)
        bag.used(1)
        assert result == 0, f"wrong result after reset: {result}"
    test("12. reset() -- chain resets, guard() proceeds again", t_reset)

    # ------------------------------------------------------------------
    # 13. audit()
    # ------------------------------------------------------------------
    print("\n[ audit() ]")
    def t_audit():
        client = RedlynrClient(base_url=base_url, tenant_id=tenant_id,
            owner_token=owner_token, chain_id=f"audit-{run_id}",
            trial_token=bag.get())
        resp = client.audit()
        assert "entries" in resp, f"missing entries: {resp}"
        assert "count" in resp
        assert resp["count"] > 0, "audit log empty"
        entry = resp["entries"][0]
        for f in ("decision", "reason", "agent_id", "chain_id", "ts"):
            assert f in entry, f"missing field: {f}"
    test("13. audit() -- decisions logged, entry shape correct", t_audit)

    # ------------------------------------------------------------------
    # 14-15. Chain signals (in-process only, 1 /run call each)
    # ------------------------------------------------------------------
    print("\n[ Chain signals ]")
    def t_is_retry():
        client = RedlynrClient(base_url=base_url, tenant_id=tenant_id,
            owner_token=owner_token, chain_id=f"retry-{run_id}",
            trial_token=bag.get(1))
        def my_tool(n=0): return n
        client.guard(my_tool, {"n": 1}, step_cost=0.0)
        bag.used(1)
        is_retry, _, _ = client._state.compute_signals("my_tool", {"n": 1})
        assert not is_retry, "is_retry should be False after success"
        client._state.record_attempt("my_tool", {"n": 2}, success=False)
        is_retry2, _, _ = client._state.compute_signals("my_tool", {"n": 2})
        assert is_retry2, "is_retry should be True after failure"
    test("14. is_retry signal -- correct after success and failure", t_is_retry)

    def t_is_novel():
        client = RedlynrClient(base_url=base_url, tenant_id=tenant_id,
            owner_token=owner_token, chain_id=f"novel-{run_id}",
            trial_token=bag.get(1))
        def my_tool(query=""): return query
        args = {"query": "same"}
        _, is_novel, _ = client._state.compute_signals("my_tool", args)
        assert is_novel, "First call should be novel"
        client.guard(my_tool, args, step_cost=0.0)
        bag.used(1)
        _, is_novel2, _ = client._state.compute_signals("my_tool", args)
        assert not is_novel2, "Repeated args should not be novel"
        _, is_novel3, _ = client._state.compute_signals("my_tool", {"query": "diff"})
        assert is_novel3, "Different args should be novel"
    test("15. is_novel_target -- repeats detected, new args novel", t_is_novel)

    # ------------------------------------------------------------------
    # 16. chain_type declared (2 /run calls)
    # ------------------------------------------------------------------
    print("\n[ chain_type ]")
    def t_chain_type():
        client = RedlynrClient(base_url=base_url, tenant_id=tenant_id,
            owner_token=owner_token, chain_id=f"ctype-{run_id}",
            trial_token=bag.get(2), chain_type="batch")
        def my_tool(): return "ok"
        result = client.guard(my_tool, {}, step_cost=0.0)
        bag.used(1)
        assert result == "ok"
        resp = client.run_raw(depth=0, step_cost=0.0, is_retry=False,
                              is_novel_target=True, tool_fixation_signal=0,
                              chain_type="batch")
        bag.used(1)
        declared = resp.get("checks", {}).get("chain_type", {}).get("declared")
        assert declared == "batch", f"chain_type not in response: {resp}"
    test("16. chain_type declared -- reflected in response", t_chain_type)

    # ------------------------------------------------------------------
    # 17. audit_analyze()
    # ------------------------------------------------------------------
    print("\n[ audit_analyze() ]")
    def t_audit_analyze():
        client = RedlynrClient(base_url=base_url, tenant_id=tenant_id,
            owner_token=owner_token, chain_id="x", trial_token=bag.get())
        resp = client.audit_analyze(min_chains=1)
        assert isinstance(resp, dict) and len(resp) > 0, f"bad resp: {resp}"
    test("17. audit_analyze() -- returns classification structure", t_audit_analyze)

    # ------------------------------------------------------------------
    # 18. pause() / unpause() (2 /run calls)
    # ------------------------------------------------------------------
    print("\n[ pause() / unpause() ]")
    def t_pause():
        client = RedlynrClient(base_url=base_url, tenant_id=tenant_id,
            owner_token=owner_token, chain_id=f"pause-{run_id}",
            agent_id=f"agent-{run_id}", trial_token=bag.get(2))
        def my_tool(): return "ok"
        resp = client.pause(paused=True)
        assert resp.get("paused") is True, f"pause resp: {resp}"
        try:
            client.guard(my_tool, {}, step_cost=0.0)
            bag.used(1)
            raise AssertionError("Expected RedlynrBlocked(agent_paused)")
        except RedlynrBlocked as e:
            bag.used(1)
            assert e.reason == "agent_paused", f"wrong reason: {e.reason}"
            assert e.response["checks"].get("paused") is True
            assert "steps_retries" not in e.response["checks"]
        resp = client.pause(paused=False)
        assert resp.get("paused") is False, f"unpause resp: {resp}"
        result = client.guard(my_tool, {}, step_cost=0.0)
        bag.used(1)
        assert result == "ok", f"wrong result after unpause: {result}"
    test("18. pause()/unpause() -- agent_paused fires, clears, proceeds", t_pause)

    _summary()


def _summary():
    print(f"\n{'=' * 60}")
    print("RESULTS\n")
    passed = sum(1 for _, s, _ in results if s == PASS)
    failed = sum(1 for _, s, _ in results if s == FAIL)
    skipped = sum(1 for _, s, _ in results if s == SKIP)
    for name, status, msg in results:
        marker = "+" if status == PASS else ("-" if status == FAIL else "o")
        line = f"  [{marker}] {name}"
        if status == FAIL and msg:
            line += f"\n      -> {msg}"
        print(line)
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped out of {len(results)} tests")
    if failed:
        print("\nSome tests FAILED -- do not publish to PyPI.")
        sys.exit(1)
    else:
        print("\nAll tests passed -- safe to publish.")


if __name__ == "__main__":
    main()
