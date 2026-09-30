"""Collision-aware API smoke test, run from the one-shot ``verify`` service.

It waits for readiness and exercises:

1. a feasible 6-arm / 6-target adjudication, checking the returned pairings,
   per-arm lengths, uniqueness and *every* pairwise clearance against the
   requested integer clearance;
2. an instance where pairwise collisions make the requested minimum
   impossible: the API must return the maximum achievable count and a clear
   reason (HTTP 200, status minimum_not_met);
3. an illegal payload that must be rejected with HTTP 422 without solving;
4. single-target-loss takeover certification: a certifiable instance with one
   standby per arm (per-target reassignment, preserved fixed pairings and
   standby clearance evidence), a non-certifiable instance (maximum certified
   count plus the first blocking scenario), and backward compatibility when
   the feature is not enabled.

Exits 0 on success, 1 on any failure.
"""
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
EPS = 1e-9


def request(method, path, payload=None, expect=(200,)):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            code, body = resp.status, resp.read()
    except urllib.error.HTTPError as e:
        code, body = e.code, e.read()
    if code not in expect:
        raise AssertionError(f"{method} {path}: expected {expect}, got {code}: "
                             f"{body[:300]!r}")
    return code, json.loads(body)


def wait_ready(deadline=60.0):
    end = time.time() + deadline
    last = None
    while time.time() < end:
        try:
            code, body = request("GET", "/health/ready")
            if code == 200 and body.get("status") == "ready":
                return True
        except Exception as e:  # noqa: BLE001 - connection may not exist yet
            last = e
        time.sleep(1)
    raise AssertionError(f"API did not become ready in {deadline}s: {last}")


def feasible_payload():
    return {
        "arms": [{"id": f"A{i}", "x": i * 100, "y": 0, "max_extension": 500}
                 for i in range(6)],
        "targets": [{"id": f"T{i}", "x": i * 100 + 10, "y": 10,
                     "priority": i + 1} for i in range(6)],
        "clearance": 5,
        "minimum_allocations": 6,
    }


def collision_payload():
    # Two arms/targets per group; both pairings of a group have segments
    # closer than clearance 3 (converging or crossing), so at most one
    # allocation per group is possible -> maximum 3 of requested 6.
    arms, targets = [], []
    for g in range(3):
        gx = 200 * g
        arms += [{"id": f"L{g}", "x": gx, "y": 0, "max_extension": 100},
                 {"id": f"R{g}", "x": gx, "y": 10, "max_extension": 100}]
        targets += [{"id": f"lo{g}", "x": gx + 10, "y": 4, "priority": 5 + g},
                    {"id": f"hi{g}", "x": gx + 10, "y": 6, "priority": 1 + g}]
    return {"arms": arms, "targets": targets,
            "clearance": 3, "minimum_allocations": 6}


def takeover_certifiable_payload():
    # Six far-apart arms with a main target and a dedicated standby each.
    arms = [{"id": f"A{i}", "x": i * 1000, "y": 0, "max_extension": 500}
            for i in range(6)]
    targets = [{"id": f"T{i}", "x": i * 1000 + 10, "y": 0, "priority": i + 1}
               for i in range(6)]
    targets += [{"id": f"S{i}", "x": i * 1000 - 10, "y": 0, "priority": 1}
                for i in range(6)]
    return {"arms": arms, "targets": targets, "clearance": 5,
            "minimum_allocations": 6, "takeover": {"enabled": True}}


def takeover_uncertifiable_payload():
    # Each arm reaches exactly one target: the six-pair plain plan cannot be
    # certified at all, so the maximum certified allocation count is zero.
    arms = [{"id": f"A{i}", "x": i * 1000, "y": 0, "max_extension": 500}
            for i in range(6)]
    targets = [{"id": f"T{i}", "x": i * 1000 + 10, "y": 0, "priority": i + 1}
               for i in range(6)]
    return {"arms": arms, "targets": targets, "clearance": 5,
            "minimum_allocations": 6, "takeover": {"enabled": True}}


def main():
    wait_ready()
    print("readiness OK")

    # 1) feasible instance + evidence verification
    _, body = request("POST", "/api/v1/adjudicate", feasible_payload())
    assert body["status"] == "ok" and body["feasible"] is True, body
    obj = body["objective"]
    assert obj["num_allocations"] == 6, obj
    assert obj["priority_sum"] == 21, obj

    arms_used = [p["arm_id"] for p in body["assignments"]]
    tgts_used = [p["target_id"] for p in body["assignments"]]
    assert len(set(arms_used)) == len(arms_used) == 6
    assert len(set(tgts_used)) == len(tgts_used) == 6
    assert body["unassigned_arms"] == [] and body["unassigned_targets"] == []

    # per-arm lengths present and consistent
    for p in body["assignments"]:
        length = body["arm_lengths"][p["arm_id"]]
        assert math.isclose(length, p["extension"], abs_tol=1e-12)
        assert length > 0

    # every pair of closed segments respects the integer clearance
    evidence = body["clearance"]["pair_evidence"]
    assert len(evidence) == math.comb(6, 2) == 15
    for rec in evidence:
        assert rec["satisfied"] is True, rec
        assert rec["distance"] + EPS >= rec["required_clearance"], rec
        assert rec["witness"]["point_on_first"]
    assert body["clearance"]["all_pairs_satisfied"] is True
    print("feasible case OK: 6 pairs, 15 clearances verified")

    # 2) collision-driven shortfall
    _, body = request("POST", "/api/v1/adjudicate", collision_payload())
    assert body["status"] == "minimum_not_met", body
    assert body["feasible"] is False
    assert body["objective"]["num_allocations"] == 3, body["objective"]
    assert "minimum_allocations=6" in body["reason"]
    bp = body["witness"]["blocking_pair"]
    assert bp is not None and bp["distance"] < bp["required_clearance"]
    for rec in body["clearance"]["pair_evidence"]:
        assert rec["satisfied"] is True
        assert rec["distance"] + EPS >= rec["required_clearance"]
    print("collision shortfall OK: max=3 reported with reason and witness")

    # 3) illegal input rejected before solving
    bad = feasible_payload()
    bad["clearance"] = 0
    request("POST", "/api/v1/adjudicate", bad, expect=(422,))
    bad = feasible_payload()
    bad["arms"] = bad["arms"][:5]
    request("POST", "/api/v1/adjudicate", bad, expect=(422,))
    print("illegal input rejected with 422")

    # 4a) takeover: certifiable plan with one standby per arm
    _, body = request("POST", "/api/v1/adjudicate",
                      takeover_certifiable_payload())
    assert body["status"] == "ok" and body["feasible"] is True, body
    tk = body["takeover"]
    assert tk["enabled"] is True and tk["fully_certified"] is True, tk
    assert tk["num_scenarios"] == 6
    assert tk["max_certified_allocations"] is None
    assert tk["first_blocking_scenario"] is None
    plan = {p["arm_id"]: p["target_id"] for p in body["assignments"]}
    assert len(plan) == 6
    for sc in tk["scenarios"]:
        arm, ra = sc["arm_id"], sc["reassignment"]
        assert ra is not None and ra["arm_id"] == arm
        assert ra["lost_target_id"] == plan[arm]
        assert ra["spare_target_id"].startswith("S")
        assert math.isclose(ra["extension"] ** 2, ra["extension_sq"],
                            rel_tol=1e-12)
        fixed = {p["arm_id"]: p["target_id"] for p in sc["fixed_pairings"]}
        assert fixed == {a: t for a, t in plan.items() if a != arm}
        assert len(sc["clearance_evidence"]) == 5
        for ev in sc["clearance_evidence"]:
            assert ev["satisfied"] is True
            assert ev["distance"] + EPS >= ev["required_clearance"]
    print("takeover certifiable case OK: 6 scenarios with standby + evidence")

    # 4b) takeover: nothing certifiable -> max certified count + blocker
    _, body = request("POST", "/api/v1/adjudicate",
                      takeover_uncertifiable_payload())
    assert body["status"] == "minimum_not_met", body
    assert body["feasible"] is False
    assert body["objective"]["num_allocations"] == 0, body["objective"]
    tk = body["takeover"]
    assert tk["max_certified_allocations"] == 0
    fbs = tk["first_blocking_scenario"]
    assert fbs is not None and fbs["arm_id"] == "A0"
    assert fbs["lost_target_id"] == "T0"
    assert fbs["reassignment"] is None
    print("takeover shortfall OK: max certified=0 with first blocking "
          "scenario")

    # 4c) backward compatibility: no takeover key in request/response
    payload = takeover_certifiable_payload()
    payload.pop("takeover")
    _, body = request("POST", "/api/v1/adjudicate", payload)
    assert body.get("takeover") is None
    assert body["objective"]["takeover_certification"] == "disabled"
    print("takeover backward compatibility OK")

    print("SMOKE: PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        print(f"SMOKE: FAIL: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        print(f"SMOKE: ERROR: {e!r}", file=sys.stderr)
        sys.exit(1)
