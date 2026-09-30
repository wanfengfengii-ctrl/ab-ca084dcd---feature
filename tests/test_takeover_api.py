"""End-to-end API tests for single-target-loss takeover."""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def takeover_payload():
    # Every arm has a dedicated spare on the other side of y=0, so the
    # full 6-pair plan is certifiable.
    return {
        "arms": [{"id": f"A{i}", "x": i * 100, "y": 0,
                  "max_extension": 500} for i in range(6)],
        "targets": [
            *[{"id": f"T{i}", "x": i * 100 + 10, "y": 10, "priority": 2}
              for i in range(6)],
            *[{"id": f"U{i}", "x": i * 100 + 10, "y": -10, "priority": 1}
              for i in range(6)],
        ],
        "clearance": 5,
        "minimum_allocations": 6,
    }


def test_flag_disabled_response_has_no_takeover_section(client):
    body = takeover_payload()
    r = client.post("/api/v1/adjudicate", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "ok"
    assert data.get("takeover") is None


def test_certified_takeover_report(client):
    body = takeover_payload()
    body["single_target_loss_takeover"] = True
    r = client.post("/api/v1/adjudicate", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "ok" and data["feasible"] is True
    rep = data["takeover"]
    assert rep["enabled"] is True and rep["certified"] is True
    assert rep["max_certifiable_allocations"] == 6
    assert rep["requested_minimum_allocations"] == 6
    assert rep["first_blocking_scenario"] is None
    assert len(rep["scenarios"]) == 6

    main_pairs = {p["arm_id"]: p["target_id"]
                  for p in data["assignments"]}
    for sc in rep["scenarios"]:
        lost = sc["lost_target_id"]
        arm = sc["arm_id"]
        assert main_pairs[arm] == lost
        # all other pairs preserved verbatim
        fixed = {p["arm_id"]: p["target_id"] for p in sc["fixed_pairs"]}
        expected = dict(main_pairs)
        del expected[arm]
        assert fixed == expected
        # replacement is a target unused by the main plan, reachable
        assert sc["replacement"]["arm_id"] == arm
        assert sc["replacement"]["target_id"] not in main_pairs.values()
        assert sc["replacement"]["extension_sq"] >= 1
        # evidence against every fixed pair, all satisfied exactly
        assert len(sc["pair_evidence"]) == 5
        for ev in sc["pair_evidence"]:
            assert ev["satisfied"] is True
            assert ev["distance"] + 1e-9 >= ev["required_clearance"]
            assert {sc["replacement"]["target_id"]} <= {
                ev["target_a"], ev["target_b"]}


def test_uncertifiable_request_reports_max_and_first_blocker(client):
    # 6 arms / 6 targets only: a full plan has no spare target at all.
    body = {
        "arms": [{"id": f"A{i}", "x": i * 100, "y": 0,
                  "max_extension": 50} for i in range(6)],
        "targets": [{"id": f"T{i}", "x": i * 100 + 10, "y": 10,
                     "priority": 1} for i in range(6)],
        "clearance": 5,
        "minimum_allocations": 6,
        "single_target_loss_takeover": True,
    }
    r = client.post("/api/v1/adjudicate", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "takeover_not_certified"
    assert data["feasible"] is False
    assert data["objective"]["num_allocations"] == 6
    rep = data["takeover"]
    assert rep["certified"] is False
    assert rep["max_certifiable_allocations"] == 0
    assert rep["scenarios"] == []
    blocker = rep["first_blocking_scenario"]
    assert blocker["arm_id"] == "A0" and blocker["lost_target_id"] == "T0"
    assert blocker["blocked_candidates"] == []
    assert "minimum_allocations=6" in data["reason"]


def test_geometric_shortfall_with_flag_reports_minimum_not_met(client):
    # Three collision groups: at most 3 pairs geometrically, with or
    # without takeover certification.
    arms, targets = [], []
    for g in range(3):
        gx = 200 * g
        arms += [{"id": f"L{g}", "x": gx, "y": 0, "max_extension": 100},
                 {"id": f"R{g}", "x": gx, "y": 10, "max_extension": 100}]
        targets += [{"id": f"lo{g}", "x": gx + 10, "y": 4, "priority": 5 + g},
                    {"id": f"hi{g}", "x": gx + 10, "y": 6, "priority": 1 + g}]
    body = {"arms": arms, "targets": targets,
            "clearance": 3, "minimum_allocations": 6,
            "single_target_loss_takeover": True}
    r = client.post("/api/v1/adjudicate", json=body)
    data = r.json()
    assert data["status"] == "minimum_not_met"
    assert data["feasible"] is False
    assert data["objective"]["num_allocations"] == 3
    assert data["witness"]["blocking_pair"] is not None


def test_non_boolean_flag_rejected(client):
    body = takeover_payload()
    body["single_target_loss_takeover"] = "yes"
    assert client.post("/api/v1/adjudicate", json=body).status_code == 422
