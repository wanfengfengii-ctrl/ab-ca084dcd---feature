"""End-to-end API tests: validation gating, adjudication and evidence."""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def far_apart_payload(**overrides):
    payload = {
        "arms": [
            {"id": f"A{i}", "x": i * 100, "y": 0, "max_extension": 500}
            for i in range(6)
        ],
        "targets": [
            {"id": f"T{i}", "x": i * 100 + 10, "y": 10, "priority": i + 1}
            for i in range(6)
        ],
        "clearance": 5,
        "minimum_allocations": 6,
    }
    payload.update(overrides)
    return payload


def test_readiness_passed_startup_self_check(client):
    r = client.get("/health/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"
    assert client.get("/health/live").status_code == 200


def test_valid_request_returns_optimal_assignment_with_evidence(client):
    r = client.post("/api/v1/adjudicate", json=far_apart_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok" and body["feasible"] is True
    assert body["objective"]["num_allocations"] == 6
    assert body["objective"]["priority_sum"] == 21
    assert len(body["assignments"]) == 6
    assert body["unassigned_arms"] == []
    assert body["unassigned_targets"] == []
    # every submitted arm has a reported length
    assert set(body["arm_lengths"]) == {f"A{i}" for i in range(6)}
    # pairwise clearance evidence: 15 pairs, all satisfied
    evidence = body["clearance"]["pair_evidence"]
    assert len(evidence) == 15
    for p in evidence:
        assert p["satisfied"] is True
        assert p["distance"] + 1e-9 >= p["required_clearance"]
        assert p["witness"]["point_on_first"]
    assert body["clearance"]["all_pairs_satisfied"] is True


def test_minimum_not_met_reports_maximum_and_reason(client):
    # Three groups; within each group both pairings of two arms/targets
    # violate clearance 3 (converging gap 2 / crossing), so at most one
    # allocation per group -> maximum 3 although 6 are requested.
    arms, targets = [], []
    for g in range(3):
        gx = 200 * g
        arms += [{"id": f"L{g}", "x": gx, "y": 0, "max_extension": 100},
                 {"id": f"R{g}", "x": gx, "y": 10, "max_extension": 100}]
        targets += [{"id": f"lo{g}", "x": gx + 10, "y": 4, "priority": 5 + g},
                    {"id": f"hi{g}", "x": gx + 10, "y": 6, "priority": 1 + g}]
    payload = {"arms": arms, "targets": targets,
               "clearance": 3, "minimum_allocations": 6}
    r = client.post("/api/v1/adjudicate", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "minimum_not_met"
    assert body["feasible"] is False
    assert body["objective"]["num_allocations"] == 3
    assert "minimum_allocations=6" in body["reason"]
    assert "3" in body["reason"]
    assert body["witness"]["blocking_pair"] is not None
    bp = body["witness"]["blocking_pair"]
    assert bp["distance"] < bp["required_clearance"]
    # the returned partial assignment is itself collision free
    for p in body["clearance"]["pair_evidence"]:
        assert p["satisfied"] is True


def test_invalid_inputs_are_rejected_before_solving(client):
    bad_cases = [
        {"arms": far_apart_payload()["arms"][:5]},  # too few arms
        {"clearance": 0},
        {"clearance": -2},
        {"minimum_allocations": 0},
    ]
    for patch in bad_cases:
        payload = far_apart_payload()
        payload.update(patch)
        r = client.post("/api/v1/adjudicate", json=payload)
        assert r.status_code == 422, (patch, r.status_code)

    # non-positive priority embedded in a target
    payload = far_apart_payload()
    payload["targets"][0]["priority"] = 0
    assert client.post("/api/v1/adjudicate", json=payload).status_code == 422

    # duplicate target ids
    payload = far_apart_payload()
    payload["targets"][1]["id"] = payload["targets"][0]["id"]
    assert client.post("/api/v1/adjudicate", json=payload).status_code == 422

    # float where integer base coordinate is required
    payload = far_apart_payload()
    payload["arms"][0]["x"] = 1.5
    assert client.post("/api/v1/adjudicate", json=payload).status_code == 422

    # booleans must not sneak in as integers
    payload = far_apart_payload()
    payload["clearance"] = True
    assert client.post("/api/v1/adjudicate", json=payload).status_code == 422
    payload = far_apart_payload()
    payload["targets"][0]["priority"] = False
    assert client.post("/api/v1/adjudicate", json=payload).status_code == 422

    # unknown field
    payload = far_apart_payload()
    payload["bogus"] = 1
    assert client.post("/api/v1/adjudicate", json=payload).status_code == 422


def test_too_many_targets_rejected(client):
    payload = far_apart_payload()
    payload["targets"] = [
        {"id": f"T{i}", "x": i, "y": 0, "priority": 1} for i in range(17)
    ]
    assert client.post("/api/v1/adjudicate", json=payload).status_code == 422


def test_maximum_size_instance_accepted(client):
    payload = {
        "arms": [{"id": f"A{i}", "x": i * 40, "y": 0, "max_extension": 45}
                 for i in range(12)],
        "targets": [{"id": f"T{j}", "x": j * 30, "y": 6,
                     "priority": (j % 9) + 1} for j in range(16)],
        "clearance": 2,
        "minimum_allocations": 1,
    }
    r = client.post("/api/v1/adjudicate", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["objective"]["num_allocations"] >= 1
    for p in body["clearance"]["pair_evidence"]:
        assert p["distance"] + 1e-9 >= p["required_clearance"]
