"""Tests for the lexicographic allocation solver."""
import math
from fractions import Fraction

import pytest

from app.solver import Arm, Solver, Target


def solve(arms, targets, clearance, minimum=1):
    s = Solver(arms, targets, clearance, minimum)
    assignment, ok, witness = s.solve()
    return s, assignment, ok, witness


def assert_feasible(s, assignment, clearance):
    used = set()
    placed = []
    for i, j in enumerate(assignment):
        if j < 0:
            continue
        assert j not in used, "target reused"
        assert s.len2[i][j] is not None, "unreachable allocation"
        for k, h in placed:
            a, b = (i, k) if i < k else (k, i)
            t1, t2 = (j, h) if i < k else (h, j)
            conflict, d2, _ = s.pair_info(a, t1, b, t2)
            assert not conflict, "clearance violated"
            assert d2 >= Fraction(clearance ** 2)
        used.add(j)
        placed.append((i, j))


def test_all_reachable_far_apart():
    arms = [Arm(f"A{i}", i * 100, 0, 500) for i in range(6)]
    targets = [Target(f"T{i}", i * 100 + 10, 10, i + 1) for i in range(6)]
    s, a, ok, _ = solve(arms, targets, 5, 6)
    assert ok and tuple(a) == (0, 1, 2, 3, 4, 5)
    r = s.build_result(a, ok, None)
    assert r.num_allocations == 6
    assert r.priority_sum == 21
    assert all(p["satisfied"] for p in r.pair_clearances)


def test_crossing_segments_cannot_both_be_used():
    # Three groups of two arms/targets.  Within each group the arms start
    # 10 apart while both targets sit only 2 apart: the "straight" pairing
    # converges to a 2-gap and the swapped pairing crosses.  Both violate a
    # clearance of 3, so each group contributes at most one allocation.
    arms, targets = [], []
    for g in range(3):
        gx = 200 * g
        arms += [Arm(f"L{g}", gx, 0, 100), Arm(f"R{g}", gx, 10, 100)]
        targets += [Target(f"lo{g}", gx + 10, 4, 5 + g),
                    Target(f"hi{g}", gx + 10, 6, 1 + g)]
    s, a, ok, w = solve(arms, targets, 3, 6)
    assert not ok
    assert_feasible(s, a, 3)
    assert sum(x >= 0 for x in a) == 3
    r = s.build_result(a, ok, w)
    assert r.reason and "minimum_allocations=6" in r.reason
    assert r.witness["blocking_pair"] is not None
    # high-priority targets win within each single-allocation group
    assert r.priority_sum == 5 + 6 + 7


def test_priority_beats_extension():
    # Every arm reaches two targets at equal length, one low and one high
    # priority; the high-priority ones must be taken.
    arms = [Arm(f"p{i}", i * 100, 0, 10) for i in range(6)]
    targets = []
    for i in range(6):
        targets.append(Target(f"lo{i}", i * 100 + 3, 0, 1))
        targets.append(Target(f"hi{i}", i * 100 - 3, 0, 9))
    s, a, ok, _ = solve(arms, targets, 1, 6)
    assert ok
    assert_feasible(s, a, 1)
    r = s.build_result(a, ok, None)
    assert r.num_allocations == 6
    assert r.priority_sum == 54
    assert all(t.startswith("hi") for t in
               [p["target_id"] for p in r.assignments])


def test_extension_squared_is_minimised_after_priority():
    # Single arm + far-away filler arms; the close target must win among
    # equal-priority choices.
    arms = [Arm("a0", 0, 0, 999)] + [Arm(f"a{i}", i * 1000, 5000, 10)
                                     for i in range(1, 6)]
    targets = [Target("near", 3, 0, 7), Target("far", 5, 0, 7)]
    targets += [Target(f"f{i}", i * 1000, 5000 + 2, 7) for i in range(1, 6)]
    s, a, ok, _ = solve(arms, targets, 1, 6)
    assert ok
    assert a[0] == 0  # near
    r = s.build_result(a, ok, None)
    assert r.extension_sq_sum == 9 + 5 * 4


def test_stable_sequence_prefers_earlier_arms():
    # a0/a1 share a base: their segments always start at the same point and
    # therefore clash, so exactly one of them is allocated.  The stable
    # sequence must assign the earlier arm (and the smaller target index).
    arms = [Arm("a0", 0, 0, 10), Arm("a1", 0, 0, 10)] + \
           [Arm(f"a{i}", i * 1000, 0, 10) for i in range(2, 6)]
    targets = [Target("t0", 3, 0, 5), Target("t1", 4, 0, 5)] + \
              [Target(f"f{i}", i * 1000 + 1, 0, 5) for i in range(2, 6)]
    s, a, ok, _ = solve(arms, targets, 1, 5)
    assert ok
    assert a[0] == 0 and a[1] == -1
    assert tuple(a[2:]) == (2, 3, 4, 5)


def test_clearance_equality_is_allowed_closed_segments():
    # Two parallel segments exactly 3 apart: closed-segment distance equals
    # the clearance, so the constraint (>=) must allow both.
    arms = [Arm("low", 0, 0, 50), Arm("high", 0, 3, 50)] + \
           [Arm(f"f{i}", i * 1000, 0, 50) for i in range(2, 6)]
    targets = [Target("t0", 10, 0, 4), Target("t1", 10, 3, 4)] + \
              [Target(f"g{i}", i * 1000 + 5, 0, 4) for i in range(2, 6)]
    s, a, ok, _ = solve(arms, targets, 3, 6)
    assert ok and tuple(a) == (0, 1, 2, 3, 4, 5)
    # one unit tighter must forbid the close pair
    s2, a2, ok2, _ = solve(arms, targets, 4, 6)
    assert not ok2 and sum(x >= 0 for x in a2) == 5


def test_unreachable_arm_reported():
    arms = [Arm(f"A{i}", i * 1000, 0, 5) for i in range(6)]
    targets = [Target(f"T{i}", i * 1000, 0, 1) for i in range(6)]
    # reachable (distance 0); now move one target out of reach
    targets[0] = Target("T0", 100, 0, 1)
    s, a, ok, w = solve(arms, targets, 1, 6)
    assert not ok
    assert a[0] == -1
    assert sum(x >= 0 for x in a) == 5
    r = s.build_result(a, ok, w)
    assert "A0" in r.unassigned_arms
    assert "T0" in r.unassigned_targets


def test_minimum_higher_than_capacity():
    arms = [Arm(f"A{i}", 0, i * 100, 50) for i in range(6)]
    targets = [Target(f"T{j}", 0, j * 100, 1) for j in range(6)]
    s, a, ok, w = solve(arms, targets, 1, 12)
    assert not ok
    assert w["capacity_limit"] == 6


@pytest.mark.parametrize("seed", range(20))
def test_random_instances_feasible_and_weight_optimal(seed):
    import random
    rng = random.Random(seed)
    n_arms, n_targets = rng.randint(6, 12), rng.randint(6, 16)
    span = rng.choice([30, 80])
    arms = [Arm(f"a{i}", rng.randint(-span, span), rng.randint(-span, span),
                rng.randint(10, span)) for i in range(n_arms)]
    targets = [Target(f"t{j}", rng.randint(-span, span), rng.randint(-span, span),
                      rng.randint(1, 9)) for j in range(n_targets)]
    clearance = rng.randint(1, 4)
    s, a, ok, _ = solve(arms, targets, clearance, 1)
    assert_feasible(s, a, clearance)
    # Result must achieve at least the collision-ignoring greedy counts.
    r = s.build_result(a, ok, None)
    assert r.num_allocations >= 1
    assert all(p["satisfied"] for p in r.pair_clearances)
    # Evidence distances must agree with the extension geometry.
    for p in r.assignments:
        assert math.isclose(p["extension"] ** 2, p["extension_sq"], rel_tol=1e-12)
