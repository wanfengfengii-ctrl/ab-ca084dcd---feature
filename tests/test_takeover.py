"""Tests for single-target-loss takeover certification."""
from fractions import Fraction

from app.solver import Arm, Solver, Target


def assert_feasible(s, assignment, clearance):
    used = set()
    placed = []
    for i, j in enumerate(assignment):
        if j < 0:
            continue
        assert j not in used
        assert s.len2[i][j] is not None
        for k, h in placed:
            a, b = (i, k) if i < k else (k, i)
            t1, t2 = (j, h) if i < k else (h, j)
            conflict, d2, _ = s.pair_info(a, t1, b, t2)
            assert not conflict
            assert d2 >= Fraction(clearance ** 2)
        used.add(j)
        placed.append((i, j))
    return placed


def assert_certifiable(s, assignment):
    """Every fixed allocation has a free, reachable, compatible standby."""
    fixed = sorted((i, j) for i, j in enumerate(assignment) if j >= 0)
    used = [False] * s.n_targets
    for _, j in fixed:
        used[j] = True
    options = s._takeover_options(used, fixed)
    for edge in fixed:
        assert options[edge], f"{edge} has no takeover"
    return options


def far_arms(n=6):
    return [Arm(f"A{i}", i * 1000, 0, 500) for i in range(n)]


def filler_set(start_index=2, base_x=5000):
    """Four far-apart arms, each with its own main and a dedicated standby."""
    arms = [Arm(f"F{k}", base_x + k * 1000, 0, 60)
            for k in range(start_index, start_index + 4)]
    targets = []
    for k in range(start_index, start_index + 4):
        targets.append(Target(f"g{k}", base_x + k * 1000 + 10, 0, 5))
        targets.append(Target(f"h{k}", base_x + k * 1000 - 10, 0, 1))
    return arms, targets


def test_disabled_flag_keeps_plain_behaviour():
    arms = far_arms()
    targets = [Target(f"T{i}", i * 1000 + 10, 0, i + 1) for i in range(6)]
    s_plain = Solver(arms, targets, 5, 6, require_takeover=False)
    a_plain, ok_plain, _ = s_plain.solve()
    s_take = Solver(arms, targets, 5, 6, require_takeover=True)
    a_take, ok_take, _ = s_take.solve()
    assert ok_plain and tuple(a_plain) == tuple(range(6))
    # Same instance, no spare anywhere: certification must not certify the
    # plain optimum, so the certified maximum collapses to zero.
    assert not ok_take and sum(x >= 0 for x in a_take) == 0


def test_certified_plan_with_dedicated_spares():
    arms = far_arms()
    # Each arm has its main target and a far-away spare on the opposite side.
    targets = [Target(f"T{i}", i * 1000 + 10, 0, i + 1) for i in range(6)]
    targets += [Target(f"S{i}", i * 1000 - 10, 0, 1) for i in range(6)]
    s = Solver(arms, targets, 5, 6, require_takeover=True)
    a, ok, _ = s.solve()
    assert ok
    assert_feasible(s, a, 5)
    options = assert_certifiable(s, a)
    assert sum(x >= 0 for x in a) == 6
    # exactly the main targets are used, the spares stay free
    used = {j for j in a if j >= 0}
    assert used == set(range(6))
    # the selected standby for arm i is its dedicated spare S_i
    for i in range(6):
        assert options[(i, a[i])][0][1] == 6 + i


def test_certification_restricts_the_count():
    # Two cluster arms each have a main and a standby; four filler arms have
    # a single reachable target each and therefore can never be certified.
    arms = [Arm("L", 0, 0, 50), Arm("R", 1000, 0, 50)]
    arms += [Arm(f"X{k}", 5000 + k * 1000, 0, 50) for k in range(4)]
    targets = [
        Target("lo", 10, 0, 9),    # L main
        Target("ls", 12, 1, 1),    # L standby
        Target("ro", 1010, 0, 9),  # R main
        Target("rs", 1008, 1, 1),  # R standby
    ]
    targets += [Target(f"x{k}", 5000 + k * 1000 + 10, 0, 1) for k in range(4)]
    s_plain = Solver(arms, targets, 5, 6, require_takeover=False)
    a_plain, ok_plain, _ = s_plain.solve()
    assert ok_plain and a_plain[0] == 0 and a_plain[1] == 2
    s = Solver(arms, targets, 5, 6, require_takeover=True)
    a, ok, _ = s.solve()
    assert not ok
    assert_feasible(s, a, 5)
    options = assert_certifiable(s, a)
    # the two cluster arms survive with their standbys; the single-target
    # filler arms are all dropped.
    assert sum(x >= 0 for x in a) == 2
    assert options[(0, a[0])][0][1] == 1
    assert options[(1, a[1])][0][1] == 3


def test_same_spare_reused_across_scenarios_but_not_inside_one():
    # A0 and A1 both reach a single shared standby.  Across the two loss
    # scenarios that spare may serve either arm; within each scenario it is
    # free (the other arm keeps its own target).
    arms = [Arm("A0", 0, 0, 120), Arm("A1", 0, 200, 120)]
    fill_arms, fill_targets = filler_set()
    arms += fill_arms
    targets = [
        Target("m0", 50, 0, 5),
        Target("m1", 50, 200, 5),
        Target("shared", 0, 100, 1),
    ] + fill_targets
    s = Solver(arms, targets, 3, 6, require_takeover=True)
    a, ok, _ = s.solve()
    assert ok
    fixed = sorted((i, j) for i, j in enumerate(a) if j >= 0)
    used = [False] * s.n_targets
    for _, j in fixed:
        used[j] = True
    options = s._takeover_options(used, fixed)
    e0 = next(e for e in fixed if e[0] == 0)
    e1 = next(e for e in fixed if e[0] == 1)
    assert options[e0][0][1] == 2
    assert options[e1][0][1] == 2
    assert used[2] is False


def test_clearance_equality_is_takeover_allowed():
    # a0's standby segment runs horizontally at y=0 to the left; a1's fixed
    # main segment runs horizontally at y=3: the two are exactly 3 apart, and
    # equality must be acceptable for closed segments.
    arms = [Arm("a0", 0, 0, 60), Arm("a1", 0, 3, 60)]
    fill_arms, fill_targets = filler_set()
    arms += fill_arms
    targets = [
        Target("m0", 10, 0, 5),
        Target("m1", 10, 3, 5),
        Target("s0", -10, 0, 1),
        Target("s1", -10, 3, 1),
    ] + fill_targets
    s = Solver(arms, targets, 3, 6, require_takeover=True)
    a, ok, _ = s.solve()
    assert ok
    options = assert_certifiable(s, a)
    assert options[(0, a[0])][0][1] == 2
    # one unit tighter must invalidate it
    s2 = Solver(arms, targets, 4, 6, require_takeover=True)
    a2, ok2, _ = s2.solve()
    assert not ok2


def test_standby_tie_break_priority_then_extension_then_index():
    arms = [Arm("a0", 0, 0, 999)]
    fill_arms, fill_targets = filler_set()
    arms += fill_arms
    targets = [
        Target("m0", 100, 0, 50),
        Target("low_near", 10, 0, 1),
        Target("hi_far", 20, 0, 9),
        Target("hi_near", 8, 0, 9),       # same priority, shortest
        Target("hi_near_idx", 9, 0, 9),   # same priority, index tie-break
    ] + fill_targets
    s = Solver(arms, targets, 1, 5, require_takeover=True)
    a, ok, _ = s.solve()
    assert ok and a[0] == 0
    used = [False] * s.n_targets
    for j in a:
        if j >= 0:
            used[j] = True
    options = s._takeover_options(
        used, sorted((i, j) for i, j in enumerate(a) if j >= 0))
    spares = options[(0, 0)]
    assert [t for _, t in spares[:3]] == [3, 4, 2]


def test_takeover_report_contains_reassignment_fixed_and_evidence():
    arms = far_arms()
    targets = [Target(f"T{i}", i * 1000 + 10, 0, i + 1) for i in range(6)]
    targets += [Target(f"S{i}", i * 1000 - 10, 0, 1) for i in range(6)]
    s = Solver(arms, targets, 5, 6, require_takeover=True)
    a, ok, _ = s.solve()
    assert ok
    report = s.build_takeover_report(a, ok)
    assert report["fully_certified"] is True
    assert report["num_scenarios"] == 6
    for sc in report["scenarios"]:
        assert sc["reassignment"] is not None
        assert sc["reassignment"]["arm_id"] == sc["arm_id"]
        assert sc["reassignment"]["spare_target_id"].startswith("S")
        # the other five pairings are preserved exactly
        assert len(sc["fixed_pairings"]) == 5
        fixed_ids = {(p["arm_id"], p["target_id"])
                     for p in sc["fixed_pairings"]}
        plan = {(arms[i].id, targets[j].id)
                for i, j in enumerate(a)
                if j >= 0 and arms[i].id != sc["arm_id"]}
        assert fixed_ids == plan
        # five exact clearance records, all satisfied at >= clearance
        assert len(sc["clearance_evidence"]) == 5
        for ev in sc["clearance_evidence"]:
            assert ev["satisfied"] is True
            assert ev["distance"] + 1e-9 >= 5
            assert ev["witness"]["point_on_first"]


def test_report_when_no_certifiable_plan_gives_max_and_blocker():
    # Every arm reaches only its own target: certified maximum is 0 and the
    # first blocking scenario names the lost target with no reassignment.
    arms = far_arms()
    targets = [Target(f"T{i}", i * 1000 + 10, 0, i + 1) for i in range(6)]
    s = Solver(arms, targets, 5, 6, require_takeover=True)
    a, ok, _ = s.solve()
    assert not ok
    report = s.build_takeover_report(a, ok)
    assert report["fully_certified"] is True  # vacuous: zero allocations
    assert report["max_certified_allocations"] == 0
    fbs = report["first_blocking_scenario"]
    assert fbs is not None and fbs["reassignment"] is None
    assert fbs["lost_target_id"] == "T0"


def test_blocking_scenario_carries_exact_blocker_evidence():
    # A0's only standby clashes (below clearance 3) with A1's fixed segment;
    # the plain optimum has both arms, the certified optimum cannot.
    arms = [Arm("A0", 0, 0, 20), Arm("A1", 10, 0, 20)]
    fill_arms, fill_targets = filler_set(base_x=8000)
    arms += fill_arms
    targets = [
        Target("t0", 0, 5, 9),    # A0 main (short high-priority)
        Target("t1", 5, 0, 9),    # A1 main
        Target("t2", 5, -1, 5),   # A0 standby: 1 from t1 endpoint
    ] + fill_targets
    s = Solver(arms, targets, 3, 6, require_takeover=True)
    a, ok, _ = s.solve()
    assert not ok
    report = s.build_takeover_report(a, ok)
    fbs = report["first_blocking_scenario"]
    assert fbs is not None and fbs["arm_id"] == "A0"
    blocked = fbs["blocked_candidates"]
    assert blocked and blocked[0]["spare_target_id"] == "t2"
    assert blocked[0]["blocked_by_arm"] == "A1"
    cl = blocked[0]["clearance"]
    assert cl["satisfied"] is False
    assert cl["distance"] < 3
