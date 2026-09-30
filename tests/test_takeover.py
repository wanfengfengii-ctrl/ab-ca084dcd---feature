"""Tests for single-target-loss takeover certification."""
import itertools
import random
from fractions import Fraction

import pytest

from app.solver import Arm, Solver, Target


def verify_scenarios(s, assignment, clearance):
    """Independently validate every reported takeover scenario."""
    placed = sorted((i, j) for i, j in enumerate(assignment) if j >= 0)
    used = {j for _, j in placed}
    scenarios = s.takeover_scenarios(assignment)
    assert len(scenarios) == len(placed)
    for sc in scenarios:
        i0 = next(i for i, j in placed
                  if s.arms[i].id == sc["arm_id"]
                  and s.targets[j].id == sc["lost_target_id"])
        j0 = next(j for i, j in placed if i == i0)
        fixed = [e for e in placed if e != (i0, j0)]

        # every other pair is preserved verbatim
        assert {(p["arm_id"], p["target_id"]) for p in sc["fixed_pairs"]} == {
            (s.arms[i].id, s.targets[j].id) for i, j in fixed}

        jp = next(j for j, t in enumerate(s.targets)
                  if t.id == sc["replacement"]["target_id"])
        assert jp not in used, "backup target already occupied in the plan"
        assert s.len2[i0][jp] is not None, "backup unreachable"
        assert sc["replacement"]["extension_sq"] == s.len2[i0][jp]

        for k, h in fixed:
            a, b = (i0, k) if i0 < k else (k, i0)
            t1, t2 = (jp, h) if i0 < k else (h, jp)
            conflict, d2, _ = s.pair_info(a, t1, b, t2)
            assert not conflict
            assert d2 >= Fraction(clearance ** 2)

        assert len(sc["pair_evidence"]) == len(fixed)
        for rec in sc["pair_evidence"]:
            assert rec["satisfied"] is True
            assert Fraction(rec["distance_squared_exact"]) >= \
                Fraction(rec["required_clearance"] ** 2)

        # tie-breaking: priority desc, extension^2 asc, target index asc
        cands = s.replacement_targets(placed, (i0, j0))
        key = lambda j: (-s.targets[j].priority, s.len2[i0][j], j)
        assert jp == min(cands, key=key)


def far_arms():
    return [Arm(f"A{i}", i * 100, 0, 500) for i in range(6)]


def far_targets_with_spares():
    targets = [Target(f"T{i}", i * 100 + 10, 10, 2) for i in range(6)]
    targets += [Target(f"U{i}", i * 100 + 10, -10, 1) for i in range(6)]
    return targets


def test_certified_plan_with_spares_keeps_full_size():
    s = Solver(far_arms(), far_targets_with_spares(), 5, 6)
    plain, _, _ = s.solve()
    assert tuple(plain) == tuple(range(6))
    cert = s.solve_certified()
    assert tuple(cert) == tuple(range(6))
    assert s.cert_stats.get("fast_path") is True
    verify_scenarios(s, cert, 5)


def test_backup_tie_breaking_priority_then_extension_then_index():
    # A0 has two unused reachable spares; highest priority wins.
    arms = [Arm("A0", 0, 0, 500)] + \
           [Arm(f"A{i}", i * 1000, 0, 500) for i in range(1, 6)]
    targets = [Target("T0", 10, 0, 9),
               Target("lo", 20, 0, 1), Target("hi", 5, 0, 5)]
    for i in range(1, 6):
        targets.append(Target(f"T{i}", i * 1000 + 10, 0, 9))
        targets.append(Target(f"U{i}", i * 1000 + 10, 5, 1))
    s = Solver(arms, targets, 1, 6)
    cert = s.solve_certified()
    sc = s.takeover_scenarios(cert)
    a0 = next(x for x in sc if x["arm_id"] == "A0")
    assert a0["replacement"]["target_id"] == "hi"  # priority 5 > 1

    # equal priority -> smaller squared extension (distance 5 vs 20)
    targets[1] = Target("lo", 20, 0, 5)
    s = Solver(arms, targets, 1, 6)
    cert = s.solve_certified()
    a0 = next(x for x in s.takeover_scenarios(cert) if x["arm_id"] == "A0")
    assert a0["replacement"]["target_id"] == "hi"


def test_same_spare_reused_across_scenarios_but_not_inside_one():
    arms = [Arm("A0", 0, 0, 2000), Arm("A1", 1000, 0, 2000)] + \
           [Arm(f"A{i}", i * 5000, 0, 100) for i in range(2, 6)]
    targets = [Target("T0", 5, 0, 5), Target("T1", 1005, 0, 5),
               Target("SP", 500, 0, 5)]
    for i in range(2, 6):
        targets.append(Target(f"T{i}", i * 5000 + 5, 0, 5))
        targets.append(Target(f"S{i}", i * 5000 + 8, 0, 5))
    s = Solver(arms, targets, 1, 6)
    cert = s.solve_certified()
    assert sum(x >= 0 for x in cert) == 6
    scenarios = s.takeover_scenarios(cert)
    sp = [x for x in scenarios if x["replacement"]["target_id"] == "SP"]
    assert len(sp) == 2  # reused in two different loss scenarios
    # inside each scenario the backup is never a fixed target
    used = {j for i, j in enumerate(cert) if j >= 0}
    sp_idx = next(j for j, t in enumerate(targets) if t.id == "SP")
    assert sp_idx not in used
    verify_scenarios(s, cert, 1)


def test_clearance_equality_is_certifiable():
    # backup and fixed segments exactly 5 apart -> takeover allowed
    arms = [Arm("A0", 0, 0, 500), Arm("A1", 0, 5, 500)]
    targets = [Target("T0", 8, 0, 1), Target("T1", 8, 5, 1),
               Target("S0", 10, 0, 1), Target("S1", 10, 5, 1)]
    for i in range(2, 6):
        bx = i * 1000
        arms.append(Arm(f"A{i}", bx, 0, 500))
        targets.append(Target(f"T{i}", bx + 8, 0, 1))
        targets.append(Target(f"S{i}", bx + 10, 0, 1))
    s = Solver(arms, targets, 5, 6)
    cert = s.solve_certified()
    assert sum(x >= 0 for x in cert) == 6
    sc0 = next(x for x in s.takeover_scenarios(cert)
               if x["lost_target_id"] == "T0")
    assert sc0["replacement"]["target_id"] == "S0"
    rec = next(e for e in sc0["pair_evidence"] if e["target_b"] == "T1")
    assert rec["satisfied"] is True
    assert Fraction(rec["distance_squared_exact"]) == 25

    # one unit tighter: the full (otherwise optimal) plan cannot be
    # certified; its first blocker names both reachable spares as blocked
    # by a fixed pair
    s6 = Solver(arms, targets, 6, 6)
    cert6 = s6.solve_certified()
    assert sum(x >= 0 for x in cert6) < 6
    forced_full = [0, 1, 4, 6, 8, 10]  # A0->T0, A1->T1, A2.. -> T2..T5
    blocker = s6.first_blocking_scenario(forced_full)
    assert blocker["arm_id"] == "A0" and blocker["lost_target_id"] == "T0"
    blocked_ids = {c["target_id"] for c in blocker["blocked_candidates"]}
    assert {"S0", "S1"} <= blocked_ids
    for cand in blocker["blocked_candidates"]:
        assert cand["blocked_by"] is not None
        assert cand["blocked_by"]["satisfied"] is False


def test_no_spare_targets_collapses_certified_size():
    # 6 arms / 6 targets, each arm only reaches its own far-away target:
    # with a full plan every arm has no backup, and dropping any pairs
    # leaves the surviving arms without any reachable unused target, so
    # even a single pair cannot be certified.
    arms = [Arm(f"A{i}", i * 100, 0, 50) for i in range(6)]
    targets = [Target(f"T{i}", i * 100 + 10, 10, 1) for i in range(6)]
    s = Solver(arms, targets, 5, 6)
    cert = s.solve_certified()
    assert sum(x >= 0 for x in cert) == 0
    verify_scenarios(s, cert, 5)

    plain, _, _ = s.solve()
    blocker = s.first_blocking_scenario(plain)
    assert blocker["arm_id"] == "A0" and blocker["lost_target_id"] == "T0"
    assert blocker["blocked_candidates"] == []
    assert "used by the main plan" in blocker["reason"]


def test_certified_optimality_matches_brute_force():
    def brute(s, n, m):
        best_key = None
        best = tuple([-1] * n)
        for combo in itertools.product(range(-1, m), repeat=n):
            used = set()
            placed = []
            ok = True
            ext = prio = 0
            for i, j in enumerate(combo):
                if j < 0:
                    continue
                if j in used or s.len2[i][j] is None:
                    ok = False
                    break
                for k, h in placed:
                    a, b = (i, k) if i < k else (k, i)
                    t1, t2 = (j, h) if i < k else (h, j)
                    if s.pair_info(a, t1, b, t2)[0]:
                        ok = False
                        break
                if not ok:
                    break
                used.add(j)
                placed.append((i, j))
                ext += s.len2[i][j]
                prio += s.targets[j].priority
            if not ok or not all(s.replacement_targets(placed, e)
                                 for e in placed):
                continue
            stab = tuple(j if j >= 0 else m for j in combo)
            key = (len(placed), prio, -ext, tuple(-x for x in stab))
            if best_key is None or key > best_key:
                best_key, best = key, combo
        return tuple(best)

    rng = random.Random(2026)
    for _ in range(25):
        n, m = rng.randint(2, 4), rng.randint(2, 5)
        cr = rng.choice([3, 6, 12])
        arms = [Arm(f"a{i}", rng.randint(-cr, cr), rng.randint(-cr, cr),
                    rng.randint(2, 8)) for i in range(n)]
        targets = [Target(f"t{j}", rng.randint(-cr, cr),
                          rng.randint(-cr, cr), rng.randint(1, 9))
                   for j in range(m)]
        cl = rng.randint(1, 4)
        s = Solver(arms, targets, cl, 1)
        assert tuple(s.solve_certified()) == brute(s, n, m)
