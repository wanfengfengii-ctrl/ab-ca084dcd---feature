"""Brute-force differential test for takeover-certified solving."""
import itertools
import random

from app.solver import Arm, Solver, Target


def stability_tuple(combo, n_targets):
    return tuple(j if j >= 0 else n_targets for j in combo)


def certifiable(s, combo):
    placed = [(i, j) for i, j in enumerate(combo) if j >= 0]
    used = {j for _, j in placed}
    for (i, j) in placed:
        ok = False
        for jp in range(s.n_targets):
            if jp == j or jp in used or s.len2[i][jp] is None:
                continue
            standby = (i, jp)
            if all(s._spare_ok(standby, e) for e in placed if e != (i, j)):
                ok = True
                break
        if not ok:
            return False
    return True


def feasible(s, combo):
    used = set()
    placed = []
    for i, j in enumerate(combo):
        if j < 0:
            continue
        if j in used or s.len2[i][j] is None:
            return False
        for e in placed:
            a, b = (i, e[0]) if i < e[0] else (e[0], i)
            t1, t2 = (j, e[1]) if i < e[0] else (e[1], j)
            if s.pair_info(a, t1, b, t2)[0]:
                return False
        used.add(j)
        placed.append((i, j))
    return True


def brute_certified(s, n_arms, n_targets):
    best = None
    for combo in itertools.product(range(-1, n_targets), repeat=n_arms):
        if not feasible(s, combo) or not certifiable(s, combo):
            continue
        ext = sum(s.len2[i][j] for i, j in enumerate(combo) if j >= 0)
        prio = sum(s.targets[j].priority for j in combo if j >= 0)
        n = sum(1 for j in combo if j >= 0)
        key = (n, prio, -ext,
               tuple(-x for x in stability_tuple(combo, n_targets)))
        if best is None or key > best[0]:
            best = (key, combo)
    return best[1] if best else tuple([-1] * n_arms)


def main():
    rng = random.Random(123)
    fails = 0
    for trial in range(400):
        n_arms = rng.randint(2, 5)
        n_targets = rng.randint(2, 6)
        cr = rng.choice([4, 8, 15])
        arms = [Arm(f"a{i}", rng.randint(-cr, cr), rng.randint(-cr, cr),
                    rng.randint(3, cr)) for i in range(n_arms)]
        targets = [Target(f"t{j}", rng.randint(-cr, cr),
                          rng.randint(-cr, cr), rng.randint(1, 9))
                   for j in range(n_targets)]
        clearance = rng.randint(1, 4)
        s = Solver(arms, targets, clearance, 1, require_takeover=True)
        a, ok, _ = s.solve()
        assert feasible(s, a), "infeasible result"
        assert certifiable(s, a), "uncertified result!"
        b = brute_certified(s, n_arms, n_targets)
        if tuple(a) != tuple(b):
            fails += 1
            print("MISMATCH", trial, "got", a, "want", b)
            if fails <= 3:
                print(" arms:", arms)
                print(" targets:", targets, "clearance", clearance)
    print(f"takeover fuzz: 400 trials, fails={fails}")


if __name__ == "__main__":
    main()
