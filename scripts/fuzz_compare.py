"""Brute-force differential test + performance scenarios."""
import itertools
import random
import time

from app.solver import Arm, Solver, Target


def stability_tuple(combo, n_targets):
    return tuple(j if j >= 0 else n_targets for j in combo)


def brute(s, n_arms, n_targets, targets):
    best = None
    for combo in itertools.product(range(-1, n_targets), repeat=n_arms):
        used = set()
        placed = []
        ok = True
        ext = 0
        prio = 0
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
            prio += targets[j].priority
        if not ok:
            continue
        key = (len(placed), prio, -ext, tuple(-x for x in stability_tuple(combo, n_targets)))
        if best is None or key > best[0]:
            best = (key, combo)
    return best[1]


def random_instance(rng, n_arms=None, n_targets=None):
    n_arms = n_arms or rng.randint(2, 4)
    n_targets = n_targets or rng.randint(2, 5)
    cr = rng.choice([3, 6, 12])
    arms = [Arm(f"a{i}", rng.randint(-cr, cr), rng.randint(-cr, cr),
                rng.randint(2, 8)) for i in range(n_arms)]
    targets = [Target(f"t{j}", rng.randint(-cr, cr), rng.randint(-cr, cr),
                      rng.randint(1, 9)) for j in range(n_targets)]
    clearance = rng.randint(1, 4)
    return arms, targets, clearance


def brute_certified(s, n_arms, n_targets, targets):
    best = None
    for combo in itertools.product(range(-1, n_targets), repeat=n_arms):
        used = set()
        placed = []
        ok = True
        ext = 0
        prio = 0
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
            prio += targets[j].priority
        if not ok:
            continue
        if not all(s.replacement_targets(placed, e) for e in placed):
            continue
        key = (len(placed), prio, -ext,
               tuple(-x for x in stability_tuple(combo, n_targets)))
        if best is None or key > best[0]:
            best = (key, combo)
    return best[1] if best else [-1] * n_arms


def main():
    rng = random.Random(42)
    fails = 0
    slow = []
    for trial in range(300):
        arms, targets, clearance = random_instance(rng)
        s = Solver(arms, targets, clearance, 1)
        t0 = time.time()
        a, _, _ = s.solve()
        dt = time.time() - t0
        if dt > 1.0:
            slow.append((trial, dt, s.solve_stats["nodes"]))
        b = brute(s, len(arms), len(targets), targets)
        if tuple(a) != tuple(b):
            fails += 1
            print("MISMATCH", trial, "got", a, "want", b)
            if fails <= 3:
                print(" arms:", arms)
                print(" targets:", targets, "clearance", clearance)
    print(f"fuzz: 300 trials, fails={fails}, slow={slow}")

    cert_fails = 0
    for trial in range(150):
        arms, targets, clearance = random_instance(rng)
        s = Solver(arms, targets, clearance, 1)
        a = s.solve_certified()
        b = brute_certified(s, len(arms), len(targets), targets)
        if tuple(a) != tuple(b):
            cert_fails += 1
            print("CERT MISMATCH", trial, "got", a, "want", b)
            if cert_fails <= 3:
                print(" arms:", arms)
                print(" targets:", targets, "clearance", clearance)
    print(f"cert fuzz: 150 trials, fails={cert_fails}")
    fails += cert_fails

    random.seed(1)
    arms = [Arm(f"a{i}", random.randint(0, 40), random.randint(0, 40),
                random.randint(15, 40)) for i in range(12)]
    targets = [Target(f"t{j}", random.randint(0, 40), random.randint(0, 40),
                      random.randint(1, 9)) for j in range(16)]
    for cl in (1, 3, 6, 10):
        s = Solver(arms, targets, cl, 1)
        t0 = time.time()
        a, ok, _ = s.solve()
        print(f"dense cl={cl}: n={sum(x >= 0 for x in a)} "
              f"nodes={s.solve_stats['nodes']} time={time.time()-t0:.3f}s")

    arms = [Arm(f"a{i}", i, 0, 100) for i in range(12)]
    targets = [Target(f"t{j}", j % 4, j // 4, 9) for j in range(16)]
    s = Solver(arms, targets, 1, 1)
    t0 = time.time()
    a, _, _ = s.solve()
    print(f"grid: n={sum(x >= 0 for x in a)} nodes={s.solve_stats['nodes']} "
          f"time={time.time()-t0:.3f}s")


if __name__ == "__main__":
    main()
