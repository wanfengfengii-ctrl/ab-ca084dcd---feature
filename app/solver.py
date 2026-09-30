"""Fibre-arm / target allocation solver.

Given fibre arms (integer base + maximum extension) and unique targets with
positive integer priorities, find the lexicographically optimal assignment:

    1. maximise the number of allocations;
    2. maximise the sum of priorities of allocated targets;
    3. minimise the sum of squared arm extensions;
    4. the stable sequence: among remaining ties, prefer allocating the
       earlier arm (submission order), and to the lowest-indexed target.
       Concretely minimise the arm-order tuple whose entry is the assigned
       target index and ``n_targets`` for an unused arm.

Hard constraints:

* an arm connects to at most one reachable target (distance <= reach);
* a target is used at most once;
* the closed segment of every pair of allocations keeps at least ``clearance``
  distance from every other allocation segment.

Algorithm
---------
The whole lexicographic objective is encoded as one *integer* edge weight
(with dominance constants derived from the instance bounds), so a single
maximum-weight bipartite matching (Hungarian method) solves the relaxed
problem where pairwise collisions are ignored.  Branch-and-bound then only
branches on collisions: if the relaxed matching contains a conflicting pair
of allocations, every feasible solution omits at least one of the two edges,
so we recurse with one edge forbidden at a time.  The relaxed matching value
is an upper bound used for pruning.  All geometry stays exact (Fraction).
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from .geometry import segment_distance_sq

TargetId = str
ArmId = str
Edge = Tuple[int, int]  # (arm index, target index)


@dataclass(frozen=True)
class Target:
    id: TargetId
    x: int
    y: int
    priority: int


@dataclass(frozen=True)
class Arm:
    id: ArmId
    x: int
    y: int
    max_extension: int


@dataclass
class SolveResult:
    assignments: List[dict]
    unassigned_arms: List[ArmId]
    unassigned_targets: List[TargetId]
    arm_lengths: Dict[ArmId, float]
    pair_clearances: List[dict]
    minimum_clearance: Optional[float]
    num_allocations: int
    priority_sum: int
    extension_sq_sum: int
    reachable_count: int
    minimum_requested: int
    feasible: bool
    reason: Optional[str] = None
    witness: Optional[dict] = None


INF = 10**100


class Solver:
    def __init__(
        self,
        arms: Sequence[Arm],
        targets: Sequence[Target],
        clearance: int,
        minimum_allocations: int,
    ) -> None:
        self.arms = list(arms)
        self.targets = list(targets)
        self.clearance = clearance
        self.minimum_allocations = minimum_allocations
        self.n_arms = len(self.arms)
        self.n_targets = len(self.targets)
        self.clearance_sq = Fraction(clearance * clearance)

        # len2[i][j] is None when target j is unreachable for arm i.
        self.len2: List[List[Optional[int]]] = [
            [None] * self.n_targets for _ in range(self.n_arms)
        ]
        self.edges: List[Edge] = []
        for i, arm in enumerate(self.arms):
            for j, tgt in enumerate(self.targets):
                d2 = (arm.x - tgt.x) ** 2 + (arm.y - tgt.y) ** 2
                if d2 <= arm.max_extension**2:
                    self.len2[i][j] = d2
                    self.edges.append((i, j))
        self.total_reachable = len(self.edges)

        self._pair_cache: Dict[Tuple[int, int, int, int],
                               Tuple[bool, Fraction, dict]] = {}
        self._make_weights()

    # ------------------------------------------------------------------
    # Lexicographic objective as one integer edge weight
    # ------------------------------------------------------------------
    def _make_weights(self) -> None:
        n, m = self.n_arms, self.n_targets
        base = m + 1  # stability digit base (digits 0..m)

        # Stability magnitude bound: sum of digit values * place value.
        place = [base ** (n - 1 - i) for i in range(n)]
        s_max = m * sum(place)

        # Extension magnitude bound (worst case all arms at max reachable d2).
        e_max = sum(
            max((d2 for d2 in row if d2 is not None), default=0)
            for row in self.len2
        )
        p_max = n * max((t.priority for t in self.targets), default=1)

        # Maximise: C3*count + C2*priority - C1*ext2 + stability_term.
        self.c1 = s_max + 1
        self.c2 = e_max * self.c1 + s_max + 1
        self.c3 = p_max * self.c2 + e_max * self.c1 + s_max + 1
        self._place = place

        self.weight: Dict[Edge, int] = {}
        for i, j in self.edges:
            stability_gain = (m - j) * place[i]  # vs. leaving arm i unused
            self.weight[(i, j)] = (
                self.c3
                + self.c2 * self.targets[j].priority
                - self.c1 * self.len2[i][j]
                + stability_gain
            )

    def matching_weight(self, assignment: Sequence[int]) -> int:
        return sum(self.weight[(i, j)] for i, j in enumerate(assignment) if j >= 0)

    # ------------------------------------------------------------------
    # Pairwise segment feasibility (exact arithmetic)
    # ------------------------------------------------------------------
    def pair_info(self, i: int, j: int, k: int, h: int):
        """Info for simultaneous allocations (i->j),(k->h), with i < k."""
        key = (i, j, k, h)
        cached = self._pair_cache.get(key)
        if cached is not None:
            return cached

        arm_i, tgt_j = self.arms[i], self.targets[j]
        arm_k, tgt_h = self.arms[k], self.targets[h]
        dist_sq, u, v = segment_distance_sq(
            (arm_i.x, arm_i.y), (tgt_j.x, tgt_j.y),
            (arm_k.x, arm_k.y), (tgt_h.x, tgt_h.y),
        )
        witness = {
            "point_on_first": [
                _frac_str(arm_i.x + u * (tgt_j.x - arm_i.x)),
                _frac_str(arm_i.y + u * (tgt_j.y - arm_i.y)),
            ],
            "point_on_second": [
                _frac_str(arm_k.x + v * (tgt_h.x - arm_k.x)),
                _frac_str(arm_k.y + v * (tgt_h.y - arm_k.y)),
            ],
        }
        result = (dist_sq < self.clearance_sq, dist_sq, witness)
        self._pair_cache[key] = result
        return result

    def _first_conflict(self, used_edges: List[Edge]
                        ) -> Optional[Tuple[Edge, Edge]]:
        used_edges = sorted(used_edges)
        for idx in range(len(used_edges)):
            for jdx in range(idx + 1, len(used_edges)):
                (i, j), (k, h) = used_edges[idx], used_edges[jdx]
                if self.pair_info(i, j, k, h)[0]:
                    return used_edges[idx], used_edges[jdx]
        return None

    def _branch_edge(self, chosen: List[Edge]) -> Edge:
        """Pick the chosen edge participating in the most internal clashes;
        ties broken by arm then target order."""
        degrees = {}
        chosen_sorted = sorted(chosen)
        for idx in range(len(chosen_sorted)):
            for jdx in range(idx + 1, len(chosen_sorted)):
                (i, j), (k, h) = chosen_sorted[idx], chosen_sorted[jdx]
                if self.pair_info(i, j, k, h)[0]:
                    degrees[(i, j)] = degrees.get((i, j), 0) + 1
                    degrees[(k, h)] = degrees.get((k, h), 0) + 1
        if not degrees:
            return chosen[0]
        return min(degrees, key=lambda e: (-degrees[e], e))

    # ------------------------------------------------------------------
    # Relaxed maximum-weight matching with a forbidden-edge set
    # ------------------------------------------------------------------
    def _relaxed_match(self, forbidden: FrozenSet[Edge]
                       ) -> Tuple[int, List[Edge]]:
        n, m = self.n_arms, self.n_targets
        size = max(n, m, 1)
        cost = [[0] * size for _ in range(size)]
        for (i, j), w in self.weight.items():
            if (i, j) not in forbidden:
                cost[i][j] = -w  # Hungarian minimises
        row_owner, _ = _hungarian(cost)
        chosen: List[Edge] = []
        total = 0
        for i in range(n):
            j = row_owner[i]
            if 0 <= j < m and self.len2[i][j] is not None and (i, j) not in forbidden:
                chosen.append((i, j))
                total += self.weight[(i, j)]
        return total, chosen

    # ------------------------------------------------------------------
    # Branch and bound on collision pairs
    # ------------------------------------------------------------------
    def solve(self) -> Tuple[Tuple[int, ...], bool, Optional[dict]]:
        # Feasible incumbent from deterministic greedy seeds (enables pruning
        # at the root); then exact collision branching proves optimality.
        incumbent = self._greedy_seed()
        best_value = self.matching_weight(incumbent)
        best_assignment = incumbent[:]
        stats = {"nodes": 0}
        memo: Dict[FrozenSet[Edge], Tuple[int, List[Edge]]] = {}

        def recurse(forbidden: FrozenSet[Edge]) -> None:
            nonlocal best_value, best_assignment
            stats["nodes"] += 1
            cached = memo.get(forbidden)
            if cached is not None:
                bound, chosen = cached
            else:
                bound, chosen = self._relaxed_match(forbidden)
                memo[forbidden] = (bound, chosen)
            if bound <= best_value:
                return  # cannot improve the incumbent
            if self._first_conflict(chosen) is None:
                # Relaxed optimum is collision-free -> true optimum of node.
                assignment = [-1] * self.n_arms
                for i, j in chosen:
                    assignment[i] = j
                best_value = bound
                best_assignment = assignment
                return

            edge = self._branch_edge(chosen)
            ei, ej = edge
            # Branch 1: include ``edge`` -> forbid every other edge on its
            # arm or target, and every edge in the whole graph that collides
            # with it.
            forced: set = set(forbidden)
            for jj in range(self.n_targets):
                if jj != ej and self.len2[ei][jj] is not None:
                    forced.add((ei, jj))
            for ii in range(self.n_arms):
                if ii != ei and self.len2[ii][ej] is not None:
                    forced.add((ii, ej))
            for (k, h) in self.edges:
                if k == ei:
                    continue
                a, b = (ei, k) if ei < k else (k, ei)
                t1, t2 = (ej, h) if ei < k else (h, ej)
                if self.pair_info(a, t1, b, t2)[0]:
                    forced.add((k, h))
            recurse(frozenset(forced))
            if best_value == bound:
                return  # reached the node's upper bound
            # Branch 2: exclude ``edge``.
            recurse(forbidden | {edge})

        recurse(frozenset())
        self.solve_stats = stats

        best_tuple = tuple(best_assignment)
        count = sum(1 for x in best_tuple if x >= 0)
        witness = None
        if count < self.minimum_allocations:
            witness = self._shortfall_witness()
        result = (best_tuple, count >= self.minimum_allocations, witness)
        self._last_plain = result
        return result

    def last_plain_solution(self) -> Tuple[Tuple[int, ...], bool,
                                           Optional[dict]]:
        """Cached result of the latest :meth:`solve` call (run it if the
        cached value is absent)."""
        cached = getattr(self, "_last_plain", None)
        if cached is None:
            return self.solve()
        return cached

    # ------------------------------------------------------------------
    # Greedy feasible seeds
    # ------------------------------------------------------------------
    def _greedy_seed(self) -> List[int]:
        """Best of several deterministic greedy, always-feasible passes."""
        n, m = self.n_arms, self.n_targets
        candidates: List[List[int]] = []

        def greedy(arm_order, target_key):
            assignment = [-1] * n
            used = [False] * m
            placed: List[Edge] = []
            for i in arm_order:
                for j in sorted((j for j in range(m)
                                 if self.len2[i][j] is not None and not used[j]),
                                key=lambda j, i=i: target_key(i, j)):
                    clash = False
                    for k, h in placed:
                        a, b = (i, k) if i < k else (k, i)
                        t1, t2 = (j, h) if i < k else (h, j)
                        if self.pair_info(a, t1, b, t2)[0]:
                            clash = True
                            break
                    if not clash:
                        assignment[i] = j
                        used[j] = True
                        placed.append((i, j))
                        break
            return assignment

        candidates.append(greedy(range(n), lambda i, j: j))
        # Process the most constrained arms first.
        constrained = sorted(
            range(n),
            key=lambda i: (sum(1 for d2 in self.len2[i] if d2 is not None), i))
        candidates.append(greedy(constrained,
                                 lambda i, j: -self.targets[j].priority))
        candidates.append(greedy(constrained, lambda i, j: self.len2[i][j]))
        return max(candidates, key=self.matching_weight)

    # ------------------------------------------------------------------
    # Shortfall explanation
    # ------------------------------------------------------------------
    def _shortfall_witness(self) -> dict:
        """Structural evidence when the minimum cannot be met: the tightest
        violating pair among individually reachable allocations (if any)."""
        blocking = None
        best_dist: Optional[Fraction] = None
        for i in range(self.n_arms):
            for k in range(i + 1, self.n_arms):
                for j in range(self.n_targets):
                    if self.len2[i][j] is None:
                        continue
                    for h in range(self.n_targets):
                        if h == j or self.len2[k][h] is None:
                            continue
                        conflict, dist_sq, w = self.pair_info(i, j, k, h)
                        if conflict and (best_dist is None or dist_sq > best_dist):
                            best_dist = dist_sq
                            blocking = {
                                "arm_a": self.arms[i].id,
                                "target_a": self.targets[j].id,
                                "arm_b": self.arms[k].id,
                                "target_b": self.targets[h].id,
                                "distance_exact": _sqrt_str(dist_sq),
                                "distance": float(dist_sq ** Fraction(1, 2)),
                                "required_clearance": self.clearance,
                                "witness": w,
                            }
        return {
            "blocking_pair": blocking,
            "capacity_limit": min(self.n_arms, self.n_targets),
            "reachable_arm_target_links": self.total_reachable,
        }

    # ------------------------------------------------------------------
    # Single-target-loss takeover certification
    # ------------------------------------------------------------------
    # A main plan is *certifiable* when every allocated edge (i0, j0) has a
    # takeover target j' that is unused by the whole main plan, reachable from
    # arm i0, and keeps the required clearance from every fixed pair (all
    # pairs except (i0, j0)).  The same spare target may serve as backup in
    # several different loss scenarios; it only has to stay unoccupied inside
    # one concrete scenario.
    def replacement_targets(self, placed: Sequence[Edge],
                            edge: Edge) -> List[int]:
        """Candidate substitute target indices for losing ``edge``.

        ``placed`` is the main plan (including ``edge`` itself).  A candidate
        must be unused by the main plan, reachable from the arm that owned the
        lost target, and clear of every fixed pair.  Equality with the
        clearance limit counts as feasible (conflict is strict ``<``).
        """
        i0, _ = edge
        used = {j for _, j in placed}
        fixed = [g for g in placed if g != edge]
        candidates: List[int] = []
        for jp in range(self.n_targets):
            if jp in used or self.len2[i0][jp] is None:
                continue
            clash = False
            for (k, h) in fixed:
                a, b = (i0, k) if i0 < k else (k, i0)
                t1, t2 = (jp, h) if i0 < k else (h, jp)
                if self.pair_info(a, t1, b, t2)[0]:
                    clash = True
                    break
            if not clash:
                candidates.append(jp)
        return candidates

    @staticmethod
    def _substitute_key(targets: Sequence[Target], len2, arm: int):
        # Scenario tie-breaking: priority desc, squared extension asc,
        # then submitted target index asc.
        return lambda jp: (-targets[jp].priority, len2[arm][jp], jp)

    def _cert_plan_is_valid(self, placed: List[Edge]) -> bool:
        return all(self.replacement_targets(placed, e) for e in placed)

    def _cert_seed_from_plan(self, assignment: Sequence[int]) -> List[int]:
        """Drop non-certifiable edges from a feasible plan until what remains
        is certifiable (a feasible lower bound for the branch-and-bound)."""
        result = list(assignment)
        while True:
            placed = sorted((i, j) for i, j in enumerate(result) if j >= 0)
            failing = next((e for e in placed
                            if not self.replacement_targets(placed, e)), None)
            if failing is None:
                return result
            result[failing[0]] = -1

    def solve_certified(self) -> Tuple[int, ...]:
        """Lexicographically optimal *certifiable* main plan.

        Same four-level order as :meth:`solve`, but the search space is
        restricted to plans where each placed edge keeps at least one live
        takeover target.  Branch-and-bound extends the collision branching:

        * viability prune -- once an edge is forced in, its possible backup
          set can only shrink as more edges join, so a forced edge without a
          geometrically possible backup prunes the whole node;
        * rescue branching -- when the collision-free relaxed optimum places
          an edge ``e`` without any surviving backup, a certifiable solution
          either drops ``e`` or keeps one concrete backup target alive (which
          forbids every relaxed-optimum edge that occupies that target or
          collides with the backup segment).  Branches yielding the same
          forbidden set are merged: the backup target itself is never part
          of the plan, so only the existence of *some* surviving backup
          matters.
        """
        # Fast path: the ordinary optimum is usually certifiable.  Solving it
        # first also yields a strong feasible seed for the restricted search.
        plain, _, _ = self.solve()
        plain_placed = sorted((i, j) for i, j in enumerate(plain) if j >= 0)
        if all(self.replacement_targets(plain_placed, e) for e in plain_placed):
            self.cert_stats = {"nodes": 0, "fast_path": True}
            return plain

        seeds = [self._cert_seed_from_plan(plain)]
        seeds += [
            self._cert_greedy_seed(),
        ]
        best_assignment = max(seeds, key=self.matching_weight)
        best_value = self.matching_weight(best_assignment)
        relaxed_memo: Dict[FrozenSet[Edge], Tuple[int, List[Edge]]] = {}
        viability_memo: Dict[FrozenSet[Edge], bool] = {}
        stats = {"nodes": 0}

        def viable(included: FrozenSet[Edge]) -> bool:
            cached = viability_memo.get(included)
            if cached is not None:
                return cached
            ok = True
            if included:
                placed = sorted(included)
                ok = all(self.replacement_targets(placed, e) for e in placed)
            viability_memo[included] = ok
            return ok

        def recurse(forbidden: FrozenSet[Edge],
                    included: FrozenSet[Edge]) -> None:
            nonlocal best_value, best_assignment
            stats["nodes"] += 1
            if not viable(included):
                return
            cached = relaxed_memo.get(forbidden)
            if cached is not None:
                bound, chosen = cached
            else:
                bound, chosen = self._relaxed_match(forbidden)
                relaxed_memo[forbidden] = (bound, chosen)
            if bound <= best_value:
                return
            if self._first_conflict(chosen) is not None:
                edge = self._branch_edge(chosen)
                ei, ej = edge
                forced: set = set(forbidden)
                for jj in range(self.n_targets):
                    if jj != ej and self.len2[ei][jj] is not None:
                        forced.add((ei, jj))
                for ii in range(self.n_arms):
                    if ii != ei and self.len2[ii][ej] is not None:
                        forced.add((ii, ej))
                for (k, h) in self.edges:
                    if k == ei:
                        continue
                    a, b = (ei, k) if ei < k else (k, ei)
                    t1, t2 = (ej, h) if ei < k else (h, ej)
                    if self.pair_info(a, t1, b, t2)[0]:
                        forced.add((k, h))
                recurse(frozenset(forced), included | {edge})
                if best_value == bound:
                    return
                recurse(forbidden | {edge}, included)
                return

            # Collision-free relaxed optimum -> true optimum of this node
            # among ordinary plans; it additionally has to be certifiable.
            chosen_sorted = sorted(chosen)
            failing = next(
                (e for e in chosen_sorted
                 if not self.replacement_targets(chosen_sorted, e)),
                None,
            )
            if failing is None:
                assignment = [-1] * self.n_arms
                for i, j in chosen:
                    assignment[i] = j
                best_value = bound
                best_assignment = assignment
                return

            ei, ej = failing
            possible = self.replacement_targets(
                sorted(included | {failing}), failing)
            # Each rescue branch keeps ``failing`` and reserves one concrete
            # backup target jp: forbid every other edge on arm ei / target ej
            # (failing is forced), every edge occupying jp, and every edge in
            # the whole graph whose segment collides with the would-be backup
            # segment.  Any certifiable solution that keeps failing certifies
            # it via exactly such a jp, so the branches plus the final
            # exclude-failing branch exhaust the solution space.  Identical
            # forbidden sets (different jp, same effect) are merged.
            branch_sets: Dict[FrozenSet[Edge], int] = {}
            for jp in possible:
                forb: set = set(forbidden)
                for jj in range(self.n_targets):
                    if jj != ej and self.len2[ei][jj] is not None:
                        forb.add((ei, jj))
                for ii in range(self.n_arms):
                    if ii != ei and self.len2[ii][ej] is not None:
                        forb.add((ii, ej))
                for ii in range(self.n_arms):
                    if ii != ei and self.len2[ii][jp] is not None:
                        forb.add((ii, jp))
                for (k, h) in self.edges:
                    if k == ei:
                        continue
                    a, b = (ei, k) if ei < k else (k, ei)
                    t1, t2 = (jp, h) if ei < k else (h, jp)
                    if self.pair_info(a, t1, b, t2)[0]:
                        forb.add((k, h))
                bf = frozenset(forb)
                # The reserved backup edge (ei, jp) itself is reachable and
                # differs from failing, so at least that edge is newly
                # forbidden; a non-growing branch would mean jp already
                # certified failing, contradicting its selection.
                if bf == forbidden:
                    continue
                key_jp = (-self.targets[jp].priority, self.len2[ei][jp], jp)
                if bf not in branch_sets:
                    branch_sets[bf] = key_jp
                else:
                    branch_sets[bf] = min(branch_sets[bf], key_jp)
            # Most promising rescue first: fewest added restrictions, then
            # the best backup representative by scenario priority.
            ordered = sorted(
                branch_sets,
                key=lambda bf: (len(bf), branch_sets[bf]))
            for forb in ordered:
                recurse(forb, included | {failing})
                if best_value == bound:
                    return
            # Last resort: the uncertifiable edge must leave the plan.
            recurse(forbidden | {failing}, included)

        recurse(frozenset(), frozenset())
        stats["fast_path"] = False
        self.cert_stats = stats
        return tuple(best_assignment)

    def _cert_greedy_seed(self) -> List[int]:
        """Deterministic greedy seeds that are certifiable at every step."""
        n, m = self.n_arms, self.n_targets

        def greedy(arm_order, target_key) -> List[int]:
            assignment = [-1] * n
            used = [False] * m
            placed: List[Edge] = []
            for i in arm_order:
                for j in sorted((j for j in range(m)
                                 if self.len2[i][j] is not None and not used[j]),
                                key=lambda j, i=i: target_key(i, j)):
                    clash = False
                    for k, h in placed:
                        a, b = (i, k) if i < k else (k, i)
                        t1, t2 = (j, h) if i < k else (h, j)
                        if self.pair_info(a, t1, b, t2)[0]:
                            clash = True
                            break
                    if clash:
                        continue
                    trial = placed + [(i, j)]
                    if all(self.replacement_targets(trial, e)
                           for e in trial):
                        assignment[i] = j
                        used[j] = True
                        placed = trial
                        break
            return assignment

        candidates = [greedy(range(n), lambda i, j: j)]
        constrained = sorted(
            range(n),
            key=lambda i: (sum(1 for d2 in self.len2[i] if d2 is not None), i))
        candidates.append(greedy(constrained,
                                 lambda i, j: -self.targets[j].priority))
        candidates.append(greedy(constrained, lambda i, j: self.len2[i][j]))
        return max(candidates, key=self.matching_weight)

    def takeover_scenarios(self, assignment: Sequence[int]) -> List[dict]:
        """Per-lost-target scenario report for a certified main plan."""
        placed = sorted((i, j) for i, j in enumerate(assignment) if j >= 0)
        scenarios: List[dict] = []
        for (i0, j0) in placed:
            fixed = [e for e in placed if e != (i0, j0)]
            candidates = self.replacement_targets(placed, (i0, j0))
            key = self._substitute_key(self.targets, self.len2, i0)
            jp = min(candidates, key=key)
            evidence = [self._clearance_record(i0, jp, k, h)
                        for (k, h) in fixed]
            scenarios.append({
                "arm_id": self.arms[i0].id,
                "lost_target_id": self.targets[j0].id,
                "replacement": {
                    "arm_id": self.arms[i0].id,
                    "target_id": self.targets[jp].id,
                    "extension": float(self.len2[i0][jp] ** 0.5),
                    "extension_sq": self.len2[i0][jp],
                },
                "fixed_pairs": [
                    {"arm_id": self.arms[k].id,
                     "target_id": self.targets[h].id}
                    for (k, h) in fixed
                ],
                "pair_evidence": evidence,
            })
        return scenarios

    def first_blocking_scenario(self, assignment: Sequence[int]) -> dict:
        """First loss scenario (arm submission order) without a takeover.

        Built from an unconstrained optimum that reaches the requested size,
        to explain why that plan cannot be certified.  Lists every unused
        reachable target together with the first fixed pair that blocks it.
        """
        placed = sorted((i, j) for i, j in enumerate(assignment) if j >= 0)
        used = {j for _, j in placed}
        for (i0, j0) in placed:
            fixed = [e for e in placed if e != (i0, j0)]
            reachable_unused = [jp for jp in range(self.n_targets)
                                if jp not in used
                                and self.len2[i0][jp] is not None]
            blocked: List[dict] = []
            for jp in reachable_unused:
                blocker = None
                for (k, h) in fixed:
                    a, b = (i0, k) if i0 < k else (k, i0)
                    t1, t2 = (jp, h) if i0 < k else (h, jp)
                    conflict, _, _ = self.pair_info(a, t1, b, t2)
                    if conflict:
                        blocker = self._clearance_record(i0, jp, k, h)
                        break
                if blocker is None:
                    break  # a valid takeover exists -> scenario is certified
                blocked.append({
                    "target_id": self.targets[jp].id,
                    "extension_sq": self.len2[i0][jp],
                    "blocked_by": blocker,
                })
            else:
                if not reachable_unused:
                    reason = ("no substitute target: every target is already "
                              "used by the main plan or unreachable from "
                              f"arm {self.arms[i0].id}")
                else:
                    reason = (
                        f"arm {self.arms[i0].id} losing target "
                        f"{self.targets[j0].id} has no reachable unused "
                        "target clear of all fixed pairs"
                    )
                return {
                    "arm_id": self.arms[i0].id,
                    "lost_target_id": self.targets[j0].id,
                    "reason": reason,
                    "blocked_candidates": blocked,
                }
        raise ValueError("assignment is certifiable; no blocking scenario")

    # ------------------------------------------------------------------
    # Result assembly with evidence
    # ------------------------------------------------------------------
    def _clearance_record(self, i: int, j: int, k: int, h: int) -> dict:
        """Pairwise clearance evidence for allocations (i->j),(k->h)."""
        a, b = (i, k) if i < k else (k, i)
        t1, t2 = (j, h) if i < k else (h, j)
        conflict, dist_sq, w = self.pair_info(a, t1, b, t2)
        dist = dist_sq ** Fraction(1, 2)
        return {
            "arm_a": self.arms[a].id,
            "target_a": self.targets[t1].id,
            "arm_b": self.arms[b].id,
            "target_b": self.targets[t2].id,
            "distance": float(dist),
            "distance_exact": _sqrt_str(dist_sq),
            "distance_squared_exact": _frac_str(dist_sq),
            "required_clearance": self.clearance,
            "satisfied": not conflict,
            "witness": w,
        }

    def build_result(self, assignment: Sequence[int], feasible: bool,
                     witness: Optional[dict]) -> SolveResult:
        pairs: List[dict] = []
        used_targets = set()
        arm_lengths: Dict[ArmId, float] = {}
        ext_sq_sum = 0
        prio_sum = 0
        placed: List[Edge] = []
        for i, j in enumerate(assignment):
            if j < 0:
                continue
            arm, tgt = self.arms[i], self.targets[j]
            length = (self.len2[i][j] or 0) ** 0.5
            pairs.append({
                "arm_id": arm.id,
                "target_id": tgt.id,
                "extension": float(length),
                "extension_sq": self.len2[i][j],
            })
            arm_lengths[arm.id] = float(length)
            used_targets.add(j)
            ext_sq_sum += self.len2[i][j] or 0
            prio_sum += tgt.priority
            placed.append((i, j))

        pair_clearances: List[dict] = []
        min_d2: Optional[Fraction] = None
        for idx in range(len(placed)):
            for jdx in range(idx + 1, len(placed)):
                i, j = placed[idx]
                k, h = placed[jdx]
                a, b = (i, k) if i < k else (k, i)
                t1, t2 = (j, h) if i < k else (h, j)
                _, dist_sq, _ = self.pair_info(a, t1, b, t2)
                record = self._clearance_record(i, j, k, h)
                pair_clearances.append(record)
                if min_d2 is None or dist_sq < min_d2:
                    min_d2 = dist_sq

        unassigned_arms = [self.arms[i].id
                           for i, j in enumerate(assignment) if j < 0]
        unassigned_targets = [self.targets[j].id
                              for j in range(self.n_targets)
                              if j not in used_targets]

        count = len(pairs)
        if not feasible:
            if min(self.n_arms, self.n_targets) < self.minimum_allocations:
                cause = "fewer arms/targets than the requested minimum"
            elif self.total_reachable < self.minimum_allocations:
                cause = "not every arm has a reachable target"
            else:
                cause = ("pairwise segment clearance / reachability restricts "
                         "the compatible set below the requested minimum")
            reason = (
                f"minimum_allocations={self.minimum_allocations} not met: the "
                f"maximum non-colliding reachable allocation has {count} "
                f"pair(s); {cause}"
            )
        else:
            reason = None

        return SolveResult(
            assignments=pairs,
            unassigned_arms=unassigned_arms,
            unassigned_targets=unassigned_targets,
            arm_lengths=arm_lengths,
            pair_clearances=pair_clearances,
            minimum_clearance=(float(min_d2 ** Fraction(1, 2))
                               if min_d2 is not None else None),
            num_allocations=count,
            priority_sum=prio_sum,
            extension_sq_sum=ext_sq_sum,
            reachable_count=self.total_reachable,
            minimum_requested=self.minimum_allocations,
            feasible=feasible,
            reason=reason,
            witness=witness,
        )


# ----------------------------------------------------------------------
# Hungarian method (minimisation, square integer cost matrix)
# ----------------------------------------------------------------------
def _hungarian(cost: List[List[int]]) -> Tuple[List[int], List[int]]:
    """Matched column per row and matched row per column (1-indexed
    potentials formulation)."""
    n = len(cost)
    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)  # p[col] = row matched to column
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used_col = [False] * (n + 1)
        while True:
            used_col[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, n + 1):
                if not used_col[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used_col[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    row_owner = [-1] * n
    col_owner = [-1] * n
    for j in range(1, n + 1):
        i = p[j]
        if i >= 1:
            row_owner[i - 1] = j - 1
            col_owner[j - 1] = i - 1
    return row_owner, col_owner


# ----------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------
def _frac_str(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _sqrt_str(value: Fraction) -> str:
    """Exact rendering of a distance whose square is the rational ``value``."""
    if value == 0:
        return "0"
    # Render rational roots directly when the fraction is a perfect square.
    num_sq = _isqrt(value.numerator)
    den_sq = _isqrt(value.denominator)
    if num_sq * num_sq == value.numerator and den_sq * den_sq == value.denominator:
        q = Fraction(num_sq, den_sq)
        return _frac_str(q)
    return f"sqrt({_frac_str(value)})"


def _isqrt(n: int) -> int:
    """Integer square root (floor) via Newton's method."""
    if n < 2:
        return n
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x
