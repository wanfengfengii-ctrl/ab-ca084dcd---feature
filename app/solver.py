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

Optional single-target-loss takeover certification
---------------------------------------------------
When ``require_takeover`` is enabled, the selected main plan must additionally
be *certifiable*: for every allocated edge ``(i, j)`` there exists a spare
target ``j'`` (reachable from arm ``i``, unused by the main plan) such that
reconnecting only arm ``i`` from the lost target ``j`` to ``j'`` keeps every
closed-segment clearance against the unchanged pairings.  The same spare may
back several different loss scenarios; it only has to stay free within one
scenario.  Certification is part of the branch-and-bound search (a relaxed
optimum that is conflict-free but not certifiable is branched further), so the
lexicographic order is optimised *among certifiable plans* in one pass rather
than patching an unconstrained optimum afterwards.

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
    takeover: Optional[dict] = None


INF = 10**100


class Solver:
    def __init__(
        self,
        arms: Sequence[Arm],
        targets: Sequence[Target],
        clearance: int,
        minimum_allocations: int,
        require_takeover: bool = False,
    ) -> None:
        self.arms = list(arms)
        self.targets = list(targets)
        self.clearance = clearance
        self.minimum_allocations = minimum_allocations
        self.require_takeover = require_takeover
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
        # spare_compat[(i,j')][(k,h)] == True when the standby edge (i->j')
        # respects the clearance against the fixed allocation (k->h).
        self._compat_cache: Dict[Tuple[Edge, Edge], bool] = {}
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

    # ------------------------------------------------------------------
    # Single-target-loss takeover certification
    # ------------------------------------------------------------------
    def _spare_ok(self, standby: Edge, fixed: Edge) -> bool:
        """Whether the standby segment ``standby`` (arm i -> unused target j')
        keeps the required clearance against the fixed segment ``fixed``."""
        key = (standby, fixed)
        ok = self._compat_cache.get(key)
        if ok is None:
            (i, jp), (k, h) = standby, fixed
            a, b = (i, k) if i < k else (k, i)
            t1, t2 = (jp, h) if i < k else (h, jp)
            ok = not self.pair_info(a, t1, b, t2)[0]
            self._compat_cache[key] = ok
        return ok

    def _spare_witness(self, standby: Edge, fixed: Edge) -> dict:
        """Full clearance evidence record for a standby/fixed segment pair."""
        (i, jp), (k, h) = standby, fixed
        a, b = (i, k) if i < k else (k, i)
        t1, t2 = (jp, h) if i < k else (h, jp)
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

    def _viable_spares(self, bad: Edge, fixed: List[Edge],
                       reserved: Dict[Edge, Edge]) -> List[Edge]:
        """Standby edges for ``bad``'s arm when its target alone is lost.

        A standby must be reachable, on ``bad``'s arm, unused by the fixed
        plan, and compatible with every other fixed segment (checked
        directly; an edge merely excluded from the *main* matching may still
        serve as a standby).  Ordered by the takeover tie-break.
        """
        i, j = bad
        used_targets = {h for _, h in fixed}
        reserved_spare = reserved.get(bad)
        if (reserved_spare is not None
                and reserved_spare[1] not in used_targets):
            # The reservation's closure already forbids every main edge that
            # could clash with the standby segment (and any other arm using
            # its target), so it stays a valid rescue in every descendant.
            return [reserved_spare]
        spares = []
        for jp in range(self.n_targets):
            if jp == j or jp in used_targets or self.len2[i][jp] is None:
                continue
            standby = (i, jp)
            if all(self._spare_ok(standby, e) for e in fixed if e != bad):
                spares.append(standby)
        spares.sort(key=lambda e: (-self.targets[e[1]].priority,
                                  self.len2[e[0]][e[1]], e[1]))
        return spares

    def _takeover_options(self, used_targets: Sequence[bool],
                         fixed: List[Edge]) -> Dict[Edge, List[Edge]]:
        """For every fixed allocation ``(i, j)`` the ordered list of viable
        standby edges ``(i, j')`` when target ``j`` alone is lost: ``j'`` must
        be reachable from arm ``i``, unused by the main plan, and compatible
        with every fixed segment.  Ordered per the takeover tie-break: spare
        target priority (descending), extension squared (ascending), target
        submission index (ascending)."""
        options: Dict[Edge, List[Edge]] = {}
        for (i, j) in fixed:
            spares = []
            for jp in range(self.n_targets):
                if jp == j or used_targets[jp] or self.len2[i][jp] is None:
                    continue
                standby = (i, jp)
                if all(self._spare_ok(standby, e) for e in fixed
                       if e != (i, j)):
                    spares.append(standby)
            spares.sort(key=lambda e: (-self.targets[e[1]].priority,
                                      self.len2[e[0]][e[1]], e[1]))
            options[(i, j)] = spares
        return options

    def _uncertified(self, chosen: List[Edge],
                     reserved: Dict[Edge, Edge]
                     ) -> Optional[Tuple[Edge, List[Edge]]]:
        """The first fixed allocation (arm submission order) that has no viable
        takeover, plus *all* reachable standby candidates for its arm
        (tie-break ordered).  Standby targets currently owned by another edge
        are included: reserving such a standby carries a closure that forbids
        the owner edge (and every edge clashing with the standby segment), so
        every candidate child strictly restricts the node."""
        if not self.require_takeover:
            return None
        chosen = sorted(chosen)
        for edge in chosen:
            if self._viable_spares(edge, chosen, reserved):
                continue
            i, j = edge
            # All reachable standbys are enumerated: standbys blocked by an
            # owner or a clashing fixed edge carry a closure that forbids that
            # edge, so each reservation genuinely restricts the node.
            candidates = [(i, jp) for jp in range(self.n_targets)
                          if jp != j and self.len2[i][jp] is not None]
            candidates.sort(key=lambda e: (-self.targets[e[1]].priority,
                                          self.len2[e[0]][e[1]], e[1]))
            return edge, candidates
        return None

    def _spare_closure(self, bad: Edge, spare: Edge) -> FrozenSet[Edge]:
        """Restrictions imposed on a main plan that keeps ``bad`` and rescues
        it with ``spare`` when its target is lost: arm ``bad``'s arm carries
        ``bad`` (every other edge on that arm is out), the spare target stays
        free (no other arm may use it), and nothing in the plan may collide
        with the standby segment."""
        bi, bj = bad
        si, sj = spare
        assert si == bi
        forced: set = set()
        for jj in range(self.n_targets):
            if jj != bj and self.len2[bi][jj] is not None:
                forced.add((bi, jj))
        for ii in range(self.n_arms):
            if ii != bi and self.len2[ii][sj] is not None:
                forced.add((ii, sj))
        for (k, h) in self.edges:
            if k == bi:
                continue
            a, b = (bi, k) if bi < k else (k, bi)
            t1, t2 = (sj, h) if bi < k else (h, sj)
            if self.pair_info(a, t1, b, t2)[0]:
                forced.add((k, h))
        return frozenset(forced)

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
    # Branch and bound on collision pairs (and takeover certification)
    # ------------------------------------------------------------------
    def solve(self) -> Tuple[Tuple[int, ...], bool, Optional[dict]]:
        # Feasible incumbent from deterministic greedy seeds (enables pruning
        # at the root); then exact collision branching proves optimality.
        if self.require_takeover:
            # Search only admits certifiable solutions; start from the best
            # certifiable incumbent (certified greedy passes + the trimmed
            # unconstrained seed) so the root prunes aggressively.
            cert_seeds = self._certified_greedy_seeds()
            cert_seeds.append(self._certifiable_seed(self._greedy_seed()))
            incumbent = max(cert_seeds, key=self.matching_weight)
        else:
            incumbent = self._greedy_seed()
        best_value = self.matching_weight(incumbent)
        best_assignment = incumbent[:]
        stats = {"nodes": 0}
        matching_cache: Dict[FrozenSet[Edge], Tuple[int, List[Edge]]] = {}
        Reserved = Dict[Edge, Edge]

        def recurse(forbidden: FrozenSet[Edge],
                    reserved: Reserved) -> None:
            nonlocal best_value, best_assignment
            stats["nodes"] += 1
            # A reservation is void once its main-plan edge is excluded.
            # (The reserved standby edge itself is expected to be forbidden
            # from the main matching -- that does not void it.)
            reserved = {b: s for b, s in reserved.items()
                        if b not in forbidden}

            cached = matching_cache.get(forbidden)
            if cached is not None:
                bound, chosen = cached
            else:
                bound, chosen = self._relaxed_match(forbidden)
                matching_cache[forbidden] = (bound, chosen)
            if bound <= best_value:
                return  # cannot improve the incumbent
            conflict = self._first_conflict(chosen)
            uncertified = (self._uncertified(chosen, reserved)
                           if conflict is None else None)
            if conflict is None and uncertified is None:
                # Relaxed optimum is collision-free and takeover-certified ->
                # the true optimum among certifiable plans at this node.
                assignment = [-1] * self.n_arms
                for i, j in chosen:
                    assignment[i] = j
                best_value = bound
                best_assignment = assignment
                return

            if conflict is not None:
                edge = self._branch_edge(chosen)
                ei, ej = edge
                # Branch 1: include ``edge`` -> forbid every other edge on
                # its arm or target, and every edge that collides with it.
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
                children_forbidden = [frozenset(forced),
                                      forbidden | {edge}]
                children_reserved = [reserved, reserved]
            else:
                # Collision-free but ``bad`` lacks a takeover.  Every
                # certifiable completion either reserves one of the spare
                # candidates (tie-break order; the closure forbids the spare
                # from the main plan and every segment clashing with it) or
                # drops ``bad``.  Each child strictly grows ``forbidden``.
                bad, candidates = uncertified
                children_forbidden = []
                children_reserved = []
                for spare in candidates:
                    new_reserved = dict(reserved)
                    new_reserved[bad] = spare
                    children_forbidden.append(
                        forbidden | self._spare_closure(bad, spare))
                    children_reserved.append(new_reserved)
                children_forbidden.append(forbidden | {bad})
                children_reserved.append(reserved)

            for child_forbidden, child_reserved in zip(children_forbidden,
                                                       children_reserved):
                recurse(child_forbidden, child_reserved)
                if best_value == bound:
                    return  # reached the node's upper bound

        recurse(frozenset(), {})
        self.solve_stats = stats

        best_tuple = tuple(best_assignment)
        count = sum(1 for x in best_tuple if x >= 0)
        witness = None
        if count < self.minimum_allocations:
            witness = self._shortfall_witness()
        return best_tuple, count >= self.minimum_allocations, witness

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

    def _certifiable_seed(self, seed: List[int]) -> List[int]:
        """Trim a feasible seed until every surviving allocation has a
        takeover.  Removing an edge only frees targets and removes fixed
        segments, so this terminates (the empty plan is trivially certifiable);
        it merely supplies a pruning incumbent for the certified search."""
        assignment = seed[:]
        while True:
            fixed = sorted((i, j) for i, j in enumerate(assignment) if j >= 0)
            used_targets = [False] * self.n_targets
            for _, j in fixed:
                used_targets[j] = True
            options = self._takeover_options(used_targets, fixed)
            bad_edge = next((e for e in fixed if not options[e]), None)
            if bad_edge is None:
                return assignment
            assignment[bad_edge[0]] = -1

    def _certified_greedy_seeds(self) -> List[List[int]]:
        """Deterministic greedy passes that only accept an edge when the whole
        partial plan stays collision-free *and* certifiable, yielding strong
        (often near-optimal) incumbents for aggressive pruning."""
        n, m = self.n_arms, self.n_targets
        seeds: List[List[int]] = []

        def run(arm_order, target_key):
            assignment = [-1] * n
            used = [False] * m
            placed: List[Edge] = []

            def stays_certified():
                used_flags = [False] * m
                for _, jj in placed:
                    used_flags[jj] = True
                options = self._takeover_options(used_flags, placed)
                return all(options[e] for e in placed)

            for i in arm_order:
                for j in sorted((jj for jj in range(m)
                                 if self.len2[i][jj] is not None
                                 and not used[jj]),
                                key=lambda jj, i=i: target_key(i, jj)):
                    clash = any(
                        self.pair_info(*self._ordered(i, j, k, h))[0]
                        for k, h in placed)
                    if clash:
                        continue
                    placed.append((i, j))
                    if stays_certified():
                        assignment[i] = j
                        used[j] = True
                    else:
                        placed.pop()
            return assignment

        seeds.append(run(range(n), lambda i, j: j))
        constrained = sorted(
            range(n),
            key=lambda i: (sum(1 for d2 in self.len2[i] if d2 is not None), i))
        seeds.append(run(constrained, lambda i, j: -self.targets[j].priority))
        seeds.append(run(constrained, lambda i, j: self.len2[i][j]))
        # Arms with many reachable options first: they are least likely to
        # monopolise a unique standby later.
        flexible = sorted(
            range(n),
            key=lambda i: (-sum(1 for d2 in self.len2[i] if d2 is not None), i))
        seeds.append(run(flexible, lambda i, j: j))
        return seeds

    @staticmethod
    def _ordered(i: int, j: int, k: int, h: int) -> Tuple[int, int, int, int]:
        a, b = (i, k) if i < k else (k, i)
        t1, t2 = (j, h) if i < k else (h, j)
        return a, t1, b, t2

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
    # Result assembly with evidence
    # ------------------------------------------------------------------
    def _scenario(self, edge: Edge, fixed: List[Edge],
                  spares: List[Edge], with_blockers: bool = False) -> dict:
        """Assemble the single-target-loss scenario for one fixed allocation.

        ``spares`` is the ordered list of viable standby edges (empty when the
        allocation cannot be certified).  With ``with_blockers`` the empty case
        additionally reports, for every free reachable standby candidate, the
        first fixed pairing that blocks it with exact clearance evidence.
        """
        i, j = edge
        fixed_pairs = [{
            "arm_id": self.arms[k].id,
            "target_id": self.targets[h].id,
            "extension_sq": self.len2[k][h],
        } for k, h in fixed if (k, h) != edge]

        reassignment = None
        evidence: List[dict] = []
        if spares:
            si, sj = spares[0]
            d2 = self.len2[si][sj]
            reassignment = {
                "arm_id": self.arms[si].id,
                "lost_target_id": self.targets[j].id,
                "spare_target_id": self.targets[sj].id,
                "extension": float(d2 ** 0.5),
                "extension_sq": d2,
                "priority": self.targets[sj].priority,
            }
            for e in fixed:
                if e == edge:
                    continue
                evidence.append(self._spare_witness((si, sj), e))

        blocked = None
        if with_blockers and not spares:
            used_targets = {h for _, h in fixed}
            candidates = []
            for jp in range(self.n_targets):
                if jp in used_targets or self.len2[i][jp] is None:
                    continue
                candidates.append((i, jp))
            candidates.sort(key=lambda e: (-self.targets[e[1]].priority,
                                          self.len2[e[0]][e[1]], e[1]))
            blocked = []
            for standby in candidates:
                si, sj = standby
                blocker = next(e for e in fixed
                               if e != edge and not self._spare_ok(standby, e))
                bk, bh = blocker
                blocked.append({
                    "arm_id": self.arms[si].id,
                    "spare_target_id": self.targets[sj].id,
                    "extension_sq": self.len2[si][sj],
                    "priority": self.targets[sj].priority,
                    "blocked_by_arm": self.arms[bk].id,
                    "blocked_by_target": self.targets[bh].id,
                    "clearance": self._spare_witness(standby, blocker),
                })

        return {
            "arm_id": self.arms[i].id,
            "lost_target_id": self.targets[j].id,
            "reassignment": reassignment,
            "fixed_pairings": fixed_pairs,
            "clearance_evidence": evidence,
            "blocked_candidates": blocked,
        }

    def _plain_optimum(self) -> Tuple[int, ...]:
        """The certification-free (but collision-aware) lexicographic optimum
        of the same instance, solved once and cached."""
        cached = getattr(self, "_plain_optimum_cache", None)
        if cached is None:
            plain = Solver(self.arms, self.targets, self.clearance,
                           self.minimum_allocations, require_takeover=False)
            cached, _, _ = plain.solve()
            self._plain_optimum_cache = cached
        return cached

    def build_takeover_report(self, assignment: Sequence[int],
                              feasible: bool) -> dict:
        """Per-target-loss report for the certified search result.

        Always lists one scenario per fixed allocation (arm submission order):
        the selected standby pairing, the unchanged fixed pairings and exact
        per-pair clearance evidence.  When the certified optimum is below the
        requested minimum, additionally reports that maximum certified
        allocation count and the first blocking scenario of a maximal
        conflict-free candidate, with every free reachable standby candidate
        and the fixed pairing that blocks it.
        """
        fixed = sorted((i, j) for i, j in enumerate(assignment) if j >= 0)
        used_targets = [False] * self.n_targets
        for _, j in fixed:
            used_targets[j] = True
        options = self._takeover_options(used_targets, fixed)
        scenarios = [self._scenario(e, fixed, options[e]) for e in fixed]
        fully = all(options[e] for e in fixed)

        first_blocking = None
        max_certified: Optional[int] = None
        if not feasible:
            # Exhibit why a larger plan cannot be certified.  The
            # certification-free optimum, if it exceeds the certified optimum,
            # must be non-certifiable, so its first uncertifiable scenario is
            # the blocking witness.
            plain_assignment = self._plain_optimum()
            candidate = sorted((i, j) for i, j in enumerate(plain_assignment)
                               if j >= 0)
            if len(candidate) > len(fixed):
                cand_used = [False] * self.n_targets
                for _, j in candidate:
                    cand_used[j] = True
                cand_options = self._takeover_options(cand_used, candidate)
                uncert = [e for e in candidate if not cand_options[e]]
                if uncert:
                    first_blocking = self._scenario(
                        uncert[0], candidate, [], with_blockers=True)
            max_certified = len(fixed)

        return {
            "enabled": True,
            "fully_certified": fully,
            "num_scenarios": len(scenarios),
            "scenarios": scenarios,
            "first_blocking_scenario": first_blocking,
            "max_certified_allocations": max_certified,
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
                conflict, dist_sq, w = self.pair_info(a, t1, b, t2)
                dist = dist_sq ** Fraction(1, 2)
                record = {
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
            elif self.require_takeover and \
                    sum(1 for x in self._plain_optimum() if x >= 0) \
                    < self.minimum_allocations:
                cause = ("pairwise segment clearance / reachability restricts "
                         "the compatible set below the requested minimum")
            elif self.require_takeover:
                cause = ("single-target-loss takeover certification "
                         "(every fixed pairing must keep a reachable spare "
                         "compatible with the unchanged pairings) restricts "
                         "the certifiable set below the requested minimum")
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
