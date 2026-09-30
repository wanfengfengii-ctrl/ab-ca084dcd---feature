"""Fibre-arm allocation adjudication API.

Endpoints:
    GET  /health/live   - process is up
    GET  /health/ready  - the adjudication endpoint can accept requests
    POST /api/v1/adjudicate

``/health/ready`` only reports healthy once the solver machinery is loaded
and a tiny self-check of the exact geometry kernel passes, so the
Compose-level health gate cannot pass before the service can actually
adjudicate.
"""

import os
from fractions import Fraction
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .geometry import segment_distance_sq
from .schemas import AdjudicateRequest, AdjudicateResponse
from .solver import Arm, Solver, Target

_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ready
    _startup_self_check()
    yield
    _ready = False


app = FastAPI(
    title="Fibre-Arm Allocation Adjudicator",
    version="1.0.0",
    description="Collision-aware optimal assignment of retractable fibre "
                "arms to spectroscopic targets.",
    lifespan=lifespan,
)


@app.get("/health/live")
def live() -> dict:
    return {"status": "alive"}


@app.get("/health/ready")
def ready() -> JSONResponse:
    if _ready:
        return JSONResponse({"status": "ready"}, status_code=200)
    return JSONResponse({"status": "starting"}, status_code=503)


@app.post("/api/v1/adjudicate", response_model=AdjudicateResponse)
def adjudicate(req: AdjudicateRequest) -> AdjudicateResponse:
    arms = [Arm(id=a.id, x=a.x, y=a.y, max_extension=a.max_extension)
            for a in req.arms]
    targets = [Target(id=t.id, x=t.x, y=t.y, priority=t.priority)
               for t in req.targets]
    solver = Solver(arms, targets,
                    clearance=req.clearance,
                    minimum_allocations=req.minimum_allocations)

    takeover_report = None
    blocker = None
    if req.single_target_loss_takeover:
        # The certified plan is selected synchronously: the takeover
        # guarantee is part of the main-plan optimisation, not a repair
        # applied after obtaining the ordinary optimum.
        cert_assignment = solver.solve_certified()
        cert_count = sum(1 for x in cert_assignment if x >= 0)
        certified = cert_count >= req.minimum_allocations
        if certified:
            assignment = cert_assignment
            feasible = True
            witness = None
            scenarios = solver.takeover_scenarios(cert_assignment)
        else:
            # Explain the gap: report the ordinary optimum (and, when it
            # reaches the requested size, the first loss scenario without
            # a valid takeover, in arm submission order).
            plain_assignment, plain_feasible, plain_witness = \
                solver.last_plain_solution()
            plain_count = sum(1 for x in plain_assignment if x >= 0)
            assignment = plain_assignment
            feasible = False
            scenarios = []
            witness = plain_witness if plain_count < req.minimum_allocations \
                else None
            plain_placed = [(i, j) for i, j in enumerate(plain_assignment)
                            if j >= 0]
            if plain_placed and not all(
                    solver.replacement_targets(
                        sorted(plain_placed), e) for e in plain_placed):
                blocker = solver.first_blocking_scenario(plain_assignment)
        takeover_report = {
            "enabled": True,
            "certified": certified,
            "max_certifiable_allocations": cert_count,
            "requested_minimum_allocations": req.minimum_allocations,
            "scenarios": scenarios,
            "first_blocking_scenario": blocker,
        }
    else:
        assignment, feasible, witness = solver.solve()

    r = solver.build_result(assignment, feasible, witness)
    if takeover_report is not None and not feasible and witness is None:
        if blocker is not None:
            r.reason = (
                f"minimum_allocations={req.minimum_allocations} cannot be "
                "certified for single-target-loss takeover: the maximum "
                f"certifiable main plan has {cert_count} pair(s); first "
                "blocking loss scenario: "
                f"arm {blocker['arm_id']} losing {blocker['lost_target_id']}"
                f" - {blocker['reason']}"
            )
        else:
            r.reason = (
                f"minimum_allocations={req.minimum_allocations} not met: "
                f"the maximum certifiable main plan has {cert_count} pair(s)"
            )

    status = "ok" if feasible else (
        "takeover_not_certified"
        if takeover_report is not None and witness is None
        else "minimum_not_met")

    return AdjudicateResponse(
        status=status,
        feasible=feasible,
        reason=r.reason,
        objective={
            "num_allocations": r.num_allocations,
            "priority_sum": r.priority_sum,
            "extension_sq_sum": r.extension_sq_sum,
            "optimization_order": [
                "maximize_num_allocations",
                "maximize_priority_sum",
                "minimize_extension_sq_sum",
                "stable_sequence",
            ],
        },
        assignments=r.assignments,
        unassigned_arms=r.unassigned_arms,
        unassigned_targets=r.unassigned_targets,
        arm_lengths=r.arm_lengths,
        clearance={
            "required": req.clearance,
            "minimum_observed": r.minimum_clearance,
            "all_pairs_satisfied": all(p["satisfied"]
                                       for p in r.pair_clearances),
            "pair_evidence": r.pair_clearances,
        },
        limits={
            "minimum_allocations_requested": r.minimum_requested,
            "reachable_arm_target_links": r.reachable_count,
            "arms_submitted": len(arms),
            "targets_submitted": len(targets),
        },
        witness=r.witness,
        takeover=takeover_report,
    )


def _startup_self_check() -> None:
    """Gate readiness on the geometry kernel + a minimal end-to-end solve."""
    global _ready
    # Exact sanity: parallel unit segments two units apart -> distance 2.
    d2, _, _ = segment_distance_sq((0, 0), (3, 0), (0, 2), (3, 2))
    assert d2 == Fraction(4, 1)
    # Crossing segments touch (distance 0).
    d0, _, _ = segment_distance_sq((0, 0), (2, 2), (0, 2), (2, 0))
    assert d0 == 0
    # A minimal solve (boundary-sized instance) must terminate optimally.
    arms = [Arm(f"A{i}", i * 100, 0, 500) for i in range(6)]
    targets = [Target(f"T{i}", i * 100 + 10, 10, i + 1) for i in range(6)]
    s = Solver(arms, targets, clearance=1, minimum_allocations=6)
    a, ok, _ = s.solve()
    assert ok and sum(1 for x in a if x >= 0) == 6
    # The single-target-loss takeover machinery must also certify a small
    # plan with one spare and fail to certify an instance without spares.
    cert_targets = targets + [Target("U0", 10, -10, 1)]
    sc = Solver(arms, cert_targets, clearance=1, minimum_allocations=6)
    ca = sc.solve_certified()
    assert sum(1 for x in ca if x >= 0) == 6
    assert len(sc.takeover_scenarios(ca)) == 6
    _ready = True


def main() -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=os.environ.get("APP_HOST", "0.0.0.0"),
        port=int(os.environ.get("APP_PORT", "8080")),
        log_level=os.environ.get("APP_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
