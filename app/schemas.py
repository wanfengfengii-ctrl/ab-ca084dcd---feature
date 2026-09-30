"""Request/response schemas with strict input validation.

Illegal payloads (wrong counts, non-positive values, duplicate ids, ...) are
rejected with HTTP 422 *before* the solver is invoked.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    # Strict typing: int fields reject strings/floats/booleans rather than
    # coercing them, so illegal payloads never reach the solver.
    model_config = ConfigDict(extra="forbid", strict=True)


class ArmIn(StrictModel):
    id: str = Field(..., min_length=1)
    x: int
    y: int
    max_extension: int = Field(..., ge=1)


class TargetIn(StrictModel):
    id: str = Field(..., min_length=1)
    x: int
    y: int
    priority: int = Field(..., ge=1)


class TakeoverOptions(StrictModel):
    # Opt-in: when omitted or false the request behaves exactly as before.
    enabled: bool = False


class AdjudicateRequest(StrictModel):
    arms: List[ArmIn] = Field(..., min_length=6, max_length=12)
    targets: List[TargetIn] = Field(..., min_length=6, max_length=16)
    clearance: int = Field(..., ge=1)
    minimum_allocations: int = Field(..., ge=1)
    takeover: Optional[TakeoverOptions] = None

    @model_validator(mode="after")
    def _unique_ids(self) -> "AdjudicateRequest":
        arm_ids = [a.id for a in self.arms]
        target_ids = [t.id for t in self.targets]
        if len(set(arm_ids)) != len(arm_ids):
            dup = sorted({x for x in arm_ids if arm_ids.count(x) > 1})
            raise ValueError(f"duplicate arm ids: {dup}")
        if len(set(target_ids)) != len(target_ids):
            dup = sorted({x for x in target_ids if target_ids.count(x) > 1})
            raise ValueError(f"duplicate target ids: {dup}")
        return self


class AssignmentOut(BaseModel):
    arm_id: str
    target_id: str
    extension: float
    extension_sq: int


class PairClearance(BaseModel):
    arm_a: str
    target_a: str
    arm_b: str
    target_b: str
    distance: float
    distance_exact: str
    distance_squared_exact: str
    required_clearance: int
    satisfied: bool
    witness: dict


class WitnessPair(BaseModel):
    arm_a: str
    target_a: str
    arm_b: str
    target_b: str
    distance: float
    distance_exact: str
    required_clearance: int
    witness: dict


class ShortfallWitness(BaseModel):
    blocking_pair: Optional[WitnessPair] = None
    capacity_limit: int
    reachable_arm_target_links: int


class ReassignmentOut(BaseModel):
    arm_id: str
    lost_target_id: str
    spare_target_id: str
    extension: float
    extension_sq: int
    priority: int


class FixedPairingOut(BaseModel):
    arm_id: str
    target_id: str
    extension_sq: int


class BlockedStandbyOut(BaseModel):
    arm_id: str
    spare_target_id: str
    extension_sq: int
    priority: int
    blocked_by_arm: str
    blocked_by_target: str
    clearance: dict


class TakeoverScenarioOut(BaseModel):
    arm_id: str
    lost_target_id: str
    reassignment: Optional[ReassignmentOut] = None
    fixed_pairings: List[FixedPairingOut]
    clearance_evidence: List[PairClearance]
    blocked_candidates: Optional[List[BlockedStandbyOut]] = None


class TakeoverReport(BaseModel):
    enabled: bool
    fully_certified: bool
    num_scenarios: int
    scenarios: List[TakeoverScenarioOut]
    first_blocking_scenario: Optional[TakeoverScenarioOut] = None
    max_certified_allocations: Optional[int] = None


class AdjudicateResponse(BaseModel):
    status: str  # "ok" | "minimum_not_met"
    feasible: bool
    reason: Optional[str] = None
    objective: dict
    assignments: List[AssignmentOut]
    unassigned_arms: List[str]
    unassigned_targets: List[str]
    arm_lengths: dict
    clearance: dict
    limits: dict
    witness: Optional[ShortfallWitness] = None
    takeover: Optional[TakeoverReport] = None
