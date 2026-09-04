"""Verification contracts: what survived, what was removed, and why.

The verifier is the authority between the agent and the report. Its output is itself an
evidence artifact (``verification_…``) so a report can cite *why* its confidence is what
it is.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.agents.schemas import EvidenceKind, Hypothesis

DISCLAIMER = (
    "AIP provides engineering assistance, not certified safety tooling. Safety-critical "
    "conclusions require review by a qualified engineer. Findings derived from simulation "
    "or synthetic data do not transfer to road safety without validation on real data. "
    "AIP aligns with the vocabulary of ISO 26262 and ISO 21448 (SOTIF) but certifies "
    "against neither."
)

# Metrics without which an AEB late-braking diagnosis cannot be asserted (spec §28.1:
# critical signal missing → Low / Blocked).
CRITICAL_METRICS: frozenset[str] = frozenset(
    {"braking_latency_s", "ttc_threshold_crossing_s", "brake_command_time_s", "min_ttc_s"}
)


class ReportConfidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BLOCKED = "blocked"


_ORDER = {
    ReportConfidence.BLOCKED: 0,
    ReportConfidence.LOW: 1,
    ReportConfidence.MEDIUM: 2,
    ReportConfidence.HIGH: 3,
}


def cap(level: ReportConfidence, ceiling: ReportConfidence) -> ReportConfidence:
    return level if _ORDER[level] <= _ORDER[ceiling] else ceiling


def band(score: float) -> ReportConfidence:
    if score >= 0.7:
        return ReportConfidence.HIGH
    if score >= 0.4:
        return ReportConfidence.MEDIUM
    return ReportConfidence.LOW


class ClaimStatus(StrEnum):
    SUPPORTED = "supported"  # every cited ID resolves, timestamped evidence present
    PARTIAL = "partial"  # some IDs dropped, still timestamped support
    UNSUPPORTED = "unsupported"  # removed from the ranked list


class EvidenceRef(BaseModel):
    """A resolved evidence artifact as the verifier sees it."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str
    kind: EvidenceKind
    source: str  # "telemetry", "quality", "file", or "doc:<source_type>"
    summary: str
    t_s: float | None = None
    passed: bool | None = None
    stale: bool = False
    injection_flagged: bool = False


class AppliedRule(BaseModel):
    model_config = ConfigDict(frozen=True)

    rule: str
    effect: str
    detail: str
    hypothesis_index: int | None = None


class VerifiedHypothesis(BaseModel):
    model_config = ConfigDict(frozen=True)

    hypothesis: Hypothesis  # as the agent produced it
    status: ClaimStatus
    resolved_ids: tuple[str, ...]
    dropped_ids: tuple[str, ...]
    independent_sources: tuple[str, ...]
    agent_confidence: float
    adjusted_confidence: float
    confidence_label: ReportConfidence
    notes: tuple[str, ...]


class StrippedHypothesis(BaseModel):
    model_config = ConfigDict(frozen=True)

    hypothesis: Hypothesis
    reason: str


class VerificationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    verification_id: str
    run_id: str
    report_confidence: ReportConfidence
    human_review_required: bool
    review_reasons: tuple[str, ...]
    hypotheses: tuple[VerifiedHypothesis, ...]  # ranked by adjusted confidence
    stripped: tuple[StrippedHypothesis, ...]
    applied_rules: tuple[AppliedRule, ...]
    flagged_observations: tuple[str, ...]
    missing_evidence: tuple[str, ...]  # agent's list ∪ pipeline's unavailable metrics
    recommended_next_tests: tuple[str, ...]
    limitations: tuple[str, ...]
    disclaimer: str = DISCLAIMER
    cited_ids_total: int
    cited_ids_resolved: int
    evidence_support_rate: float = Field(ge=0, le=1)
    unsupported_claim_rate: float = Field(ge=0, le=1)

    @property
    def top(self) -> VerifiedHypothesis | None:
        return self.hypotheses[0] if self.hypotheses else None

    @property
    def cited_evidence_ids(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for h in self.hypotheses:
            for i in h.resolved_ids:
                seen.setdefault(i, None)
        return tuple(seen)
