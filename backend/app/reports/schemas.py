"""Report contracts following the spec §27.3 template.

A report is assembled from *verified* results only. It never sees the raw agent output,
so a stripped hypothesis cannot reappear here. Every row that states a fact carries the
evidence ID it came from.
"""

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from app.agents.schemas import EvidenceKind, FailureClass
from app.verification.schemas import DISCLAIMER, ClaimStatus, ReportConfidence

REPORT_TEMPLATE_VERSION = "aeb-diagnostic-report/1"


class ReportMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    report_id: str
    template_version: str = REPORT_TEMPLATE_VERSION
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    file_id: str
    quality_id: str
    scenario_id: str | None
    data_origin: str
    source_path: str
    run_id: str | None  # None when no agent diagnosis was run
    verification_id: str | None
    agent: str | None
    provider: str | None
    model: str | None


class TimelineEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    t_s: float
    evidence_id: str
    kind: EvidenceKind
    description: str


class MetricRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    metric_id: str | None  # None when the metric could not be computed
    name: str
    observed: str
    unit: str
    threshold: str
    passed: bool | None
    t_s: float | None
    missing_reason: str | None = None


class HypothesisRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    rank: int
    cause: str
    failure_class: FailureClass
    status: ClaimStatus
    confidence_label: ReportConfidence
    adjusted_confidence: float
    agent_confidence: float
    evidence_ids: tuple[str, ...]
    sources: tuple[str, ...]
    notes: tuple[str, ...]


class EvidenceAppendixEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    evidence_id: str
    kind: EvidenceKind
    source: str
    summary: str
    t_s: float | None


class ApprovalSection(BaseModel):
    """Spec §15: approvals record reviewer, timestamp, decision, reason. Empty until reviewed."""

    model_config = ConfigDict(frozen=True)

    status: str = "pending_review"
    human_review_required: bool
    review_reasons: tuple[str, ...]
    reviewer: str | None = None
    decision: str | None = None
    reason: str | None = None
    decided_at: datetime | None = None


class DiagnosticReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    metadata: ReportMetadata
    report_confidence: ReportConfidence
    executive_summary: tuple[str, ...]  # short factual sentences, each with its evidence IDs
    event_metadata: dict[str, str]
    timeline: tuple[TimelineEntry, ...]
    metrics_table: tuple[MetricRow, ...]
    hypotheses: tuple[HypothesisRow, ...]
    stripped_hypotheses: tuple[str, ...]  # "cause (reason)" for transparency
    missing_evidence: tuple[str, ...]
    recommended_next_tests: tuple[str, ...]
    limitations: tuple[str, ...]
    disclaimer: str = DISCLAIMER
    approval: ApprovalSection
    evidence_appendix: tuple[EvidenceAppendixEntry, ...]
    evidence_support_rate: float | None
    unsupported_claim_rate: float | None
