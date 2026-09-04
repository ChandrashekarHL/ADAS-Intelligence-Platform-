"""Request/response models for the HTTP API (spec §18)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.rag.schemas import AccessLevel


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class ProjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    created_at: datetime


class FileOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    original_name: str
    data_origin: str
    scenario_id: str | None
    row_count: int
    duration_s: float | None
    quality_id: str | None
    quality_verdict: str | None
    uploaded_at: datetime


class IngestionJobCreate(BaseModel):
    file_id: str


class IngestionJobOut(BaseModel):
    job_id: str
    file_id: str
    status: Literal["completed", "blocked"]
    quality_verdict: str
    events: int
    metrics_available: int
    metrics_missing: int
    primary_window_id: str | None


class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    file_id: str
    project_id: str
    event_type: str
    t_s: float
    description: str


class QueryCreate(BaseModel):
    project_id: str
    file_id: str
    question: str | None = None
    access_level: AccessLevel = AccessLevel.PUBLIC
    top_k: int = Field(default=6, ge=1, le=20)


class QueryOut(BaseModel):
    """Spec §18.2 contract plus the IDs needed to trace and follow up."""

    answer: str
    confidence: str
    evidence_ids: list[str]
    unsupported_claims: list[str]
    recommended_next_tests: list[str]
    human_review_required: bool
    run_id: str
    verification_id: str
    evidence_support_rate: float


class ReportCreate(BaseModel):
    project_id: str
    file_id: str
    run_id: str | None = None  # None → metrics-only report
    access_level: AccessLevel = AccessLevel.PUBLIC


class ReportOut(BaseModel):
    report_id: str
    project_id: str
    file_id: str
    run_id: str | None
    report_confidence: str
    approval_id: str
    human_review_required: bool
    markdown_url: str
    json_url: str


class ApprovalDecision(BaseModel):
    reviewer: str = Field(min_length=1, max_length=200)
    decision: Literal["approved", "rejected"]
    reason: str = Field(min_length=1)


class ApprovalOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    report_id: str
    status: str
    human_review_required: bool
    review_reasons: list[str]
    reviewer: str | None
    decision: str | None
    reason: str | None
    decided_at: datetime | None


class DashboardOut(BaseModel):
    projects: int
    files: int
    agent_runs: int
    reports: int
    reports_by_confidence: dict[str, int]
    approvals_pending: int
    avg_evidence_support_rate: float | None
    llm_prompt_tokens: int
    llm_completion_tokens: int
    llm_provider: str
    rag_index_loaded: bool


class MetricOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    file_id: str
    name: str
    value: float | None
    unit: str
    passed: bool | None
    t_s: float | None
    window_id: str | None


class SignalsOut(BaseModel):
    """Downsampled telemetry for plotting. Never the source of truth for metrics."""

    file_id: str
    rows_total: int
    step: int
    columns: list[str]
    data: dict[str, list[float | None]]


class RunSummaryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    file_id: str
    agent: str
    provider: str
    model: str
    question: str
    verification_id: str
    report_confidence: str
    human_review_required: bool
    evidence_support_rate: float
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    created_at: datetime


class RunDetailOut(RunSummaryOut):
    run: dict[str, object]
    verification: dict[str, object]


class ChunkOut(BaseModel):
    chunk_id: str
    document_title: str
    heading: str
    text: str
    source_type: str
    access_level: str
    version: str | None
    requirement_ids: list[str]


class ReportListItem(BaseModel):
    report_id: str
    project_id: str
    file_id: str
    run_id: str | None
    report_confidence: str
    created_at: datetime
    approval_id: str | None
    approval_status: str | None
