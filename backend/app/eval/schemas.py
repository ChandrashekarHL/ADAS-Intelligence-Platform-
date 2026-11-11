"""Pydantic models for the evaluation suite: cases, expectations, results, gate."""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.agents.schemas import FailureClass
from app.metrics.schemas import EventType
from app.quality.report import QualityVerdict
from app.synthetic.schemas import AebScenarioConfig
from app.verification.schemas import ReportConfidence


class ExpectedOutcome(StrEnum):
    """What a correct diagnosis of the case looks like."""

    FAULT = "fault"  # a requirement is violated; the agent must name an acceptable cause
    NOMINAL = "nominal"  # everything within spec; the agent must not invent a fault
    BLOCKED = "blocked"  # data-quality gates must refuse the file before any analysis


class EvalCase(BaseModel):
    """One locked scenario plus the expectations it is scored against.

    Expectations are pinned by hand from the generator's ground truth, never derived from
    the code under test — otherwise the suite could not detect a regression.
    """

    model_config = ConfigDict(frozen=True)

    case_id: str = Field(pattern=r"^[a-z0-9_]+$")
    description: str
    config: AebScenarioConfig
    expected_outcome: ExpectedOutcome
    expected_quality: QualityVerdict
    expected_events: tuple[EventType, ...] = ()
    forbidden_events: tuple[EventType, ...] = ()
    # metric name -> expected pass/fail verdict (only metrics worth locking)
    expected_metric_verdicts: dict[str, bool] = Field(default_factory=dict)
    # FAULT cases: the top-ranked verified hypothesis must carry one of these classes
    acceptable_failure_classes: frozenset[FailureClass] = frozenset()
    # verifier must not report above this (DEGRADED quality caps at medium)
    max_report_confidence: ReportConfidence | None = None
    rationale: str


class CaseStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"  # the pipeline or provider raised; counts as a failure for the gate


class CheckResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    passed: bool
    detail: str


class AgentCaseMetrics(BaseModel):
    """Numbers recorded for one agent run; gate inputs and the latency dashboard."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    verification_id: str
    report_confidence: ReportConfidence
    human_review_required: bool
    evidence_support_rate: float = Field(ge=0, le=1)
    unsupported_claim_rate: float = Field(ge=0, le=1)
    latency_s: float = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    attempts: int = Field(ge=1)
    hypotheses_verified: int = Field(ge=0)
    hypotheses_stripped: int = Field(ge=0)
    unresolved_ids: int = Field(ge=0)
    injection_flags: int = Field(ge=0)
    top_failure_class: FailureClass | None
    acceptable_class_rank: int | None  # 1-based rank of the first acceptable class


class CaseResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    expected_outcome: ExpectedOutcome
    status: CaseStatus
    scenario_id: str | None
    deterministic_checks: tuple[CheckResult, ...]
    agent_checks: tuple[CheckResult, ...]
    agent_ran: bool
    agent: AgentCaseMetrics | None = None
    error: str | None = None

    @property
    def failed_checks(self) -> tuple[CheckResult, ...]:
        return tuple(c for c in (*self.deterministic_checks, *self.agent_checks) if not c.passed)


class Aggregates(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_cases: int
    n_passed: int
    n_failed: int
    n_errors: int
    n_agent_cases: int
    n_fault_cases_scored: int
    n_nominal_cases_scored: int
    top1_accuracy: float | None  # over FAULT cases the agent answered
    false_fault_rate: float | None  # over NOMINAL cases the agent answered
    mean_evidence_support_rate: float | None
    max_unsupported_claim_rate: float | None
    latency_p50_s: float | None
    latency_p95_s: float | None
    prompt_tokens: int
    completion_tokens: int
    n_human_review_required: int


class GateThresholds(BaseModel):
    """Release gate (spec §19). Accuracy is gated by the baseline, not an absolute bar,
    unless a threshold is set explicitly."""

    model_config = ConfigDict(frozen=True)

    max_unsupported_claim_rate: float = Field(default=0.10, ge=0, le=1)
    min_evidence_support_rate: float = Field(default=0.90, ge=0, le=1)
    min_top1_accuracy: float | None = Field(default=None, ge=0, le=1)
    max_false_fault_rate: float | None = Field(default=None, ge=0, le=1)


class GateResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool
    failures: tuple[str, ...]
    thresholds: GateThresholds
    baseline_eval_id: str | None = None


class EvalReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    eval_id: str
    suite_name: str
    suite_sha256: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    provider: str
    model: str
    agent_enabled: bool
    question_sha256: str
    cases: tuple[CaseResult, ...]
    aggregates: Aggregates
    gate: GateResult
