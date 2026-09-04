"""Agent contracts. ``AgentOutput`` is the fixed §11.4 schema; nothing else leaves an agent.

The schema is deliberately narrow so the API can enforce it (OpenAI structured output) and
the verifier (M7) can check every ``evidence_ids`` entry mechanically.
"""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.llm.schemas import TokenUsage


class FailureClass(StrEnum):
    """Spec §13.4 failure taxonomy plus ``unknown`` for honest uncertainty."""

    SENSOR_LIMITATION = "sensor_limitation"
    PERCEPTION_ERROR = "perception_error"
    FUSION_ERROR = "fusion_error"
    PLANNER_LOGIC_ERROR = "planner_logic_error"
    CONTROL_ISSUE = "control_issue"
    CALIBRATION_CONFIG_ISSUE = "calibration_config_issue"
    DATA_LOGGING_ISSUE = "data_logging_issue"
    UNKNOWN = "unknown"


class Hypothesis(BaseModel):
    model_config = ConfigDict(frozen=True)

    cause: str = Field(min_length=1)
    failure_class: FailureClass = FailureClass.UNKNOWN
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class AgentOutput(BaseModel):
    """The §11.4 contract: observations, hypotheses, missing_evidence, recommended_next_tests."""

    model_config = ConfigDict(frozen=True)

    observations: list[str] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    recommended_next_tests: list[str] = Field(default_factory=list)

    @property
    def cited_ids(self) -> frozenset[str]:
        return frozenset(i for h in self.hypotheses for i in h.evidence_ids)


class EvidenceKind(StrEnum):
    QUALITY = "quality"
    EVENT = "event"
    WINDOW = "window"
    METRIC = "metric"
    CHUNK = "chunk"
    FILE = "file"


class EvidenceItem(BaseModel):
    """One citable artifact, rendered compactly for the prompt."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str
    kind: EvidenceKind
    summary: str
    t_s: float | None = None
    passed: bool | None = None


class InjectionFlag(BaseModel):
    """A retrieved text matched an instruction-like pattern. Recorded, never hidden."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str
    pattern: str
    excerpt: str


class AgentRun(BaseModel):
    """Everything needed to replay and audit one agent invocation."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    agent: str
    provider: str
    model: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    question: str
    offered_evidence_ids: tuple[str, ...]
    missing_evidence_offered: tuple[str, ...]
    injection_flags: tuple[InjectionFlag, ...]
    prompt_sha256: str
    attempts: int
    unresolved_ids: tuple[str, ...]  # cited but not offered, after any repair round
    usage: TokenUsage
    latency_s: float
    output: AgentOutput
    data_origin: str
