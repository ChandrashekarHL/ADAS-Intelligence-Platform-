"""Run every gate and reduce to one verdict.

* PASS      – all gates pass; analysis may claim up to High confidence.
* DEGRADED  – warnings only; analysis proceeds but confidence is capped (M7 rules).
* BLOCKED   – at least one FAIL; no metric or AI analysis may run on this file.

The report carries a ``quality_`` evidence ID so later stages can cite *why* they were
blocked or downgraded instead of silently doing less.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from app.core.errors import DataQualityError
from app.core.ids import stable_id
from app.ingestion.schemas import IngestedTelemetry
from app.quality.gates import ALL_GATES, Gate, GateResult, GateStatus, QualityPolicy


class QualityVerdict(StrEnum):
    PASS = "pass"
    DEGRADED = "degraded"
    BLOCKED = "blocked"


class QualityReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    quality_id: str
    file_id: str
    verdict: QualityVerdict
    gates: tuple[GateResult, ...]

    @property
    def failed(self) -> tuple[GateResult, ...]:
        return tuple(g for g in self.gates if g.status is GateStatus.FAIL)

    @property
    def warned(self) -> tuple[GateResult, ...]:
        return tuple(g for g in self.gates if g.status is GateStatus.WARN)

    @property
    def analyzable(self) -> bool:
        return self.verdict is not QualityVerdict.BLOCKED


def evaluate_gates(
    telemetry: IngestedTelemetry,
    policy: QualityPolicy | None = None,
    gates: tuple[Gate, ...] = ALL_GATES,
) -> QualityReport:
    """Run all gates on an ingested frame. Deterministic apart from the minted ID."""
    policy = policy or QualityPolicy()
    results = tuple(gate(telemetry.frame, policy) for gate in gates)
    statuses = {r.status for r in results}
    if GateStatus.FAIL in statuses:
        verdict = QualityVerdict.BLOCKED
    elif GateStatus.WARN in statuses:
        verdict = QualityVerdict.DEGRADED
    else:
        verdict = QualityVerdict.PASS
    return QualityReport(
        # Stable for the same file and policy, so stored runs/reports keep resolving it.
        quality_id=stable_id("quality", telemetry.provenance.file_id, policy.model_dump_json()),
        file_id=telemetry.provenance.file_id,
        verdict=verdict,
        gates=results,
    )


def require_analyzable(report: QualityReport) -> None:
    """Raise :class:`DataQualityError` when the report is BLOCKED. Never swallow it."""
    if report.analyzable:
        return
    reasons = "; ".join(f"{g.gate}: {g.message}" for g in report.failed)
    raise DataQualityError(
        f"analysis blocked by data-quality gates ({report.quality_id}): {reasons}"
    )
