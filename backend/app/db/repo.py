"""Repository functions: the only place that writes rows. Keeps the API thin and testable."""

from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agents.schemas import AgentRun
from app.core.ids import new_id
from app.db.models import AgentRunRow, ApprovalTask, Event, LogFile, Metric, Project, ReportRow
from app.ingestion.schemas import TelemetryProvenance
from app.metrics.schemas import AebMetricsReport
from app.quality.report import QualityReport
from app.reports.schemas import DiagnosticReport
from app.verification.schemas import VerificationReport


def new_project_id() -> str:
    # Projects are not evidence; keep them out of the evidence ID namespace.
    return "proj_" + new_id("run").split("_", 1)[1]


def new_approval_id() -> str:
    return "appr_" + new_id("run").split("_", 1)[1]


# --- projects -----------------------------------------------------------------------------------


def create_project(session: Session, name: str) -> Project:
    project = Project(id=new_project_id(), name=name)
    session.add(project)
    session.flush()
    return project


def get_project(session: Session, project_id: str) -> Project | None:
    return session.get(Project, project_id)


# --- files ----------------------------------------------------------------------------------------


def add_log_file(
    session: Session,
    *,
    project_id: str,
    original_name: str,
    path: str,
    sidecar_path: str | None,
    provenance: TelemetryProvenance,
    quality: QualityReport,
) -> LogFile:
    row = LogFile(
        id=provenance.file_id,
        project_id=project_id,
        original_name=original_name,
        path=path,
        sidecar_path=sidecar_path,
        sha256=provenance.sha256,
        data_origin=provenance.data_origin,
        scenario_id=provenance.scenario_id,
        row_count=provenance.row_count,
        duration_s=provenance.duration_s,
        quality_id=quality.quality_id,
        quality_verdict=quality.verdict.value,
        quality_json=quality.model_dump(mode="json"),
    )
    session.add(row)
    session.flush()
    return row


def get_log_file(session: Session, file_id: str) -> LogFile | None:
    return session.get(LogFile, file_id)


def store_metrics(session: Session, log_file: LogFile, metrics: AebMetricsReport) -> None:
    """Replace stored events/metrics for a file with the given report."""
    for old in session.scalars(select(Event).where(Event.file_id == log_file.id)):
        session.delete(old)
    for old_metric in session.scalars(select(Metric).where(Metric.file_id == log_file.id)):
        session.delete(old_metric)
    for e in metrics.events:
        session.add(
            Event(
                id=e.event_id,
                file_id=log_file.id,
                project_id=log_file.project_id,
                event_type=e.event_type.value,
                t_s=e.t_s,
                description=e.description,
            )
        )
    for m in metrics.metrics:
        if not m.available:
            continue
        value = float(m.value) if isinstance(m.value, (int, float)) else None
        session.add(
            Metric(
                id=m.metric_id,
                file_id=log_file.id,
                name=m.name,
                value=value,
                unit=m.unit,
                passed=m.passed,
                t_s=m.t_s,
                window_id=m.window_id,
            )
        )
    log_file.metrics_json = metrics.model_dump(mode="json")
    session.flush()


def list_events(
    session: Session, *, project_id: str | None = None, file_id: str | None = None
) -> list[Event]:
    stmt = select(Event).order_by(Event.t_s)
    if project_id:
        stmt = stmt.where(Event.project_id == project_id)
    if file_id:
        stmt = stmt.where(Event.file_id == file_id)
    return list(session.scalars(stmt))


# --- runs ---------------------------------------------------------------------------------------


def save_run(
    session: Session,
    *,
    project_id: str,
    file_id: str,
    run: AgentRun,
    verification: VerificationReport,
) -> AgentRunRow:
    row = AgentRunRow(
        id=run.run_id,
        project_id=project_id,
        file_id=file_id,
        agent=run.agent,
        provider=run.provider,
        model=run.model,
        question=run.question,
        verification_id=verification.verification_id,
        report_confidence=verification.report_confidence.value,
        human_review_required=verification.human_review_required,
        evidence_support_rate=verification.evidence_support_rate,
        prompt_tokens=run.usage.prompt_tokens,
        completion_tokens=run.usage.completion_tokens,
        latency_s=run.latency_s,
        run_json=run.model_dump(mode="json"),
        verification_json=verification.model_dump(mode="json"),
    )
    session.add(row)
    session.flush()
    return row


def get_run(session: Session, run_id: str) -> AgentRunRow | None:
    return session.get(AgentRunRow, run_id)


# --- reports and approvals ------------------------------------------------------------------


def save_report(
    session: Session,
    *,
    project_id: str,
    file_id: str,
    report: DiagnosticReport,
    markdown_path: str,
    json_path: str,
) -> tuple[ReportRow, ApprovalTask]:
    row = ReportRow(
        id=report.metadata.report_id,
        project_id=project_id,
        file_id=file_id,
        run_id=report.metadata.run_id,
        report_confidence=report.report_confidence.value,
        markdown_path=markdown_path,
        json_path=json_path,
        report_json=report.model_dump(mode="json"),
    )
    session.add(row)
    session.flush()  # the approval task's FK needs the report row to exist first
    task = ApprovalTask(
        id=new_approval_id(),
        project_id=project_id,
        report_id=row.id,
        human_review_required=report.approval.human_review_required,
        review_reasons=list(report.approval.review_reasons),
    )
    session.add(task)
    session.flush()
    return row, task


def get_report(session: Session, report_id: str) -> ReportRow | None:
    return session.get(ReportRow, report_id)


def get_approval(session: Session, approval_id: str) -> ApprovalTask | None:
    return session.get(ApprovalTask, approval_id)


def approval_for_report(session: Session, report_id: str) -> ApprovalTask | None:
    return session.scalars(select(ApprovalTask).where(ApprovalTask.report_id == report_id)).first()


def decide_approval(
    session: Session, task: ApprovalTask, *, reviewer: str, decision: str, reason: str
) -> ApprovalTask:
    task.reviewer = reviewer
    task.decision = decision
    task.reason = reason
    task.decided_at = datetime.now(UTC)
    task.status = decision
    session.flush()
    return task


# --- dashboard ----------------------------------------------------------------------------------


def dashboard_summary(session: Session) -> dict[str, object]:
    def count(model: type[Project] | type[LogFile] | type[AgentRunRow] | type[ReportRow]) -> int:
        return int(session.scalar(select(func.count()).select_from(model)) or 0)

    by_conf = {
        conf: int(n)
        for conf, n in session.execute(
            select(ReportRow.report_confidence, func.count()).group_by(ReportRow.report_confidence)
        )
    }
    pending = int(
        session.scalar(
            select(func.count())
            .select_from(ApprovalTask)
            .where(ApprovalTask.status == "pending_review")
        )
        or 0
    )
    avg_support = session.scalar(select(func.avg(AgentRunRow.evidence_support_rate)))
    tokens = session.execute(
        select(func.sum(AgentRunRow.prompt_tokens), func.sum(AgentRunRow.completion_tokens))
    ).one()
    return {
        "projects": count(Project),
        "files": count(LogFile),
        "agent_runs": count(AgentRunRow),
        "reports": count(ReportRow),
        "reports_by_confidence": by_conf,
        "approvals_pending": pending,
        "avg_evidence_support_rate": float(avg_support) if avg_support is not None else None,
        "llm_prompt_tokens": int(tokens[0] or 0),
        "llm_completion_tokens": int(tokens[1] or 0),
    }
