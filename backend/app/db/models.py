"""ORM tables (spec §17 subset for the AEB slice).

Rules that keep the PostgreSQL swap cheap: portable column types only, application-generated
string IDs, JSON via SQLAlchemy's dialect-neutral ``JSON`` type, and bulk telemetry
referenced by path rather than stored.
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    files: Mapped[list["LogFile"]] = relationship(back_populates="project")


class LogFile(Base):
    __tablename__ = "log_files"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # file_id from ingestion
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    original_name: Mapped[str] = mapped_column(String(300))
    path: Mapped[str] = mapped_column(Text)  # telemetry stays on disk
    sidecar_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    data_origin: Mapped[str] = mapped_column(String(32))
    scenario_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    row_count: Mapped[int] = mapped_column(Integer)
    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    quality_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    quality_verdict: Mapped[str | None] = mapped_column(String(16), nullable=True)
    quality_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    metrics_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    project: Mapped[Project] = relationship(back_populates="files")


class Event(Base):
    __tablename__ = "events"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    file_id: Mapped[str] = mapped_column(ForeignKey("log_files.id"), index=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    event_type: Mapped[str] = mapped_column(String(64))
    t_s: Mapped[float] = mapped_column(Float)
    description: Mapped[str] = mapped_column(Text)


class Metric(Base):
    __tablename__ = "metrics"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    file_id: Mapped[str] = mapped_column(ForeignKey("log_files.id"), index=True)
    name: Mapped[str] = mapped_column(String(64), index=True)
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    unit: Mapped[str] = mapped_column(String(16))
    passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    t_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    window_id: Mapped[str | None] = mapped_column(String(32), nullable=True)


class AgentRunRow(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # run_id
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    file_id: Mapped[str] = mapped_column(ForeignKey("log_files.id"), index=True)
    agent: Mapped[str] = mapped_column(String(64))
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(64))
    question: Mapped[str] = mapped_column(Text)
    verification_id: Mapped[str] = mapped_column(String(32), index=True)
    report_confidence: Mapped[str] = mapped_column(String(16))
    human_review_required: Mapped[bool] = mapped_column(Boolean)
    evidence_support_rate: Mapped[float] = mapped_column(Float)
    prompt_tokens: Mapped[int] = mapped_column(Integer)
    completion_tokens: Mapped[int] = mapped_column(Integer)
    latency_s: Mapped[float] = mapped_column(Float)
    run_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    verification_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ReportRow(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # report_id
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    file_id: Mapped[str] = mapped_column(ForeignKey("log_files.id"), index=True)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("agent_runs.id"), nullable=True)
    report_confidence: Mapped[str] = mapped_column(String(16))
    markdown_path: Mapped[str] = mapped_column(Text)
    json_path: Mapped[str] = mapped_column(Text)
    report_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ApprovalTask(Base):
    __tablename__ = "approval_tasks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    report_id: Mapped[str] = mapped_column(ForeignKey("reports.id"), index=True)
    status: Mapped[str] = mapped_column(String(16), default="pending_review")
    human_review_required: Mapped[bool] = mapped_column(Boolean)
    review_reasons: Mapped[list[str]] = mapped_column(JSON, default=list)
    reviewer: Mapped[str | None] = mapped_column(String(200), nullable=True)
    decision: Mapped[str | None] = mapped_column(String(16), nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
