"""FastAPI application factory.

Everything is injected through :class:`AppState` so tests can run the whole API in-process
with a FakeProvider, a temporary SQLite file and a temporary workspace. The MVP executes
ingestion, diagnosis and report generation synchronously; the endpoint shapes follow the
async-first spec so jobs can move to a queue later without changing clients.
"""

import json
import math
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from sqlalchemy.orm import Session, sessionmaker

from app.agents.diagnostic import DEFAULT_QUESTION, DiagnosticAgent
from app.agents.pipeline import DiagnosisInputs, prepare_diagnosis
from app.agents.schemas import AgentRun
from app.api.schemas import (
    ApprovalDecision,
    ApprovalOut,
    ChunkOut,
    DashboardOut,
    EventOut,
    FileOut,
    IngestionJobCreate,
    IngestionJobOut,
    MetricOut,
    ProjectCreate,
    ProjectOut,
    QueryCreate,
    QueryOut,
    ReportCreate,
    ReportListItem,
    ReportOut,
    RunDetailOut,
    RunSummaryOut,
    SignalsOut,
)
from app.core.config import Settings, get_settings
from app.core.errors import (
    DataQualityError,
    EvidenceResolutionError,
    IngestionError,
    ProviderError,
)
from app.core.ids import new_id
from app.db import repo
from app.db.base import init_db, make_engine, make_session_factory, session_scope
from app.ingestion.csv_loader import SIDECAR_NAME, load_telemetry_csv
from app.ingestion.schemas import IngestedTelemetry
from app.llm.factory import build_provider
from app.llm.provider import CallLog, LLMProvider
from app.metrics.aeb import compute_aeb_metrics
from app.quality.report import evaluate_gates
from app.rag.cli import access_up_to
from app.rag.index import ChunkIndex, IndexError_
from app.rag.schemas import RetrievalFilters
from app.reports.builder import build_report
from app.reports.render import write_report
from app.reports.schemas import ApprovalSection, DiagnosticReport
from app.verification.registry import EvidenceRegistry
from app.verification.schemas import VerificationReport
from app.verification.verifier import verify_diagnosis

API_VERSION = "0.1.0"


@dataclass
class AppState:
    settings: Settings
    provider: LLMProvider
    index: ChunkIndex | None
    sessions: sessionmaker[Session]
    workspace: Path
    call_log: CallLog


def _load_index(path: Path) -> ChunkIndex | None:
    try:
        return ChunkIndex.load(path)
    except IndexError_:
        return None


def create_app(
    settings: Settings | None = None,
    *,
    provider: LLMProvider | None = None,
    index: ChunkIndex | None = None,
    load_index: bool = True,
) -> FastAPI:
    settings = settings or get_settings()
    call_log = CallLog()
    provider = provider or build_provider(settings, call_log)
    if index is None and load_index:
        index = _load_index(settings.index_dir)
    engine = make_engine(settings.database_url)
    init_db(engine)
    workspace = settings.workspace_dir
    workspace.mkdir(parents=True, exist_ok=True)
    state = AppState(
        settings=settings,
        provider=provider,
        index=index,
        sessions=make_session_factory(engine),
        workspace=workspace,
        call_log=call_log,
    )

    app = FastAPI(title="ADAS Intelligence Platform API", version=API_VERSION)
    app.state.aip = state
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
    _register_routes(app)
    return app


def _state(request: Request) -> AppState:
    state: AppState = request.app.state.aip
    return state


StateDep = Annotated[AppState, Depends(_state)]


def _not_found(what: str, ident: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"{what} {ident!r} not found")


def _filters(state: AppState, access: str) -> RetrievalFilters:
    from app.rag.schemas import AccessLevel

    return RetrievalFilters(allowed_access=access_up_to(AccessLevel(access)))


def _diagnosis_inputs(
    state: AppState, log_path: Path, access: str, top_k: int, file_id: str
) -> DiagnosisInputs:
    try:
        return prepare_diagnosis(
            log_path,
            state.provider,
            index=state.index,
            filters=_filters(state, access),
            top_k=top_k,
            file_id=file_id,  # keep the identity assigned at upload
        )
    except DataQualityError as exc:
        raise HTTPException(
            status_code=422, detail=f"blocked by data-quality gates: {exc}"
        ) from exc


def _register_routes(app: FastAPI) -> None:
    @app.get("/api/health")
    def health(state: StateDep) -> dict[str, object]:
        return {
            "status": "ok",
            "version": API_VERSION,
            "llm_provider": state.provider.name,
            "rag_index_loaded": state.index is not None,
            "time": datetime.now(UTC).isoformat(),
        }

    # --- projects ---------------------------------------------------------------------------

    @app.post("/api/projects", response_model=ProjectOut, status_code=201)
    def create_project(body: ProjectCreate, state: StateDep) -> ProjectOut:
        with session_scope(state.sessions) as s:
            return ProjectOut.model_validate(repo.create_project(s, body.name))

    @app.get("/api/projects/{project_id}", response_model=ProjectOut)
    def get_project(project_id: str, state: StateDep) -> ProjectOut:
        with session_scope(state.sessions) as s:
            p = repo.get_project(s, project_id)
            if p is None:
                raise _not_found("project", project_id)
            return ProjectOut.model_validate(p)

    # --- files ------------------------------------------------------------------------------

    @app.post("/api/projects/{project_id}/files", response_model=FileOut, status_code=201)
    def upload_file(
        project_id: str,
        state: StateDep,
        telemetry: Annotated[UploadFile, File(description="telemetry CSV")],
        sidecar: Annotated[UploadFile | None, File(description="optional scenario.json")] = None,
    ) -> FileOut:
        with session_scope(state.sessions) as s:
            if repo.get_project(s, project_id) is None:
                raise _not_found("project", project_id)
        staging = state.workspace / project_id / "files" / new_id("file")
        staging.mkdir(parents=True, exist_ok=True)
        csv_path = staging / "telemetry.csv"
        with csv_path.open("wb") as out:
            shutil.copyfileobj(telemetry.file, out)
        sidecar_path: Path | None = None
        if sidecar is not None:
            sidecar_path = staging / SIDECAR_NAME
            with sidecar_path.open("wb") as out:
                shutil.copyfileobj(sidecar.file, out)
        try:
            staged = load_telemetry_csv(csv_path)
        except (IngestionError, ValueError) as exc:
            shutil.rmtree(staging, ignore_errors=True)
            raise HTTPException(status_code=422, detail=f"cannot ingest file: {exc}") from exc
        # Store under the ingestion-assigned file_id so paths and IDs agree; keep the same
        # provenance (and therefore the same file_id), only the paths change.
        final = staging.with_name(staged.provenance.file_id)
        staging.rename(final)
        csv_path = final / "telemetry.csv"
        sidecar_path = final / SIDECAR_NAME if sidecar_path else None
        ingested = IngestedTelemetry(
            frame=staged.frame,
            provenance=staged.provenance.model_copy(
                update={
                    "source_path": str(csv_path),
                    "sidecar_path": str(sidecar_path) if sidecar_path else None,
                }
            ),
        )
        quality = evaluate_gates(ingested)
        with session_scope(state.sessions) as s:
            row = repo.add_log_file(
                s,
                project_id=project_id,
                original_name=telemetry.filename or "telemetry.csv",
                path=str(csv_path),
                sidecar_path=str(sidecar_path) if sidecar_path else None,
                provenance=ingested.provenance,
                quality=quality,
            )
            return FileOut.model_validate(row)

    @app.get("/api/files/{file_id}", response_model=FileOut)
    def get_file(file_id: str, state: StateDep) -> FileOut:
        with session_scope(state.sessions) as s:
            f = repo.get_log_file(s, file_id)
            if f is None:
                raise _not_found("file", file_id)
            return FileOut.model_validate(f)

    # --- ingestion job: gates + metrics + events ---------------------------------------------

    @app.post("/api/ingestion/jobs", response_model=IngestionJobOut, status_code=201)
    def run_ingestion_job(body: IngestionJobCreate, state: StateDep) -> IngestionJobOut:
        with session_scope(state.sessions) as s:
            f = repo.get_log_file(s, body.file_id)
            if f is None:
                raise _not_found("file", body.file_id)
            path = Path(f.path)
        ingested = load_telemetry_csv(path, file_id=body.file_id)
        quality = evaluate_gates(ingested)
        job_id = "job_" + new_id("run").split("_", 1)[1]
        try:
            metrics = compute_aeb_metrics(ingested, quality)
        except DataQualityError:
            return IngestionJobOut(
                job_id=job_id,
                file_id=body.file_id,
                status="blocked",
                quality_verdict=quality.verdict.value,
                events=0,
                metrics_available=0,
                metrics_missing=0,
                primary_window_id=None,
            )
        with session_scope(state.sessions) as s:
            f = repo.get_log_file(s, body.file_id)
            assert f is not None
            repo.store_metrics(s, f, metrics)
        available = sum(1 for m in metrics.metrics if m.available)
        return IngestionJobOut(
            job_id=job_id,
            file_id=body.file_id,
            status="completed",
            quality_verdict=quality.verdict.value,
            events=len(metrics.events),
            metrics_available=available,
            metrics_missing=len(metrics.metrics) - available,
            primary_window_id=metrics.primary_window_id,
        )

    @app.get("/api/events", response_model=list[EventOut])
    def list_events(
        state: StateDep, project_id: str | None = None, file_id: str | None = None
    ) -> list[EventOut]:
        with session_scope(state.sessions) as s:
            return [
                EventOut.model_validate(e)
                for e in repo.list_events(s, project_id=project_id, file_id=file_id)
            ]

    # --- query: the diagnostic agent -------------------------------------------------------------

    @app.post("/api/query", response_model=QueryOut)
    def query(body: QueryCreate, state: StateDep) -> QueryOut:
        with session_scope(state.sessions) as s:
            f = repo.get_log_file(s, body.file_id)
            if f is None or f.project_id != body.project_id:
                raise _not_found("file", body.file_id)
            path = Path(f.path)
        inputs = _diagnosis_inputs(state, path, body.access_level.value, body.top_k, body.file_id)
        agent = DiagnosticAgent(state.provider)
        try:
            run = agent.run(inputs.bundle, body.question or DEFAULT_QUESTION)
        except ProviderError as exc:
            raise HTTPException(status_code=503, detail=f"LLM provider unavailable: {exc}") from exc
        verification = verify_diagnosis(run, inputs)
        with session_scope(state.sessions) as s:
            repo.save_run(
                s,
                project_id=body.project_id,
                file_id=body.file_id,
                run=run,
                verification=verification,
            )
        top = verification.top
        answer = (
            f"{top.hypothesis.cause} (failure class: {top.hypothesis.failure_class.value}; "
            f"confidence {top.confidence_label.value})"
            if top is not None
            else "No hypothesis survived evidence verification."
        )
        return QueryOut(
            answer=answer,
            confidence=verification.report_confidence.value,
            evidence_ids=list(verification.cited_evidence_ids),
            unsupported_claims=[
                f"{s.hypothesis.cause} ({s.reason})" for s in verification.stripped
            ],
            recommended_next_tests=list(verification.recommended_next_tests),
            human_review_required=verification.human_review_required,
            run_id=run.run_id,
            verification_id=verification.verification_id,
            evidence_support_rate=verification.evidence_support_rate,
        )

    # --- reports ------------------------------------------------------------------------------

    @app.post("/api/reports", response_model=ReportOut, status_code=201)
    def create_report(body: ReportCreate, state: StateDep) -> ReportOut:
        with session_scope(state.sessions) as s:
            f = repo.get_log_file(s, body.file_id)
            if f is None or f.project_id != body.project_id:
                raise _not_found("file", body.file_id)
            path = Path(f.path)
            run_row = repo.get_run(s, body.run_id) if body.run_id else None
            if body.run_id and (run_row is None or run_row.file_id != body.file_id):
                raise _not_found("run", body.run_id)
            run = AgentRun.model_validate(run_row.run_json) if run_row else None
            verification = (
                VerificationReport.model_validate(run_row.verification_json) if run_row else None
            )
        inputs = _diagnosis_inputs(state, path, body.access_level.value, 6, body.file_id)
        registry = EvidenceRegistry.from_bundle(inputs.bundle, inputs.retrieval)
        try:
            report = build_report(inputs, run, verification, registry=registry)
            out_dir = state.workspace / body.project_id / "reports" / report.metadata.report_id
            md_path, json_path = write_report(report, out_dir, registry)
        except EvidenceResolutionError as exc:
            raise HTTPException(status_code=500, detail=f"report not traceable: {exc}") from exc
        with session_scope(state.sessions) as s:
            row, task = repo.save_report(
                s,
                project_id=body.project_id,
                file_id=body.file_id,
                report=report,
                markdown_path=str(md_path),
                json_path=str(json_path),
            )
            return ReportOut(
                report_id=row.id,
                project_id=row.project_id,
                file_id=row.file_id,
                run_id=row.run_id,
                report_confidence=row.report_confidence,
                approval_id=task.id,
                human_review_required=task.human_review_required,
                markdown_url=f"/api/reports/{row.id}?format=md",
                json_url=f"/api/reports/{row.id}",
            )

    @app.get("/api/reports/{report_id}", response_model=None)
    def get_report(report_id: str, state: StateDep, format: str = "json") -> Response:
        with session_scope(state.sessions) as s:
            row = repo.get_report(s, report_id)
            if row is None:
                raise _not_found("report", report_id)
            md_path, payload = Path(row.markdown_path), dict(row.report_json)
        if format == "md":
            return PlainTextResponse(
                md_path.read_text(encoding="utf-8"), media_type="text/markdown"
            )
        return JSONResponse(payload)

    # --- approvals ----------------------------------------------------------------------------

    @app.get("/api/approvals/{approval_id}", response_model=ApprovalOut)
    def get_approval(approval_id: str, state: StateDep) -> ApprovalOut:
        with session_scope(state.sessions) as s:
            t = repo.get_approval(s, approval_id)
            if t is None:
                raise _not_found("approval", approval_id)
            return ApprovalOut.model_validate(t)

    @app.post("/api/approvals/{approval_id}/decision", response_model=ApprovalOut)
    def decide(approval_id: str, body: ApprovalDecision, state: StateDep) -> ApprovalOut:
        with session_scope(state.sessions) as s:
            t = repo.get_approval(s, approval_id)
            if t is None:
                raise _not_found("approval", approval_id)
            if t.status != "pending_review":
                raise HTTPException(status_code=409, detail=f"approval already {t.status}")
            t = repo.decide_approval(
                s, t, reviewer=body.reviewer, decision=body.decision, reason=body.reason
            )
            row = repo.get_report(s, t.report_id)
            assert row is not None
            # Re-render the stored report with the approval section filled in (spec §15).
            report = DiagnosticReport.model_validate(row.report_json)
            approved = report.model_copy(
                update={
                    "approval": ApprovalSection(
                        status=t.status,
                        human_review_required=t.human_review_required,
                        review_reasons=tuple(t.review_reasons),
                        reviewer=t.reviewer,
                        decision=t.decision,
                        reason=t.reason,
                        decided_at=t.decided_at,
                    )
                }
            )
            row.report_json = approved.model_dump(mode="json")
            from app.reports.render import render_json, render_markdown

            Path(row.markdown_path).write_text(render_markdown(approved), encoding="utf-8")
            Path(row.json_path).write_text(render_json(approved), encoding="utf-8")
            return ApprovalOut.model_validate(t)

    # --- listings and detail views for the dashboard -------------------------------------------

    @app.get("/api/projects", response_model=list[ProjectOut])
    def list_projects(state: StateDep) -> list[ProjectOut]:
        with session_scope(state.sessions) as s:
            return [ProjectOut.model_validate(p) for p in repo.list_projects(s)]

    @app.get("/api/projects/{project_id}/files", response_model=list[FileOut])
    def list_files(project_id: str, state: StateDep) -> list[FileOut]:
        with session_scope(state.sessions) as s:
            if repo.get_project(s, project_id) is None:
                raise _not_found("project", project_id)
            return [FileOut.model_validate(f) for f in repo.list_files(s, project_id)]

    @app.get("/api/files/{file_id}/metrics", response_model=list[MetricOut])
    def file_metrics(file_id: str, state: StateDep) -> list[MetricOut]:
        with session_scope(state.sessions) as s:
            if repo.get_log_file(s, file_id) is None:
                raise _not_found("file", file_id)
            return [MetricOut.model_validate(m) for m in repo.list_metrics(s, file_id)]

    @app.get("/api/files/{file_id}/signals", response_model=SignalsOut)
    def file_signals(file_id: str, state: StateDep, max_points: int = 1000) -> SignalsOut:
        """Downsampled numeric telemetry for plots; metrics never use this path."""
        with session_scope(state.sessions) as s:
            f = repo.get_log_file(s, file_id)
            if f is None:
                raise _not_found("file", file_id)
            path = Path(f.path)
        frame = load_telemetry_csv(path, file_id=file_id).frame
        step = max(1, math.ceil(len(frame) / max(1, max_points)))
        sampled = frame.iloc[::step]
        numeric = [c for c in sampled.columns if sampled[c].dtype.kind in "fiub"]
        data: dict[str, list[float | None]] = {}
        for col in numeric:
            values = sampled[col].astype("float64").tolist()
            data[col] = [None if (v != v) else float(v) for v in values]  # NaN → null
        return SignalsOut(
            file_id=file_id, rows_total=int(len(frame)), step=step, columns=numeric, data=data
        )

    @app.get("/api/runs", response_model=list[RunSummaryOut])
    def list_runs(
        state: StateDep, project_id: str | None = None, file_id: str | None = None
    ) -> list[RunSummaryOut]:
        with session_scope(state.sessions) as s:
            return [
                RunSummaryOut.model_validate(r)
                for r in repo.list_runs(s, project_id=project_id, file_id=file_id)
            ]

    @app.get("/api/runs/{run_id}", response_model=RunDetailOut)
    def get_run(run_id: str, state: StateDep) -> RunDetailOut:
        with session_scope(state.sessions) as s:
            r = repo.get_run(s, run_id)
            if r is None:
                raise _not_found("run", run_id)
            summary = RunSummaryOut.model_validate(r)
            return RunDetailOut(
                **summary.model_dump(), run=dict(r.run_json), verification=dict(r.verification_json)
            )

    @app.get("/api/chunks/{chunk_id}", response_model=ChunkOut)
    def get_chunk(chunk_id: str, state: StateDep, access: str = "public") -> ChunkOut:
        """Resolve a cited requirement chunk. The access wall applies here too."""
        if state.index is None:
            raise HTTPException(status_code=404, detail="no RAG index loaded")
        chunk = state.index.get(chunk_id)
        if chunk is None:
            raise _not_found("chunk", chunk_id)
        from app.rag.schemas import AccessLevel

        if chunk.access_level not in access_up_to(AccessLevel(access)):
            raise HTTPException(status_code=403, detail="access level insufficient for this chunk")
        return ChunkOut(
            chunk_id=chunk.chunk_id,
            document_title=chunk.document_title,
            heading=chunk.heading,
            text=chunk.text,
            source_type=chunk.source_type.value,
            access_level=chunk.access_level.value,
            version=chunk.version,
            requirement_ids=list(chunk.requirement_ids),
        )

    @app.get("/api/reports", response_model=list[ReportListItem])
    def list_reports(
        state: StateDep, project_id: str | None = None, file_id: str | None = None
    ) -> list[ReportListItem]:
        with session_scope(state.sessions) as s:
            return [
                ReportListItem(
                    report_id=r.id,
                    project_id=r.project_id,
                    file_id=r.file_id,
                    run_id=r.run_id,
                    report_confidence=r.report_confidence,
                    created_at=r.created_at,
                    approval_id=t.id if t else None,
                    approval_status=t.status if t else None,
                )
                for r, t in repo.list_reports(s, project_id=project_id, file_id=file_id)
            ]

    @app.get("/api/approvals", response_model=list[ApprovalOut])
    def list_approvals(state: StateDep, status: str | None = None) -> list[ApprovalOut]:
        with session_scope(state.sessions) as s:
            return [ApprovalOut.model_validate(t) for t in repo.list_approvals(s, status=status)]

    # --- dashboard ----------------------------------------------------------------------------

    @app.get("/api/dashboard/summary", response_model=DashboardOut)
    def dashboard(state: StateDep) -> DashboardOut:
        with session_scope(state.sessions) as s:
            data = repo.dashboard_summary(s)
        return DashboardOut(
            **{k: v for k, v in data.items()},  # type: ignore[arg-type]
            llm_provider=state.provider.name,
            rag_index_loaded=state.index is not None,
        )


def dump_state_for_debug(state: AppState) -> str:  # pragma: no cover
    return json.dumps({"provider": state.provider.name, "index": state.index is not None}, indent=2)
