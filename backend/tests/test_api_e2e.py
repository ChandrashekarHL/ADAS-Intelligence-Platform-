"""M9 end-to-end acceptance test through the HTTP API.

This is the gate for every later milestone: the whole AEB late-braking slice, from upload
to approved report, in-process with a FakeProvider and a temporary SQLite database.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.agents.schemas import AgentOutput, FailureClass, Hypothesis
from app.api.app import create_app
from app.core.config import Settings
from app.llm.fake import FakeProvider
from app.llm.schemas import LLMRequest, Role
from app.rag.index import ChunkIndex, build_index
from app.synthetic.aeb_generator import generate_aeb_scenario
from app.synthetic.io import CSV_NAME, METADATA_NAME, write_scenario
from app.synthetic.schemas import AebScenarioConfig, ScenarioVariant

DOCS = Path(__file__).resolve().parents[2] / "data" / "demo_docs"
ID_RE = r"^(metric|event|window|chunk|quality|file)_[0-9a-f]{12}$"


def scripted_diagnosis(request: LLMRequest) -> str:
    """Pretend-model that behaves: cites only IDs it was shown, with timestamped evidence."""
    import re

    # Like a real model, read the whole transcript (the repair round appends messages).
    text = "\n".join(m.content for m in request.messages if m.role is Role.USER)
    # Work from the evidence section only; never quote IDs from anywhere else.
    section = text.split("EVIDENCE (cite only these IDs):", 1)[-1]
    metrics = re.findall(r"- (metric_[0-9a-f]{12}) t=[\d.]+s FAIL: (\w+)", section)
    chunks = re.findall(r"- (chunk_[0-9a-f]{12}): requirement", section)
    latency = [m for m, name in metrics if name == "braking_latency_s"]
    dropout = [m for m, name in metrics if name == "confidence_dropout_during_risk_s"]
    evidence = latency + dropout + chunks[:1]
    out = AgentOutput(
        observations=["Brake command lagged the TTC threshold crossing."],
        hypotheses=[
            Hypothesis(
                cause="Perception confidence dropout during the risk phase delayed the valid "
                "target and therefore the brake request.",
                failure_class=FailureClass.PERCEPTION_ERROR,
                evidence_ids=evidence,
                confidence=0.8,
            ),
            # Fabricated evidence, kept even after the agent's repair round (a stubborn
            # model), so the verifier has to strip it and the report shows the removal.
            Hypothesis(
                cause="Brake actuator was slow.",
                failure_class=FailureClass.CONTROL_ISSUE,
                evidence_ids=["metric_000000000000"],
                confidence=0.6,
            ),
        ],
        missing_evidence=["fusion track-hold state"],
        recommended_next_tests=["Repeat SCN-AEB-LVSB-01 with fusion trace logging (TC-AEB-011)."],
    )
    return out.model_dump_json()


@pytest.fixture(scope="module")
def provider() -> FakeProvider:
    return FakeProvider(scripted=scripted_diagnosis, embedding_dim=32)


@pytest.fixture(scope="module")
def index(provider: FakeProvider) -> ChunkIndex:
    return build_index(DOCS, provider)


@pytest.fixture(scope="module")
def client(
    tmp_path_factory: pytest.TempPathFactory, provider: FakeProvider, index: ChunkIndex
) -> Iterator[TestClient]:
    root = tmp_path_factory.mktemp("api")
    settings = Settings(
        _env_file=None,
        llm_provider="fake",
        database_url=f"sqlite:///{(root / 'aip.sqlite').as_posix()}",
        workspace_dir=root / "workspace",
        index_dir=root / "no-index-here",
    )
    app = create_app(settings, provider=provider, index=index)
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def scenario_dirs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for variant in ScenarioVariant:
        d = tmp_path_factory.mktemp(variant.value)
        write_scenario(generate_aeb_scenario(AebScenarioConfig(variant=variant, seed=42)), d)
        out[variant.value] = d
    return out


def upload(
    client: TestClient, project_id: str, d: Path, *, with_sidecar: bool = True
) -> dict[str, object]:
    with (d / CSV_NAME).open("rb") as csv:
        files: dict[str, tuple[str, object, str]] = {"telemetry": (CSV_NAME, csv, "text/csv")}
        if with_sidecar:
            with (d / METADATA_NAME).open("rb") as sc:
                files["sidecar"] = (METADATA_NAME, sc, "application/json")
                r = client.post(f"/api/projects/{project_id}/files", files=files)
        else:
            r = client.post(f"/api/projects/{project_id}/files", files=files)
    assert r.status_code == 201, r.text
    data: dict[str, object] = r.json()
    return data


# --- the acceptance flow ------------------------------------------------------------------------


def test_e2e_late_braking_upload_to_approved_report(
    client: TestClient, scenario_dirs: dict[str, Path]
) -> None:
    assert client.get("/api/health").json()["rag_index_loaded"] is True

    # 1. project
    p = client.post("/api/projects", json={"name": "AEB late braking"})
    assert p.status_code == 201
    project_id = p.json()["id"]
    assert project_id.startswith("proj_")
    assert client.get(f"/api/projects/{project_id}").json()["name"] == "AEB late braking"

    # 2. upload with sidecar → ingested, gated, stored on disk under its file_id
    f = upload(client, project_id, scenario_dirs["late_braking"])
    file_id = str(f["id"])
    assert file_id.startswith("file_") and f["quality_verdict"] == "pass"
    assert f["data_origin"] == "synthetic" and str(f["scenario_id"]).startswith("scenario_")
    assert f["row_count"] == 500
    assert client.get(f"/api/files/{file_id}").json()["id"] == file_id

    # 3. ingestion job → events + metrics persisted
    job = client.post("/api/ingestion/jobs", json={"file_id": file_id})
    assert job.status_code == 201, job.text
    j = job.json()
    assert j["status"] == "completed" and j["events"] == 3 and j["metrics_missing"] == 0
    assert j["primary_window_id"].startswith("window_")
    events = client.get("/api/events", params={"file_id": file_id}).json()
    assert [e["event_type"] for e in events] == [
        "ttc_threshold_crossing",
        "aeb_brake_command",
        "collision",
    ]
    assert [e["t_s"] for e in events] == sorted(e["t_s"] for e in events)
    assert client.get("/api/events", params={"project_id": project_id}).json() == events

    # 4. query → verified answer in the §18.2 contract
    q = client.post(
        "/api/query",
        json={"project_id": project_id, "file_id": file_id, "access_level": "internal"},
    )
    assert q.status_code == 200, q.text
    qa = q.json()
    assert "confidence dropout" in qa["answer"].lower()
    assert qa["confidence"] in {"high", "medium"}
    assert qa["evidence_ids"] and all(
        i.split("_")[0] in {"metric", "chunk", "event", "window"} for i in qa["evidence_ids"]
    )
    assert "metric_000000000000" not in qa["evidence_ids"]
    assert qa["unsupported_claims"] == [
        "Brake actuator was slow. (none of the cited evidence IDs resolve)"
    ]
    assert qa["recommended_next_tests"] == [
        "Repeat SCN-AEB-LVSB-01 with fusion trace logging (TC-AEB-011)."
    ]
    assert qa["human_review_required"] is False
    assert qa["run_id"].startswith("run_") and qa["verification_id"].startswith("verification_")
    assert 0.5 <= qa["evidence_support_rate"] < 1.0

    # 5. report from the run → files on disk, approval task pending
    r = client.post(
        "/api/reports",
        json={
            "project_id": project_id,
            "file_id": file_id,
            "run_id": qa["run_id"],
            "access_level": "internal",
        },
    )
    assert r.status_code == 201, r.text
    rep = r.json()
    assert rep["report_id"].startswith("report_") and rep["run_id"] == qa["run_id"]
    assert rep["report_confidence"] == qa["confidence"]
    assert rep["approval_id"].startswith("appr_")

    md = client.get(f"/api/reports/{rep['report_id']}", params={"format": "md"})
    assert md.status_code == 200 and md.headers["content-type"].startswith("text/markdown")
    text = md.text
    assert text.startswith("# AEB Diagnostic Report")
    assert "FAIL" in text and "Perception confidence dropout" in text
    assert "Brake actuator was slow." in text and "Removed by the verifier" in text
    assert "Status: **pending_review**" in text
    body = client.get(f"/api/reports/{rep['report_id']}").json()
    assert body["metadata"]["report_id"] == rep["report_id"]
    assert body["metadata"]["run_id"] == qa["run_id"]
    assert body["hypotheses"][0]["failure_class"] == "perception_error"
    assert all(h["cause"] != "Brake actuator was slow." for h in body["hypotheses"])

    # 6. approval decision → recorded and re-rendered into the report (spec §15)
    a = client.get(f"/api/approvals/{rep['approval_id']}").json()
    assert a["status"] == "pending_review" and a["reviewer"] is None
    d = client.post(
        f"/api/approvals/{rep['approval_id']}/decision",
        json={
            "reviewer": "lead validation engineer",
            "decision": "approved",
            "reason": "Evidence consistent with INC-2041.",
        },
    )
    assert d.status_code == 200, d.text
    dec = d.json()
    assert dec["status"] == "approved" and dec["reviewer"] == "lead validation engineer"
    assert dec["decided_at"] is not None
    text2 = client.get(f"/api/reports/{rep['report_id']}", params={"format": "md"}).text
    assert "Status: **approved**" in text2 and "lead validation engineer" in text2
    again = client.post(
        f"/api/approvals/{rep['approval_id']}/decision",
        json={"reviewer": "x", "decision": "rejected", "reason": "y"},
    )
    assert again.status_code == 409

    # 7. dashboard
    s = client.get("/api/dashboard/summary").json()
    assert s["projects"] >= 1 and s["files"] >= 1 and s["agent_runs"] >= 1 and s["reports"] >= 1
    assert s["reports_by_confidence"].get(qa["confidence"], 0) >= 1
    assert s["llm_provider"] == "fake" and s["rag_index_loaded"] is True
    assert s["avg_evidence_support_rate"] is not None
    assert s["llm_prompt_tokens"] > 0


def test_nominal_passes_and_metrics_only_report(
    client: TestClient, scenario_dirs: dict[str, Path]
) -> None:
    project_id = client.post("/api/projects", json={"name": "nominal"}).json()["id"]
    f = upload(client, project_id, scenario_dirs["nominal"], with_sidecar=False)
    assert f["data_origin"] == "unknown" and f["scenario_id"] is None  # no sidecar
    job = client.post("/api/ingestion/jobs", json={"file_id": f["id"]}).json()
    assert job["status"] == "completed" and job["events"] == 2  # no collision
    r = client.post("/api/reports", json={"project_id": project_id, "file_id": f["id"]})
    assert r.status_code == 201
    rep = r.json()
    assert rep["run_id"] is None and rep["report_confidence"] == "low"
    text = client.get(f"/api/reports/{rep['report_id']}", params={"format": "md"}).text
    assert "No AI diagnosis was run" in text
    assert "against a limit of 0.3 s: PASS" in text
    assert "No collision was recorded" in text


def test_blocked_file_is_rejected_at_query_and_job(
    client: TestClient, scenario_dirs: dict[str, Path], tmp_path: Path
) -> None:
    project_id = client.post("/api/projects", json={"name": "bad data"}).json()["id"]
    src = scenario_dirs["late_braking"] / CSV_NAME
    bad = tmp_path / "bad"
    bad.mkdir()
    pd.read_csv(src).drop(columns=["object_confidence"]).to_csv(bad / CSV_NAME, index=False)
    f = upload(client, project_id, bad, with_sidecar=False)
    assert f["quality_verdict"] == "blocked"
    job = client.post("/api/ingestion/jobs", json={"file_id": f["id"]}).json()
    assert job["status"] == "blocked" and job["events"] == 0
    q = client.post("/api/query", json={"project_id": project_id, "file_id": f["id"]})
    assert q.status_code == 422 and "data-quality" in q.json()["detail"]
    r = client.post("/api/reports", json={"project_id": project_id, "file_id": f["id"]})
    assert r.status_code == 422


def test_not_found_and_validation(client: TestClient) -> None:
    assert client.get("/api/projects/proj_nope").status_code == 404
    assert client.get("/api/files/file_nope").status_code == 404
    assert client.post("/api/ingestion/jobs", json={"file_id": "file_nope"}).status_code == 404
    assert client.get("/api/reports/report_nope").status_code == 404
    assert client.get("/api/approvals/appr_nope").status_code == 404
    assert client.post("/api/projects", json={"name": ""}).status_code == 422
    project_id = client.post("/api/projects", json={"name": "p"}).json()["id"]
    assert (
        client.post(
            "/api/query", json={"project_id": project_id, "file_id": "file_nope"}
        ).status_code
        == 404
    )
    r = client.post(
        f"/api/projects/{project_id}/files",
        files={"telemetry": ("x.csv", b"not,a\n1,2,3\n", "text/csv")},
    )
    assert r.status_code in {
        201,
        422,
    }  # unparseable → 422; parseable-but-unknown columns → 201 blocked later


def test_unscripted_fake_provider_yields_503(
    tmp_path: Path, scenario_dirs: dict[str, Path], index: ChunkIndex
) -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="fake",
        database_url=f"sqlite:///{(tmp_path / 'aip.sqlite').as_posix()}",
        workspace_dir=tmp_path / "ws",
    )
    app = create_app(settings, provider=FakeProvider(embedding_dim=32), index=index)
    with TestClient(app) as c:
        project_id = c.post("/api/projects", json={"name": "p"}).json()["id"]
        f = upload(c, project_id, scenario_dirs["late_braking"])
        q = c.post("/api/query", json={"project_id": project_id, "file_id": f["id"]})
        assert q.status_code == 503


def test_report_json_matches_disk(client: TestClient, scenario_dirs: dict[str, Path]) -> None:
    project_id = client.post("/api/projects", json={"name": "disk"}).json()["id"]
    f = upload(client, project_id, scenario_dirs["late_braking"])
    rep = client.post("/api/reports", json={"project_id": project_id, "file_id": f["id"]}).json()
    body = client.get(f"/api/reports/{rep['report_id']}").json()
    # the API's JSON is the same document that was written to the workspace
    ws_files = list(Path(client.app.state.aip.workspace).rglob(f"{rep['report_id']}/report.json"))  # type: ignore[attr-defined]
    assert len(ws_files) == 1
    assert (
        json.loads(ws_files[0].read_text(encoding="utf-8"))["metadata"]["report_id"]
        == body["metadata"]["report_id"]
    )
