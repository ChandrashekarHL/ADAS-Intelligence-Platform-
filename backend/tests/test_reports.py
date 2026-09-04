"""M8 report generator: built from verified results only, traceable, §27.3 sections."""

import json
import re
from pathlib import Path

import pytest

from app.agents.cli import main as cli_main
from app.agents.diagnostic import DiagnosticAgent
from app.agents.pipeline import DiagnosisInputs, prepare_diagnosis
from app.agents.schemas import AgentOutput, AgentRun, EvidenceKind, FailureClass, Hypothesis
from app.core.errors import EvidenceResolutionError
from app.llm.fake import FakeProvider
from app.metrics.aeb import M_BRAKING_LATENCY, M_CONF_DROPOUT
from app.rag.index import ChunkIndex, build_index
from app.rag.schemas import AccessLevel, RetrievalFilters, SourceType
from app.reports.builder import assert_traceable, build_report
from app.reports.render import (
    JSON_NAME,
    MARKDOWN_NAME,
    own_ids,
    render_json,
    render_markdown,
    write_report,
)
from app.reports.schemas import REPORT_TEMPLATE_VERSION, DiagnosticReport
from app.synthetic.aeb_generator import generate_aeb_scenario
from app.synthetic.io import CSV_NAME, write_scenario
from app.synthetic.schemas import AebScenarioConfig, ScenarioVariant
from app.verification.registry import EvidenceRegistry
from app.verification.schemas import DISCLAIMER, ReportConfidence, VerificationReport
from app.verification.verifier import verify_diagnosis

DOCS = Path(__file__).resolve().parents[2] / "data" / "demo_docs"
INTERNAL = RetrievalFilters(allowed_access=frozenset({AccessLevel.PUBLIC, AccessLevel.INTERNAL}))
ID_TOKEN = re.compile(r"\b(?:metric|event|window|chunk|quality|file)_[0-9a-f]{12}\b")


@pytest.fixture(scope="module")
def fake() -> FakeProvider:
    return FakeProvider(embedding_dim=32)


@pytest.fixture(scope="module")
def index(fake: FakeProvider) -> ChunkIndex:
    return build_index(DOCS, fake)


@pytest.fixture(scope="module")
def inputs(
    tmp_path_factory: pytest.TempPathFactory, fake: FakeProvider, index: ChunkIndex
) -> DiagnosisInputs:
    d = tmp_path_factory.mktemp("late")
    write_scenario(
        generate_aeb_scenario(AebScenarioConfig(variant=ScenarioVariant.LATE_BRAKING, seed=41)), d
    )
    return prepare_diagnosis(d / CSV_NAME, fake, index=index, filters=INTERNAL, top_k=6)


@pytest.fixture(scope="module")
def verified(inputs: DiagnosisInputs) -> tuple[AgentRun, VerificationReport]:
    m = inputs.metrics
    assert inputs.retrieval is not None
    req = next(
        c.chunk_id for c in inputs.retrieval.chunks if c.chunk.source_type is SourceType.REQUIREMENT
    )
    out = AgentOutput(
        observations=["Brake command lagged the TTC crossing by 0.80 s."],
        hypotheses=[
            Hypothesis(
                cause="Perception confidence dropout delayed the valid target.",
                failure_class=FailureClass.PERCEPTION_ERROR,
                evidence_ids=[
                    m.metric(M_BRAKING_LATENCY).metric_id,
                    m.metric(M_CONF_DROPOUT).metric_id,
                    req,
                ],
                confidence=0.85,
            ),
            Hypothesis(
                cause="Made-up cause with fake evidence.",
                evidence_ids=["metric_000000000000"],
                confidence=0.9,
            ),
        ],
        missing_evidence=["camera exposure log"],
        recommended_next_tests=["Rerun SCN-AEB-LVSB-01 with fusion trace enabled."],
    )
    run = DiagnosticAgent(FakeProvider([out]), max_repair_rounds=0).run(inputs.bundle)
    return run, verify_diagnosis(run, inputs)


@pytest.fixture(scope="module")
def report(
    inputs: DiagnosisInputs, verified: tuple[AgentRun, VerificationReport]
) -> DiagnosticReport:
    run, v = verified
    return build_report(inputs, run, v)


@pytest.fixture(scope="module")
def registry(inputs: DiagnosisInputs) -> EvidenceRegistry:
    return EvidenceRegistry.from_bundle(inputs.bundle, inputs.retrieval)


# --- structure ----------------------------------------------------------------------------------


def test_report_metadata_and_confidence(
    report: DiagnosticReport, verified: tuple[AgentRun, VerificationReport]
) -> None:
    run, v = verified
    m = report.metadata
    assert m.report_id.startswith("report_") and m.template_version == REPORT_TEMPLATE_VERSION
    assert m.run_id == run.run_id and m.verification_id == v.verification_id
    assert m.data_origin == "synthetic" and m.scenario_id is not None
    assert m.agent == "aeb_diagnostic_v1" and m.provider == "fake"
    assert report.report_confidence is v.report_confidence is ReportConfidence.HIGH
    assert report.disclaimer == DISCLAIMER
    assert report.approval.status == "pending_review" and report.approval.reviewer is None
    assert report.evidence_support_rate == pytest.approx(3 / 4)
    assert report.unsupported_claim_rate == pytest.approx(0.5)


def test_hypotheses_come_from_verification_only(
    report: DiagnosticReport, verified: tuple[AgentRun, VerificationReport]
) -> None:
    run, v = verified
    assert len(run.output.hypotheses) == 2 and len(v.hypotheses) == 1
    assert [h.cause for h in report.hypotheses] == [
        "Perception confidence dropout delayed the valid target."
    ]
    h = report.hypotheses[0]
    assert h.rank == 1 and h.failure_class is FailureClass.PERCEPTION_ERROR
    assert h.agent_confidence == 0.85 and h.adjusted_confidence == 0.85
    assert h.sources == ("telemetry", "doc:requirement")
    assert report.stripped_hypotheses == (
        "Made-up cause with fake evidence. (none of the cited evidence IDs resolve)",
    )
    assert "Made-up cause" not in " ".join(x.cause for x in report.hypotheses)


def test_executive_summary_is_templated_from_metrics(
    report: DiagnosticReport, inputs: DiagnosisInputs
) -> None:
    text = " ".join(report.executive_summary)
    lat = inputs.metrics.metric(M_BRAKING_LATENCY)
    assert f"by {lat.value:.3f} s against a limit of 0.3 s: FAIL" in text
    assert lat.metric_id in text
    assert "A collision was recorded at" in text
    assert "Top verified hypothesis (perception_error, high)" in text
    assert "Report confidence: HIGH; human review not triggered" in text
    # every sentence that states a fact carries at least one evidence ID
    for sentence in report.executive_summary:
        assert ID_TOKEN.search(sentence) or "verification_" in sentence, sentence


def test_timeline_and_metrics_table(report: DiagnosticReport, inputs: DiagnosisInputs) -> None:
    ts = [t.t_s for t in report.timeline]
    assert ts == sorted(ts) and len(report.timeline) >= len(inputs.metrics.events) + 3
    assert {t.kind for t in report.timeline} == {EvidenceKind.EVENT, EvidenceKind.METRIC}
    assert len(report.metrics_table) == len(inputs.metrics.metrics)
    latency_row = next(r for r in report.metrics_table if r.name == M_BRAKING_LATENCY)
    assert (
        latency_row.passed is False and latency_row.threshold == "<= 0.3" and latency_row.metric_id
    )
    collision_row = next(r for r in report.metrics_table if r.name == "collision")
    assert collision_row.threshold == "== false" and collision_row.observed == "yes"


def test_appendix_covers_every_cited_and_timeline_id(
    report: DiagnosticReport, registry: EvidenceRegistry
) -> None:
    appendix_ids = {e.evidence_id for e in report.evidence_appendix}
    for h in report.hypotheses:
        assert set(h.evidence_ids) <= appendix_ids
    assert {t.evidence_id for t in report.timeline} <= appendix_ids
    assert all(registry.resolve(e) is not None for e in appendix_ids)
    assert len(appendix_ids) == len(report.evidence_appendix)  # unique


def test_limitations_and_event_metadata(report: DiagnosticReport) -> None:
    lims = " ".join(report.limitations)
    assert "synthetic" in lims and "withheld by access control" in lims
    assert "removed as unsupported" in lims
    assert "clipped to the log bounds" in lims  # 10 s log, ±5 s window
    em = report.event_metadata
    assert em["data_origin"] == "synthetic" and em["sample_rate_hz"] == "50.0"
    assert "ego_speed_kmh [km/h] -> ego_speed_mps [m/s]" in em["unit_conversions"]
    assert em["quality_verdict"].startswith("pass (quality_")
    assert report.missing_evidence == ("camera exposure log",)


# --- rendering ------------------------------------------------------------------------------------


def test_markdown_has_all_sections_in_order_and_is_traceable(
    report: DiagnosticReport, registry: EvidenceRegistry
) -> None:
    md = render_markdown(report)
    headings = [line for line in md.splitlines() if line.startswith("## ")]
    assert headings == [
        "## 1. Executive summary",
        "## 2. Event metadata",
        "## 3. Evidence timeline",
        "## 4. Metrics",
        "## 5. Root-cause hypotheses (verified)",
        "## 6. Missing evidence",
        "## 7. Recommended next tests",
        "## 8. Limitations",
        "## 9. Approval",
        "## Appendix A. Evidence",
    ]
    assert "**Report confidence: HIGH**" in md
    assert DISCLAIMER in md
    assert "Made-up cause" in md and "Removed by the verifier" in md  # transparency
    assert "Evidence support rate: 75%" in md and "Unsupported claim rate: 50%" in md
    assert md == render_markdown(report)  # pure
    assert_traceable(md, registry, allowed=own_ids(report))  # no unknown IDs anywhere
    # the fake ID the agent invented must not appear as a citation in the markdown
    assert "metric_000000000000" not in md


def test_json_roundtrip(report: DiagnosticReport) -> None:
    text = render_json(report)
    data = json.loads(text)
    assert set(data) >= {"metadata", "executive_summary", "metrics_table", "hypotheses", "approval"}
    assert DiagnosticReport.model_validate_json(text) == report


def test_write_report_refuses_untraceable(
    report: DiagnosticReport, registry: EvidenceRegistry, tmp_path: Path
) -> None:
    md, js = write_report(report, tmp_path / "out", registry)
    assert md.name == MARKDOWN_NAME and js.name == JSON_NAME and md.exists() and js.exists()
    assert md.read_text(encoding="utf-8").startswith("# AEB Diagnostic Report")

    doctored = report.model_copy(
        update={"executive_summary": (*report.executive_summary, "See metric_beefbeefbeef.")}
    )
    with pytest.raises(EvidenceResolutionError, match="metric_beefbeefbeef"):
        write_report(doctored, tmp_path / "bad", registry)
    assert not (tmp_path / "bad").exists()


def test_build_report_rejects_unknown_hypothesis_evidence(
    inputs: DiagnosisInputs, verified: tuple[AgentRun, VerificationReport]
) -> None:
    run, v = verified
    bad_v = v.model_copy(
        update={
            "hypotheses": (
                v.hypotheses[0].model_copy(update={"resolved_ids": ("metric_cafecafecafe",)}),
            )
        }
    )
    with pytest.raises(EvidenceResolutionError, match="metric_cafecafecafe"):
        build_report(inputs, run, bad_v)
    with pytest.raises(ValueError, match="together"):
        build_report(inputs, run, None)


# --- metrics-only mode --------------------------------------------------------------------------


def test_metrics_only_report(
    inputs: DiagnosisInputs, registry: EvidenceRegistry, tmp_path: Path
) -> None:
    r = build_report(inputs, None, None)
    assert (
        r.metadata.run_id is None
        and r.metadata.verification_id is None
        and r.metadata.agent is None
    )
    assert r.hypotheses == () and r.recommended_next_tests == ()
    assert r.report_confidence is ReportConfidence.LOW
    assert any("No AI diagnosis was run" in s for s in r.executive_summary)
    assert any("No AI diagnosis was run" in lim for lim in r.limitations)
    assert any("synthetic" in lim for lim in r.limitations)
    assert r.evidence_support_rate is None
    md = render_markdown(r)
    assert "_No verified hypotheses._ No AI diagnosis was run." in md
    assert "| model | n/a |" in md
    write_report(r, tmp_path, registry)


# --- CLI ------------------------------------------------------------------------------------------


def test_cli_dry_run_writes_metrics_only_report(
    tmp_path: Path,
    index: ChunkIndex,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    write_scenario(
        generate_aeb_scenario(AebScenarioConfig(variant=ScenarioVariant.LATE_BRAKING, seed=5)),
        tmp_path,
    )
    index.save(tmp_path / "idx")
    out_dir = tmp_path / "report"
    rc = cli_main(
        [
            str(tmp_path / CSV_NAME),
            "--index",
            str(tmp_path / "idx"),
            "--access",
            "internal",
            "--dry-run",
            "--report-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    assert "report written:" in capsys.readouterr().out
    md = (out_dir / MARKDOWN_NAME).read_text(encoding="utf-8")
    assert "# AEB Diagnostic Report" in md and "No AI diagnosis was run" in md
    data = json.loads((out_dir / JSON_NAME).read_text(encoding="utf-8"))
    assert data["metadata"]["run_id"] is None
    assert any(
        row["name"] == M_BRAKING_LATENCY and row["passed"] is False for row in data["metrics_table"]
    )
