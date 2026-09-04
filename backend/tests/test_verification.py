"""M7 verifier: ID resolution, stripping, source counting, §28.1 confidence rules."""

from pathlib import Path

import pytest

from app.agents.diagnostic import DiagnosticAgent
from app.agents.pipeline import DiagnosisInputs, prepare_diagnosis
from app.agents.schemas import (
    AgentOutput,
    AgentRun,
    EvidenceKind,
    FailureClass,
    Hypothesis,
    InjectionFlag,
)
from app.llm.fake import FakeProvider
from app.llm.schemas import TokenUsage
from app.metrics.aeb import M_BRAKING_LATENCY, M_CONF_DROPOUT, M_MAX_DECEL
from app.quality.gates import GateResult, GateStatus
from app.quality.report import QualityReport, QualityVerdict
from app.rag.index import ChunkIndex, build_index
from app.rag.schemas import AccessLevel, RetrievalFilters, SourceType
from app.synthetic.aeb_generator import generate_aeb_scenario
from app.synthetic.io import CSV_NAME, write_scenario
from app.synthetic.schemas import AebScenarioConfig, ScenarioVariant
from app.verification.registry import EvidenceRegistry
from app.verification.schemas import (
    DISCLAIMER,
    ClaimStatus,
    ReportConfidence,
    band,
    cap,
)
from app.verification.verifier import verify, verify_diagnosis

DOCS = Path(__file__).resolve().parents[2] / "data" / "demo_docs"
INTERNAL = RetrievalFilters(allowed_access=frozenset({AccessLevel.PUBLIC, AccessLevel.INTERNAL}))


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
        generate_aeb_scenario(AebScenarioConfig(variant=ScenarioVariant.LATE_BRAKING, seed=31)), d
    )
    return prepare_diagnosis(d / CSV_NAME, fake, index=index, filters=INTERNAL, top_k=6)


def ids(inputs: DiagnosisInputs) -> dict[str, str]:
    """Handy IDs: failing metrics, a requirement chunk, an issue chunk."""
    m = inputs.metrics
    out = {
        "latency": m.metric(M_BRAKING_LATENCY).metric_id,
        "dropout": m.metric(M_CONF_DROPOUT).metric_id,
        "decel_pass": m.metric(M_MAX_DECEL).metric_id,
        "quality": inputs.quality.quality_id,
        "file": inputs.telemetry.provenance.file_id,
    }
    assert inputs.retrieval is not None
    for rc in inputs.retrieval.chunks:
        if rc.chunk.source_type is SourceType.REQUIREMENT and "req" not in out:
            out["req"] = rc.chunk_id
        if rc.chunk.source_type is SourceType.TEST_SPEC and "test" not in out:
            out["test"] = rc.chunk_id
        if rc.chunk.source_type is SourceType.DBC and "dbc" not in out:
            out["dbc"] = rc.chunk_id
    return out


def run_with(inputs: DiagnosisInputs, output: AgentOutput) -> AgentRun:
    return DiagnosticAgent(FakeProvider([output]), max_repair_rounds=0).run(inputs.bundle)


def hyp(
    cause: str, evidence: list[str], conf: float, fc: FailureClass = FailureClass.PERCEPTION_ERROR
) -> Hypothesis:
    return Hypothesis(cause=cause, failure_class=fc, evidence_ids=evidence, confidence=conf)


# --- registry ---------------------------------------------------------------------------------


def test_registry_resolves_every_offered_id_with_sources(inputs: DiagnosisInputs) -> None:
    reg = EvidenceRegistry.from_bundle(inputs.bundle, inputs.retrieval)
    assert reg.ids == inputs.bundle.offered_ids and len(reg) == len(inputs.bundle.items)
    i = ids(inputs)
    lat = reg.resolve(i["latency"])
    assert lat is not None and lat.kind is EvidenceKind.METRIC and lat.source == "telemetry"
    assert lat.t_s is not None and lat.passed is False
    req = reg.resolve(i["req"])
    assert req is not None and req.source == "doc:requirement" and req.t_s is None
    assert reg.resolve(i["quality"]).source == "quality"  # type: ignore[union-attr]
    assert reg.resolve("metric_000000000000") is None and "metric_000000000000" not in reg
    with pytest.raises(ValueError, match="duplicate"):
        EvidenceRegistry([lat, lat])


# --- happy path ---------------------------------------------------------------------------------


def test_multi_source_supported_hypothesis_yields_high(inputs: DiagnosisInputs) -> None:
    i = ids(inputs)
    out = AgentOutput(
        observations=["Brake command lagged the TTC crossing by 0.80 s."],
        hypotheses=[
            hyp(
                "Confidence dropout delayed the valid target.",
                [i["latency"], i["dropout"], i["req"], i["test"]],
                0.85,
            )
        ],
        missing_evidence=["camera exposure log"],
        recommended_next_tests=["Rerun SCN-AEB-LVSB-01 with fusion trace enabled."],
    )
    run = run_with(inputs, out)
    v = verify_diagnosis(run, inputs)
    assert v.verification_id.startswith("verification_") and v.run_id == run.run_id
    assert v.report_confidence is ReportConfidence.HIGH
    assert not v.human_review_required and v.review_reasons == ()
    top = v.top
    assert top is not None and top.status is ClaimStatus.SUPPORTED
    assert top.independent_sources == ("telemetry", "doc:requirement", "doc:test_spec")
    assert top.adjusted_confidence == 0.85 and top.confidence_label is ReportConfidence.HIGH
    assert set(top.resolved_ids) == {i["latency"], i["dropout"], i["req"], i["test"]}
    assert v.stripped == () and v.flagged_observations == ()
    assert v.evidence_support_rate == 1.0 and v.unsupported_claim_rate == 0.0
    assert v.cited_ids_total == 4 and v.cited_ids_resolved == 4
    # pipeline's missing metrics are merged with the agent's list (none missing here)
    assert v.missing_evidence == ("camera exposure log",)
    assert v.recommended_next_tests == ("Rerun SCN-AEB-LVSB-01 with fusion trace enabled.",)
    assert any("synthetic" in lim for lim in v.limitations)
    assert any("withheld by access control" in lim for lim in v.limitations)
    assert v.disclaimer == DISCLAIMER and "not certified safety tooling" in v.disclaimer
    assert v.cited_evidence_ids == top.resolved_ids


# --- stripping ------------------------------------------------------------------------------------


def test_unresolvable_ids_are_dropped_and_partial(inputs: DiagnosisInputs) -> None:
    i = ids(inputs)
    out = AgentOutput(hypotheses=[hyp("x", [i["latency"], "metric_ffffffffffff", i["req"]], 0.8)])
    v = verify_diagnosis(run_with(inputs, out), inputs)
    top = v.top
    assert top is not None and top.status is ClaimStatus.PARTIAL
    assert top.dropped_ids == ("metric_ffffffffffff",)
    assert any("did not resolve" in n for n in top.notes)
    assert v.evidence_support_rate == pytest.approx(2 / 3)
    assert [r.rule for r in v.applied_rules if r.hypothesis_index == 0][0] == "unresolvable_ids"


def test_hypotheses_without_resolvable_or_timestamped_evidence_are_stripped(
    inputs: DiagnosisInputs,
) -> None:
    i = ids(inputs)
    out = AgentOutput(
        hypotheses=[
            hyp("all fake", ["metric_aaaaaaaaaaaa", "chunk_bbbbbbbbbbbb"], 0.9),
            hyp("docs only", [i["req"], i["test"]], 0.9),
            hyp("metadata only", [i["quality"], i["file"]], 0.9),
            Hypothesis(cause="nothing cited", confidence=0.9),
            hyp("real one", [i["latency"], i["req"]], 0.7),
        ]
    )
    v = verify_diagnosis(run_with(inputs, out), inputs)
    assert [s.hypothesis.cause for s in v.stripped] == [
        "all fake",
        "docs only",
        "metadata only",
        "nothing cited",
    ]
    assert [s.reason for s in v.stripped] == [
        "none of the cited evidence IDs resolve",
        "no timestamped evidence for a root cause",
        "no timestamped evidence for a root cause",
        "no evidence cited",
    ]
    assert [h.hypothesis.cause for h in v.hypotheses] == ["real one"]
    assert v.unsupported_claim_rate == pytest.approx(0.8)
    assert v.evidence_support_rate == pytest.approx(6 / 8)
    assert any("4 agent hypothesis(es) removed" in lim for lim in v.limitations)
    rules = {r.rule for r in v.applied_rules}
    assert {"evidence_required", "timestamped_evidence_required", "unresolvable_ids"} <= rules


def test_no_supported_hypothesis_is_low(inputs: DiagnosisInputs) -> None:
    out = AgentOutput(hypotheses=[hyp("guess", ["metric_aaaaaaaaaaaa"], 0.95)])
    v = verify_diagnosis(run_with(inputs, out), inputs)
    assert v.hypotheses == () and v.report_confidence is ReportConfidence.LOW
    assert any(r.rule == "no_supported_hypothesis" for r in v.applied_rules)


# --- confidence rules -----------------------------------------------------------------------------


def test_single_source_caps_at_medium(inputs: DiagnosisInputs) -> None:
    i = ids(inputs)
    out = AgentOutput(hypotheses=[hyp("telemetry only", [i["latency"], i["dropout"]], 0.95)])
    v = verify_diagnosis(run_with(inputs, out), inputs)
    top = v.top
    assert top is not None
    assert top.independent_sources == ("telemetry",)
    assert top.agent_confidence == 0.95 and top.adjusted_confidence == 0.6
    assert top.confidence_label is ReportConfidence.MEDIUM
    assert v.report_confidence is ReportConfidence.MEDIUM
    assert any(r.rule == "single_source" and r.hypothesis_index == 0 for r in v.applied_rules)
    assert "single evidence source" in top.notes


def test_passing_metrics_only_is_penalised(inputs: DiagnosisInputs) -> None:
    i = ids(inputs)
    out = AgentOutput(
        hypotheses=[hyp("brakes were fine so it must be X", [i["decel_pass"], i["req"]], 0.8)]
    )
    v = verify_diagnosis(run_with(inputs, out), inputs)
    top = v.top
    assert top is not None and top.adjusted_confidence == pytest.approx(0.65)
    assert any(r.rule == "passing_metrics_only" for r in v.applied_rules)


def test_degraded_quality_caps_at_medium(inputs: DiagnosisInputs) -> None:
    i = ids(inputs)
    degraded = QualityReport(
        quality_id=inputs.quality.quality_id,
        file_id=inputs.quality.file_id,
        verdict=QualityVerdict.DEGRADED,
        gates=(
            *inputs.quality.gates[:-1],
            GateResult(gate="scenario_completeness", status=GateStatus.WARN, message="w"),
        ),
    )
    out = AgentOutput(
        hypotheses=[hyp("strong", [i["latency"], i["dropout"], i["req"], i["test"]], 0.9)]
    )
    run = run_with(inputs, out)
    registry = EvidenceRegistry.from_bundle(inputs.bundle, inputs.retrieval)
    v = verify(run, inputs.bundle, registry, degraded)
    assert v.report_confidence is ReportConfidence.MEDIUM
    assert any(r.rule == "quality_degraded" for r in v.applied_rules)
    assert any("Data quality degraded" in lim for lim in v.limitations)


def test_blocked_quality_is_blocked(inputs: DiagnosisInputs) -> None:
    i = ids(inputs)
    blocked = inputs.quality.model_copy(update={"verdict": QualityVerdict.BLOCKED})
    out = AgentOutput(hypotheses=[hyp("x", [i["latency"], i["req"]], 0.9)])
    registry = EvidenceRegistry.from_bundle(inputs.bundle, inputs.retrieval)
    v = verify(run_with(inputs, out), inputs.bundle, registry, blocked)
    assert v.report_confidence is ReportConfidence.BLOCKED


def test_missing_critical_metric_caps_at_low(
    tmp_path: Path, fake: FakeProvider, index: ChunkIndex
) -> None:
    import pandas as pd

    # AEB never braked: brake command time / latency are unavailable → critical metrics missing
    write_scenario(generate_aeb_scenario(AebScenarioConfig(seed=8)), tmp_path)
    csv = tmp_path / CSV_NAME
    df = pd.read_csv(csv)
    df["brake_command"] = 0
    df.to_csv(csv, index=False)
    inp = prepare_diagnosis(csv, fake, index=index, filters=INTERNAL)
    assert any(m.startswith(M_BRAKING_LATENCY) for m in inp.bundle.missing)
    i_ttc = inp.metrics.metric("ttc_threshold_crossing_s").metric_id
    assert inp.retrieval is not None
    req = inp.retrieval.chunks[0].chunk_id
    out = AgentOutput(
        hypotheses=[hyp("AEB never triggered", [i_ttc, req], 0.9, FailureClass.PLANNER_LOGIC_ERROR)]
    )
    v = verify_diagnosis(run_with(inp, out), inp)
    assert v.report_confidence is ReportConfidence.LOW
    assert any(r.rule == "critical_metric_missing" for r in v.applied_rules)
    assert any(M_BRAKING_LATENCY in lim for lim in v.limitations)
    # the pipeline's missing metrics reach the report even though the agent listed none
    assert any(m.startswith(M_BRAKING_LATENCY) for m in v.missing_evidence)


def test_competing_hypotheses_require_human_review(inputs: DiagnosisInputs) -> None:
    i = ids(inputs)
    out = AgentOutput(
        hypotheses=[
            hyp(
                "perception",
                [i["latency"], i["dropout"], i["req"]],
                0.8,
                FailureClass.PERCEPTION_ERROR,
            ),
            hyp("controller", [i["latency"], i["test"]], 0.75, FailureClass.CONTROL_ISSUE),
        ]
    )
    v = verify_diagnosis(run_with(inputs, out), inputs)
    assert v.human_review_required
    assert any("competing high-confidence hypotheses" in r for r in v.review_reasons)
    assert any(r.rule == "contradictory_evidence" for r in v.applied_rules)
    # same class at high confidence is not a contradiction
    same = AgentOutput(
        hypotheses=[
            hyp("a", [i["latency"], i["req"]], 0.8),
            hyp("b", [i["dropout"], i["test"]], 0.75),
        ]
    )
    assert not verify_diagnosis(run_with(inputs, same), inputs).human_review_required


def test_real_world_language_on_synthetic_data_is_flagged(inputs: DiagnosisInputs) -> None:
    i = ids(inputs)
    out = AgentOutput(
        observations=["This proves the real-world AEB is unsafe."],
        hypotheses=[hyp("x", [i["latency"], i["req"]], 0.7)],
    )
    v = verify_diagnosis(run_with(inputs, out), inputs)
    assert v.human_review_required
    assert any("real-world language" in r for r in v.review_reasons)
    assert any(r.rule == "synthetic_no_real_world_claims" for r in v.applied_rules)


def test_observations_with_unknown_ids_are_flagged(inputs: DiagnosisInputs) -> None:
    i = ids(inputs)
    out = AgentOutput(
        observations=[f"Latency {i['latency']} exceeded.", "See metric_123456789abc for proof."],
        hypotheses=[hyp("x", [i["latency"], i["req"]], 0.7)],
    )
    v = verify_diagnosis(run_with(inputs, out), inputs)
    assert v.flagged_observations == ("See metric_123456789abc for proof.",)


def test_injection_flagged_and_stale_sources_are_penalised(inputs: DiagnosisInputs) -> None:
    i = ids(inputs)
    bundle = inputs.bundle
    flagged_bundle = bundle.__class__(
        items=bundle.items,
        missing=bundle.missing,
        chunk_texts=bundle.chunk_texts,
        injection_flags=(
            InjectionFlag(evidence_id=i["req"], pattern="role_override", excerpt="…"),
        ),
        data_origin=bundle.data_origin,
        excluded_by_access=bundle.excluded_by_access,
        stale_chunk_ids=bundle.stale_chunk_ids,
    )
    assert inputs.retrieval is not None
    stale_retrieval = inputs.retrieval.model_copy(
        update={
            "chunks": tuple(
                rc.model_copy(update={"stale": rc.chunk_id == i["test"]})
                for rc in inputs.retrieval.chunks
            )
        }
    )
    registry = EvidenceRegistry.from_bundle(flagged_bundle, stale_retrieval)
    out = AgentOutput(hypotheses=[hyp("x", [i["latency"], i["req"], i["test"]], 0.9)])
    run = DiagnosticAgent(FakeProvider([out]), max_repair_rounds=0).run(flagged_bundle)
    v = verify(run, flagged_bundle, registry, inputs.quality)
    top = v.top
    assert top is not None and top.adjusted_confidence == pytest.approx(0.6)  # 0.9 - 0.2 - 0.1
    assert v.report_confidence is ReportConfidence.MEDIUM
    assert v.human_review_required and any("instruction-like text" in r for r in v.review_reasons)
    assert {r.rule for r in v.applied_rules} >= {"stale_document", "injection_flagged_source"}


def test_unresolved_after_repair_is_recorded(inputs: DiagnosisInputs) -> None:
    i = ids(inputs)
    out = AgentOutput(hypotheses=[hyp("x", [i["latency"], i["req"], "chunk_deadbeef0000"], 0.7)])
    run = run_with(inputs, out)
    assert run.unresolved_ids == ("chunk_deadbeef0000",)
    v = verify_diagnosis(run, inputs)
    assert any(r.rule == "agent_unresolved_after_repair" for r in v.applied_rules)


def test_ranking_and_determinism(inputs: DiagnosisInputs) -> None:
    i = ids(inputs)
    out = AgentOutput(
        hypotheses=[
            hyp("weaker first", [i["latency"]], 0.9),  # single source → 0.6
            hyp("stronger second", [i["dropout"], i["req"]], 0.7),
        ]
    )
    run = run_with(inputs, out)
    a, b = verify_diagnosis(run, inputs), verify_diagnosis(run, inputs)
    assert [h.hypothesis.cause for h in a.hypotheses] == ["stronger second", "weaker first"]
    assert a.model_dump(exclude={"verification_id"}) == b.model_dump(exclude={"verification_id"})
    assert a.verification_id != b.verification_id


def test_band_and_cap_helpers() -> None:
    assert band(0.7) is ReportConfidence.HIGH and band(0.69) is ReportConfidence.MEDIUM
    assert band(0.4) is ReportConfidence.MEDIUM and band(0.39) is ReportConfidence.LOW
    assert cap(ReportConfidence.HIGH, ReportConfidence.MEDIUM) is ReportConfidence.MEDIUM
    assert cap(ReportConfidence.LOW, ReportConfidence.MEDIUM) is ReportConfidence.LOW
    assert cap(ReportConfidence.BLOCKED, ReportConfidence.HIGH) is ReportConfidence.BLOCKED


def test_token_usage_helper_unused_guard() -> None:
    assert TokenUsage().total_tokens == 0
