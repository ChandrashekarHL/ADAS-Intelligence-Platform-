"""M11 evaluation suite: locked cases, scoring rules, gate, baseline regression, CLI."""

import json
import re
from pathlib import Path

import pytest

from app.agents.schemas import AgentOutput, FailureClass, Hypothesis
from app.eval.cli import EXIT_GATE_FAILED, EXIT_NO_PROVIDER
from app.eval.cli import main as cli_main
from app.eval.runner import (
    RunOptions,
    load_report,
    render_markdown,
    run_case,
    run_suite,
    write_report,
)
from app.eval.schemas import (
    AgentCaseMetrics,
    CaseResult,
    CaseStatus,
    CheckResult,
    EvalReport,
    ExpectedOutcome,
    GateThresholds,
)
from app.eval.scoring import aggregate, apply_gate, check_deterministic
from app.eval.suite import LOCKED_SUITE, select_cases, suite_sha256
from app.llm.fake import FakeProvider
from app.llm.schemas import LLMRequest, Role
from app.metrics.schemas import EventType
from app.quality.report import QualityVerdict
from app.rag.index import ChunkIndex, build_index
from app.rag.schemas import AccessLevel, RetrievalFilters
from app.verification.schemas import ReportConfidence

DOCS = Path(__file__).resolve().parents[2] / "data" / "demo_docs"
INTERNAL = RetrievalFilters(allowed_access=frozenset({AccessLevel.PUBLIC, AccessLevel.INTERNAL}))
BY_ID = {c.case_id: c for c in LOCKED_SUITE}

# ----------------------------------------------------------------- scripted models


def _evidence_section(request: LLMRequest) -> str:
    text = "\n".join(m.content for m in request.messages if m.role is Role.USER)
    return text.split("EVIDENCE (cite only these IDs):", 1)[-1]


def good_model(request: LLMRequest) -> str:
    """Behaves like the engineer we want: names the perception cause on a fault, and
    reports nothing wrong on a nominal log."""
    section = _evidence_section(request)
    failing = re.findall(r"- (metric_[0-9a-f]{12}) t=[\d.]+s FAIL: (\w+)", section)
    chunks = re.findall(r"- (chunk_[0-9a-f]{12}): requirement", section)
    if not failing:
        return AgentOutput(
            observations=["All AEB metrics pass their thresholds; no requirement violated."],
            hypotheses=[],
            missing_evidence=[],
            recommended_next_tests=["Repeat with a longer confidence dropout to probe margins."],
        ).model_dump_json()
    latency = [m for m, n in failing if n == "braking_latency_s"]
    dropout = [m for m, n in failing if n == "confidence_dropout_during_risk_s"]
    return AgentOutput(
        observations=["Brake command lagged the TTC threshold crossing."],
        hypotheses=[
            Hypothesis(
                cause="Perception confidence collapsed during the risk phase, delaying the "
                "valid target and therefore the brake request.",
                failure_class=FailureClass.PERCEPTION_ERROR,
                evidence_ids=latency + dropout + chunks[:1],
                confidence=0.85,
            )
        ],
        missing_evidence=[],
        recommended_next_tests=["Replay with perception confidence held at nominal."],
    ).model_dump_json()


def bad_model(request: LLMRequest) -> str:
    """Blames the planner on every log, including nominal ones, citing real IDs."""
    section = _evidence_section(request)
    metrics = re.findall(r"- (metric_[0-9a-f]{12})", section)[:3]
    chunks = re.findall(r"- (chunk_[0-9a-f]{12}): requirement", section)[:1]
    return AgentOutput(
        observations=["The planner issued the brake command late."],
        hypotheses=[
            Hypothesis(
                cause="Planner logic delayed the brake command.",
                failure_class=FailureClass.PLANNER_LOGIC_ERROR,
                evidence_ids=metrics + chunks,
                confidence=0.95,
            )
        ],
        missing_evidence=[],
        recommended_next_tests=["Inspect planner timing."],
    ).model_dump_json()


@pytest.fixture(scope="module")
def fake() -> FakeProvider:
    return FakeProvider(embedding_dim=32)


@pytest.fixture(scope="module")
def index(fake: FakeProvider) -> ChunkIndex:
    return build_index(DOCS, fake)


SUBSET = ("late_braking_seed42", "nominal_seed42", "degraded_confidence_nan_burst")


def _run(model, index: ChunkIndex, work: Path, *, baseline: EvalReport | None = None,
         gate: GateThresholds | None = None) -> EvalReport:  # type: ignore[no-untyped-def]
    return run_suite(
        select_cases(list(SUBSET)),
        FakeProvider(scripted=model, embedding_dim=32),
        index=index,
        filters=INTERNAL,
        work_dir=work,
        options=RunOptions(run_agent=True),
        gate=gate,
        baseline=baseline,
    )


@pytest.fixture(scope="module")
def good_report(index: ChunkIndex, tmp_path_factory: pytest.TempPathFactory) -> EvalReport:
    return _run(good_model, index, tmp_path_factory.mktemp("good"))


@pytest.fixture(scope="module")
def bad_report(index: ChunkIndex, tmp_path_factory: pytest.TempPathFactory) -> EvalReport:
    return _run(bad_model, index, tmp_path_factory.mktemp("bad"))


# ----------------------------------------------------------------------- the suite


def test_suite_is_well_formed_and_locked() -> None:
    ids = [c.case_id for c in LOCKED_SUITE]
    assert len(ids) == len(set(ids)) and len(ids) >= 8
    for c in LOCKED_SUITE:
        if c.expected_outcome is ExpectedOutcome.FAULT:
            assert c.acceptable_failure_classes, c.case_id
            assert c.expected_metric_verdicts.get("braking_latency_s") is False, c.case_id
        elif c.expected_outcome is ExpectedOutcome.NOMINAL:
            assert EventType.COLLISION in c.forbidden_events, c.case_id
            assert all(c.expected_metric_verdicts.values()), c.case_id
        else:
            assert c.expected_quality is QualityVerdict.BLOCKED
            assert not c.expected_metric_verdicts
        if c.expected_quality is QualityVerdict.DEGRADED:
            assert c.max_report_confidence is ReportConfidence.MEDIUM
    # The hash is a function of the definitions only, and any edit changes it.
    assert suite_sha256() == suite_sha256()
    edited = (LOCKED_SUITE[0].model_copy(update={"description": "x"}), *LOCKED_SUITE[1:])
    assert suite_sha256(edited) != suite_sha256()


def test_suite_hash_is_stable_across_hash_seeds() -> None:
    """frozenset fields serialise in PYTHONHASHSEED order; the hash must not depend on it."""
    import os
    import subprocess
    import sys

    code = "from app.eval.suite import suite_sha256; print(suite_sha256())"
    seen = set()
    for seed in ("1", "2", "random"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
        )
        seen.add(out.stdout.strip())
    assert seen == {suite_sha256()}


def test_select_cases_keeps_suite_order_and_rejects_unknown() -> None:
    picked = select_cases(["nominal_seed42", "late_braking_seed42"])
    assert [c.case_id for c in picked] == ["late_braking_seed42", "nominal_seed42"]
    assert select_cases(None) == LOCKED_SUITE
    with pytest.raises(KeyError, match="unknown eval case"):
        select_cases(["nope"])


def test_deterministic_suite_passes_without_a_model(fake: FakeProvider, tmp_path: Path) -> None:
    """The CI gate: every locked expectation holds for the current pipeline."""
    report = run_suite(
        LOCKED_SUITE,
        fake,
        index=None,
        filters=INTERNAL,
        work_dir=tmp_path,
        options=RunOptions(run_agent=False),
    )
    assert report.gate.passed, report.gate.failures
    assert report.aggregates.n_cases == len(LOCKED_SUITE)
    assert report.aggregates.n_passed == len(LOCKED_SUITE)
    assert report.aggregates.n_agent_cases == 0
    assert report.aggregates.top1_accuracy is None
    by_id = {r.case_id: r for r in report.cases}
    blocked = by_id["blocked_missing_ego_speed"]
    assert blocked.deterministic_checks[0].name == "quality_blocked"
    assert "ego_speed_mps" in blocked.deterministic_checks[0].detail
    assert not blocked.agent_ran
    names = {c.name for c in by_id["late_braking_seed42"].deterministic_checks}
    assert {"quality_verdict", "event_present:collision", "metric:braking_latency_s",
            "ground_truth:collision", "ground_truth:braking_latency_s"} <= names


def test_wrong_expectations_are_detected(fake: FakeProvider, tmp_path: Path) -> None:
    """Scoring compares against the case, not against whatever the code produced."""
    from app.agents.pipeline import prepare_diagnosis
    from app.synthetic.aeb_generator import generate_aeb_scenario
    from app.synthetic.io import CSV_NAME, write_scenario

    nominal = BY_ID["nominal_seed42"]
    scenario = generate_aeb_scenario(nominal.config)
    write_scenario(scenario, tmp_path)
    inputs = prepare_diagnosis(tmp_path / CSV_NAME, fake, index=None, filters=INTERNAL)
    # Pretend this nominal log was expected to be the collision case.
    wrong = BY_ID["late_braking_seed42"]
    checks = check_deterministic(wrong, inputs.quality, inputs.metrics, scenario.ground_truth)
    failed = {c.name: c.detail for c in checks if not c.passed}
    assert "event_present:collision" in failed
    assert failed["metric:braking_latency_s"].startswith("expected FAIL, got PASS")
    # Ground truth from the generator still agrees with the metrics for this log.
    assert all(c.passed for c in checks if c.name.startswith("ground_truth:"))


# ------------------------------------------------------------------- agent scoring


def test_good_model_scores_perfectly(good_report: EvalReport) -> None:
    a = good_report.aggregates
    assert good_report.gate.passed, good_report.gate.failures
    assert a.n_agent_cases == 3 and a.n_fault_cases_scored == 2 and a.n_nominal_cases_scored == 1
    assert a.top1_accuracy == 1.0 and a.false_fault_rate == 0.0
    assert a.mean_evidence_support_rate == 1.0 and a.max_unsupported_claim_rate == 0.0
    assert a.latency_p50_s is not None and a.latency_p95_s is not None
    by_id = {r.case_id: r for r in good_report.cases}
    late = by_id["late_braking_seed42"]
    assert late.status is CaseStatus.PASS and late.agent is not None
    assert late.agent.top_failure_class is FailureClass.PERCEPTION_ERROR
    assert late.agent.acceptable_class_rank == 1
    degraded = by_id["degraded_confidence_nan_burst"]
    cap = next(c for c in degraded.agent_checks if c.name == "confidence_cap")
    assert cap.passed and degraded.agent is not None
    assert degraded.agent.report_confidence in (ReportConfidence.MEDIUM, ReportConfidence.LOW)
    nominal = by_id["nominal_seed42"]
    assert next(c for c in nominal.agent_checks if c.name == "no_false_fault").passed
    assert nominal.agent is not None and nominal.agent.hypotheses_verified == 0


def test_bad_model_is_caught_by_outcome_checks(bad_report: EvalReport) -> None:
    a = bad_report.aggregates
    assert a.top1_accuracy == 0.0 and a.false_fault_rate == 1.0
    by_id = {r.case_id: r for r in bad_report.cases}
    late = by_id["late_braking_seed42"]
    assert late.status is CaseStatus.FAIL
    top = next(c for c in late.agent_checks if c.name == "top_hypothesis_class")
    assert not top.passed and "top=planner_logic_error" in top.detail
    nominal = by_id["nominal_seed42"]
    ff = next(c for c in nominal.agent_checks if c.name == "no_false_fault")
    assert not ff.passed and "planner_logic_error@" in ff.detail
    # Deterministic checks are untouched by the model's quality.
    assert all(c.passed for r in bad_report.cases for c in r.deterministic_checks)
    # Without absolute accuracy bars the evidence gate alone still passes: the bad model
    # cited only real IDs. The regression gate below is what catches it.
    assert bad_report.gate.passed


def test_absolute_accuracy_thresholds_fail_the_gate(bad_report: EvalReport) -> None:
    gate = apply_gate(
        bad_report.cases,
        bad_report.aggregates,
        GateThresholds(min_top1_accuracy=0.5, max_false_fault_rate=0.0),
        suite_sha256=bad_report.suite_sha256,
    )
    assert not gate.passed
    assert any(f.startswith("top1_accuracy 0.00 < 0.50") for f in gate.failures)
    assert any(f.startswith("false_fault_rate 1.00 > 0.00") for f in gate.failures)


def test_regression_against_baseline(good_report: EvalReport, bad_report: EvalReport) -> None:
    gate = apply_gate(
        bad_report.cases,
        bad_report.aggregates,
        GateThresholds(),
        suite_sha256=bad_report.suite_sha256,
        baseline=good_report,
    )
    assert not gate.passed and gate.baseline_eval_id == good_report.eval_id
    joined = "\n".join(gate.failures)
    assert "top1_accuracy regressed: 0.000 vs baseline 1.000" in joined
    assert "false_fault_rate regressed: 1.000 vs baseline 0.000" in joined
    assert "late_braking_seed42 passed in baseline, now fail" in joined
    # The other direction is not a regression.
    improved = apply_gate(
        good_report.cases,
        good_report.aggregates,
        GateThresholds(),
        suite_sha256=good_report.suite_sha256,
        baseline=bad_report,
    )
    assert improved.passed


def test_baseline_from_another_suite_is_refused(good_report: EvalReport) -> None:
    foreign = good_report.model_copy(update={"suite_sha256": "0" * 64})
    gate = apply_gate(
        good_report.cases,
        good_report.aggregates,
        GateThresholds(),
        suite_sha256=good_report.suite_sha256,
        baseline=foreign,
    )
    assert not gate.passed and len(gate.failures) == 1
    assert "re-baseline" in gate.failures[0]


def test_provider_without_answers_is_an_error_not_a_pass(
    fake: FakeProvider, index: ChunkIndex, tmp_path: Path
) -> None:
    result = run_case(
        BY_ID["late_braking_seed42"],
        fake,  # no scripted answers: the agent raises ProviderError
        index=index,
        filters=INTERNAL,
        work_dir=tmp_path,
        options=RunOptions(run_agent=True),
    )
    assert result.status is CaseStatus.ERROR and result.agent_ran and result.agent is None
    assert result.error is not None and "ProviderError" in result.error
    gate = apply_gate([result], aggregate([result]), GateThresholds(), suite_sha256="x")
    assert not gate.passed and gate.failures[0].startswith("late_braking_seed42: error:")


def _agent_metrics(latency: float) -> AgentCaseMetrics:
    return AgentCaseMetrics(
        run_id="run_000000000001",
        verification_id="verification_000000000001",
        report_confidence=ReportConfidence.HIGH,
        human_review_required=False,
        evidence_support_rate=1.0,
        unsupported_claim_rate=0.0,
        latency_s=latency,
        prompt_tokens=10,
        completion_tokens=5,
        attempts=1,
        hypotheses_verified=1,
        hypotheses_stripped=0,
        unresolved_ids=0,
        injection_flags=0,
        top_failure_class=FailureClass.PERCEPTION_ERROR,
        acceptable_class_rank=1,
    )


def test_aggregate_percentiles_and_empty_input() -> None:
    empty = aggregate([])
    assert empty.n_cases == 0 and empty.top1_accuracy is None and empty.latency_p95_s is None
    results = [
        CaseResult(
            case_id=f"c{i}",
            expected_outcome=ExpectedOutcome.FAULT,
            status=CaseStatus.PASS,
            scenario_id=None,
            deterministic_checks=(),
            agent_checks=(CheckResult(name="top_hypothesis_class", passed=True, detail=""),),
            agent_ran=True,
            agent=_agent_metrics(lat),
        )
        for i, lat in enumerate([1.0, 2.0, 3.0, 4.0, 10.0])
    ]
    a = aggregate(results)
    assert a.latency_p50_s == 3.0 and a.latency_p95_s == 10.0
    assert a.prompt_tokens == 50 and a.completion_tokens == 25
    assert a.top1_accuracy == 1.0 and a.false_fault_rate is None


# ------------------------------------------------------------------ files and CLI


def test_write_and_reload_report_roundtrip(good_report: EvalReport, tmp_path: Path) -> None:
    md, js = write_report(good_report, tmp_path)
    assert load_report(js) == good_report
    text = md.read_text(encoding="utf-8")
    assert "gate: PASS" in text and "late_braking_seed42" in text
    assert "Synthetic scenarios only" in text
    assert "## Gate failures" not in text
    data = json.loads(js.read_text(encoding="utf-8"))
    assert data["suite_sha256"] == suite_sha256()


def test_render_markdown_lists_gate_failures(bad_report: EvalReport) -> None:
    gate = apply_gate(
        bad_report.cases,
        bad_report.aggregates,
        GateThresholds(min_top1_accuracy=1.0),
        suite_sha256=bad_report.suite_sha256,
    )
    text = render_markdown(bad_report.model_copy(update={"gate": gate}))
    assert "**gate: FAIL**" in text and "## Gate failures" in text
    assert "top1_accuracy 0.00 < 1.00" in text


def test_cli_no_agent_needs_no_provider_or_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CI runs the deterministic suite with the default provider config and no key."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # no backend/.env in reach
    code = cli_main(
        ["run", "--no-agent", "--out", str(tmp_path / "out"), "--work-dir", str(tmp_path / "w"),
         "--cases", "blocked_missing_ego_speed"]
    )
    assert code == 0


def test_cli_list_run_and_exit_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    assert cli_main(["list"]) == 0
    assert "suite sha256" in capsys.readouterr().out

    out = tmp_path / "out"
    code = cli_main(
        [
            "run",
            "--no-agent",
            "--out",
            str(out),
            "--work-dir",
            str(tmp_path / "work"),
            "--cases",
            "nominal_seed42,blocked_missing_ego_speed",
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "GATE: PASS" in printed and "blocked_missing_ego_speed" in printed
    written = sorted(out.glob("eval_*.json"))
    assert len(written) == 1
    report = load_report(written[0])
    assert [c.case_id for c in report.cases] == ["nominal_seed42", "blocked_missing_ego_speed"]
    assert not report.agent_enabled

    # Agent requested with the fake provider: refuse rather than pretend.
    assert cli_main(["run", "--out", str(out), "--work-dir", str(tmp_path / "w2")]) == (
        EXIT_NO_PROVIDER
    )
    assert cli_main(["run", "--no-agent", "--out", str(out), "--cases", "nope"]) == 2

    # A baseline from a different suite fails the gate (exit 1).
    foreign = report.model_copy(update={"suite_sha256": "0" * 64})
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(foreign.model_dump(mode="json")), encoding="utf-8")
    code = cli_main(
        [
            "run",
            "--no-agent",
            "--out",
            str(out),
            "--work-dir",
            str(tmp_path / "w3"),
            "--cases",
            "nominal_seed42",
            "--baseline",
            str(baseline),
        ]
    )
    assert code == EXIT_GATE_FAILED
    assert "re-baseline" in capsys.readouterr().out
