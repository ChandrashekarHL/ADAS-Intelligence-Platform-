"""Run the locked suite: generate each scenario, run the pipeline, optionally the agent."""

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.agents.diagnostic import DEFAULT_QUESTION, DiagnosticAgent
from app.agents.pipeline import prepare_diagnosis
from app.core.errors import AipError, DataQualityError
from app.eval.schemas import (
    CaseResult,
    CaseStatus,
    CheckResult,
    EvalCase,
    EvalReport,
    ExpectedOutcome,
    GateThresholds,
)
from app.eval.scoring import (
    aggregate,
    apply_gate,
    check_agent,
    check_blocked,
    check_deterministic,
    status_of,
)
from app.eval.suite import SUITE_NAME, suite_sha256
from app.llm.provider import LLMProvider
from app.metrics.schemas import AebThresholds
from app.rag.index import ChunkIndex
from app.rag.schemas import RetrievalFilters
from app.synthetic.aeb_generator import generate_aeb_scenario
from app.synthetic.io import CSV_NAME, write_scenario
from app.verification.verifier import verify_diagnosis


@dataclass(frozen=True)
class RunOptions:
    run_agent: bool = True
    question: str = DEFAULT_QUESTION
    top_k: int = 6
    thresholds: AebThresholds | None = None


def run_case(
    case: EvalCase,
    provider: LLMProvider,
    *,
    index: ChunkIndex | None,
    filters: RetrievalFilters,
    work_dir: Path,
    options: RunOptions,
) -> CaseResult:
    """Score one case. Generation and the deterministic stages never call a model."""
    scenario = generate_aeb_scenario(case.config)
    case_dir = work_dir / case.case_id
    metadata = write_scenario(scenario, case_dir)

    try:
        inputs = prepare_diagnosis(
            case_dir / CSV_NAME,
            provider,
            index=index,
            filters=filters,
            thresholds=options.thresholds,
            top_k=options.top_k,
        )
    except DataQualityError as exc:
        checks = check_blocked(case, str(exc))
        return CaseResult(
            case_id=case.case_id,
            expected_outcome=case.expected_outcome,
            status=status_of(checks, None),
            scenario_id=metadata.scenario_id,
            deterministic_checks=tuple(checks),
            agent_checks=(),
            agent_ran=False,
        )

    det: list[CheckResult]
    if case.expected_outcome is ExpectedOutcome.BLOCKED:
        det = check_blocked(case, None)
    else:
        det = check_deterministic(case, inputs.quality, inputs.metrics, scenario.ground_truth)

    if not options.run_agent or case.expected_outcome is ExpectedOutcome.BLOCKED:
        return CaseResult(
            case_id=case.case_id,
            expected_outcome=case.expected_outcome,
            status=status_of(det, None),
            scenario_id=metadata.scenario_id,
            deterministic_checks=tuple(det),
            agent_checks=(),
            agent_ran=False,
        )

    try:
        run = DiagnosticAgent(provider).run(inputs.bundle, options.question)
    except AipError as exc:
        return CaseResult(
            case_id=case.case_id,
            expected_outcome=case.expected_outcome,
            status=CaseStatus.ERROR,
            scenario_id=metadata.scenario_id,
            deterministic_checks=tuple(det),
            agent_checks=(),
            agent_ran=True,
            error=f"{type(exc).__name__}: {exc}",
        )
    verification = verify_diagnosis(run, inputs)
    agent_checks, agent_metrics = check_agent(case, run, verification)
    return CaseResult(
        case_id=case.case_id,
        expected_outcome=case.expected_outcome,
        status=status_of([*det, *agent_checks], None),
        scenario_id=metadata.scenario_id,
        deterministic_checks=tuple(det),
        agent_checks=tuple(agent_checks),
        agent_ran=True,
        agent=agent_metrics,
    )


def run_suite(
    cases: tuple[EvalCase, ...],
    provider: LLMProvider,
    *,
    index: ChunkIndex | None,
    filters: RetrievalFilters,
    work_dir: Path,
    options: RunOptions | None = None,
    gate: GateThresholds | None = None,
    baseline: EvalReport | None = None,
) -> EvalReport:
    options = options if options is not None else RunOptions()
    gate = gate if gate is not None else GateThresholds()
    work_dir.mkdir(parents=True, exist_ok=True)
    results = tuple(
        run_case(
            case, provider, index=index, filters=filters, work_dir=work_dir, options=options
        )
        for case in cases
    )
    agg = aggregate(results)
    # Identity of the suite *definitions*, not of the subset that ran: a subset run can
    # still be compared per case against a full baseline.
    sha = suite_sha256()
    return EvalReport(
        eval_id=f"eval_{uuid.uuid4().hex[:12]}",
        suite_name=SUITE_NAME,
        suite_sha256=sha,
        provider=provider.name,
        model=provider.model,
        agent_enabled=options.run_agent,
        question_sha256=hashlib.sha256(options.question.encode("utf-8")).hexdigest(),
        cases=results,
        aggregates=agg,
        gate=apply_gate(results, agg, gate, suite_sha256=sha, baseline=baseline),
    )


def load_report(path: Path) -> EvalReport:
    return EvalReport.model_validate_json(path.read_text(encoding="utf-8"))


def write_report(report: EvalReport, out_dir: Path) -> tuple[Path, Path]:
    """``<out_dir>/<eval_id>.json`` (full record) and ``.md`` (human summary)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    js = out_dir / f"{report.eval_id}.json"
    md = out_dir / f"{report.eval_id}.md"
    js.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md.write_text(render_markdown(report), encoding="utf-8")
    return md, js


def _fmt(x: float | None, pct: bool = False) -> str:
    if x is None:
        return "n/a"
    return f"{x:.0%}" if pct else f"{x:.2f}"


def render_markdown(report: EvalReport) -> str:
    a = report.aggregates
    lines = [
        f"# Eval `{report.eval_id}` — {report.suite_name}",
        "",
        f"- generated: {report.generated_at.isoformat(timespec='seconds')}",
        f"- provider/model: {report.provider} / {report.model}"
        + ("" if report.agent_enabled else " (agent disabled: deterministic checks only)"),
        f"- suite sha256: `{report.suite_sha256[:16]}`  question sha256: "
        f"`{report.question_sha256[:16]}`",
        f"- **gate: {'PASS' if report.gate.passed else 'FAIL'}**"
        + (f" (baseline {report.gate.baseline_eval_id})" if report.gate.baseline_eval_id else ""),
        "",
        "## Aggregates",
        "",
        "| cases | pass | fail | error | agent cases | top-1 accuracy | false-fault rate | "
        "evidence support | max unsupported | p50 latency | p95 latency | tokens in/out |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
        f"| {a.n_cases} | {a.n_passed} | {a.n_failed} | {a.n_errors} | {a.n_agent_cases} | "
        f"{_fmt(a.top1_accuracy, True)} | {_fmt(a.false_fault_rate, True)} | "
        f"{_fmt(a.mean_evidence_support_rate, True)} | "
        f"{_fmt(a.max_unsupported_claim_rate, True)} | {_fmt(a.latency_p50_s)} s | "
        f"{_fmt(a.latency_p95_s)} s | {a.prompt_tokens}/{a.completion_tokens} |",
        "",
        "## Cases",
        "",
        "| case | expected | status | top class | report confidence | review | support | "
        "latency | failed checks |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report.cases:
        ag = r.agent
        failed = "; ".join(f"{c.name}: {c.detail}" for c in r.failed_checks) or (
            r.error or "—"
        )
        lines.append(
            f"| {r.case_id} | {r.expected_outcome.value} | {r.status.value} | "
            f"{ag.top_failure_class.value if ag and ag.top_failure_class else '—'} | "
            f"{ag.report_confidence.value if ag else '—'} | "
            f"{'yes' if ag and ag.human_review_required else ('no' if ag else '—')} | "
            f"{_fmt(ag.evidence_support_rate, True) if ag else '—'} | "
            f"{_fmt(ag.latency_s) + ' s' if ag else '—'} | {failed} |"
        )
    if report.gate.failures:
        lines += ["", "## Gate failures", ""]
        lines += [f"- {f}" for f in report.gate.failures]
    lines += [
        "",
        "_Synthetic scenarios only; these numbers measure the diagnostic pipeline, not "
        "real-world AEB performance._",
        "",
    ]
    return "\n".join(lines)
