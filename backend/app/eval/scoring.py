"""Pure scoring: expectations vs what the pipeline, agent and verifier produced.

Nothing here calls a model or touches disk, so every rule is unit-testable with fixtures.
"""

import statistics
from collections.abc import Sequence

from app.agents.schemas import AgentRun, FailureClass
from app.eval.schemas import (
    AgentCaseMetrics,
    Aggregates,
    CaseResult,
    CaseStatus,
    CheckResult,
    EvalCase,
    EvalReport,
    ExpectedOutcome,
    GateResult,
    GateThresholds,
)
from app.metrics.schemas import AebMetricsReport
from app.quality.report import QualityReport
from app.synthetic.schemas import ScenarioGroundTruth
from app.verification.schemas import ReportConfidence, VerificationReport

# One sample at the generator's default 50 Hz is 0.02 s; allow two samples of slack
# between the generator's continuous-time ground truth and the sampled metric.
GROUND_TRUTH_TOLERANCE_S = 0.05

_CONFIDENCE_ORDER = {
    ReportConfidence.BLOCKED: 0,
    ReportConfidence.LOW: 1,
    ReportConfidence.MEDIUM: 2,
    ReportConfidence.HIGH: 3,
}
# A fault hypothesis at or above this label on a nominal log is a false alarm.
FALSE_FAULT_MIN_LABEL = ReportConfidence.MEDIUM


def check_blocked(case: EvalCase, error: str | None) -> list[CheckResult]:
    """Deterministic checks for a case where the gates were expected to refuse the file."""
    if case.expected_outcome is ExpectedOutcome.BLOCKED:
        return [
            CheckResult(
                name="quality_blocked",
                passed=error is not None,
                detail=error or "gates let the file through; expected BLOCKED",
            )
        ]
    return [
        CheckResult(
            name="quality_verdict",
            passed=False,
            detail=f"expected {case.expected_quality.value}, gates blocked: {error}",
        )
    ]


def check_deterministic(
    case: EvalCase,
    quality: QualityReport,
    metrics: AebMetricsReport,
    ground_truth: ScenarioGroundTruth,
) -> list[CheckResult]:
    """Quality verdict, event set, locked metric verdicts, and metrics vs generator truth."""
    checks = [
        CheckResult(
            name="quality_verdict",
            passed=quality.verdict is case.expected_quality,
            detail=f"expected {case.expected_quality.value}, got {quality.verdict.value}",
        )
    ]
    present = {e.event_type for e in metrics.events}
    for ev in case.expected_events:
        checks.append(
            CheckResult(
                name=f"event_present:{ev.value}",
                passed=ev in present,
                detail="present" if ev in present else "missing",
            )
        )
    for ev in case.forbidden_events:
        checks.append(
            CheckResult(
                name=f"event_absent:{ev.value}",
                passed=ev not in present,
                detail="absent" if ev not in present else "present but forbidden",
            )
        )
    by_name = {m.name: m for m in metrics.metrics}
    for name, expected in case.expected_metric_verdicts.items():
        m = by_name.get(name)
        if m is None or m.passed is None:
            got = "missing" if m is None else f"no verdict ({m.missing_reason or 'no threshold'})"
            checks.append(CheckResult(name=f"metric:{name}", passed=False, detail=got))
            continue
        checks.append(
            CheckResult(
                name=f"metric:{name}",
                passed=m.passed is expected,
                detail=f"expected {'PASS' if expected else 'FAIL'}, got "
                f"{'PASS' if m.passed else 'FAIL'} (value={m.value})",
            )
        )
    checks.extend(_ground_truth_checks(metrics, ground_truth))
    return checks


def _ground_truth_checks(
    metrics: AebMetricsReport, truth: ScenarioGroundTruth
) -> list[CheckResult]:
    """The generator knows what it produced; the metrics must agree within one or two
    samples. This is the only oracle independent of the metrics code."""
    by_name = {m.name: m for m in metrics.metrics}
    out: list[CheckResult] = []
    collision = by_name.get("collision")
    got_collision = bool(collision.value) if collision is not None else None
    out.append(
        CheckResult(
            name="ground_truth:collision",
            passed=got_collision is truth.collision,
            detail=f"generator={truth.collision}, metrics={got_collision}",
        )
    )
    latency = by_name.get("braking_latency_s")
    if truth.braking_latency_s is not None:
        got = latency.value if latency is not None else None
        ok = isinstance(got, float | int) and not isinstance(got, bool) and (
            abs(float(got) - truth.braking_latency_s) <= GROUND_TRUTH_TOLERANCE_S
        )
        out.append(
            CheckResult(
                name="ground_truth:braking_latency_s",
                passed=bool(ok),
                detail=f"generator={truth.braking_latency_s:.3f}, metrics={got}",
            )
        )
    return out


def _label_at_least(label: ReportConfidence, floor: ReportConfidence) -> bool:
    return _CONFIDENCE_ORDER[label] >= _CONFIDENCE_ORDER[floor]


def check_agent(
    case: EvalCase, run: AgentRun, verification: VerificationReport
) -> tuple[list[CheckResult], AgentCaseMetrics]:
    """Outcome-specific checks on the *verified* ranking plus the recorded numbers."""
    ranked = verification.hypotheses
    classes = [h.hypothesis.failure_class for h in ranked]
    top = classes[0] if classes else None
    rank = next(
        (i + 1 for i, c in enumerate(classes) if c in case.acceptable_failure_classes), None
    )
    checks: list[CheckResult] = []

    if case.expected_outcome is ExpectedOutcome.FAULT:
        if top is None:
            checks.append(
                CheckResult(
                    name="top_hypothesis_class",
                    passed=False,
                    detail="no supported hypothesis survived verification",
                )
            )
        else:
            checks.append(
                CheckResult(
                    name="top_hypothesis_class",
                    passed=top in case.acceptable_failure_classes,
                    detail=f"top={top.value}, acceptable="
                    f"{sorted(c.value for c in case.acceptable_failure_classes)}, "
                    f"first acceptable at rank {rank}",
                )
            )
    elif case.expected_outcome is ExpectedOutcome.NOMINAL:
        false_faults = [
            f"{h.hypothesis.failure_class.value}@{h.confidence_label.value}"
            for h in ranked
            if h.hypothesis.failure_class is not FailureClass.UNKNOWN
            and _label_at_least(h.confidence_label, FALSE_FAULT_MIN_LABEL)
        ]
        checks.append(
            CheckResult(
                name="no_false_fault",
                passed=not false_faults,
                detail="no fault hypothesis at or above "
                f"{FALSE_FAULT_MIN_LABEL.value}"
                if not false_faults
                else f"fault hypotheses on a nominal log: {false_faults}",
            )
        )

    if case.max_report_confidence is not None:
        cap_ok = (
            _CONFIDENCE_ORDER[verification.report_confidence]
            <= _CONFIDENCE_ORDER[case.max_report_confidence]
        )
        checks.append(
            CheckResult(
                name="confidence_cap",
                passed=cap_ok,
                detail=f"report {verification.report_confidence.value}, "
                f"cap {case.max_report_confidence.value}",
            )
        )

    agent = AgentCaseMetrics(
        run_id=run.run_id,
        verification_id=verification.verification_id,
        report_confidence=verification.report_confidence,
        human_review_required=verification.human_review_required,
        evidence_support_rate=verification.evidence_support_rate,
        unsupported_claim_rate=verification.unsupported_claim_rate,
        latency_s=run.latency_s,
        prompt_tokens=run.usage.prompt_tokens,
        completion_tokens=run.usage.completion_tokens,
        attempts=run.attempts,
        hypotheses_verified=len(ranked),
        hypotheses_stripped=len(verification.stripped),
        unresolved_ids=len(run.unresolved_ids),
        injection_flags=len(run.injection_flags),
        top_failure_class=top,
        acceptable_class_rank=rank,
    )
    return checks, agent


def status_of(checks: Sequence[CheckResult], error: str | None) -> CaseStatus:
    if error is not None:
        return CaseStatus.ERROR
    return CaseStatus.PASS if all(c.passed for c in checks) else CaseStatus.FAIL


def _percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile; exact for the small samples an eval run produces."""
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round(q * (len(ordered) - 1))))
    return ordered[k]


def aggregate(results: Sequence[CaseResult]) -> Aggregates:
    scored = [r for r in results if r.agent is not None]
    faults = [
        r
        for r in scored
        if r.expected_outcome is ExpectedOutcome.FAULT
        and any(c.name == "top_hypothesis_class" for c in r.agent_checks)
    ]
    nominals = [
        r
        for r in scored
        if r.expected_outcome is ExpectedOutcome.NOMINAL
        and any(c.name == "no_false_fault" for c in r.agent_checks)
    ]
    top1 = (
        sum(
            1
            for r in faults
            if any(c.name == "top_hypothesis_class" and c.passed for c in r.agent_checks)
        )
        / len(faults)
        if faults
        else None
    )
    false_fault = (
        sum(
            1
            for r in nominals
            if any(c.name == "no_false_fault" and not c.passed for c in r.agent_checks)
        )
        / len(nominals)
        if nominals
        else None
    )
    metrics = [r.agent for r in scored if r.agent is not None]
    latencies = [m.latency_s for m in metrics]
    return Aggregates(
        n_cases=len(results),
        n_passed=sum(1 for r in results if r.status is CaseStatus.PASS),
        n_failed=sum(1 for r in results if r.status is CaseStatus.FAIL),
        n_errors=sum(1 for r in results if r.status is CaseStatus.ERROR),
        n_agent_cases=len(scored),
        n_fault_cases_scored=len(faults),
        n_nominal_cases_scored=len(nominals),
        top1_accuracy=top1,
        false_fault_rate=false_fault,
        mean_evidence_support_rate=(
            statistics.fmean(m.evidence_support_rate for m in metrics) if metrics else None
        ),
        max_unsupported_claim_rate=(
            max(m.unsupported_claim_rate for m in metrics) if metrics else None
        ),
        latency_p50_s=_percentile(latencies, 0.5) if latencies else None,
        latency_p95_s=_percentile(latencies, 0.95) if latencies else None,
        prompt_tokens=sum(m.prompt_tokens for m in metrics),
        completion_tokens=sum(m.completion_tokens for m in metrics),
        n_human_review_required=sum(1 for m in metrics if m.human_review_required),
    )


def apply_gate(
    results: Sequence[CaseResult],
    agg: Aggregates,
    thresholds: GateThresholds,
    *,
    suite_sha256: str,
    baseline: EvalReport | None = None,
) -> GateResult:
    """Release gate: every deterministic check, the evidence gate, optional absolute
    accuracy bars, and "no worse than the baseline" when one is supplied."""
    failures: list[str] = []
    for r in results:
        if r.status is CaseStatus.ERROR:
            failures.append(f"{r.case_id}: error: {r.error}")
        for c in r.deterministic_checks:
            if not c.passed:
                failures.append(f"{r.case_id}: {c.name}: {c.detail}")
    if (
        agg.max_unsupported_claim_rate is not None
        and agg.max_unsupported_claim_rate > thresholds.max_unsupported_claim_rate
    ):
        failures.append(
            f"unsupported_claim_rate {agg.max_unsupported_claim_rate:.2f} > "
            f"{thresholds.max_unsupported_claim_rate:.2f}"
        )
    if (
        agg.mean_evidence_support_rate is not None
        and agg.mean_evidence_support_rate < thresholds.min_evidence_support_rate
    ):
        failures.append(
            f"evidence_support_rate {agg.mean_evidence_support_rate:.2f} < "
            f"{thresholds.min_evidence_support_rate:.2f}"
        )
    if (
        thresholds.min_top1_accuracy is not None
        and agg.top1_accuracy is not None
        and agg.top1_accuracy < thresholds.min_top1_accuracy
    ):
        failures.append(
            f"top1_accuracy {agg.top1_accuracy:.2f} < {thresholds.min_top1_accuracy:.2f}"
        )
    if (
        thresholds.max_false_fault_rate is not None
        and agg.false_fault_rate is not None
        and agg.false_fault_rate > thresholds.max_false_fault_rate
    ):
        failures.append(
            f"false_fault_rate {agg.false_fault_rate:.2f} > {thresholds.max_false_fault_rate:.2f}"
        )
    if baseline is not None:
        failures.extend(compare_with_baseline(results, agg, suite_sha256, baseline))
    return GateResult(
        passed=not failures,
        failures=tuple(failures),
        thresholds=thresholds,
        baseline_eval_id=baseline.eval_id if baseline is not None else None,
    )


def compare_with_baseline(
    results: Sequence[CaseResult], agg: Aggregates, suite_sha256: str, baseline: EvalReport
) -> list[str]:
    """Regression rule: no case that passed before fails now and, when the same set of
    cases ran, no aggregate worse than before. Aggregates over different case sets are not
    comparable, so a subset run is judged per case only."""
    if baseline.suite_sha256 != suite_sha256:
        return [
            f"baseline {baseline.eval_id} was produced by a different suite "
            f"({baseline.suite_sha256[:12]} vs {suite_sha256[:12]}); re-baseline first"
        ]
    out: list[str] = []
    b = baseline.aggregates
    same_cases = {r.case_id for r in results} == {r.case_id for r in baseline.cases}

    def worse(
        name: str, now: float | None, before: float | None, *, higher_is_better: bool
    ) -> None:
        if now is None or before is None:
            return
        regressed = now < before if higher_is_better else now > before
        if regressed:
            out.append(f"{name} regressed: {now:.3f} vs baseline {before:.3f}")

    if same_cases:
        worse("top1_accuracy", agg.top1_accuracy, b.top1_accuracy, higher_is_better=True)
        worse(
            "false_fault_rate", agg.false_fault_rate, b.false_fault_rate, higher_is_better=False
        )
        worse(
            "mean_evidence_support_rate",
            agg.mean_evidence_support_rate,
            b.mean_evidence_support_rate,
            higher_is_better=True,
        )
        worse(
            "max_unsupported_claim_rate",
            agg.max_unsupported_claim_rate,
            b.max_unsupported_claim_rate,
            higher_is_better=False,
        )
    before_status = {r.case_id: r.status for r in baseline.cases}
    for r in results:
        if before_status.get(r.case_id) is CaseStatus.PASS and r.status is not CaseStatus.PASS:
            out.append(f"{r.case_id} passed in baseline, now {r.status.value}")
    return out


__all__ = [
    "FALSE_FAULT_MIN_LABEL",
    "GROUND_TRUTH_TOLERANCE_S",
    "aggregate",
    "apply_gate",
    "check_agent",
    "check_blocked",
    "check_deterministic",
    "compare_with_baseline",
    "status_of",
]
