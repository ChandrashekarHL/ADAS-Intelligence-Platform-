"""Assemble a DiagnosticReport from pipeline inputs plus (optionally) a verified agent run.

The executive summary is templated from measured facts, never written by the model, so it
cannot say anything the metrics do not support. When no agent run is supplied the report
is a metrics-only report and says so; the pipeline is useful offline.
"""

import re

from app.agents.pipeline import DiagnosisInputs
from app.agents.schemas import AgentRun, EvidenceKind
from app.core.errors import EvidenceResolutionError
from app.core.ids import new_id
from app.metrics.aeb import (
    M_BRAKE_CMD,
    M_BRAKING_LATENCY,
    M_COLLISION,
    M_COLLISION_TIME,
    M_CONF_DROPOUT,
    M_MIN_TTC,
    M_TTC_CROSSING,
    fmt_value,
)
from app.metrics.schemas import MetricResult
from app.reports.schemas import (
    ApprovalSection,
    DiagnosticReport,
    EvidenceAppendixEntry,
    HypothesisRow,
    MetricRow,
    ReportMetadata,
    TimelineEntry,
)
from app.verification.registry import EvidenceRegistry
from app.verification.schemas import ReportConfidence, VerificationReport

_ID_TOKEN = re.compile(
    r"\b(?:metric|event|window|chunk|quality|file|doc|run|verification|report|scenario)_"
    r"[0-9a-f]{12}\b"
)


def _metric_or_none(inputs: DiagnosisInputs, name: str) -> MetricResult | None:
    try:
        m = inputs.metrics.metric(name)
    except KeyError:
        return None
    return m if m.available else None


def _threshold_str(m: MetricResult) -> str:
    if m.comparator is None or m.threshold is None:
        return ""
    thr = "false" if m.threshold is False else "true" if m.threshold is True else str(m.threshold)
    return f"{m.comparator.value} {thr}"


def executive_summary(
    inputs: DiagnosisInputs, verification: VerificationReport | None
) -> tuple[str, ...]:
    lines: list[str] = []
    crossing = _metric_or_none(inputs, M_TTC_CROSSING)
    brake = _metric_or_none(inputs, M_BRAKE_CMD)
    latency = _metric_or_none(inputs, M_BRAKING_LATENCY)
    collision = _metric_or_none(inputs, M_COLLISION)
    min_ttc = _metric_or_none(inputs, M_MIN_TTC)
    dropout = _metric_or_none(inputs, M_CONF_DROPOUT)

    if crossing and brake and latency:
        verdict = "PASS" if latency.passed else "FAIL"
        lines.append(
            f"The AEB brake command at {fmt_value(brake)} s followed the TTC threshold crossing "
            f"at {fmt_value(crossing)} s by {fmt_value(latency)} s against a limit of "
            f"{latency.threshold} s: {verdict} [{crossing.metric_id}, {brake.metric_id}, "
            f"{latency.metric_id}]."
        )
    elif crossing and not brake:
        lines.append(
            f"TTC crossed the trigger threshold at {fmt_value(crossing)} s but no brake command "
            f"was logged [{crossing.metric_id}]."
        )
    else:
        lines.append(
            "No TTC threshold crossing was detected in this log; no AEB episode to assess."
        )

    if collision is not None:
        if collision.value is True:
            ct = _metric_or_none(inputs, M_COLLISION_TIME)
            when = f" at {fmt_value(ct)} s" if ct else ""
            ids = ", ".join(i for i in (collision.metric_id, ct.metric_id if ct else None) if i)
            lines.append(f"A collision was recorded{when} [{ids}].")
        else:
            lines.append(f"No collision was recorded [{collision.metric_id}].")
    if min_ttc is not None:
        lines.append(
            f"Minimum TTC was {fmt_value(min_ttc)} s "
            f"({'PASS' if min_ttc.passed else 'FAIL'} against >= {min_ttc.threshold} s) "
            f"[{min_ttc.metric_id}]."
        )
    if dropout is not None and isinstance(dropout.value, float) and dropout.value > 0:
        lines.append(
            f"Perception confidence was below the validity threshold for {fmt_value(dropout)} s "
            f"of the risk phase [{dropout.metric_id}]."
        )

    if verification is None:
        lines.append(
            "No AI diagnosis was run; this is a metrics-only report. Root-cause hypotheses "
            "are not available."
        )
    elif verification.top is None:
        lines.append(
            "The AI diagnosis produced no hypothesis that survived evidence verification; "
            f"report confidence is {verification.report_confidence.value.upper()}."
        )
    else:
        top = verification.top
        lines.append(
            f"Top verified hypothesis ({top.hypothesis.failure_class.value}, "
            f"{top.confidence_label.value}): {top.hypothesis.cause} "
            f"[{', '.join(top.resolved_ids)}]."
        )
        review = "required" if verification.human_review_required else "not triggered"
        lines.append(
            f"Report confidence: {verification.report_confidence.value.upper()}; "
            f"human review {review} [{verification.verification_id}]."
        )
    return tuple(lines)


def build_timeline(inputs: DiagnosisInputs) -> tuple[TimelineEntry, ...]:
    entries: list[TimelineEntry] = []
    for e in inputs.metrics.events:
        entries.append(
            TimelineEntry(
                t_s=e.t_s,
                evidence_id=e.event_id,
                kind=EvidenceKind.EVENT,
                description=f"{e.event_type.value}: {e.description}",
            )
        )
    for m in inputs.metrics.metrics:
        if m.available and m.t_s is not None and m.comparator is not None:
            entries.append(
                TimelineEntry(
                    t_s=m.t_s,
                    evidence_id=m.metric_id,
                    kind=EvidenceKind.METRIC,
                    description=(
                        f"{m.name} = {fmt_value(m)} {m.unit} "
                        f"({'PASS' if m.passed else 'FAIL'} {_threshold_str(m)})"
                    ),
                )
            )
    entries.sort(key=lambda t: (t.t_s, t.kind.value, t.evidence_id))
    return tuple(entries)


def build_metrics_table(inputs: DiagnosisInputs) -> tuple[MetricRow, ...]:
    rows: list[MetricRow] = []
    for m in inputs.metrics.metrics:
        rows.append(
            MetricRow(
                metric_id=m.metric_id if m.available else None,
                name=m.name,
                observed=fmt_value(m),
                unit=m.unit,
                threshold=_threshold_str(m),
                passed=m.passed,
                t_s=m.t_s,
                missing_reason=m.missing_reason,
            )
        )
    return tuple(rows)


def build_report(
    inputs: DiagnosisInputs,
    run: AgentRun | None,
    verification: VerificationReport | None,
    *,
    registry: EvidenceRegistry | None = None,
) -> DiagnosticReport:
    if (run is None) != (verification is None):
        raise ValueError("run and verification must be supplied together")
    registry = registry or EvidenceRegistry.from_bundle(inputs.bundle, inputs.retrieval)
    prov = inputs.telemetry.provenance

    hypotheses: list[HypothesisRow] = []
    stripped: tuple[str, ...] = ()
    cited: list[str] = []
    if verification is not None:
        for rank, v in enumerate(verification.hypotheses, 1):
            hypotheses.append(
                HypothesisRow(
                    rank=rank,
                    cause=v.hypothesis.cause,
                    failure_class=v.hypothesis.failure_class,
                    status=v.status,
                    confidence_label=v.confidence_label,
                    adjusted_confidence=v.adjusted_confidence,
                    agent_confidence=v.agent_confidence,
                    evidence_ids=v.resolved_ids,
                    sources=v.independent_sources,
                    notes=v.notes,
                )
            )
            cited.extend(v.resolved_ids)
        stripped = tuple(f"{s.hypothesis.cause} ({s.reason})" for s in verification.stripped)

    timeline = build_timeline(inputs)
    appendix_ids = dict.fromkeys([*cited, *(t.evidence_id for t in timeline)])
    appendix: list[EvidenceAppendixEntry] = []
    for eid in appendix_ids:
        ref = registry.resolve(eid)
        if ref is None:
            raise EvidenceResolutionError(f"report references unknown evidence {eid!r}")
        appendix.append(
            EvidenceAppendixEntry(
                evidence_id=ref.evidence_id,
                kind=ref.kind,
                source=ref.source,
                summary=ref.summary,
                t_s=ref.t_s,
            )
        )

    limitations: list[str] = list(verification.limitations) if verification else []
    if verification is None:
        limitations.append("No AI diagnosis was run; hypotheses and next tests are not available.")
        if prov.data_origin == "synthetic":
            limitations.append(
                "Evidence is synthetic/simulation-only. No claim here describes real-world "
                "vehicle behaviour."
            )
    if inputs.quality.warned:
        limitations.append(
            "Data-quality warnings: "
            + "; ".join(f"{g.gate}: {g.message}" for g in inputs.quality.warned)
        )
    for w in inputs.metrics.windows:
        if w.window_id == inputs.metrics.primary_window_id and (w.clipped_start or w.clipped_end):
            limitations.append(
                f"The primary analysis window ({w.window_id}) was clipped to the log bounds; "
                "context before or after the event is incomplete."
            )

    missing = tuple(verification.missing_evidence) if verification else tuple(inputs.bundle.missing)
    event_metadata = {
        "source_file": prov.source_path,
        "file_id": prov.file_id,
        "scenario_id": prov.scenario_id or "n/a",
        "data_origin": prov.data_origin,
        "rows": str(prov.row_count),
        "duration_s": f"{prov.duration_s}" if prov.duration_s is not None else "n/a",
        "sample_rate_hz": (f"{1.0 / prov.nominal_dt_s:.1f}" if prov.nominal_dt_s else "n/a"),
        "quality_verdict": f"{inputs.quality.verdict.value} ({inputs.quality.quality_id})",
        "primary_window": inputs.metrics.primary_window_id or "none",
        "unit_conversions": ", ".join(
            f"{c.source_column} [{c.source_unit}] -> {c.target_column} [{c.target_unit}]"
            for c in prov.conversions
        )
        or "none",
    }

    report_confidence = (
        verification.report_confidence if verification is not None else ReportConfidence.LOW
    )
    return DiagnosticReport(
        metadata=ReportMetadata(
            report_id=new_id("report"),
            file_id=prov.file_id,
            quality_id=inputs.quality.quality_id,
            scenario_id=prov.scenario_id,
            data_origin=prov.data_origin,
            source_path=prov.source_path,
            run_id=run.run_id if run else None,
            verification_id=verification.verification_id if verification else None,
            agent=run.agent if run else None,
            provider=run.provider if run else None,
            model=run.model if run else None,
        ),
        report_confidence=report_confidence,
        executive_summary=executive_summary(inputs, verification),
        event_metadata=event_metadata,
        timeline=timeline,
        metrics_table=build_metrics_table(inputs),
        hypotheses=tuple(hypotheses),
        stripped_hypotheses=stripped,
        missing_evidence=missing,
        recommended_next_tests=tuple(verification.recommended_next_tests) if verification else (),
        limitations=tuple(dict.fromkeys(limitations)),
        approval=ApprovalSection(
            human_review_required=bool(verification and verification.human_review_required),
            review_reasons=tuple(verification.review_reasons) if verification else (),
        ),
        evidence_appendix=tuple(appendix),
        evidence_support_rate=verification.evidence_support_rate if verification else None,
        unsupported_claim_rate=verification.unsupported_claim_rate if verification else None,
    )


def assert_traceable(text: str, registry: EvidenceRegistry, *, allowed: frozenset[str]) -> None:
    """Every evidence-looking ID in ``text`` must resolve in ``registry`` or be in ``allowed``
    (the report's own IDs: report_, run_, verification_, scenario_). Raises otherwise."""
    unknown = sorted({t for t in _ID_TOKEN.findall(text) if t not in registry and t not in allowed})
    if unknown:
        raise EvidenceResolutionError(f"report cites unknown evidence IDs: {', '.join(unknown)}")
