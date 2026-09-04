"""Render a DiagnosticReport as Markdown (§27.3 section order) or JSON, and write both.

Rendering is pure: the same report gives the same text. The Markdown is checked for
traceability before it is returned, so a report that cites an unknown ID cannot be written.
"""

from pathlib import Path

from app.agents.evidence import neutralise
from app.reports.builder import assert_traceable
from app.reports.schemas import DiagnosticReport
from app.verification.registry import EvidenceRegistry

MARKDOWN_NAME = "report.md"
JSON_NAME = "report.json"


def _yes_no(v: bool | None) -> str:
    return "—" if v is None else ("PASS" if v else "FAIL")


def _t(t_s: float | None) -> str:
    return "" if t_s is None else f"{t_s:.2f}"


def render_markdown(report: DiagnosticReport) -> str:
    m = report.metadata
    out: list[str] = []
    out.append("# AEB Diagnostic Report")
    out.append("")
    out.append(
        f"`{m.report_id}` · template `{m.template_version}` · generated "
        f"{m.generated_at.isoformat(timespec='seconds')}"
    )
    out.append("")
    out.append(f"**Report confidence: {report.report_confidence.value.upper()}**")
    if report.approval.human_review_required:
        out.append("")
        out.append("> **Human review required.** " + " ".join(report.approval.review_reasons))
    out.append("")
    out.append(f"> {report.disclaimer}")
    out.append("")

    out.append("## 1. Executive summary")
    out.append("")
    out.extend(f"- {neutralise(s)}" for s in report.executive_summary)
    out.append("")

    out.append("## 2. Event metadata")
    out.append("")
    out.append("| Field | Value |")
    out.append("|---|---|")
    for k, v in report.event_metadata.items():
        out.append(f"| {k} | {neutralise(v)} |")
    out.append(f"| run_id | {m.run_id or 'n/a'} |")
    out.append(f"| verification_id | {m.verification_id or 'n/a'} |")
    out.append(f"| agent | {m.agent or 'n/a'} |")
    out.append(f"| model | {m.provider + '/' + m.model if m.provider and m.model else 'n/a'} |")
    out.append("")

    out.append("## 3. Evidence timeline")
    out.append("")
    out.append("| t [s] | Evidence | Kind | Description |")
    out.append("|---|---|---|---|")
    for t in report.timeline:
        out.append(
            f"| {_t(t.t_s)} | `{t.evidence_id}` | {t.kind.value} | {neutralise(t.description)} |"
        )
    out.append("")

    out.append("## 4. Metrics")
    out.append("")
    out.append("| Metric | Observed | Unit | Threshold | Result | t [s] | Evidence |")
    out.append("|---|---|---|---|---|---|---|")
    for r in report.metrics_table:
        if r.metric_id is None:
            out.append(
                f"| {r.name} | — | {r.unit} | {r.threshold} | MISSING: {r.missing_reason} | | |"
            )
        else:
            out.append(
                f"| {r.name} | {r.observed} | {r.unit} | {r.threshold} | {_yes_no(r.passed)} | "
                f"{_t(r.t_s)} | `{r.metric_id}` |"
            )
    out.append("")

    out.append("## 5. Root-cause hypotheses (verified)")
    out.append("")
    if not report.hypotheses:
        out.append(
            "_No verified hypotheses._" + (" No AI diagnosis was run." if m.run_id is None else "")
        )
    for h in report.hypotheses:
        out.append(f"### {h.rank}. {neutralise(h.cause)}")
        out.append("")
        out.append(
            f"- Failure class: `{h.failure_class.value}` · "
            f"Confidence: **{h.confidence_label.value}** "
            f"(agent {h.agent_confidence:.2f} → adjusted {h.adjusted_confidence:.2f}) · "
            f"Status: {h.status.value}"
        )
        out.append(f"- Sources: {', '.join(h.sources) or '—'}")
        out.append("- Evidence: " + ", ".join(f"`{e}`" for e in h.evidence_ids))
        for note in h.notes:
            out.append(f"- Note: {note}")
        out.append("")
    if report.stripped_hypotheses:
        out.append("**Removed by the verifier (unsupported):**")
        out.append("")
        out.extend(f"- {neutralise(s)}" for s in report.stripped_hypotheses)
        out.append("")
    if report.evidence_support_rate is not None and report.unsupported_claim_rate is not None:
        out.append(
            f"Evidence support rate: {report.evidence_support_rate:.0%} · "
            f"Unsupported claim rate: {report.unsupported_claim_rate:.0%}"
        )
        out.append("")

    out.append("## 6. Missing evidence")
    out.append("")
    if report.missing_evidence:
        out.extend(f"- {neutralise(x)}" for x in report.missing_evidence)
    else:
        out.append("- none")
    out.append("")

    out.append("## 7. Recommended next tests")
    out.append("")
    if report.recommended_next_tests:
        out.extend(f"- {neutralise(x)}" for x in report.recommended_next_tests)
    else:
        out.append("- none recorded")
    out.append("")

    out.append("## 8. Limitations")
    out.append("")
    if report.limitations:
        out.extend(f"- {neutralise(x)}" for x in report.limitations)
    else:
        out.append("- none")
    out.append("")

    out.append("## 9. Approval")
    out.append("")
    a = report.approval
    out.append(f"- Status: **{a.status}**")
    out.append(f"- Human review required: {'yes' if a.human_review_required else 'no'}")
    out.append(f"- Reviewer: {a.reviewer or '_unassigned_'}")
    out.append(f"- Decision: {a.decision or '_pending_'}")
    out.append(f"- Reason: {a.reason or '_pending_'}")
    out.append("")

    out.append("## Appendix A. Evidence")
    out.append("")
    out.append("| Evidence | Kind | Source | t [s] | Summary |")
    out.append("|---|---|---|---|---|")
    for e in report.evidence_appendix:
        out.append(
            f"| `{e.evidence_id}` | {e.kind.value} | {e.source} | {_t(e.t_s)} | "
            f"{neutralise(e.summary)} |"
        )
    out.append("")
    return "\n".join(out)


def render_json(report: DiagnosticReport) -> str:
    return report.model_dump_json(indent=2) + "\n"


def own_ids(report: DiagnosticReport) -> frozenset[str]:
    m = report.metadata
    return frozenset(i for i in (m.report_id, m.run_id, m.verification_id, m.scenario_id) if i)


def write_report(
    report: DiagnosticReport, out_dir: Path, registry: EvidenceRegistry
) -> tuple[Path, Path]:
    """Write ``report.md`` and ``report.json``. Refuses to write an untraceable report."""
    markdown = render_markdown(report)
    assert_traceable(markdown, registry, allowed=own_ids(report))  # raises on unknown IDs
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / MARKDOWN_NAME
    json_path = out_dir / JSON_NAME
    md_path.write_text(markdown, encoding="utf-8", newline="\n")
    json_path.write_text(render_json(report), encoding="utf-8", newline="\n")
    return md_path, json_path
