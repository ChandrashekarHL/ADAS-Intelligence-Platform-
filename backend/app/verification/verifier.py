"""Verify an AgentRun against the evidence registry and apply the §28.1 confidence rules.

Mechanical, deterministic, and explicit: every adjustment is recorded as an AppliedRule.

Per hypothesis
  * every cited ID must resolve; unresolved IDs are dropped and noted
  * at least one cited artifact must be timestamped (metric/event/window), otherwise the
    hypothesis is stripped: no root cause without timestamped evidence
  * independent sources = telemetry + each distinct document source type cited;
    a single source caps the hypothesis at Medium (0.6)
  * citing a stale document, or a document that raised injection flags, lowers confidence
  * citing only PASSing metrics as telemetry support is noted (weak causal support)

Report level
  * BLOCKED quality → BLOCKED; DEGRADED quality → cap Medium
  * a critical metric missing → cap Low
  * top hypothesis single-source → cap Medium
  * two hypotheses ≥ 0.6 with different failure classes → human review
  * injection flags on cited documents → human review + cap Medium
  * synthetic origin → limitation + flag any "real-world" language
"""

import re

from app.agents.evidence import EvidenceBundle
from app.agents.pipeline import DiagnosisInputs
from app.agents.schemas import AgentRun, EvidenceKind, Hypothesis
from app.core.ids import new_id
from app.quality.report import QualityReport, QualityVerdict
from app.verification.registry import EvidenceRegistry
from app.verification.schemas import (
    CRITICAL_METRICS,
    AppliedRule,
    ClaimStatus,
    EvidenceRef,
    ReportConfidence,
    StrippedHypothesis,
    VerificationReport,
    VerifiedHypothesis,
    band,
    cap,
)

SINGLE_SOURCE_CAP = 0.6
STALE_PENALTY = 0.10
INJECTION_PENALTY = 0.20
PASSING_ONLY_PENALTY = 0.15
COMPETING_THRESHOLD = 0.6

_ID_TOKEN = re.compile(r"\b(?:metric|event|window|chunk|quality|file|doc|run)_[0-9a-f]{12}\b")
_REAL_WORLD = re.compile(
    r"\b(real[- ]world|on[- ]road|in the field|production vehicle|customer vehicle|road safety)\b",
    re.I,
)


def _telemetry_only_passing(refs: list[EvidenceRef]) -> bool:
    telemetry = [r for r in refs if r.kind is EvidenceKind.METRIC]
    return bool(telemetry) and all(r.passed is True for r in telemetry)


def verify_hypothesis(
    index: int, h: Hypothesis, registry: EvidenceRegistry, rules: list[AppliedRule]
) -> VerifiedHypothesis | StrippedHypothesis:
    resolved: list[EvidenceRef] = []
    dropped: list[str] = []
    for eid in dict.fromkeys(h.evidence_ids):
        ref = registry.resolve(eid)
        if ref is None:
            dropped.append(eid)
        else:
            resolved.append(ref)

    if not h.evidence_ids:
        rules.append(
            AppliedRule(
                rule="evidence_required",
                effect="stripped",
                detail="hypothesis cites no evidence",
                hypothesis_index=index,
            )
        )
        return StrippedHypothesis(hypothesis=h, reason="no evidence cited")
    if dropped:
        rules.append(
            AppliedRule(
                rule="unresolvable_ids",
                effect="ids dropped",
                detail=", ".join(dropped),
                hypothesis_index=index,
            )
        )
    if not resolved:
        return StrippedHypothesis(hypothesis=h, reason="none of the cited evidence IDs resolve")
    if not any(r.t_s is not None for r in resolved):
        rules.append(
            AppliedRule(
                rule="timestamped_evidence_required",
                effect="stripped",
                detail="only untimestamped evidence (documents/metadata) cited",
                hypothesis_index=index,
            )
        )
        return StrippedHypothesis(hypothesis=h, reason="no timestamped evidence for a root cause")

    notes: list[str] = []
    score = h.confidence
    sources = tuple(
        dict.fromkeys(r.source for r in resolved if r.source not in {"file", "quality"})
    )

    if len(sources) <= 1:
        if score > SINGLE_SOURCE_CAP:
            rules.append(
                AppliedRule(
                    rule="single_source",
                    effect=f"confidence capped at {SINGLE_SOURCE_CAP}",
                    detail=f"sources: {', '.join(sources) or 'none'}",
                    hypothesis_index=index,
                )
            )
        score = min(score, SINGLE_SOURCE_CAP)
        notes.append("single evidence source")
    if dropped:
        notes.append(f"{len(dropped)} cited ID(s) did not resolve and were removed")
    if any(r.stale for r in resolved):
        score -= STALE_PENALTY
        rules.append(
            AppliedRule(
                rule="stale_document",
                effect=f"confidence -{STALE_PENALTY}",
                detail=", ".join(r.evidence_id for r in resolved if r.stale),
                hypothesis_index=index,
            )
        )
        notes.append("cites a stale document")
    if any(r.injection_flagged for r in resolved):
        score -= INJECTION_PENALTY
        rules.append(
            AppliedRule(
                rule="injection_flagged_source",
                effect=f"confidence -{INJECTION_PENALTY}",
                detail=", ".join(r.evidence_id for r in resolved if r.injection_flagged),
                hypothesis_index=index,
            )
        )
        notes.append("cites a document that contained instruction-like text")
    if _telemetry_only_passing(resolved):
        score -= PASSING_ONLY_PENALTY
        rules.append(
            AppliedRule(
                rule="passing_metrics_only",
                effect=f"confidence -{PASSING_ONLY_PENALTY}",
                detail="all cited metrics pass their thresholds",
                hypothesis_index=index,
            )
        )
        notes.append("cited metrics all pass; weak causal support")

    score = max(0.0, min(1.0, round(score, 4)))
    return VerifiedHypothesis(
        hypothesis=h,
        status=ClaimStatus.PARTIAL if dropped else ClaimStatus.SUPPORTED,
        resolved_ids=tuple(r.evidence_id for r in resolved),
        dropped_ids=tuple(dropped),
        independent_sources=sources,
        agent_confidence=h.confidence,
        adjusted_confidence=score,
        confidence_label=band(score),
        notes=tuple(notes),
    )


def verify(
    run: AgentRun,
    bundle: EvidenceBundle,
    registry: EvidenceRegistry,
    quality: QualityReport,
) -> VerificationReport:
    rules: list[AppliedRule] = []
    kept: list[VerifiedHypothesis] = []
    stripped: list[StrippedHypothesis] = []
    for i, h in enumerate(run.output.hypotheses):
        result = verify_hypothesis(i, h, registry, rules)
        if isinstance(result, VerifiedHypothesis):
            kept.append(result)
        else:
            stripped.append(result)
    kept.sort(key=lambda v: (-v.adjusted_confidence, v.hypothesis.cause))

    # --- observations: free text may not smuggle in unknown IDs -----------------------
    flagged_obs: list[str] = []
    for obs in run.output.observations:
        unknown = [t for t in _ID_TOKEN.findall(obs) if t not in registry]
        if unknown:
            flagged_obs.append(obs)
            rules.append(
                AppliedRule(
                    rule="observation_unknown_ids",
                    effect="observation flagged",
                    detail=", ".join(unknown),
                )
            )

    # --- report-level confidence ---------------------------------------------------------
    review: list[str] = []
    limitations: list[str] = []

    if quality.verdict is QualityVerdict.BLOCKED:
        level = ReportConfidence.BLOCKED
        rules.append(
            AppliedRule(rule="quality_blocked", effect="BLOCKED", detail=quality.quality_id)
        )
    elif not kept:
        level = ReportConfidence.LOW
        rules.append(
            AppliedRule(
                rule="no_supported_hypothesis",
                effect="LOW",
                detail=f"{len(stripped)} hypothesis(es) stripped",
            )
        )
    else:
        top = kept[0]
        level = band(top.adjusted_confidence)
        if len(top.independent_sources) <= 1:
            level = cap(level, ReportConfidence.MEDIUM)
        if quality.verdict is QualityVerdict.DEGRADED:
            level = cap(level, ReportConfidence.MEDIUM)
            warned = ", ".join(g.gate for g in quality.warned)
            rules.append(AppliedRule(rule="quality_degraded", effect="cap MEDIUM", detail=warned))
            limitations.append(f"Data quality degraded ({warned}); confidence capped at Medium.")

    missing_critical = sorted(
        m.split(":", 1)[0] for m in bundle.missing if m.split(":", 1)[0] in CRITICAL_METRICS
    )
    if missing_critical and level is not ReportConfidence.BLOCKED:
        level = cap(level, ReportConfidence.LOW)
        rules.append(
            AppliedRule(
                rule="critical_metric_missing", effect="cap LOW", detail=", ".join(missing_critical)
            )
        )
        limitations.append("Critical metric(s) unavailable: " + ", ".join(missing_critical) + ".")

    strong = [v for v in kept if v.adjusted_confidence >= COMPETING_THRESHOLD]
    classes = {v.hypothesis.failure_class for v in strong}
    if len(classes) > 1:
        review.append(
            "competing high-confidence hypotheses with different failure classes: "
            + ", ".join(sorted(c.value for c in classes))
        )
        rules.append(
            AppliedRule(
                rule="contradictory_evidence",
                effect="human review",
                detail=", ".join(sorted(c.value for c in classes)),
            )
        )

    cited_flagged = {
        r
        for v in kept
        for r in v.resolved_ids
        if (ref := registry.resolve(r)) and ref.injection_flagged
    }
    if cited_flagged:
        level = cap(level, ReportConfidence.MEDIUM)
        review.append(
            "cited documents contained instruction-like text: " + ", ".join(sorted(cited_flagged))
        )

    if run.unresolved_ids:
        rules.append(
            AppliedRule(
                rule="agent_unresolved_after_repair",
                effect="recorded",
                detail=", ".join(run.unresolved_ids),
            )
        )

    if run.data_origin == "synthetic":
        limitations.append(
            "Evidence is synthetic/simulation-only. No claim here describes real-world "
            "vehicle behaviour."
        )
        texts = [v.hypothesis.cause for v in kept] + list(run.output.observations)
        offenders = [t for t in texts if _REAL_WORLD.search(t)]
        if offenders:
            review.append("real-world language used on synthetic evidence")
            rules.append(
                AppliedRule(
                    rule="synthetic_no_real_world_claims",
                    effect="human review",
                    detail=offenders[0][:120],
                )
            )
    if bundle.excluded_by_access:
        limitations.append(
            f"{bundle.excluded_by_access} document chunk(s) were withheld by access control "
            "and not consulted."
        )
    if stripped:
        limitations.append(f"{len(stripped)} agent hypothesis(es) removed as unsupported.")

    cited_total = sum(len(dict.fromkeys(h.evidence_ids)) for h in run.output.hypotheses)
    cited_resolved = sum(
        1 for h in run.output.hypotheses for e in dict.fromkeys(h.evidence_ids) if e in registry
    )
    n_hyp = len(run.output.hypotheses)

    missing = tuple(dict.fromkeys([*run.output.missing_evidence, *bundle.missing]))
    return VerificationReport(
        verification_id=new_id("verification"),
        run_id=run.run_id,
        report_confidence=level,
        human_review_required=bool(review),
        review_reasons=tuple(review),
        hypotheses=tuple(kept),
        stripped=tuple(stripped),
        applied_rules=tuple(rules),
        flagged_observations=tuple(flagged_obs),
        missing_evidence=missing,
        recommended_next_tests=tuple(dict.fromkeys(run.output.recommended_next_tests)),
        limitations=tuple(limitations),
        cited_ids_total=cited_total,
        cited_ids_resolved=cited_resolved,
        evidence_support_rate=(cited_resolved / cited_total) if cited_total else 1.0,
        unsupported_claim_rate=(len(stripped) / n_hyp) if n_hyp else 0.0,
    )


def verify_diagnosis(run: AgentRun, inputs: DiagnosisInputs) -> VerificationReport:
    """Convenience: build the registry from the pipeline inputs and verify the run."""
    registry = EvidenceRegistry.from_bundle(inputs.bundle, inputs.retrieval)
    return verify(run, inputs.bundle, registry, inputs.quality)
