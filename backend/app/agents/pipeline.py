"""Wire the slice together: CSV → gates → metrics → retrieval → evidence bundle → agent.

This is the deterministic part of a diagnosis. The only non-deterministic step is the
provider call inside :class:`DiagnosticAgent`, and even that is seeded where supported.
"""

from dataclasses import dataclass
from pathlib import Path

from app.agents.evidence import EvidenceBundle, build_evidence_bundle
from app.ingestion.csv_loader import load_telemetry_csv
from app.ingestion.schemas import IngestedTelemetry
from app.llm.provider import LLMProvider
from app.metrics.aeb import compute_aeb_metrics
from app.metrics.schemas import AebMetricsReport, AebThresholds
from app.quality.report import QualityReport, evaluate_gates
from app.rag.index import ChunkIndex
from app.rag.retrieval import retrieve
from app.rag.schemas import Feature, RetrievalFilters, RetrievalResult

# Fixed vocabulary that anchors retrieval on the AEB late-braking requirements. Failing
# metric names are appended so the query follows the evidence, not a hard-coded story.
BASE_QUERY = "AEB brake command latency TTC threshold requirement valid target confidence"


def retrieval_query(metrics: AebMetricsReport) -> str:
    failing = [m.name for m in metrics.metrics if m.passed is False]
    missing = [m.name for m in metrics.metrics if not m.available]
    terms = [BASE_QUERY, *failing]
    if missing:
        terms.append("data logging requirements " + " ".join(missing))
    return " ".join(dict.fromkeys(terms))


@dataclass(frozen=True)
class DiagnosisInputs:
    telemetry: IngestedTelemetry
    quality: QualityReport
    metrics: AebMetricsReport
    retrieval: RetrievalResult | None
    bundle: EvidenceBundle


def prepare_diagnosis(
    csv_path: Path,
    provider: LLMProvider,
    *,
    index: ChunkIndex | None,
    filters: RetrievalFilters,
    thresholds: AebThresholds | None = None,
    top_k: int = 6,
    file_id: str | None = None,
) -> DiagnosisInputs:
    """Run every deterministic stage. Raises DataQualityError when the gates block.

    ``file_id`` keeps a previously registered file identity (see the API); otherwise a
    fresh one is minted.
    """
    telemetry = load_telemetry_csv(csv_path, file_id=file_id)
    quality = evaluate_gates(telemetry)
    metrics = compute_aeb_metrics(telemetry, quality, thresholds)  # enforces the gate
    retrieval: RetrievalResult | None = None
    if index is not None:
        f = (
            filters
            if filters.feature is not None
            else filters.model_copy(update={"feature": Feature.AEB})
        )
        retrieval = retrieve(index, provider, retrieval_query(metrics), filters=f, top_k=top_k)
    bundle = build_evidence_bundle(telemetry.provenance, quality, metrics, retrieval)
    return DiagnosisInputs(
        telemetry=telemetry, quality=quality, metrics=metrics, retrieval=retrieval, bundle=bundle
    )
