"""Evidence registry: the one place that answers "does this ID exist, and what is it?"."""

from collections.abc import Iterable

from app.agents.evidence import EvidenceBundle
from app.agents.schemas import EvidenceKind
from app.rag.schemas import RetrievalResult
from app.verification.schemas import EvidenceRef

_TELEMETRY = frozenset({EvidenceKind.METRIC, EvidenceKind.EVENT, EvidenceKind.WINDOW})


class EvidenceRegistry:
    def __init__(self, refs: Iterable[EvidenceRef]) -> None:
        self._refs: dict[str, EvidenceRef] = {}
        for r in refs:
            if r.evidence_id in self._refs:
                raise ValueError(f"duplicate evidence id {r.evidence_id!r}")
            self._refs[r.evidence_id] = r

    @classmethod
    def from_bundle(
        cls, bundle: EvidenceBundle, retrieval: RetrievalResult | None
    ) -> "EvidenceRegistry":
        source_by_chunk: dict[str, str] = {}
        stale: set[str] = set()
        if retrieval is not None:
            for rc in retrieval.chunks:
                source_by_chunk[rc.chunk_id] = f"doc:{rc.chunk.source_type.value}"
                if rc.stale:
                    stale.add(rc.chunk_id)
        flagged = {f.evidence_id for f in bundle.injection_flags}
        refs: list[EvidenceRef] = []
        for item in bundle.items:
            if item.kind in _TELEMETRY:
                source = "telemetry"
            elif item.kind is EvidenceKind.CHUNK:
                source = source_by_chunk.get(item.evidence_id, "doc:unknown")
            else:
                source = item.kind.value
            refs.append(
                EvidenceRef(
                    evidence_id=item.evidence_id,
                    kind=item.kind,
                    source=source,
                    summary=item.summary,
                    t_s=item.t_s,
                    passed=item.passed,
                    stale=item.evidence_id in stale,
                    injection_flagged=item.evidence_id in flagged,
                )
            )
        return cls(refs)

    def resolve(self, evidence_id: str) -> EvidenceRef | None:
        return self._refs.get(evidence_id)

    def __contains__(self, evidence_id: object) -> bool:
        return evidence_id in self._refs

    def __len__(self) -> int:
        return len(self._refs)

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(self._refs)
