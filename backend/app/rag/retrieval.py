"""Hybrid retrieval: cosine similarity + BM25, metadata filters, trust/freshness rerank.

Access filtering is not a ranking signal, it is a hard wall: a chunk whose access level is
not in ``filters.allowed_access`` is removed before scoring and can never appear in the
result. The result records how many chunks that wall excluded, so a report can say
"3 restricted chunks were not consulted".
"""

from datetime import date

import numpy as np
import numpy.typing as npt

from app.llm.provider import LLMProvider
from app.rag.index import ChunkIndex, IndexError_
from app.rag.schemas import (
    SOURCE_TRUST,
    Chunk,
    RetrievalFilters,
    RetrievalResult,
    RetrievalWeights,
    RetrievedChunk,
)


def _passes(chunk: Chunk, f: RetrievalFilters) -> bool:
    if f.feature is not None and chunk.feature is not f.feature:
        return False
    if f.source_types is not None and chunk.source_type not in f.source_types:
        return False
    return not (f.vehicle_platform is not None and chunk.vehicle_platform != f.vehicle_platform)


def _minmax(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x) if hi <= 0 else np.ones_like(x)
    return (x - lo) / (hi - lo)


def is_stale(chunk: Chunk, today: date, stale_after_days: int) -> bool:
    return chunk.valid_from is not None and (today - chunk.valid_from).days > stale_after_days


def retrieve(
    index: ChunkIndex,
    provider: LLMProvider,
    query: str,
    *,
    filters: RetrievalFilters,
    top_k: int = 5,
    weights: RetrievalWeights | None = None,
    today: date | None = None,
) -> RetrievalResult:
    weights = weights or RetrievalWeights()
    today = today or date.today()

    if provider.name != index.manifest.embedding_provider:
        raise IndexError_(
            f"index embedded with {index.manifest.embedding_provider!r}, "
            f"query provider is {provider.name!r}"
        )

    # 1. Hard access wall, then metadata filters.
    allowed_idx = [
        i for i, c in enumerate(index.chunks) if c.access_level in filters.allowed_access
    ]
    excluded_by_access = len(index) - len(allowed_idx)
    cand = np.asarray([i for i in allowed_idx if _passes(index.chunks[i], filters)], dtype=int)
    if cand.size == 0:
        return RetrievalResult(
            query=query,
            filters=filters,
            chunks=(),
            candidates_considered=0,
            excluded_by_access=excluded_by_access,
        )

    # 2. Scores on the candidate set only.
    q = np.asarray(provider.embed([query], purpose="rag_query").vectors[0], dtype="float64")
    if q.shape[0] != index.manifest.dimension:
        raise IndexError_(
            f"query embedding dimension {q.shape[0]} != index dimension {index.manifest.dimension}"
        )
    q /= np.linalg.norm(q) or 1.0
    vec = index.vectors[cand] @ q  # cosine, rows already unit-norm
    kw = index.bm25.scores(query)[cand]
    hybrid = weights.vector * _minmax(vec) + weights.keyword * _minmax(kw)

    # 3. Rerank by source trust and freshness.
    stale = np.asarray(
        [is_stale(index.chunks[i], today, weights.stale_after_days) for i in cand], dtype=bool
    )
    trust = np.asarray([SOURCE_TRUST[index.chunks[i].source_type] for i in cand], dtype="float64")
    final = hybrid + weights.trust_boost * trust - weights.stale_penalty * stale

    order = np.argsort(-final, kind="stable")[:top_k]
    results = tuple(
        RetrievedChunk(
            chunk=index.chunks[int(cand[j])],
            rank=r + 1,
            score=float(final[j]),
            vector_score=float(vec[j]),
            keyword_score=float(kw[j]),
            stale=bool(stale[j]),
        )
        for r, j in enumerate(order)
    )
    return RetrievalResult(
        query=query,
        filters=filters,
        chunks=results,
        candidates_considered=int(cand.size),
        excluded_by_access=excluded_by_access,
    )
