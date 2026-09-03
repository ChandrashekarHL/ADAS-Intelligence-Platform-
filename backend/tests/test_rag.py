"""M5 requirement RAG: chunking, stable IDs, index persistence, hybrid retrieval, filters."""

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from pydantic import BaseModel

from app.core.ids import stable_id
from app.llm.fake import FakeProvider
from app.llm.provider import CallLog
from app.llm.schemas import EmbeddingResponse, LLMRequest, LLMResponse, TokenUsage
from app.rag.chunking import chunk_document, load_documents, parse_front_matter, split_long
from app.rag.cli import access_up_to
from app.rag.cli import main as cli_main
from app.rag.index import Bm25Stats, ChunkIndex, IndexError_, build_index, tokenize
from app.rag.retrieval import is_stale, retrieve
from app.rag.schemas import (
    AccessLevel,
    Feature,
    RetrievalFilters,
    RetrievalWeights,
    SourceType,
)

DOCS = Path(__file__).resolve().parents[2] / "data" / "demo_docs"
INTERNAL = RetrievalFilters(allowed_access=frozenset({AccessLevel.PUBLIC, AccessLevel.INTERNAL}))
ALL_ACCESS = RetrievalFilters(allowed_access=frozenset(AccessLevel))


class VocabEmbedder:
    """Bag-of-words embedder over a fixed vocabulary: semantic-ish, fully deterministic.

    Lets tests exercise the *vector* path with a tiny, inspectable vocabulary instead of
    the FakeProvider's hashed bag-of-words.
    """

    VOCAB = ("latency", "brake", "confidence", "dropout", "timestamp", "gap", "jerk", "restricted")

    def __init__(self) -> None:
        self.call_log = CallLog()

    @property
    def name(self) -> str:
        return "fake"  # matches FakeProvider so indexes are interchangeable in tests

    @property
    def model(self) -> str:
        return "vocab"

    def complete(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError

    def complete_structured[T: BaseModel](
        self, request: LLMRequest, schema: type[T]
    ) -> tuple[T, LLMResponse]:
        raise NotImplementedError

    def embed(self, texts: Sequence[str], *, purpose: str = "") -> EmbeddingResponse:
        vecs = []
        for t in texts:
            low = t.lower()
            v = np.array([low.count(w) for w in self.VOCAB], dtype="float64") + 1e-6
            vecs.append(tuple(float(x) for x in v / np.linalg.norm(v)))
        return EmbeddingResponse(
            vectors=tuple(vecs), model="vocab", provider="fake", usage=TokenUsage(), latency_s=0.0
        )


@pytest.fixture(scope="module")
def index() -> ChunkIndex:
    return build_index(DOCS, VocabEmbedder())


# --- chunking ------------------------------------------------------------------------------


def test_front_matter_and_sections() -> None:
    fields, body = parse_front_matter("---\ntitle: T\nfeature: AEB\n---\n# H\n\nbody\n")
    assert fields == {"title": "T", "feature": "AEB"} and body.startswith("# H")
    assert parse_front_matter("no front matter") == ({}, "no front matter")


def test_srs_chunks_are_requirements_with_ids_and_signals() -> None:
    path = DOCS / "AEB_SRS_v1.2.md"
    meta, chunks = chunk_document(path, path.read_text(encoding="utf-8"))
    assert meta.source_type is SourceType.REQUIREMENT
    assert meta.feature is Feature.AEB and meta.access_level is AccessLevel.INTERNAL
    assert meta.valid_from == date(2026, 5, 15) and meta.version == "1.2"
    assert meta.document_id.startswith("doc_")
    by_req = {c.requirement_ids[0]: c for c in chunks if c.requirement_ids}
    latency = by_req["REQ-AEB-003"]
    assert latency.heading.startswith("REQ-AEB-003")
    assert "300 ms" in latency.text
    assert set(latency.requirement_ids) == {
        "REQ-AEB-003",
        "REQ-AEB-001",
        "REQ-AEB-010",
        "TC-AEB-003",
    }
    assert "brake_command" in latency.related_signal_names
    assert latency.related_scenario_ids == ("SCN-AEB-LVSB-01",)
    assert latency.citation.startswith("AEB System Requirements Specification > REQ-AEB-003")
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert all(c.chunk_id.startswith("chunk_") for c in chunks)


def test_chunk_ids_are_stable_and_content_addressed() -> None:
    path = DOCS / "AEB_SRS_v1.2.md"
    text = path.read_text(encoding="utf-8")
    a = chunk_document(path, text)[1]
    b = chunk_document(path, text)[1]
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    changed = chunk_document(path, text.replace("300 ms", "250 ms"))[1]
    assert [c.chunk_id for c in changed] != [
        c.chunk_id for c in a
    ]  # doc sha changed → all ids move
    assert stable_id("chunk", "x") == stable_id("chunk", "x") != stable_id("chunk", "y")
    with pytest.raises(ValueError):
        stable_id("bogus", "x")


def test_split_long_keeps_paragraphs() -> None:
    text = "\n\n".join(f"para {i} " + "x" * 100 for i in range(6))
    pieces = split_long(text, 250)
    assert len(pieces) == 3 and all(len(p) <= 250 for p in pieces)
    assert "".join(pieces).count("para") == 6


def test_load_documents_covers_demo_corpus() -> None:
    docs = load_documents(DOCS)
    titles = {m.title for m, _ in docs}
    assert len(docs) == 5 and "AEB Issue History" in titles
    types = {m.source_type for m, _ in docs}
    assert types == {
        SourceType.REQUIREMENT,
        SourceType.DBC,
        SourceType.TEST_SPEC,
        SourceType.ISSUE,
        SourceType.MANUAL,
    }
    restricted = [m for m, _ in docs if m.access_level is AccessLevel.RESTRICTED]
    assert len(restricted) == 1


# --- index ---------------------------------------------------------------------------------


def test_tokenizer_keeps_identifiers() -> None:
    toks = tokenize("REQ-AEB-003: object_confidence >= 0.50 within 300 ms.")
    assert "req-aeb-003" in toks and "object_confidence" in toks and "0.50" in toks
    assert "ms" in toks and ":" not in toks


def test_bm25_prefers_term_matches() -> None:
    stats = Bm25Stats.build(["brake latency requirement", "jerk limit", "brake latch"])
    s = stats.scores("latency")
    assert s[0] > 0 and s[1] == 0 and s[2] == 0
    assert stats.scores("").tolist() == [0.0, 0.0, 0.0]


def test_index_build_save_load_roundtrip(tmp_path: Path, index: ChunkIndex) -> None:
    assert len(index) > 20
    assert index.manifest.embedding_provider == "fake" and index.manifest.dimension == 8
    assert np.allclose(np.linalg.norm(index.vectors, axis=1), 1.0)
    index.save(tmp_path / "idx")
    loaded = ChunkIndex.load(tmp_path / "idx")
    assert loaded.manifest == index.manifest
    assert np.array_equal(loaded.vectors, index.vectors)
    assert loaded.get(index.chunks[3].chunk_id) == index.chunks[3]
    assert loaded.get("chunk_nope") is None


def test_index_load_errors(tmp_path: Path, index: ChunkIndex) -> None:
    with pytest.raises(IndexError_, match="no index"):
        ChunkIndex.load(tmp_path / "missing")
    index.save(tmp_path / "bad")
    np.save(tmp_path / "bad" / "vectors.npy", np.zeros((2, 2)))
    with pytest.raises(IndexError_, match="shape"):
        ChunkIndex.load(tmp_path / "bad")
    (tmp_path / "bad" / "manifest.json").write_text("{", encoding="utf-8")
    with pytest.raises(IndexError_, match="corrupt"):
        ChunkIndex.load(tmp_path / "bad")


def test_build_index_requires_chunks(tmp_path: Path) -> None:
    with pytest.raises(IndexError_, match="no chunks"):
        build_index(tmp_path, FakeProvider())


def test_build_index_batches_embeddings() -> None:
    fake = FakeProvider(embedding_dim=8)
    idx = build_index(DOCS, fake, batch_size=10)
    assert len(fake.embedded) == -(-len(idx) // 10)
    assert all(len(b) <= 10 for b in fake.embedded)
    assert fake.call_log.records[0].purpose == "rag_index"


# --- retrieval -----------------------------------------------------------------------------


def test_retrieve_finds_the_latency_requirement(index: ChunkIndex) -> None:
    r = retrieve(
        index, VocabEmbedder(), "brake command latency requirement", filters=INTERNAL, top_k=3
    )
    assert r.chunks[0].rank == 1
    # With a toy embedder the exact order is not meaningful; the latency requirement (or its
    # test case, which cites it) must be in the top 3 and every hit must mention latency.
    hit_ids = {rid for c in r.chunks for rid in c.chunk.requirement_ids}
    assert "REQ-AEB-003" in hit_ids
    assert all("latency" in c.chunk.text.lower() for c in r.chunks)
    assert r.chunks[0].vector_score > 0 and r.chunks[0].keyword_score > 0
    assert r.candidates_considered == len(index) - r.excluded_by_access
    assert r.excluded_by_access >= 1  # the restricted supplier document
    assert len(r.chunk_ids) == 3 and len(set(r.chunk_ids)) == 3
    assert [c.rank for c in r.chunks] == [1, 2, 3]
    assert r.chunks[0].score >= r.chunks[1].score >= r.chunks[2].score


def test_access_wall_is_absolute(index: ChunkIndex) -> None:
    public_only = RetrievalFilters(allowed_access=frozenset({AccessLevel.PUBLIC}))
    r = retrieve(index, VocabEmbedder(), "track hold restricted", filters=public_only, top_k=10)
    assert r.chunks == () and r.candidates_considered == 0
    assert r.excluded_by_access == len(index)  # every demo doc is internal or restricted

    r = retrieve(index, VocabEmbedder(), "track hold restricted", filters=INTERNAL, top_k=50)
    assert all(c.chunk.access_level is not AccessLevel.RESTRICTED for c in r.chunks)
    assert r.excluded_by_access > 0

    r = retrieve(index, VocabEmbedder(), "track hold restricted", filters=ALL_ACCESS, top_k=3)
    assert r.excluded_by_access == 0
    assert r.chunks[0].chunk.access_level is AccessLevel.RESTRICTED


def test_metadata_filters(index: ChunkIndex) -> None:
    issues_only = INTERNAL.model_copy(update={"source_types": frozenset({SourceType.ISSUE})})
    r = retrieve(index, VocabEmbedder(), "confidence dropout brake", filters=issues_only, top_k=5)
    assert r.chunks and all(c.chunk.source_type is SourceType.ISSUE for c in r.chunks)
    assert "INC-2041" in r.chunks[0].chunk.requirement_ids

    other_feature = INTERNAL.model_copy(update={"feature": Feature.LKA})
    assert retrieve(index, VocabEmbedder(), "anything", filters=other_feature).chunks == ()

    platform = INTERNAL.model_copy(update={"vehicle_platform": "DEMO-P1"})
    assert retrieve(index, VocabEmbedder(), "latency", filters=platform).candidates_considered > 0
    wrong = INTERNAL.model_copy(update={"vehicle_platform": "OTHER"})
    assert retrieve(index, VocabEmbedder(), "latency", filters=wrong).candidates_considered == 0


def test_weights_and_staleness(index: ChunkIndex) -> None:
    kw_only = RetrievalWeights(vector=0.0, keyword=1.0, trust_boost=0.0, stale_penalty=0.0)
    r = retrieve(index, VocabEmbedder(), "REQ-AEB-011", filters=INTERNAL, top_k=1, weights=kw_only)
    assert "REQ-AEB-011" in r.chunks[0].chunk.requirement_ids

    # Everything dated before 2026 is stale when "today" is far in the future.
    far = date(2030, 1, 1)
    r = retrieve(index, VocabEmbedder(), "track hold", filters=ALL_ACCESS, top_k=50, today=far)
    assert all(c.stale for c in r.chunks if c.chunk.valid_from is not None)
    r_now = retrieve(
        index, VocabEmbedder(), "track hold", filters=ALL_ACCESS, top_k=50, today=date(2026, 9, 3)
    )
    stale_now = {c.chunk_id for c in r_now.chunks if c.stale}
    restricted_ids = {c.chunk_id for c in index.chunks if c.access_level is AccessLevel.RESTRICTED}
    assert stale_now == set()  # all demo docs are < 365 days old on 2026-09-03
    assert is_stale(index.chunks[0], far, 365) and not is_stale(
        index.chunks[0], date(2026, 9, 3), 365
    )
    assert restricted_ids  # sanity: the corpus has restricted chunks to hide


def test_provider_mismatch_is_rejected(index: ChunkIndex) -> None:
    class Other(VocabEmbedder):
        @property
        def name(self) -> str:
            return "openai"

    with pytest.raises(IndexError_, match="embedded with 'fake'"):
        retrieve(index, Other(), "x", filters=INTERNAL)


def test_dimension_mismatch_is_rejected(index: ChunkIndex) -> None:
    with pytest.raises(IndexError_, match="dimension"):
        retrieve(index, FakeProvider(embedding_dim=32), "x", filters=INTERNAL)


# --- CLI -----------------------------------------------------------------------------------


def test_access_up_to() -> None:
    assert access_up_to(AccessLevel.PUBLIC) == frozenset({AccessLevel.PUBLIC})
    assert access_up_to(AccessLevel.INTERNAL) == frozenset(
        {AccessLevel.PUBLIC, AccessLevel.INTERNAL}
    )
    assert access_up_to(AccessLevel.RESTRICTED) == frozenset(AccessLevel)


def test_cli_build_and_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert cli_main(["build", str(DOCS), "--out", str(tmp_path / "idx")]) == 0
    out = capsys.readouterr().out
    assert '"documents": 5' in out
    assert (
        cli_main(
            [
                "query",
                str(tmp_path / "idx"),
                "brake command latency",
                "--access",
                "internal",
                "--top-k",
                "2",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "#1 chunk_" in out and "excluded_by_access=" in out and "REQ-AEB" in out
