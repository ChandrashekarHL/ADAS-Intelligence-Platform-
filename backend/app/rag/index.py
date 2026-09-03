"""Chunk index: chunks + embedding matrix + BM25 statistics, persisted on the filesystem.

Vectors live in a ``.npy`` next to a JSON manifest (bulk arrays stay out of the database,
per the persistence rules). The index records which embedding model produced the vectors
so a query embedded with a different model is rejected instead of silently mis-scored.
"""

import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

from app.core.errors import AipError
from app.core.signals import FRAME_COLUMNS
from app.llm.provider import LLMProvider
from app.rag.chunking import load_documents
from app.rag.schemas import Chunk, DocumentMeta

INDEX_FORMAT = "aip-rag-index/1"
_TOKEN = re.compile(r"[a-z0-9][a-z0-9_.\-]*")


class IndexError_(AipError):
    """Index is missing, corrupt, or built with a different embedding model."""


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens; keeps ``_``, ``.`` and ``-`` so signal and requirement
    names such as ``object_confidence`` or ``REQ-AEB-003`` survive as single tokens."""
    return [t.strip(".-") for t in _TOKEN.findall(text.lower()) if len(t.strip(".-")) > 1]


class IndexManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    format: str = INDEX_FORMAT
    embedding_provider: str
    embedding_model: str
    dimension: int
    documents: tuple[DocumentMeta, ...]
    chunks: tuple[Chunk, ...]


@dataclass(frozen=True)
class Bm25Stats:
    doc_freq: dict[str, int]
    doc_len: tuple[int, ...]
    avg_len: float
    tokens: tuple[tuple[str, ...], ...]

    @classmethod
    def build(cls, texts: Sequence[str]) -> "Bm25Stats":
        toks = tuple(tuple(tokenize(t)) for t in texts)
        df: Counter[str] = Counter()
        for t in toks:
            df.update(set(t))
        lens = tuple(len(t) for t in toks)
        return cls(
            doc_freq=dict(df),
            doc_len=lens,
            avg_len=(sum(lens) / len(lens)) if lens else 0.0,
            tokens=toks,
        )

    def scores(self, query: str, *, k1: float = 1.5, b: float = 0.75) -> npt.NDArray[np.float64]:
        n = len(self.tokens)
        out = np.zeros(n, dtype="float64")
        q_terms = set(tokenize(query))
        if not n or not q_terms:
            return out
        for i, toks in enumerate(self.tokens):
            if not toks:
                continue
            tf = Counter(toks)
            norm = k1 * (1 - b + b * self.doc_len[i] / self.avg_len) if self.avg_len else k1
            s = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                df = self.doc_freq.get(term, 0)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                s += idf * f * (k1 + 1) / (f + norm)
            out[i] = s
        return out


@dataclass(frozen=True)
class ChunkIndex:
    manifest: IndexManifest
    vectors: npt.NDArray[np.float64]  # (n_chunks, dim), unit-normalised rows
    bm25: Bm25Stats

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return self.manifest.chunks

    def __len__(self) -> int:
        return len(self.manifest.chunks)

    def get(self, chunk_id: str) -> Chunk | None:
        return next((c for c in self.manifest.chunks if c.chunk_id == chunk_id), None)

    # --- persistence ------------------------------------------------------------------

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.json").write_text(
            self.manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        np.save(directory / "vectors.npy", self.vectors)

    @classmethod
    def load(cls, directory: Path) -> "ChunkIndex":
        manifest_path = directory / "manifest.json"
        vectors_path = directory / "vectors.npy"
        if not manifest_path.is_file() or not vectors_path.is_file():
            raise IndexError_(f"no index at {directory}")
        try:
            manifest = IndexManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise IndexError_(f"corrupt manifest at {manifest_path}: {exc}") from exc
        if manifest.format != INDEX_FORMAT:
            raise IndexError_(f"unsupported index format {manifest.format!r}")
        vectors = np.load(vectors_path)
        if vectors.shape != (len(manifest.chunks), manifest.dimension):
            raise IndexError_(
                f"vector shape {vectors.shape} does not match manifest "
                f"({len(manifest.chunks)}, {manifest.dimension})"
            )
        return cls(
            manifest=manifest,
            vectors=vectors.astype("float64"),
            bm25=Bm25Stats.build([c.text for c in manifest.chunks]),
        )


def _normalise(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def embed_texts(
    provider: LLMProvider, texts: Sequence[str], *, batch_size: int = 64, purpose: str
) -> tuple[npt.NDArray[np.float64], str]:
    """Embed in batches; returns ``(matrix, model_name)``."""
    rows: list[tuple[float, ...]] = []
    model = ""
    for start in range(0, len(texts), batch_size):
        resp = provider.embed(texts[start : start + batch_size], purpose=purpose)
        rows.extend(resp.vectors)
        model = resp.model
    if not rows:
        return np.zeros((0, 0), dtype="float64"), model
    return _normalise(np.asarray(rows, dtype="float64")), model


def build_index(
    docs_dir: Path,
    provider: LLMProvider,
    *,
    pattern: str = "*.md",
    batch_size: int = 64,
) -> ChunkIndex:
    """Parse every document in ``docs_dir``, chunk, embed and assemble the index."""
    docs = load_documents(docs_dir, pattern=pattern, known_signals=FRAME_COLUMNS)
    metas = tuple(m for m, _ in docs)
    chunks = tuple(c for _, cs in docs for c in cs)
    if not chunks:
        raise IndexError_(f"no chunks produced from {docs_dir} ({pattern})")
    vectors, model = embed_texts(
        provider, [c.text for c in chunks], batch_size=batch_size, purpose="rag_index"
    )
    manifest = IndexManifest(
        embedding_provider=provider.name,
        embedding_model=model,
        dimension=int(vectors.shape[1]),
        documents=metas,
        chunks=chunks,
    )
    return ChunkIndex(
        manifest=manifest, vectors=vectors, bm25=Bm25Stats.build([c.text for c in chunks])
    )


def manifest_summary(index: ChunkIndex) -> str:
    docs = index.manifest.documents
    return json.dumps(
        {
            "documents": len(docs),
            "chunks": len(index),
            "embedding": f"{index.manifest.embedding_provider}/{index.manifest.embedding_model}",
            "dimension": index.manifest.dimension,
            "by_source": dict(Counter(c.source_type.value for c in index.chunks)),
            "by_access": dict(Counter(c.access_level.value for c in index.chunks)),
        },
        indent=2,
    )
