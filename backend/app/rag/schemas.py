"""RAG contracts: document metadata, chunks (spec §12.3) and retrieval results.

Chunk and document IDs are *stable*: derived from content, not random, so re-indexing the
same document yields the same ``chunk_`` IDs and old reports keep resolving.
"""

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class SourceType(StrEnum):
    REQUIREMENT = "requirement"
    DBC = "dbc"
    MANUAL = "manual"
    TEST_SPEC = "test_spec"
    ISSUE = "issue"
    RELEASE_NOTE = "release_note"


class AccessLevel(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"


class Feature(StrEnum):
    AEB = "AEB"
    ACC = "ACC"
    LKA = "LKA"
    TSR = "TSR"
    BSD = "BSD"
    DMS = "DMS"


# How much a source is trusted when ranking ties (spec §12: rerank by source trust).
SOURCE_TRUST: dict[SourceType, float] = {
    SourceType.REQUIREMENT: 1.00,
    SourceType.TEST_SPEC: 0.95,
    SourceType.DBC: 0.90,
    SourceType.MANUAL: 0.85,
    SourceType.RELEASE_NOTE: 0.80,
    SourceType.ISSUE: 0.75,
}


class DocumentMeta(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    title: str
    path: str
    sha256: str
    source_type: SourceType
    vehicle_platform: str | None = None
    feature: Feature | None = None
    version: str | None = None
    valid_from: date | None = None
    access_level: AccessLevel = AccessLevel.INTERNAL
    related_signal_names: tuple[str, ...] = ()
    related_scenario_ids: tuple[str, ...] = ()


class Chunk(BaseModel):
    """One retrievable unit with the §12.3 metadata. ``text`` is what gets embedded."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    document_title: str
    ordinal: int
    heading: str
    text: str
    source_type: SourceType
    vehicle_platform: str | None
    feature: Feature | None
    version: str | None
    valid_from: date | None
    access_level: AccessLevel
    related_signal_names: tuple[str, ...]
    related_scenario_ids: tuple[str, ...]
    requirement_ids: tuple[str, ...]  # REQ-/TC-/INC- identifiers found in the text

    @property
    def citation(self) -> str:
        return f"{self.document_title} > {self.heading} ({self.chunk_id})"


class RetrievalFilters(BaseModel):
    """Metadata filters. ``allowed_access`` is mandatory and defaults to the least access."""

    model_config = ConfigDict(frozen=True)

    allowed_access: frozenset[AccessLevel] = frozenset({AccessLevel.PUBLIC})
    feature: Feature | None = None
    source_types: frozenset[SourceType] | None = None
    vehicle_platform: str | None = None


class RetrievalWeights(BaseModel):
    model_config = ConfigDict(frozen=True)

    vector: float = Field(default=0.6, ge=0, le=1)
    keyword: float = Field(default=0.4, ge=0, le=1)
    trust_boost: float = Field(default=0.10, ge=0)  # × SOURCE_TRUST
    stale_penalty: float = Field(default=0.15, ge=0)
    stale_after_days: int = Field(default=365, ge=1)


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    rank: int
    score: float
    vector_score: float
    keyword_score: float
    stale: bool

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


class RetrievalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    filters: RetrievalFilters
    chunks: tuple[RetrievedChunk, ...]
    candidates_considered: int  # after filters, before top-k
    excluded_by_access: int

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(c.chunk_id for c in self.chunks)
