"""Markdown → DocumentMeta + Chunks.

Semantic chunking for engineering documents is simple: one chunk per heading section.
Requirement specs put one requirement under one heading, so a chunk is a requirement.
Sections longer than ``max_chars`` are split on paragraph boundaries; each piece keeps the
heading so it stays self-describing.
"""

import hashlib
import re
from datetime import date
from pathlib import Path

from app.core.ids import stable_id
from app.rag.schemas import AccessLevel, Chunk, DocumentMeta, Feature, SourceType

_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$", re.MULTILINE)
_REQ_ID = re.compile(r"\b(?:REQ|TC|SCN)-[A-Z]+(?:-[A-Z]+)*-\d+\b|\bINC-\d+\b")
_LIST_SPLIT = re.compile(r"\s*,\s*")

SOURCE_ALIASES: dict[str, SourceType] = {
    "requirement": SourceType.REQUIREMENT,
    "requirements": SourceType.REQUIREMENT,
    "srs": SourceType.REQUIREMENT,
    "dbc": SourceType.DBC,
    "signal_definitions": SourceType.DBC,
    "manual": SourceType.MANUAL,
    "test_spec": SourceType.TEST_SPEC,
    "test": SourceType.TEST_SPEC,
    "issue": SourceType.ISSUE,
    "issues": SourceType.ISSUE,
    "release_note": SourceType.RELEASE_NOTE,
}


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Return ``(fields, body)``. Minimal ``key: value`` parser; no YAML dependency."""
    m = _FRONT.match(text)
    if not m:
        return {}, text
    fields: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, _, value = line.partition(":")
        fields[key.strip().lower()] = value.strip().strip("\"'")
    return fields, text[m.end() :]


def _list(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(v for v in _LIST_SPLIT.split(value.strip()) if v)


def _date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def document_meta(path: Path, text: str, fields: dict[str, str]) -> DocumentMeta:
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    source_raw = fields.get("document_type", fields.get("source_type", "manual")).lower()
    source = SOURCE_ALIASES.get(source_raw, SourceType.MANUAL)
    feature_raw = fields.get("feature", "").upper()
    feature = Feature(feature_raw) if feature_raw in Feature.__members__ else None
    access_raw = fields.get("access_level", "internal").lower()
    access = (
        AccessLevel(access_raw)
        if access_raw in AccessLevel._value2member_map_
        else AccessLevel.INTERNAL
    )
    title = fields.get("title") or path.stem
    return DocumentMeta(
        document_id=stable_id("doc", path.name, sha),
        title=title,
        path=str(path),
        sha256=sha,
        source_type=source,
        vehicle_platform=fields.get("vehicle_platform") or None,
        feature=feature,
        version=fields.get("version") or None,
        valid_from=_date(fields.get("valid_from")),
        access_level=access,
        related_signal_names=_list(fields.get("related_signal_names")),
        related_scenario_ids=_list(fields.get("related_scenario_ids")),
    )


def split_sections(body: str) -> list[tuple[str, str]]:
    """``[(heading, text)]`` in document order. Text before the first heading is 'Preamble'."""
    sections: list[tuple[str, str]] = []
    matches = list(_HEADING.finditer(body))
    if not matches:
        return [("Preamble", body.strip())] if body.strip() else []
    if matches[0].start() > 0:
        pre = body[: matches[0].start()].strip()
        if pre:
            sections.append(("Preamble", pre))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[m.end() : end].strip()
        if text:
            sections.append((m.group(2).strip(), text))
    return sections


def split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    current = ""
    for para in re.split(r"\n\s*\n", text):
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) > max_chars and current:
            pieces.append(current)
            current = para
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _signals_in(text: str, known: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(s for s in known if s in text)


def chunk_document(
    path: Path, text: str, *, max_chars: int = 1500, known_signals: tuple[str, ...] = ()
) -> tuple[DocumentMeta, tuple[Chunk, ...]]:
    fields, body = parse_front_matter(text)
    meta = document_meta(path, text, fields)
    signals_vocab = tuple(dict.fromkeys((*meta.related_signal_names, *known_signals)))
    chunks: list[Chunk] = []
    ordinal = 0
    for heading, section in split_sections(body):
        for piece in split_long(section, max_chars):
            chunk_text = f"{heading}\n\n{piece}"
            ids = tuple(dict.fromkeys(_REQ_ID.findall(f"{heading} {piece}")))
            chunks.append(
                Chunk(
                    chunk_id=stable_id("chunk", meta.document_id, str(ordinal), piece),
                    document_id=meta.document_id,
                    document_title=meta.title,
                    ordinal=ordinal,
                    heading=heading,
                    text=chunk_text,
                    source_type=meta.source_type,
                    vehicle_platform=meta.vehicle_platform,
                    feature=meta.feature,
                    version=meta.version,
                    valid_from=meta.valid_from,
                    access_level=meta.access_level,
                    related_signal_names=_signals_in(chunk_text, signals_vocab),
                    related_scenario_ids=tuple(
                        s for s in meta.related_scenario_ids if s in chunk_text
                    )
                    or meta.related_scenario_ids,
                    requirement_ids=ids,
                )
            )
            ordinal += 1
    return meta, tuple(chunks)


def load_documents(
    docs_dir: Path, *, pattern: str = "*.md", known_signals: tuple[str, ...] = ()
) -> tuple[tuple[DocumentMeta, tuple[Chunk, ...]], ...]:
    paths = sorted(p for p in docs_dir.glob(pattern) if p.is_file())
    return tuple(
        chunk_document(p, p.read_text(encoding="utf-8"), known_signals=known_signals) for p in paths
    )
