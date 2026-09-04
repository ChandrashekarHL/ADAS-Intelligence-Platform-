"""Stable evidence-ID generation.

Every traceable artifact gets an ID at creation time. Agents may only cite IDs
they were handed; the verifier resolves every cited ID before a claim reaches a
report. Prefixes are fixed here so the whole codebase shares one vocabulary.
"""

import hashlib
import uuid

PREFIXES = frozenset(
    {
        "window",
        "metric",
        "chunk",
        "event",
        "report",
        "run",
        "doc",
        "file",
        "scenario",
        "quality",
        "verification",
    }
)


def new_id(prefix: str) -> str:
    if prefix not in PREFIXES:
        raise ValueError(f"Unknown evidence prefix {prefix!r}; allowed: {sorted(PREFIXES)}")
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def stable_id(prefix: str, *parts: str) -> str:
    """Content-derived ID: same inputs → same ID across runs and machines.

    Used for artifacts that are re-created from the same source (documents, chunks) so
    citations in old reports keep resolving after a re-index.
    """
    if prefix not in PREFIXES:
        raise ValueError(f"Unknown evidence prefix {prefix!r}; allowed: {sorted(PREFIXES)}")
    digest = hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:12]}"
