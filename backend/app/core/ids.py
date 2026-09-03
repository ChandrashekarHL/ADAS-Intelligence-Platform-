"""Stable evidence-ID generation.

Every traceable artifact gets an ID at creation time. Agents may only cite IDs
they were handed; the verifier resolves every cited ID before a claim reaches a
report. Prefixes are fixed here so the whole codebase shares one vocabulary.
"""

import uuid

PREFIXES = frozenset(
    {"window", "metric", "chunk", "event", "report", "run", "doc", "file", "scenario", "quality"}
)


def new_id(prefix: str) -> str:
    if prefix not in PREFIXES:
        raise ValueError(f"Unknown evidence prefix {prefix!r}; allowed: {sorted(PREFIXES)}")
    return f"{prefix}_{uuid.uuid4().hex[:12]}"
