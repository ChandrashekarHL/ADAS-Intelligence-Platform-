"""Build a requirement index and query it.

    python -m app.rag.cli build ../data/demo_docs --out ../data/index
    python -m app.rag.cli query ../data/index "brake command latency requirement" --access internal

Uses LLM_PROVIDER from settings (``fake`` works fully offline: keyword scoring carries the
ranking, vectors are deterministic hashes).
"""

import argparse
import sys
from pathlib import Path

from app.core.config import get_settings
from app.llm.factory import build_provider
from app.rag.index import ChunkIndex, build_index, manifest_summary
from app.rag.retrieval import retrieve
from app.rag.schemas import AccessLevel, Feature, RetrievalFilters, SourceType

ACCESS_ORDER = (AccessLevel.PUBLIC, AccessLevel.INTERNAL, AccessLevel.RESTRICTED)


def access_up_to(level: AccessLevel) -> frozenset[AccessLevel]:
    """``internal`` grants public+internal; ``restricted`` grants everything."""
    return frozenset(ACCESS_ORDER[: ACCESS_ORDER.index(level) + 1])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aip-rag", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="parse, chunk, embed and save an index")
    b.add_argument("docs_dir", type=Path)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--pattern", default="*.md")

    q = sub.add_parser("query", help="retrieve chunks for a question")
    q.add_argument("index_dir", type=Path)
    q.add_argument("text")
    q.add_argument("--top-k", type=int, default=5)
    q.add_argument("--access", choices=[a.value for a in AccessLevel], default="public")
    q.add_argument("--feature", choices=[f.value for f in Feature], default=None)
    q.add_argument("--source", choices=[s.value for s in SourceType], action="append")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    provider = build_provider(get_settings())

    if args.cmd == "build":
        index = build_index(args.docs_dir, provider, pattern=args.pattern)
        index.save(args.out)
        print(f"index written to {args.out}")
        print(manifest_summary(index))
        return 0

    index = ChunkIndex.load(args.index_dir)
    filters = RetrievalFilters(
        allowed_access=access_up_to(AccessLevel(args.access)),
        feature=Feature(args.feature) if args.feature else None,
        source_types=frozenset(SourceType(s) for s in args.source) if args.source else None,
    )
    result = retrieve(index, provider, args.text, filters=filters, top_k=args.top_k)
    print(
        f"query={result.query!r}  candidates={result.candidates_considered}  "
        f"excluded_by_access={result.excluded_by_access}"
    )
    for r in result.chunks:
        flags = " STALE" if r.stale else ""
        ids = ",".join(r.chunk.requirement_ids) or "-"
        print(
            f"  #{r.rank} {r.chunk_id}  score={r.score:.3f} (vec {r.vector_score:.2f}, "
            f"kw {r.keyword_score:.2f}){flags}"
        )
        tag = f"{r.chunk.source_type.value}/{r.chunk.access_level.value}"
        print(f"     {r.chunk.citation}  [{tag}] ids={ids}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
