"""Diagnose an AEB log end to end.

    python -m app.agents.cli ../data/demo/aeb_late_braking_seed42/telemetry.csv \
        --index ../data/index --access internal

With LLM_PROVIDER=fake there is no model to answer, so use ``--dry-run`` to print the
exact evidence bundle and prompt the agent would send. Exit codes: 2 blocked by data
quality, 3 no usable provider.
"""

import argparse
import io
import json
import sys
from pathlib import Path

from app.agents.diagnostic import DEFAULT_QUESTION, DiagnosticAgent
from app.agents.pipeline import prepare_diagnosis
from app.core.config import get_settings
from app.core.errors import DataQualityError, ProviderError
from app.llm.factory import build_provider
from app.rag.cli import access_up_to
from app.rag.index import ChunkIndex
from app.rag.schemas import AccessLevel, RetrievalFilters

EXIT_BLOCKED = 2
EXIT_NO_PROVIDER = 3


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aip-diagnose", description=__doc__)
    p.add_argument("csv", type=Path)
    p.add_argument("--index", type=Path, default=None, help="RAG index directory")
    p.add_argument("--access", choices=[a.value for a in AccessLevel], default="public")
    p.add_argument("--question", default=DEFAULT_QUESTION)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--dry-run", action="store_true", help="print the prompt, do not call the model")
    p.add_argument("--json", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if isinstance(sys.stdout, io.TextIOWrapper):
        # Retrieved document text is arbitrary; never crash a legacy console on one glyph.
        sys.stdout.reconfigure(errors="replace")
    settings = get_settings()
    provider = build_provider(settings)
    index = ChunkIndex.load(args.index) if args.index else None
    filters = RetrievalFilters(allowed_access=access_up_to(AccessLevel(args.access)))

    try:
        inputs = prepare_diagnosis(
            args.csv, provider, index=index, filters=filters, top_k=args.top_k
        )
    except DataQualityError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return EXIT_BLOCKED

    bundle = inputs.bundle
    if args.dry_run:
        print(f"offered evidence ids: {len(bundle.offered_ids)}  missing: {len(bundle.missing)}")
        if bundle.injection_flags:
            print(f"injection flags: {[f.pattern for f in bundle.injection_flags]}")
        print(f"QUESTION: {args.question}\n")
        print(bundle.render())
        return 0

    agent = DiagnosticAgent(provider)
    try:
        run = agent.run(bundle, args.question)
    except ProviderError as exc:
        hint = (
            " (LLM_PROVIDER=fake has no answers; use --dry-run)" if provider.name == "fake" else ""
        )
        print(f"NO PROVIDER ANSWER: {exc}{hint}", file=sys.stderr)
        return EXIT_NO_PROVIDER

    if args.json:
        print(json.dumps(run.model_dump(mode="json"), indent=2))
        return 0
    print(
        f"{run.run_id}  agent={run.agent}  provider={run.provider}/{run.model}  "
        f"attempts={run.attempts}"
    )
    print(
        f"  tokens={run.usage.total_tokens}  latency={run.latency_s:.2f}s  origin={run.data_origin}"
    )
    if run.unresolved_ids:
        print(f"  UNRESOLVED IDS (verifier will strip): {', '.join(run.unresolved_ids)}")
    out = run.output
    print("observations:")
    for o in out.observations:
        print(f"  - {o}")
    print("hypotheses:")
    for i, h in enumerate(out.hypotheses, 1):
        print(f"  {i}. [{h.failure_class.value}] conf={h.confidence:.2f} {h.cause}")
        print(f"     evidence: {', '.join(h.evidence_ids) or '-'}")
    print("missing_evidence:")
    for m in out.missing_evidence:
        print(f"  - {m}")
    print("recommended_next_tests:")
    for t in out.recommended_next_tests:
        print(f"  - {t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
