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
from app.agents.evidence import neutralise
from app.agents.pipeline import prepare_diagnosis
from app.core.config import get_settings
from app.core.errors import DataQualityError, ProviderError
from app.llm.factory import build_provider
from app.rag.cli import access_up_to
from app.rag.index import ChunkIndex
from app.rag.schemas import AccessLevel, RetrievalFilters
from app.verification.verifier import verify_diagnosis

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
    agent = DiagnosticAgent(provider)
    if args.dry_run:
        print(f"offered evidence ids: {len(bundle.offered_ids)}  missing: {len(bundle.missing)}")
        if bundle.injection_flags:
            print(f"injection flags: {[f.pattern for f in bundle.injection_flags]}")
        # Show exactly what the model would receive: every message, with its role.
        for msg in agent.build_request(bundle, args.question).messages:
            print(f"\n===== {msg.role.value.upper()} MESSAGE =====")
            print(neutralise(msg.content))
        return 0

    try:
        run = agent.run(bundle, args.question)
    except ProviderError as exc:
        hint = (
            " (LLM_PROVIDER=fake has no answers; use --dry-run)" if provider.name == "fake" else ""
        )
        print(f"NO PROVIDER ANSWER: {exc}{hint}", file=sys.stderr)
        return EXIT_NO_PROVIDER

    verification = verify_diagnosis(run, inputs)
    if args.json:
        payload = {
            "run": run.model_dump(mode="json"),
            "verification": verification.model_dump(mode="json"),
        }
        print(json.dumps(payload, indent=2))
        return 0
    print(
        f"{run.run_id}  agent={run.agent}  provider={run.provider}/{run.model}  "
        f"attempts={run.attempts}"
    )
    print(
        f"  tokens={run.usage.total_tokens}  latency={run.latency_s:.2f}s  origin={run.data_origin}"
    )
    v = verification
    print(
        f"{v.verification_id}  confidence={v.report_confidence.value.upper()}  "
        f"human_review={'YES' if v.human_review_required else 'no'}  "
        f"evidence_support={v.evidence_support_rate:.0%}  "
        f"unsupported_claims={v.unsupported_claim_rate:.0%}"
    )
    for reason in v.review_reasons:
        print(f"  REVIEW: {reason}")
    print("observations:")
    for o in run.output.observations:
        flag = "  [FLAGGED: unknown ids]" if o in v.flagged_observations else ""
        print(f"  - {neutralise(o)}{flag}")
    print("hypotheses (verified, ranked):")
    for i, h in enumerate(v.hypotheses, 1):
        hy = h.hypothesis
        print(
            f"  {i}. [{hy.failure_class.value}] {h.confidence_label.value} "
            f"(agent {h.agent_confidence:.2f} -> adjusted {h.adjusted_confidence:.2f}) "
            f"{neutralise(hy.cause)}"
        )
        sources = ", ".join(h.independent_sources)
        print(f"     evidence: {', '.join(h.resolved_ids)}  sources: {sources}")
        for note in h.notes:
            print(f"     note: {note}")
    for s in v.stripped:
        print(f"  STRIPPED: {neutralise(s.hypothesis.cause)}  ({s.reason})")
    print("missing_evidence:")
    for m in v.missing_evidence:
        print(f"  - {neutralise(m)}")
    print("recommended_next_tests:")
    for t in v.recommended_next_tests:
        print(f"  - {neutralise(t)}")
    print("limitations:")
    for lim in v.limitations:
        print(f"  - {lim}")
    print(f"disclaimer: {v.disclaimer}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
