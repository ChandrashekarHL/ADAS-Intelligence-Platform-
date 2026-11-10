"""Run the locked evaluation suite and apply the release gate.

    python -m app.eval.cli list
    python -m app.eval.cli run --no-agent                      # deterministic checks (CI)
    python -m app.eval.cli run --index ../data/index_openai --access internal \
        --out ../evals/results --baseline ../evals/baseline.json

Exit codes: 0 gate passed, 1 gate failed, 3 no usable provider (LLM_PROVIDER=fake cannot
answer; pass --no-agent).
"""

import argparse
import io
import sys
import tempfile
from pathlib import Path

from app.agents.diagnostic import DEFAULT_QUESTION
from app.core.config import get_settings
from app.eval.runner import RunOptions, load_report, run_suite, write_report
from app.eval.schemas import EvalReport, GateThresholds
from app.eval.suite import LOCKED_SUITE, select_cases, suite_sha256
from app.llm.factory import build_provider
from app.rag.cli import access_up_to
from app.rag.index import ChunkIndex
from app.rag.schemas import AccessLevel, RetrievalFilters

EXIT_GATE_FAILED = 1
EXIT_NO_PROVIDER = 3


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aip-eval", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="print the locked cases and the suite hash")

    run = sub.add_parser("run", help="run the suite and apply the gate")
    run.add_argument("--out", type=Path, default=Path("../evals/results"))
    run.add_argument("--work-dir", type=Path, default=None, help="scenario scratch dir")
    run.add_argument("--index", type=Path, default=None, help="RAG index directory")
    run.add_argument("--access", choices=[a.value for a in AccessLevel], default="internal")
    run.add_argument("--cases", default=None, help="comma-separated case ids (default: all)")
    run.add_argument("--no-agent", action="store_true", help="deterministic checks only")
    run.add_argument("--question", default=DEFAULT_QUESTION)
    run.add_argument("--top-k", type=int, default=6)
    run.add_argument("--baseline", type=Path, default=None, help="previous eval JSON")
    run.add_argument("--max-unsupported", type=float, default=0.10)
    run.add_argument("--min-support", type=float, default=0.90)
    run.add_argument("--min-top1", type=float, default=None)
    run.add_argument("--max-false-fault", type=float, default=None)
    return p


def _print_summary(report: EvalReport) -> None:
    a = report.aggregates
    agent_state = "on" if report.agent_enabled else "off"
    print(
        f"{report.eval_id}  suite={report.suite_name}@{report.suite_sha256[:12]}  "
        f"provider={report.provider}/{report.model}  agent={agent_state}"
    )
    for r in report.cases:
        ag = r.agent
        extra = ""
        if ag is not None:
            top = ag.top_failure_class.value if ag.top_failure_class else "-"
            extra = (
                f"  top={top} conf={ag.report_confidence.value} "
                f"support={ag.evidence_support_rate:.0%} lat={ag.latency_s:.1f}s"
            )
        print(f"  {r.status.value.upper():5} {r.case_id:<32} {r.expected_outcome.value:<8}{extra}")
        for c in r.failed_checks:
            print(f"        - {c.name}: {c.detail}")
        if r.error:
            print(f"        - error: {r.error}")

    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.0%}"

    print(
        f"cases={a.n_cases} pass={a.n_passed} fail={a.n_failed} error={a.n_errors}  "
        f"agent_cases={a.n_agent_cases}  top1={pct(a.top1_accuracy)}  "
        f"false_fault={pct(a.false_fault_rate)}  support={pct(a.mean_evidence_support_rate)}  "
        f"unsupported_max={pct(a.max_unsupported_claim_rate)}"
    )
    if a.latency_p50_s is not None and a.latency_p95_s is not None:
        print(
            f"latency p50={a.latency_p50_s:.2f}s p95={a.latency_p95_s:.2f}s  "
            f"tokens={a.prompt_tokens}/{a.completion_tokens}"
        )
    t = report.gate.thresholds
    if (
        a.n_agent_cases
        and report.gate.baseline_eval_id is None
        and t.min_top1_accuracy is None
        and t.max_false_fault_rate is None
    ):
        print(
            "note: agent accuracy is recorded but not gated; pass --baseline, --min-top1 or "
            "--max-false-fault to gate it"
        )
    print(f"GATE: {'PASS' if report.gate.passed else 'FAIL'}")
    for f in report.gate.failures:
        print(f"  - {f}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(errors="replace")

    if args.command == "list":
        print(f"suite sha256: {suite_sha256()}")
        for c in LOCKED_SUITE:
            print(f"  {c.case_id:<32} {c.expected_outcome.value:<8} {c.description}")
        return 0

    try:
        cases = select_cases(args.cases.split(",") if args.cases else None)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    settings = get_settings()
    provider = build_provider(settings)
    run_agent = not args.no_agent
    if run_agent and provider.name == "fake":
        print(
            "LLM_PROVIDER=fake has no answers; pass --no-agent for deterministic checks only",
            file=sys.stderr,
        )
        return EXIT_NO_PROVIDER

    index = ChunkIndex.load(args.index) if args.index else None
    filters = RetrievalFilters(allowed_access=access_up_to(AccessLevel(args.access)))
    baseline = load_report(args.baseline) if args.baseline else None
    gate = GateThresholds(
        max_unsupported_claim_rate=args.max_unsupported,
        min_evidence_support_rate=args.min_support,
        min_top1_accuracy=args.min_top1,
        max_false_fault_rate=args.max_false_fault,
    )
    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="aip-eval-"))
    report = run_suite(
        cases,
        provider,
        index=index,
        filters=filters,
        work_dir=work_dir,
        options=RunOptions(run_agent=run_agent, question=args.question, top_k=args.top_k),
        gate=gate,
        baseline=baseline,
    )
    md, js = write_report(report, args.out)
    _print_summary(report)
    print(f"written: {js}  {md}")
    return 0 if report.gate.passed else EXIT_GATE_FAILED


if __name__ == "__main__":
    sys.exit(main())
