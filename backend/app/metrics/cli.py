"""Ingest → quality gates → AEB metrics, printed as an evidence table.

    python -m app.metrics.cli ../data/demo/aeb_late_braking_seed42/telemetry.csv [--json]

Exit code 2 when the quality gates block analysis.
"""

import argparse
import json
import sys
from pathlib import Path

from app.core.errors import DataQualityError
from app.ingestion.csv_loader import load_telemetry_csv
from app.metrics.aeb import compute_aeb_metrics, fmt_value
from app.metrics.schemas import AebThresholds
from app.quality.report import evaluate_gates

EXIT_BLOCKED = 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aip-metrics", description=__doc__)
    p.add_argument("csv", type=Path)
    p.add_argument("--json", action="store_true")
    p.add_argument("--trigger-ttc-s", type=float, default=None)
    p.add_argument("--max-latency-s", type=float, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides: dict[str, float] = {}
    if args.trigger_ttc_s is not None:
        overrides["trigger_ttc_s"] = args.trigger_ttc_s
    if args.max_latency_s is not None:
        overrides["max_braking_latency_s"] = args.max_latency_s
    thresholds = AebThresholds().model_copy(update=overrides)

    telemetry = load_telemetry_csv(args.csv)
    quality = evaluate_gates(telemetry)
    try:
        report = compute_aeb_metrics(telemetry, quality, thresholds)
    except DataQualityError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return EXIT_BLOCKED

    if args.json:
        print(json.dumps(report.model_dump(mode="json"), indent=2))
        return 0

    print(f"file={report.file_id}  quality={report.quality_id} ({quality.verdict.value})")
    for e in report.events:
        print(f"  {e.event_id}  {e.event_type.value:<24} t={e.t_s:.2f}s  {e.description}")
    for w in report.windows:
        flag = " (clipped)" if w.clipped_start or w.clipped_end else ""
        print(
            f"  {w.window_id}  {w.start_s:.2f}s..{w.end_s:.2f}s  {w.sample_count} samples"
            f"  event={w.event_id}{flag}"
        )
    print(f"primary window: {report.primary_window_id}")
    print(f"  {'metric':<36}{'value':>10}  {'unit':<7}{'t [s]':>7}  {'threshold':<14}{'result'}")
    for m in report.metrics:
        thr = f"{m.comparator.value} {m.threshold}" if m.comparator else ""
        res = "PASS" if m.passed else "FAIL" if m.passed is False else ""
        if m.value is None:
            res = f"MISSING: {m.missing_reason}"
        t_s = f"{m.t_s:.2f}" if m.t_s is not None else ""
        print(f"  {m.name:<36}{fmt_value(m):>10}  {m.unit:<7}{t_s:>7}  {thr:<14}{res}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
