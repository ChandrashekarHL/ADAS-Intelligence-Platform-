"""Ingest a telemetry CSV and print its provenance and quality report.

    python -m app.ingestion.cli ../data/demo/aeb_late_braking_seed42/telemetry.csv [--json]

Exit code 0 when analysis may proceed (PASS or DEGRADED), 2 when BLOCKED.
"""

import argparse
import json
import sys
from pathlib import Path

from app.ingestion.csv_loader import load_telemetry_csv
from app.quality.report import QualityVerdict, evaluate_gates

EXIT_BLOCKED = 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aip-ingest", description=__doc__)
    p.add_argument("csv", type=Path)
    p.add_argument("--json", action="store_true", help="emit provenance + report as JSON")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    telemetry = load_telemetry_csv(args.csv)
    report = evaluate_gates(telemetry)
    prov = telemetry.provenance

    if args.json:
        payload = {
            "provenance": prov.model_dump(mode="json"),
            "quality": report.model_dump(mode="json"),
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"{prov.file_id}  {prov.source_path}")
        print(f"  origin={prov.data_origin}  scenario={prov.scenario_id}  rows={prov.row_count}")
        print(f"  duration_s={prov.duration_s}  nominal_dt_s={prov.nominal_dt_s}")
        for c in prov.conversions:
            src, dst = (
                f"{c.source_column} [{c.source_unit}]",
                f"{c.target_column} [{c.target_unit}]",
            )
            print(f"  converted {src} -> {dst}")
        print(f"{report.quality_id}  verdict={report.verdict.value.upper()}")
        for g in report.gates:
            print(f"  [{g.status.value:>4}] {g.gate:<24} {g.message}")
    return EXIT_BLOCKED if report.verdict is QualityVerdict.BLOCKED else 0


if __name__ == "__main__":
    sys.exit(main())
