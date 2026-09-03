"""Generate demo AEB scenarios from the command line.

    python -m app.synthetic.cli --variant late_braking --seed 42 --out ../data/demo

Writes ``<out>/aeb_<variant>_seed<seed>/{telemetry.csv,scenario.json}`` and prints the
ground truth. ``--all`` writes both variants for the given seed.
"""

import argparse
import sys
from pathlib import Path

from app.synthetic.aeb_generator import generate_aeb_scenario
from app.synthetic.io import SpeedUnit, default_scenario_dirname, write_scenario
from app.synthetic.schemas import AebScenarioConfig, ScenarioVariant


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aip-synth", description=__doc__)
    p.add_argument(
        "--variant",
        choices=[v.value for v in ScenarioVariant],
        default=ScenarioVariant.NOMINAL.value,
    )
    p.add_argument("--all", action="store_true", help="generate every variant")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("../data/demo"))
    p.add_argument("--speed-unit", choices=["kmh", "mps"], default="kmh")
    p.add_argument("--duration-s", type=float, default=None)
    p.add_argument("--sample-rate-hz", type=float, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    variants = list(ScenarioVariant) if args.all else [ScenarioVariant(args.variant)]
    overrides: dict[str, float] = {}
    if args.duration_s is not None:
        overrides["duration_s"] = args.duration_s
    if args.sample_rate_hz is not None:
        overrides["sample_rate_hz"] = args.sample_rate_hz
    speed_unit: SpeedUnit = args.speed_unit

    for variant in variants:
        cfg = AebScenarioConfig(variant=variant, seed=args.seed).model_copy(update=overrides)
        cfg = AebScenarioConfig.model_validate(cfg.model_dump())  # re-run validators
        scenario = generate_aeb_scenario(cfg)
        out_dir = args.out / default_scenario_dirname(variant, args.seed)
        meta = write_scenario(scenario, out_dir, speed_unit=speed_unit)
        gt = scenario.ground_truth
        print(f"{meta.scenario_id}  {variant.value:<13} -> {out_dir}")
        print(f"  rows={len(scenario.frame)}  collision={gt.collision}")
        print(f"  risk_crossing_s={gt.risk_threshold_crossing_s}  brake_cmd_s={gt.brake_command_s}")
        print(f"  braking_latency_s={gt.braking_latency_s}  min_ttc_s={gt.min_ttc_s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
