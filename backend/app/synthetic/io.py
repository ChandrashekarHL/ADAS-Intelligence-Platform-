"""Export synthetic scenarios to CSV + metadata JSON.

The CSV imitates what an engineer would upload: speeds in km/h (the common OEM logging
convention), everything else SI. Ingestion (M2) converts back to SI exactly once using
the ``column_units`` map recorded in the sidecar metadata. Telemetry stays on the
filesystem; only references to it will ever enter the database.
"""

import json
from pathlib import Path
from typing import Literal

import pandas as pd

from app.core.ids import new_id
from app.core.signals import (
    CANONICAL_UNITS,
    COL_EGO_SPEED,
    COL_REL_VELOCITY,
)
from app.core.units import MPS_TO_KMH
from app.synthetic.aeb_generator import GENERATOR_VERSION, SyntheticScenario
from app.synthetic.schemas import (
    ScenarioMetadata,
    ScenarioVariant,
)

SpeedUnit = Literal["mps", "kmh"]

CSV_NAME = "telemetry.csv"
METADATA_NAME = "scenario.json"

# Column name used in the CSV when speeds are exported in km/h.
CSV_EGO_SPEED_KMH = "ego_speed_kmh"
CSV_REL_VELOCITY_KMH = "relative_velocity_kmh"


SYNTHETIC_NOTES: tuple[str, ...] = (
    "Synthetic 1-D kinematic simulation. Not real-world data; not evidence of on-road behaviour.",
    "Signals carry seeded Gaussian measurement noise; ground_truth is noise-free.",
    "Any faults listed under config.faults were injected deliberately for data-quality "
    "gate testing.",
)


def to_export_frame(scenario: SyntheticScenario, speed_unit: SpeedUnit) -> pd.DataFrame:
    """Return the frame as it will be written, plus rename speeds if exporting in km/h."""
    frame = scenario.frame.copy()
    if speed_unit == "kmh":
        renames: dict[str, str] = {}
        if COL_EGO_SPEED in frame.columns:
            frame[COL_EGO_SPEED] = frame[COL_EGO_SPEED] * MPS_TO_KMH
            renames[COL_EGO_SPEED] = CSV_EGO_SPEED_KMH
        if COL_REL_VELOCITY in frame.columns:
            frame[COL_REL_VELOCITY] = frame[COL_REL_VELOCITY] * MPS_TO_KMH
            renames[COL_REL_VELOCITY] = CSV_REL_VELOCITY_KMH
        frame = frame.rename(columns=renames)
    return frame


def column_units_for(frame: pd.DataFrame, speed_unit: SpeedUnit) -> dict[str, str]:
    units = dict(CANONICAL_UNITS)
    if speed_unit == "kmh":
        units[CSV_EGO_SPEED_KMH] = "km/h"
        units[CSV_REL_VELOCITY_KMH] = "km/h"
    return {c: units[c] for c in frame.columns if c in units}


def write_scenario(
    scenario: SyntheticScenario,
    out_dir: Path,
    *,
    speed_unit: SpeedUnit = "kmh",
    scenario_id: str | None = None,
) -> ScenarioMetadata:
    """Write ``telemetry.csv`` and ``scenario.json`` into ``out_dir`` (created if missing)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = to_export_frame(scenario, speed_unit)
    frame.to_csv(out_dir / CSV_NAME, index=False, lineterminator="\n")

    meta = ScenarioMetadata(
        scenario_id=scenario_id or new_id("scenario"),
        generator_version=GENERATOR_VERSION,
        config=scenario.config,
        ground_truth=scenario.ground_truth,
        csv_file=CSV_NAME,
        column_units=column_units_for(frame, speed_unit),
        notes=SYNTHETIC_NOTES,
    )
    (out_dir / METADATA_NAME).write_text(
        json.dumps(meta.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    return meta


def read_metadata(path: Path) -> ScenarioMetadata:
    return ScenarioMetadata.model_validate_json(path.read_text(encoding="utf-8"))


def default_scenario_dirname(variant: ScenarioVariant, seed: int) -> str:
    return f"aeb_{variant.value}_seed{seed}"
