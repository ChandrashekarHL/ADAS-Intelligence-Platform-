"""Load telemetry CSVs into the canonical SI frame.

Column resolution, in priority order:

1. explicit ``column_units`` passed by the caller,
2. ``column_units`` from a ``scenario.json`` sidecar next to the file,
3. the unit suffix on the header itself (``ego_speed_kmh`` → km/h).

Whatever the source, the unit is converted exactly once here and the conversion is
recorded in the provenance. Nothing is sorted, filled, interpolated or dropped: the
quality gates must see the data as logged.
"""

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.errors import IngestionError
from app.core.ids import new_id
from app.core.signals import (
    CANONICAL_UNITS,
    COL_TIMESTAMP,
    FRAME_COLUMNS,
    NUMERIC_COLUMNS,
    SIGNAL_BASE_NAMES,
)
from app.core.units import KMH_TO_MPS, MS_TO_S
from app.ingestion.schemas import (
    DATA_ORIGIN_UNKNOWN,
    IngestedTelemetry,
    SidecarInfo,
    TelemetryProvenance,
    UnitConversion,
)

SIDECAR_NAME = "scenario.json"

# Header suffix → unit label as written in sidecars.
_SUFFIX_UNITS: dict[str, str] = {
    "kmh": "km/h",
    "mps": "m/s",
    "mps2": "m/s^2",
    "m": "m",
    "s": "s",
    "ms": "ms",
}
_SUFFIX_RE = re.compile(r"^(?P<base>.+?)_(?P<suffix>kmh|mps2|mps|ms|m|s)$")

# (source unit, target unit) → multiplicative factor. Identity pairs are implicit.
_FACTORS: dict[tuple[str, str], float] = {
    ("km/h", "m/s"): KMH_TO_MPS,
    ("ms", "s"): MS_TO_S,
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_sidecar(csv_path: Path) -> tuple[SidecarInfo | None, Path | None]:
    """Return sidecar info if a parseable ``scenario.json`` sits next to the CSV."""
    sidecar = csv_path.with_name(SIDECAR_NAME)
    if not sidecar.is_file():
        return None, None
    try:
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
        return SidecarInfo.model_validate(raw), sidecar
    except (OSError, ValueError) as exc:  # malformed sidecar is not fatal; the CSV still loads
        raise IngestionError(f"sidecar {sidecar} is not valid: {exc}") from exc


def resolve_column(name: str, declared_unit: str | None) -> tuple[str, str | None]:
    """Map an external header to ``(canonical_column, source_unit)``.

    Returns ``(name, None)`` for columns we do not recognise; they pass through unchanged.
    """
    if name in FRAME_COLUMNS:
        return name, declared_unit or CANONICAL_UNITS[name]
    base, suffix_unit = name, None
    m = _SUFFIX_RE.match(name)
    if m and m.group("base") in SIGNAL_BASE_NAMES:
        base, suffix_unit = m.group("base"), _SUFFIX_UNITS[m.group("suffix")]
    if base not in SIGNAL_BASE_NAMES:
        return name, None
    canonical = SIGNAL_BASE_NAMES[base]
    unit = declared_unit or suffix_unit or CANONICAL_UNITS[canonical]
    return canonical, unit


def normalise_frame(
    raw: pd.DataFrame, column_units: Mapping[str, str]
) -> tuple[
    pd.DataFrame, dict[str, str], tuple[UnitConversion, ...], tuple[str, ...], dict[str, int]
]:
    """Rename to canonical columns, convert units once, coerce numerics.

    Returns ``(frame, renamed, conversions, passthrough, coerced_counts)``.
    """
    frame = raw.copy()
    renamed: dict[str, str] = {}
    conversions: list[UnitConversion] = []
    passthrough: list[str] = []
    coerced: dict[str, int] = {}

    for col in list(frame.columns):
        canonical, unit = resolve_column(str(col), column_units.get(str(col)))
        if unit is None:
            passthrough.append(str(col))
            continue
        if canonical in frame.columns and canonical != col:
            raise IngestionError(f"columns {col!r} and {canonical!r} both map to {canonical!r}")

        if canonical in NUMERIC_COLUMNS:
            before = frame[col].isna().sum()
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
            newly_nan = int(frame[col].isna().sum() - before)
            if newly_nan:
                coerced[canonical] = newly_nan

        target_unit = CANONICAL_UNITS[canonical]
        if unit != target_unit:
            factor = _FACTORS.get((unit, target_unit))
            if factor is None:
                raise IngestionError(
                    f"cannot convert {col!r} from {unit!r} to {target_unit!r}; "
                    "add the factor to app/ingestion/csv_loader.py"
                )
            frame[col] = frame[col].astype("float64") * factor
            conversions.append(
                UnitConversion(
                    source_column=str(col),
                    target_column=canonical,
                    source_unit=unit,
                    target_unit=target_unit,
                    factor=factor,
                )
            )
        if canonical != col:
            renamed[str(col)] = canonical

    frame = frame.rename(columns=renamed)
    ordered = [c for c in FRAME_COLUMNS if c in frame.columns] + passthrough
    frame = frame.loc[:, ordered].reset_index(drop=True)
    return frame, renamed, tuple(conversions), tuple(passthrough), coerced


def _timing(frame: pd.DataFrame) -> tuple[float | None, float | None]:
    if COL_TIMESTAMP not in frame.columns:
        return None, None
    t = frame[COL_TIMESTAMP].dropna().to_numpy(dtype="float64")
    if t.size < 2:
        return None, None
    diffs = np.diff(t)
    positive = diffs[diffs > 0]
    nominal_dt = float(np.median(positive)) if positive.size else None
    return float(t.max() - t.min()), nominal_dt


def ingest_frame(
    raw: pd.DataFrame,
    *,
    source_path: str,
    column_units: Mapping[str, str] | None = None,
    data_origin: str = DATA_ORIGIN_UNKNOWN,
    scenario_id: str | None = None,
    sidecar_path: str | None = None,
    sha256: str | None = None,
) -> IngestedTelemetry:
    """Normalise an already-read frame. Used by :func:`load_telemetry_csv` and by tests."""
    frame, renamed, conversions, passthrough, coerced = normalise_frame(raw, column_units or {})
    duration, nominal_dt = _timing(frame)
    provenance = TelemetryProvenance(
        file_id=new_id("file"),
        source_path=source_path,
        sha256=sha256,
        data_origin=data_origin,
        scenario_id=scenario_id,
        sidecar_path=sidecar_path,
        renamed_columns=renamed,
        conversions=conversions,
        passthrough_columns=passthrough,
        coerced_values=coerced,
        row_count=int(len(frame)),
        duration_s=duration,
        nominal_dt_s=nominal_dt,
    )
    return IngestedTelemetry(frame=frame, provenance=provenance)


def load_telemetry_csv(
    path: Path, *, column_units: Mapping[str, str] | None = None
) -> IngestedTelemetry:
    """Read a telemetry CSV (plus optional sidecar) into the canonical SI frame."""
    if not path.is_file():
        raise FileNotFoundError(path)
    sidecar, sidecar_path = read_sidecar(path)
    units: dict[str, str] = dict(sidecar.column_units) if sidecar else {}
    if column_units:
        units.update(column_units)  # caller wins
    try:
        raw = pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise IngestionError(f"cannot parse {path}: {exc}") from exc
    return ingest_frame(
        raw,
        source_path=str(path),
        column_units=units,
        data_origin=(
            sidecar.data_origin if sidecar and sidecar.data_origin else DATA_ORIGIN_UNKNOWN
        ),
        scenario_id=sidecar.scenario_id if sidecar else None,
        sidecar_path=str(sidecar_path) if sidecar_path else None,
        sha256=_sha256(path),
    )
