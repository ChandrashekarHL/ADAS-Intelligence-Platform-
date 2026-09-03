"""Ingestion contracts: what we know about a telemetry file once it is loaded.

Provenance is evidence. Every downstream artifact (quality report, metric, window) links
back to the ``file_id`` minted here, and the conversions list is the audit trail for the
one-time unit normalisation.
"""

from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

DATA_ORIGIN_UNKNOWN = "unknown"


class UnitConversion(BaseModel):
    """One column converted at the ingestion boundary."""

    model_config = ConfigDict(frozen=True)

    source_column: str
    target_column: str
    source_unit: str
    target_unit: str
    factor: float


class SidecarInfo(BaseModel):
    """The subset of a sidecar JSON that ingestion understands. Unknown keys are ignored."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    scenario_id: str | None = None
    data_origin: str | None = None
    column_units: dict[str, str] = Field(default_factory=dict)


class TelemetryProvenance(BaseModel):
    """Where a frame came from and what ingestion did to it."""

    model_config = ConfigDict(frozen=True)

    file_id: str
    source_path: str
    sha256: str | None
    data_origin: str
    scenario_id: str | None
    sidecar_path: str | None
    renamed_columns: dict[str, str]
    conversions: tuple[UnitConversion, ...]
    passthrough_columns: tuple[str, ...]
    coerced_values: dict[str, int]
    row_count: int
    duration_s: float | None
    nominal_dt_s: float | None


@dataclass(frozen=True)
class IngestedTelemetry:
    """SI-normalised frame plus its provenance. The frame is never mutated after this."""

    frame: pd.DataFrame
    provenance: TelemetryProvenance
