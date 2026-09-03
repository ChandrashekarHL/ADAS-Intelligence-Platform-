"""Configuration and ground-truth contracts for synthetic AEB scenarios.

All quantities are SI (m, m/s, m/s^2, s). Every scenario is fully determined by its
config (including ``seed``), so a scenario can be regenerated bit-for-bit from the
metadata JSON written next to its CSV.

Synthetic data is simulation-only evidence. Nothing produced here may be described as
real-world validation; the ``data_origin`` marker travels with every export.
"""

from enum import IntEnum, StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

DATA_ORIGIN_SYNTHETIC = "synthetic"

# Canonical in-memory column names (SI). Ingestion (M2) maps external CSV headers to these.
COL_TIMESTAMP = "timestamp_s"
COL_EGO_SPEED = "ego_speed_mps"
COL_EGO_ACCEL = "ego_acceleration_mps2"
COL_REL_DISTANCE = "relative_distance_m"
COL_REL_VELOCITY = "relative_velocity_mps"
COL_OBJECT_CLASS = "object_class"
COL_OBJECT_CONF = "object_confidence"
COL_BRAKE_CMD = "brake_command"
COL_AEB_STATE = "aeb_state"
COL_WEATHER = "weather"

FRAME_COLUMNS: tuple[str, ...] = (
    COL_TIMESTAMP,
    COL_EGO_SPEED,
    COL_EGO_ACCEL,
    COL_REL_DISTANCE,
    COL_REL_VELOCITY,
    COL_OBJECT_CLASS,
    COL_OBJECT_CONF,
    COL_BRAKE_CMD,
    COL_AEB_STATE,
    COL_WEATHER,
)


class ScenarioVariant(StrEnum):
    """Which AEB behaviour the scenario exhibits."""

    NOMINAL = "nominal"
    LATE_BRAKING = "late_braking"


class AebState(IntEnum):
    """Integer codes for the ADAS state signal (stored as plain ints in the frame)."""

    IDLE = 0
    WARNING = 1
    BRAKING = 2


class TimestampGap(BaseModel):
    """Remove all samples in ``[start_s, start_s + duration_s)`` — a logging dropout."""

    model_config = ConfigDict(frozen=True)

    start_s: float = Field(ge=0)
    duration_s: float = Field(gt=0)


class NanBurst(BaseModel):
    """Replace one signal with NaN in ``[start_s, start_s + duration_s)`` — a sensor dropout."""

    model_config = ConfigDict(frozen=True)

    column: str
    start_s: float = Field(ge=0)
    duration_s: float = Field(gt=0)


class FaultInjection(BaseModel):
    """Data-quality faults layered on top of the physical scenario.

    These exist so the M2 quality gates can be tested against known-bad inputs. They
    are applied after the kinematics are generated and are recorded in the metadata,
    so a downstream consumer can never mistake an injected fault for a real defect.
    """

    model_config = ConfigDict(frozen=True)

    drop_columns: tuple[str, ...] = ()
    timestamp_gap: TimestampGap | None = None
    nan_burst: NanBurst | None = None

    @model_validator(mode="after")
    def _columns_exist(self) -> "FaultInjection":
        unknown = [c for c in self.drop_columns if c not in FRAME_COLUMNS]
        if unknown:
            raise ValueError(f"drop_columns not in frame: {unknown}")
        if COL_TIMESTAMP in self.drop_columns:
            raise ValueError("timestamp column cannot be dropped; use timestamp_gap instead")
        if self.nan_burst is not None and self.nan_burst.column not in FRAME_COLUMNS:
            raise ValueError(f"nan_burst.column not in frame: {self.nan_burst.column!r}")
        return self


class AebScenarioConfig(BaseModel):
    """Full description of a lead-vehicle-sudden-braking scenario (spec §10, §26 AEB)."""

    model_config = ConfigDict(frozen=True)

    variant: ScenarioVariant = ScenarioVariant.NOMINAL
    seed: int = Field(default=0, ge=0)

    # Sampling
    duration_s: float = Field(default=10.0, gt=0)
    sample_rate_hz: float = Field(default=50.0, gt=0)

    # Initial kinematics
    ego_speed_mps: float = Field(default=50 / 3.6, gt=0)
    lead_speed_mps: float = Field(default=50 / 3.6, ge=0)
    initial_gap_m: float = Field(default=30.0, gt=0)

    # Lead vehicle behaviour
    lead_brake_onset_s: float = Field(default=3.0, ge=0)
    lead_decel_mps2: float = Field(default=6.0, gt=0)

    # AEB controller
    warning_ttc_s: float = Field(default=2.8, gt=0)
    trigger_ttc_s: float = Field(default=2.0, gt=0)
    system_latency_s: float = Field(default=0.15, ge=0)
    ego_max_decel_mps2: float = Field(default=9.0, gt=0)
    ego_jerk_limit_mps3: float = Field(default=25.0, gt=0)

    # Perception
    detection_confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    nominal_confidence: float = Field(default=0.92, ge=0, le=1)
    # Late-braking variant only: confidence collapses for this window after lead brake onset.
    confidence_drop_start_after_onset_s: float = Field(default=1.2, ge=0)
    confidence_drop_duration_s: float = Field(default=1.2, ge=0)
    dropped_confidence: float = Field(default=0.22, ge=0, le=1)

    # Measurement noise (1-sigma, applied to reported signals, not to the physics)
    distance_noise_m: float = Field(default=0.05, ge=0)
    speed_noise_mps: float = Field(default=0.03, ge=0)
    accel_noise_mps2: float = Field(default=0.05, ge=0)
    confidence_noise: float = Field(default=0.02, ge=0)

    weather: str = "clear"
    faults: FaultInjection = FaultInjection()

    @model_validator(mode="after")
    def _consistent(self) -> "AebScenarioConfig":
        if self.trigger_ttc_s >= self.warning_ttc_s:
            raise ValueError("trigger_ttc_s must be below warning_ttc_s")
        if self.lead_brake_onset_s >= self.duration_s:
            raise ValueError("lead_brake_onset_s must be inside the scenario duration")
        if self.dropped_confidence >= self.detection_confidence_threshold:
            raise ValueError("dropped_confidence must be below detection_confidence_threshold")
        if self.nominal_confidence < self.detection_confidence_threshold:
            raise ValueError("nominal_confidence must reach detection_confidence_threshold")
        return self

    @property
    def dt_s(self) -> float:
        return 1.0 / self.sample_rate_hz


class ScenarioGroundTruth(BaseModel):
    """Noise-free truth about what happened, derived from the physics, not the signals.

    Used to validate the M3 metric implementations and to label demo scenarios.
    ``None`` means the event never occurred within the scenario.
    """

    model_config = ConfigDict(frozen=True)

    lead_brake_onset_s: float
    risk_threshold_crossing_s: float | None
    perception_valid_at_risk_s: float | None
    brake_command_s: float | None
    braking_latency_s: float | None
    min_ttc_s: float | None
    min_gap_m: float
    max_deceleration_mps2: float
    collision: bool
    collision_time_s: float | None
    ego_stopped_s: float | None


class ScenarioMetadata(BaseModel):
    """Sidecar JSON written next to every exported CSV."""

    model_config = ConfigDict(frozen=True)

    scenario_id: str
    data_origin: str = DATA_ORIGIN_SYNTHETIC
    generator_version: str
    config: AebScenarioConfig
    ground_truth: ScenarioGroundTruth
    csv_file: str
    column_units: dict[str, str]
    notes: tuple[str, ...]
