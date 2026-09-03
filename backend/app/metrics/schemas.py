"""Metric and event-window contracts.

Every value produced here is evidence. A :class:`MetricResult` carries its own
``metric_`` ID, the window it was computed in, the timestamp it refers to, and the
method used, so a report can say *what* was measured, *where*, and *how*. A metric that
cannot be computed is still returned, with ``value=None`` and a ``missing_reason``: the
agent must be told what is missing rather than left to guess.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class EventType(StrEnum):
    AEB_BRAKE_COMMAND = "aeb_brake_command"  # brake_command rising edge
    TTC_THRESHOLD_CROSSING = "ttc_threshold_crossing"  # TTC first <= trigger threshold
    COLLISION = "collision"  # relative distance reached zero


class Event(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str
    event_type: EventType
    t_s: float
    row: int
    description: str


class EventWindow(BaseModel):
    """A ``T-pre .. T+post`` slice of the log around one event (spec §9.3)."""

    model_config = ConfigDict(frozen=True)

    window_id: str
    event_id: str
    t_event_s: float
    start_s: float
    end_s: float
    start_row: int
    end_row: int  # inclusive
    sample_count: int
    clipped_start: bool  # window would have started before the log
    clipped_end: bool  # window would have ended after the log


class Comparator(StrEnum):
    LE = "<="
    GE = ">="
    EQ = "=="


class MetricResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    metric_id: str
    name: str
    value: float | bool | None
    unit: str
    t_s: float | None = None  # timestamp the value refers to, when meaningful
    threshold: float | bool | None = None
    comparator: Comparator | None = None
    passed: bool | None = None  # None when no threshold or no value
    window_id: str | None = None
    method: str
    missing_reason: str | None = None
    details: dict[str, float | int | str | bool | None] = Field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.value is not None


class AebThresholds(BaseModel):
    """Pass/fail criteria. Defaults mirror the synthetic controller; the requirement RAG
    (M5) will surface the real thresholds from documents so the agent can compare."""

    model_config = ConfigDict(frozen=True)

    trigger_ttc_s: float = Field(default=2.0, gt=0)
    max_braking_latency_s: float = Field(default=0.30, gt=0)
    min_acceptable_ttc_s: float = Field(default=0.5, ge=0)
    collision_gap_m: float = Field(default=0.10, ge=0)
    detection_confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    max_deceleration_mps2: float = Field(default=10.0, gt=0)
    max_comfort_jerk_mps3: float = Field(default=30.0, gt=0)

    # Kinematics below this closing speed are treated as "not closing" (noise floor).
    closing_speed_floor_mps: float = Field(default=0.5, gt=0)
    # Event window extent (spec: T-5 s .. T+5 s).
    window_pre_s: float = Field(default=5.0, ge=0)
    window_post_s: float = Field(default=5.0, ge=0)
    # Smoothing (samples) applied to acceleration before differentiating for jerk.
    jerk_smoothing_samples: int = Field(default=5, ge=1)


class AebMetricsReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    file_id: str
    quality_id: str
    thresholds: AebThresholds
    events: tuple[Event, ...]
    windows: tuple[EventWindow, ...]
    primary_window_id: str | None
    metrics: tuple[MetricResult, ...]

    def metric(self, name: str) -> MetricResult:
        for m in self.metrics:
            if m.name == name:
                return m
        raise KeyError(name)

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        ids = [e.event_id for e in self.events] + [w.window_id for w in self.windows]
        ids += [m.metric_id for m in self.metrics if m.available]
        return tuple(ids)
