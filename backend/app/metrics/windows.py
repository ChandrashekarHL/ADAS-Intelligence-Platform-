"""Event detection and T-pre..T+post window construction.

Events are found on the *logged* signals (never on ground truth). Windows are clipped
to the log and say so; a clipped window is weaker evidence and the report must be able
to see that.
"""

import math

import numpy as np
import pandas as pd

from app.core.ids import new_id, stable_id
from app.core.signals import (
    COL_BRAKE_CMD,
    COL_REL_DISTANCE,
    COL_REL_VELOCITY,
    COL_TIMESTAMP,
)
from app.metrics.schemas import AebThresholds, Event, EventType, EventWindow


def ttc_series(frame: pd.DataFrame, closing_speed_floor_mps: float) -> pd.Series:
    """Time-to-collision per sample: gap / closing speed, NaN when not closing.

    Closing speed is ``-relative_velocity`` (relative velocity is lead minus ego, so a
    negative value means the ego is gaining on the lead).
    """
    gap = frame[COL_REL_DISTANCE].astype("float64")
    closing = -frame[COL_REL_VELOCITY].astype("float64")
    ttc = gap / closing.where(closing > closing_speed_floor_mps)
    return ttc.astype("float64")


def _first_true_row(mask: "pd.Series[bool]") -> int | None:
    idx = np.flatnonzero(mask.to_numpy())
    return int(idx[0]) if idx.size else None


def _event_id(scope: str, kind: EventType) -> str:
    """Stable within an evidence scope (file + thresholds); random when no scope is given."""
    return stable_id("event", scope, kind.value) if scope else new_id("event")


def detect_events(
    frame: pd.DataFrame, thresholds: AebThresholds, *, scope: str = ""
) -> tuple[Event, ...]:
    """Find the AEB-relevant events present in the frame, in time order.

    ``scope`` (normally ``"<file_id>:<thresholds digest>"``) makes the IDs reproducible so
    a re-computation for the same file yields the same evidence IDs.
    """
    t = frame[COL_TIMESTAMP].to_numpy(dtype="float64")
    events: list[Event] = []

    if COL_BRAKE_CMD in frame.columns:
        row = _first_true_row(frame[COL_BRAKE_CMD] == 1)
        if row is not None:
            kind = EventType.AEB_BRAKE_COMMAND
            events.append(
                Event(
                    event_id=_event_id(scope, kind),
                    event_type=EventType.AEB_BRAKE_COMMAND,
                    t_s=float(t[row]),
                    row=row,
                    description="first sample with brake_command == 1",
                )
            )

    if COL_REL_DISTANCE in frame.columns and COL_REL_VELOCITY in frame.columns:
        ttc = ttc_series(frame, thresholds.closing_speed_floor_mps)
        row = _first_true_row(ttc <= thresholds.trigger_ttc_s)
        if row is not None:
            kind = EventType.TTC_THRESHOLD_CROSSING
            events.append(
                Event(
                    event_id=_event_id(scope, kind),
                    event_type=EventType.TTC_THRESHOLD_CROSSING,
                    t_s=float(t[row]),
                    row=row,
                    description=f"first sample with TTC <= {thresholds.trigger_ttc_s} s",
                )
            )
        row = _first_true_row(frame[COL_REL_DISTANCE] <= thresholds.collision_gap_m)
        if row is not None:
            kind = EventType.COLLISION
            events.append(
                Event(
                    event_id=_event_id(scope, kind),
                    event_type=EventType.COLLISION,
                    t_s=float(t[row]),
                    row=row,
                    description=(
                        f"first sample with relative_distance <= {thresholds.collision_gap_m} m"
                    ),
                )
            )

    return tuple(sorted(events, key=lambda e: (e.t_s, e.event_type.value)))


def build_window(
    frame: pd.DataFrame, event: Event, thresholds: AebThresholds, *, scope: str = ""
) -> EventWindow:
    t = frame[COL_TIMESTAMP].to_numpy(dtype="float64")
    want_start = event.t_s - thresholds.window_pre_s
    want_end = event.t_s + thresholds.window_post_s
    start_row = int(np.searchsorted(t, want_start, side="left"))
    end_row = int(np.searchsorted(t, want_end, side="right")) - 1
    end_row = max(end_row, start_row)
    return EventWindow(
        window_id=stable_id("window", scope, event.event_id) if scope else new_id("window"),
        event_id=event.event_id,
        t_event_s=event.t_s,
        start_s=float(t[start_row]),
        end_s=float(t[end_row]),
        start_row=start_row,
        end_row=end_row,
        sample_count=end_row - start_row + 1,
        clipped_start=bool(want_start < t[0] - 1e-9),
        clipped_end=bool(want_end > t[-1] + 1e-9),
    )


def primary_event(events: tuple[Event, ...]) -> Event | None:
    """The event the AEB diagnosis is anchored on: the brake command if present, else
    the TTC threshold crossing (AEB should have acted), else nothing."""
    for wanted in (EventType.AEB_BRAKE_COMMAND, EventType.TTC_THRESHOLD_CROSSING):
        for e in events:
            if e.event_type is wanted:
                return e
    return None


def slice_window(frame: pd.DataFrame, window: EventWindow) -> pd.DataFrame:
    return frame.iloc[window.start_row : window.end_row + 1]


def is_finite(x: float | None) -> bool:
    return x is not None and math.isfinite(x)
