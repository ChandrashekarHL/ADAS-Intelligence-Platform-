"""AEB metrics (spec §13.3, §26): pure functions over an SI frame.

All metrics are computed inside the primary event window when one exists, otherwise
over the whole log; the window ID travels with each result. Signals are used as logged
(after ingestion's unit normalisation) — nothing is smoothed except acceleration before
differentiating for jerk, and that is stated in the method string.
"""

import hashlib
import math
from contextvars import ContextVar

import numpy as np
import pandas as pd

from app.core.ids import new_id, stable_id
from app.core.signals import (
    COL_BRAKE_CMD,
    COL_EGO_ACCEL,
    COL_EGO_SPEED,
    COL_OBJECT_CONF,
    COL_REL_DISTANCE,
    COL_REL_VELOCITY,
    COL_TIMESTAMP,
)
from app.ingestion.schemas import IngestedTelemetry
from app.metrics.schemas import (
    AebMetricsReport,
    AebThresholds,
    Comparator,
    Event,
    EventType,
    EventWindow,
    MetricResult,
)
from app.metrics.windows import (
    build_window,
    detect_events,
    primary_event,
    slice_window,
    ttc_series,
)
from app.quality.report import QualityReport, require_analyzable

# Metric names are part of the public contract (reports, agent prompts, tests).
M_MIN_TTC = "min_ttc_s"
M_TTC_CROSSING = "ttc_threshold_crossing_s"
M_BRAKE_CMD = "brake_command_time_s"
M_BRAKING_LATENCY = "braking_latency_s"
M_MIN_GAP = "min_relative_distance_m"
M_COLLISION = "collision"
M_COLLISION_TIME = "collision_time_s"
M_MAX_DECEL = "max_deceleration_mps2"
M_MAX_JERK = "max_abs_jerk_mps3"
M_FIRST_DETECTION = "first_detection_time_s"
M_CONF_DROPOUT = "confidence_dropout_during_risk_s"
M_MIN_CONF_IN_RISK = "min_confidence_during_risk"
M_SPEED_AT_BRAKE = "ego_speed_at_brake_command_mps"


# Evidence scope for the metrics being computed: "<file_id>:<thresholds digest>". When set,
# metric IDs are stable across re-computation of the same file, so stored agent runs and
# reports keep resolving. Unset (direct calls in tests) → random IDs.
_SCOPE: ContextVar[str] = ContextVar("aeb_metric_scope", default="")


def _metric_id(name: str) -> str:
    scope = _SCOPE.get()
    return stable_id("metric", scope, name) if scope else new_id("metric")


def evidence_scope(file_id: str, thresholds: AebThresholds) -> str:
    digest = hashlib.sha256(thresholds.model_dump_json().encode("utf-8")).hexdigest()[:8]
    return f"{file_id}:{digest}"


def _missing(name: str, unit: str, window_id: str | None, reason: str, method: str) -> MetricResult:
    return MetricResult(
        metric_id=_metric_id(name),
        name=name,
        value=None,
        unit=unit,
        window_id=window_id,
        method=method,
        missing_reason=reason,
    )


def _result(
    name: str,
    value: float | bool,
    unit: str,
    window_id: str | None,
    method: str,
    *,
    t_s: float | None = None,
    threshold: float | bool | None = None,
    comparator: Comparator | None = None,
    details: dict[str, float | int | str | bool | None] | None = None,
) -> MetricResult:
    passed: bool | None = None
    if threshold is not None and comparator is not None:
        if comparator is Comparator.LE:
            passed = bool(value <= threshold)
        elif comparator is Comparator.GE:
            passed = bool(value >= threshold)
        else:
            passed = bool(value == threshold)
    return MetricResult(
        metric_id=_metric_id(name),
        name=name,
        value=value,
        unit=unit,
        t_s=t_s,
        threshold=threshold,
        comparator=comparator,
        passed=passed,
        window_id=window_id,
        method=method,
        details=details or {},
    )


def _need(frame: pd.DataFrame, *cols: str) -> str | None:
    missing = [c for c in cols if c not in frame.columns]
    return f"signal(s) missing: {', '.join(missing)}" if missing else None


def compute_metrics_in_frame(
    frame: pd.DataFrame,
    thresholds: AebThresholds,
    window_id: str | None,
    events: tuple[Event, ...],
) -> tuple[MetricResult, ...]:
    """Compute every AEB metric over ``frame`` (already sliced to the window)."""
    out: list[MetricResult] = []
    t = frame[COL_TIMESTAMP].to_numpy(dtype="float64")
    by_type = {e.event_type: e for e in events}
    brake = by_type.get(EventType.AEB_BRAKE_COMMAND)
    crossing = by_type.get(EventType.TTC_THRESHOLD_CROSSING)
    collision = by_type.get(EventType.COLLISION)

    # --- TTC-based --------------------------------------------------------------------
    reason = _need(frame, COL_REL_DISTANCE, COL_REL_VELOCITY)
    ttc_method = (
        "relative_distance / (-relative_velocity) per sample, undefined when closing speed "
        f"<= {thresholds.closing_speed_floor_mps} m/s"
    )
    if reason:
        out.append(_missing(M_MIN_TTC, "s", window_id, reason, ttc_method))
        out.append(_missing(M_TTC_CROSSING, "s", window_id, reason, ttc_method))
    else:
        ttc = ttc_series(frame, thresholds.closing_speed_floor_mps)
        if ttc.notna().any():
            i = int(np.nanargmin(ttc.to_numpy()))
            out.append(
                _result(
                    M_MIN_TTC,
                    float(ttc.iloc[i]),
                    "s",
                    window_id,
                    ttc_method,
                    t_s=float(t[i]),
                    threshold=thresholds.min_acceptable_ttc_s,
                    comparator=Comparator.GE,
                )
            )
        else:
            out.append(
                _missing(M_MIN_TTC, "s", window_id, "ego never closes on the lead", ttc_method)
            )
        if crossing is not None:
            out.append(
                _result(
                    M_TTC_CROSSING,
                    crossing.t_s,
                    "s",
                    window_id,
                    f"first sample with TTC <= {thresholds.trigger_ttc_s} s",
                    t_s=crossing.t_s,
                    details={"event_id": crossing.event_id},
                )
            )
        else:
            out.append(
                _missing(
                    M_TTC_CROSSING,
                    "s",
                    window_id,
                    f"TTC never fell to {thresholds.trigger_ttc_s} s",
                    f"first sample with TTC <= {thresholds.trigger_ttc_s} s",
                )
            )

    # --- brake command and latency ------------------------------------------------------
    reason = _need(frame, COL_BRAKE_CMD)
    if reason:
        out.append(_missing(M_BRAKE_CMD, "s", window_id, reason, "brake_command rising edge"))
    elif brake is None:
        out.append(
            _missing(
                M_BRAKE_CMD,
                "s",
                window_id,
                "brake_command never asserted",
                "brake_command rising edge",
            )
        )
    else:
        out.append(
            _result(
                M_BRAKE_CMD,
                brake.t_s,
                "s",
                window_id,
                "first sample with brake_command == 1",
                t_s=brake.t_s,
                details={"event_id": brake.event_id},
            )
        )

    latency_method = "brake_command_time_s - ttc_threshold_crossing_s"
    if brake is not None and crossing is not None:
        out.append(
            _result(
                M_BRAKING_LATENCY,
                round(brake.t_s - crossing.t_s, 6),
                "s",
                window_id,
                latency_method,
                t_s=brake.t_s,
                threshold=thresholds.max_braking_latency_s,
                comparator=Comparator.LE,
                details={"from_event_id": crossing.event_id, "to_event_id": brake.event_id},
            )
        )
    else:
        why = "no brake command" if brake is None else "no TTC threshold crossing"
        out.append(_missing(M_BRAKING_LATENCY, "s", window_id, why, latency_method))

    if brake is not None and COL_EGO_SPEED in frame.columns:
        row = brake.row - int(frame.index[0])
        if 0 <= row < len(frame):
            out.append(
                _result(
                    M_SPEED_AT_BRAKE,
                    float(frame[COL_EGO_SPEED].iloc[row]),
                    "m/s",
                    window_id,
                    "ego_speed at brake command sample",
                    t_s=brake.t_s,
                )
            )

    # --- gap and collision --------------------------------------------------------------
    reason = _need(frame, COL_REL_DISTANCE)
    if reason:
        out.append(_missing(M_MIN_GAP, "m", window_id, reason, "min relative_distance"))
        out.append(_missing(M_COLLISION, "bool", window_id, reason, "relative_distance <= gap"))
    else:
        gap = frame[COL_REL_DISTANCE].to_numpy(dtype="float64")
        i = int(np.nanargmin(gap))
        out.append(
            _result(
                M_MIN_GAP, float(gap[i]), "m", window_id, "min relative_distance", t_s=float(t[i])
            )
        )
        out.append(
            _result(
                M_COLLISION,
                collision is not None,
                "bool",
                window_id,
                f"any sample with relative_distance <= {thresholds.collision_gap_m} m",
                t_s=collision.t_s if collision else None,
                threshold=False,
                comparator=Comparator.EQ,
                details={"event_id": collision.event_id if collision else None},
            )
        )
        if collision is not None:
            out.append(
                _result(
                    M_COLLISION_TIME,
                    collision.t_s,
                    "s",
                    window_id,
                    "first collision sample",
                    t_s=collision.t_s,
                )
            )

    # --- deceleration and jerk ------------------------------------------------------------
    if COL_EGO_ACCEL in frame.columns and frame[COL_EGO_ACCEL].notna().any():
        acc = frame[COL_EGO_ACCEL].to_numpy(dtype="float64")
        acc_method = "logged ego_acceleration"
    elif COL_EGO_SPEED in frame.columns and len(frame) > 1:
        k0 = thresholds.jerk_smoothing_samples
        speed = frame[COL_EGO_SPEED].rolling(k0, center=True, min_periods=1).mean()
        acc = np.gradient(speed.to_numpy(dtype="float64"), t)
        acc_method = (
            f"d(ego_speed)/dt with {k0}-sample centred moving average "
            "(acceleration signal not logged)"
        )
    else:
        acc = None
        acc_method = "n/a"
    if acc is None:
        out.append(
            _missing(M_MAX_DECEL, "m/s^2", window_id, "no acceleration or speed", acc_method)
        )
        out.append(_missing(M_MAX_JERK, "m/s^3", window_id, "no acceleration or speed", acc_method))
    else:
        i = int(np.nanargmin(acc))
        out.append(
            _result(
                M_MAX_DECEL,
                float(-acc[i]),
                "m/s^2",
                window_id,
                f"max(-a) from {acc_method}",
                t_s=float(t[i]),
                threshold=thresholds.max_deceleration_mps2,
                comparator=Comparator.LE,
            )
        )
        k = thresholds.jerk_smoothing_samples
        smooth = pd.Series(acc).rolling(k, center=True, min_periods=1).mean().to_numpy()
        jerk = np.gradient(smooth, t) if len(t) > 1 else np.zeros_like(smooth)
        # Comfort jerk is only meaningful while the vehicle moves: the deceleration
        # collapsing to zero at standstill is not a ride-comfort event.
        floor = thresholds.closing_speed_floor_mps
        moving = (
            frame[COL_EGO_SPEED].to_numpy(dtype="float64") > floor
            if COL_EGO_SPEED in frame.columns
            else np.ones_like(jerk, dtype=bool)
        )
        jerk_method = (
            f"max |d(a)/dt| with {k}-sample centred moving average on {acc_method}, "
            f"while ego_speed > {floor} m/s"
        )
        if moving.any():
            masked = np.where(moving, np.abs(jerk), np.nan)
            j = int(np.nanargmax(masked))
            out.append(
                _result(
                    M_MAX_JERK,
                    float(masked[j]),
                    "m/s^3",
                    window_id,
                    jerk_method,
                    t_s=float(t[j]),
                    threshold=thresholds.max_comfort_jerk_mps3,
                    comparator=Comparator.LE,
                )
            )
        else:
            out.append(_missing(M_MAX_JERK, "m/s^3", window_id, "ego never moving", jerk_method))

    # --- perception -------------------------------------------------------------------
    reason = _need(frame, COL_OBJECT_CONF)
    conf_thr = thresholds.detection_confidence_threshold
    if reason:
        out.append(_missing(M_FIRST_DETECTION, "s", window_id, reason, "confidence >= threshold"))
        out.append(_missing(M_CONF_DROPOUT, "s", window_id, reason, "confidence < threshold"))
    else:
        conf = frame[COL_OBJECT_CONF].to_numpy(dtype="float64")
        detected = np.flatnonzero(conf >= conf_thr)
        if detected.size:
            out.append(
                _result(
                    M_FIRST_DETECTION,
                    float(t[detected[0]]),
                    "s",
                    window_id,
                    f"first sample with object_confidence >= {conf_thr}",
                    t_s=float(t[detected[0]]),
                )
            )
        else:
            out.append(
                _missing(
                    M_FIRST_DETECTION,
                    "s",
                    window_id,
                    f"object_confidence never reached {conf_thr}",
                    f"first sample with object_confidence >= {conf_thr}",
                )
            )
        # Confidence during the risk phase: from TTC crossing to brake command (or to the
        # end of the window when AEB never braked). This is the evidence for "perception
        # dropout delayed the trigger".
        if crossing is not None:
            start = crossing.t_s
            end = brake.t_s if brake is not None else float(t[-1])
            in_risk = (t >= start - 1e-9) & (t <= end + 1e-9)
            if in_risk.any():
                dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
                below = (conf < conf_thr) & in_risk
                dropout_s = float(below.sum()) * dt
                risk_conf = conf[in_risk]
                out.append(
                    _result(
                        M_CONF_DROPOUT,
                        round(dropout_s, 6),
                        "s",
                        window_id,
                        f"samples with object_confidence < {conf_thr} between TTC crossing "
                        "and brake command, times nominal dt",
                        t_s=float(t[np.flatnonzero(below)[0]]) if below.any() else None,
                        threshold=0.0,
                        comparator=Comparator.LE,
                        details={
                            "risk_start_s": start,
                            "risk_end_s": end,
                            "below_threshold_samples": int(below.sum()),
                        },
                    )
                )
                out.append(
                    _result(
                        M_MIN_CONF_IN_RISK,
                        float(np.nanmin(risk_conf)),
                        "ratio",
                        window_id,
                        "min object_confidence between TTC crossing and brake command",
                        t_s=float(t[in_risk][int(np.nanargmin(risk_conf))]),
                        threshold=conf_thr,
                        comparator=Comparator.GE,
                    )
                )
        else:
            out.append(
                _missing(
                    M_CONF_DROPOUT,
                    "s",
                    window_id,
                    "no TTC threshold crossing",
                    "confidence < threshold",
                )
            )

    return tuple(out)


def compute_aeb_metrics(
    telemetry: IngestedTelemetry,
    quality: QualityReport,
    thresholds: AebThresholds | None = None,
) -> AebMetricsReport:
    """Gate-enforced entry point: raises DataQualityError when the quality verdict is BLOCKED."""
    require_analyzable(quality)
    thresholds = thresholds or AebThresholds()
    frame = telemetry.frame

    scope = evidence_scope(telemetry.provenance.file_id, thresholds)
    events = detect_events(frame, thresholds, scope=scope)
    windows: tuple[EventWindow, ...] = tuple(
        build_window(frame, e, thresholds, scope=scope) for e in events
    )
    anchor = primary_event(events)
    primary: EventWindow | None = None
    if anchor is not None:
        primary = next(w for w in windows if w.event_id == anchor.event_id)
        analysed = slice_window(frame, primary)
        # Only events inside the window are attributed to it.
        in_scope = tuple(e for e in events if primary.start_row <= e.row <= primary.end_row)
    else:
        analysed, in_scope = frame, events

    token = _SCOPE.set(scope)
    try:
        metrics = compute_metrics_in_frame(
            analysed, thresholds, primary.window_id if primary else None, in_scope
        )
    finally:
        _SCOPE.reset(token)
    return AebMetricsReport(
        file_id=telemetry.provenance.file_id,
        quality_id=quality.quality_id,
        thresholds=thresholds,
        events=events,
        windows=windows,
        primary_window_id=primary.window_id if primary else None,
        metrics=metrics,
    )


def fmt_value(m: MetricResult) -> str:
    if m.value is None:
        return "—"
    if isinstance(m.value, bool):
        return "yes" if m.value else "no"
    if math.isinf(m.value):
        return "inf"
    return f"{m.value:.3f}"
