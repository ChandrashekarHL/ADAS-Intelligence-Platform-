"""Data-quality gates (spec §28). Pure functions over an SI frame.

Each gate returns a :class:`GateResult` with PASS, WARN or FAIL. FAIL means the frame is
not analysable for AEB diagnostics; WARN means analysis may proceed with reduced
confidence. Gates never modify the frame and never hide what they found: the
``details`` dict carries the exact rows, intervals and counts for the report.
"""

from collections.abc import Callable
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from app.core.signals import (
    COL_BRAKE_CMD,
    COL_EGO_ACCEL,
    COL_EGO_SPEED,
    COL_OBJECT_CONF,
    COL_REL_DISTANCE,
    COL_REL_VELOCITY,
    COL_TIMESTAMP,
    CRITICAL_AEB_SIGNALS,
    OPTIONAL_AEB_SIGNALS,
)


class GateStatus(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class GateResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    gate: str
    status: GateStatus
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class QualityPolicy(BaseModel):
    """Thresholds for the gates. Defaults are tuned for 10–50 Hz AEB logs."""

    model_config = ConfigDict(frozen=True)

    critical_signals: tuple[str, ...] = CRITICAL_AEB_SIGNALS
    optional_signals: tuple[str, ...] = OPTIONAL_AEB_SIGNALS

    min_rows: int = Field(default=20, ge=2)
    min_duration_s: float = Field(default=2.0, gt=0)

    # A sample interval longer than this multiple of the nominal dt is a gap.
    gap_factor: float = Field(default=2.5, gt=1)
    # Gaps summing to more than this fraction of the duration block analysis.
    max_missing_time_fraction: float = Field(default=0.10, ge=0, le=1)

    # NaN fraction in a critical signal above which analysis is blocked (any NaN warns).
    max_nan_fraction: float = Field(default=0.05, ge=0, le=1)

    # Plausibility ceilings: exceeding them usually means an unconverted unit.
    max_plausible_speed_mps: float = Field(default=90.0, gt=0)  # 324 km/h
    max_plausible_distance_m: float = Field(default=500.0, gt=0)
    max_plausible_accel_mps2: float = Field(default=20.0, gt=0)


Gate = Callable[[pd.DataFrame, QualityPolicy], GateResult]


def _runs(mask: np.ndarray, t: np.ndarray) -> list[dict[str, float | int]]:
    """Contiguous True runs in ``mask`` as ``{start_s, end_s, samples}``."""
    out: list[dict[str, float | int]] = []
    if mask.size == 0:
        return out
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    for start, end in zip(edges[::2], edges[1::2], strict=True):
        out.append(
            {"start_s": float(t[start]), "end_s": float(t[end - 1]), "samples": int(end - start)}
        )
    return out


def gate_required_signals(frame: pd.DataFrame, policy: QualityPolicy) -> GateResult:
    missing_critical = [c for c in policy.critical_signals if c not in frame.columns]
    missing_optional = [c for c in policy.optional_signals if c not in frame.columns]
    details = {"missing_critical": missing_critical, "missing_optional": missing_optional}
    if missing_critical:
        return GateResult(
            gate="required_signals",
            status=GateStatus.FAIL,
            message=f"critical signals missing: {', '.join(missing_critical)}",
            details=details,
        )
    if missing_optional:
        return GateResult(
            gate="required_signals",
            status=GateStatus.WARN,
            message=f"optional signals missing: {', '.join(missing_optional)}",
            details=details,
        )
    return GateResult(
        gate="required_signals",
        status=GateStatus.PASS,
        message="all signals present",
        details=details,
    )


def gate_size(frame: pd.DataFrame, policy: QualityPolicy) -> GateResult:
    rows = int(len(frame))
    duration: float | None = None
    if COL_TIMESTAMP in frame.columns:
        t = frame[COL_TIMESTAMP].dropna()
        if len(t) >= 2:
            duration = float(t.max() - t.min())
    details: dict[str, Any] = {"rows": rows, "duration_s": duration}
    if rows < policy.min_rows:
        return GateResult(
            gate="size",
            status=GateStatus.FAIL,
            message=f"{rows} rows < minimum {policy.min_rows}",
            details=details,
        )
    if duration is not None and duration < policy.min_duration_s:
        return GateResult(
            gate="size",
            status=GateStatus.FAIL,
            message=f"duration {duration:.2f} s < minimum {policy.min_duration_s} s",
            details=details,
        )
    return GateResult(
        gate="size", status=GateStatus.PASS, message=f"{rows} rows, {duration} s", details=details
    )


def gate_timestamps(frame: pd.DataFrame, policy: QualityPolicy) -> GateResult:
    name = "timestamp_continuity"
    if COL_TIMESTAMP not in frame.columns:
        return GateResult(gate=name, status=GateStatus.FAIL, message="timestamp column missing")
    t_series = pd.to_numeric(frame[COL_TIMESTAMP], errors="coerce")
    nan_count = int(t_series.isna().sum())
    if nan_count:
        return GateResult(
            gate=name,
            status=GateStatus.FAIL,
            message=f"{nan_count} timestamps are missing or non-numeric",
            details={"nan_count": nan_count},
        )
    t = t_series.to_numpy(dtype="float64")
    if t.size < 2:
        return GateResult(gate=name, status=GateStatus.FAIL, message="fewer than two samples")
    diffs = np.diff(t)
    non_increasing = int((diffs <= 0).sum())
    if non_increasing:
        idx = np.flatnonzero(diffs <= 0)[:10] + 1
        return GateResult(
            gate=name,
            status=GateStatus.FAIL,
            message=f"{non_increasing} non-increasing timestamps (duplicates or reordering)",
            details={"non_increasing": non_increasing, "first_rows": [int(i) for i in idx]},
        )
    nominal_dt = float(np.median(diffs))
    gap_mask = diffs > policy.gap_factor * nominal_dt
    gaps = [
        {
            "start_s": float(t[i]),
            "end_s": float(t[i + 1]),
            "missing_samples": int(round(diffs[i] / nominal_dt)) - 1,
        }
        for i in np.flatnonzero(gap_mask)
    ]
    missing_time = float(sum(diffs[gap_mask] - nominal_dt))
    duration = float(t[-1] - t[0])
    fraction = missing_time / duration if duration > 0 else 0.0
    details: dict[str, Any] = {
        "nominal_dt_s": nominal_dt,
        "sample_rate_hz": 1.0 / nominal_dt if nominal_dt > 0 else None,
        "gaps": gaps,
        "missing_time_s": missing_time,
        "missing_time_fraction": fraction,
    }
    if fraction > policy.max_missing_time_fraction:
        return GateResult(
            gate=name,
            status=GateStatus.FAIL,
            message=(
                f"{len(gaps)} gap(s) covering {missing_time:.2f} s "
                f"({fraction:.0%} of log) exceed {policy.max_missing_time_fraction:.0%}"
            ),
            details=details,
        )
    if gaps:
        return GateResult(
            gate=name,
            status=GateStatus.WARN,
            message=f"{len(gaps)} timestamp gap(s) totalling {missing_time:.2f} s",
            details=details,
        )
    return GateResult(
        gate=name,
        status=GateStatus.PASS,
        message=f"continuous at {details['sample_rate_hz']:.1f} Hz",
        details=details,
    )


def gate_missing_values(frame: pd.DataFrame, policy: QualityPolicy) -> GateResult:
    name = "missing_values"
    t = (
        pd.to_numeric(frame[COL_TIMESTAMP], errors="coerce").to_numpy(dtype="float64")
        if COL_TIMESTAMP in frame.columns
        else np.arange(len(frame), dtype="float64")
    )
    per_signal: dict[str, Any] = {}
    worst_critical = 0.0
    optional_with_nan: list[str] = []
    for col in (*policy.critical_signals, *policy.optional_signals):
        if col not in frame.columns or col == COL_TIMESTAMP:
            continue
        mask = frame[col].isna().to_numpy()
        count = int(mask.sum())
        if not count:
            continue
        fraction = count / len(frame) if len(frame) else 0.0
        per_signal[col] = {"count": count, "fraction": fraction, "runs": _runs(mask, t)}
        if col in policy.critical_signals:
            worst_critical = max(worst_critical, fraction)
        else:
            optional_with_nan.append(col)
    details = {"per_signal": per_signal, "worst_critical_fraction": worst_critical}
    critical_hit = [c for c in per_signal if c in policy.critical_signals]
    if worst_critical > policy.max_nan_fraction:
        return GateResult(
            gate=name,
            status=GateStatus.FAIL,
            message=(
                f"critical signal(s) {', '.join(critical_hit)} missing "
                f"{worst_critical:.1%} of samples (> {policy.max_nan_fraction:.0%})"
            ),
            details=details,
        )
    if critical_hit or optional_with_nan:
        return GateResult(
            gate=name,
            status=GateStatus.WARN,
            message=f"missing values in: {', '.join(per_signal)}",
            details=details,
        )
    return GateResult(
        gate=name, status=GateStatus.PASS, message="no missing values", details=details
    )


def gate_value_ranges(frame: pd.DataFrame, policy: QualityPolicy) -> GateResult:
    name = "value_ranges"
    violations: dict[str, int] = {}

    def count(col: str, bad: "pd.Series[bool]") -> None:
        n = int(bad.sum())
        if n:
            violations[col] = n

    if COL_EGO_SPEED in frame.columns:
        count(COL_EGO_SPEED, frame[COL_EGO_SPEED] < 0)
    if COL_REL_DISTANCE in frame.columns:
        count(COL_REL_DISTANCE, frame[COL_REL_DISTANCE] < 0)
    if COL_OBJECT_CONF in frame.columns:
        c = frame[COL_OBJECT_CONF]
        count(COL_OBJECT_CONF, (c < 0) | (c > 1))
    if COL_BRAKE_CMD in frame.columns:
        b = frame[COL_BRAKE_CMD]
        count(COL_BRAKE_CMD, b.notna() & ~b.isin([0, 1]))
    if violations:
        return GateResult(
            gate=name,
            status=GateStatus.FAIL,
            message="out-of-range values: "
            + ", ".join(f"{c} ({n})" for c, n in violations.items()),
            details={"violations": violations},
        )
    return GateResult(gate=name, status=GateStatus.PASS, message="all values in range")


def gate_unit_plausibility(frame: pd.DataFrame, policy: QualityPolicy) -> GateResult:
    """Catch unconverted units: SI speeds above 90 m/s are almost certainly km/h."""
    name = "unit_plausibility"
    details: dict[str, Any] = {}
    fails: list[str] = []
    warns: list[str] = []
    for col, ceiling, hard in (
        (COL_EGO_SPEED, policy.max_plausible_speed_mps, True),
        (COL_REL_VELOCITY, policy.max_plausible_speed_mps, True),
        (COL_REL_DISTANCE, policy.max_plausible_distance_m, False),
        (COL_EGO_ACCEL, policy.max_plausible_accel_mps2, False),
    ):
        if col not in frame.columns:
            continue
        peak = float(frame[col].abs().max()) if frame[col].notna().any() else 0.0
        details[col] = {"peak_abs": peak, "ceiling": ceiling}
        if peak > ceiling:
            (fails if hard else warns).append(f"{col} peaks at {peak:.1f} (ceiling {ceiling})")
    if fails:
        return GateResult(
            gate=name,
            status=GateStatus.FAIL,
            message="implausible magnitudes, check units: " + "; ".join(fails),
            details=details,
        )
    if warns:
        return GateResult(
            gate=name, status=GateStatus.WARN, message="; ".join(warns), details=details
        )
    return GateResult(
        gate=name, status=GateStatus.PASS, message="magnitudes plausible for SI", details=details
    )


def gate_scenario_completeness(frame: pd.DataFrame, policy: QualityPolicy) -> GateResult:
    """Does the log actually contain an AEB-relevant episode?"""
    name = "scenario_completeness"
    notes: list[str] = []
    details: dict[str, Any] = {}
    if COL_BRAKE_CMD in frame.columns:
        brake_samples = int((frame[COL_BRAKE_CMD] == 1).sum())
        details["brake_samples"] = brake_samples
        if brake_samples == 0:
            notes.append("no brake command in log; braking-latency metrics unavailable")
    if COL_REL_VELOCITY in frame.columns:
        closing = int((frame[COL_REL_VELOCITY] < -0.5).sum())
        details["closing_samples"] = closing
        if closing == 0:
            notes.append("ego never closes on the lead; no TTC-relevant episode")
    if notes:
        return GateResult(
            gate=name, status=GateStatus.WARN, message="; ".join(notes), details=details
        )
    return GateResult(
        gate=name,
        status=GateStatus.PASS,
        message="brake event and closing phase present",
        details=details,
    )


ALL_GATES: tuple[Gate, ...] = (
    gate_required_signals,
    gate_size,
    gate_timestamps,
    gate_missing_values,
    gate_value_ranges,
    gate_unit_plausibility,
    gate_scenario_completeness,
)
