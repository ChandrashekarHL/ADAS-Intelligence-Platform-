"""M2 data-quality gates: each gate on clean and known-bad frames, verdict reduction."""

import pandas as pd
import pytest

from app.core.errors import DataQualityError
from app.core.signals import (
    COL_BRAKE_CMD,
    COL_EGO_SPEED,
    COL_OBJECT_CONF,
    COL_REL_DISTANCE,
    COL_TIMESTAMP,
    COL_WEATHER,
)
from app.ingestion.csv_loader import ingest_frame
from app.ingestion.schemas import IngestedTelemetry
from app.quality.gates import (
    ALL_GATES,
    GateStatus,
    QualityPolicy,
    gate_missing_values,
    gate_required_signals,
    gate_scenario_completeness,
    gate_timestamps,
    gate_unit_plausibility,
    gate_value_ranges,
)
from app.quality.report import QualityVerdict, evaluate_gates, require_analyzable
from app.synthetic.aeb_generator import apply_faults, generate_aeb_scenario
from app.synthetic.schemas import (
    AebScenarioConfig,
    FaultInjection,
    NanBurst,
    ScenarioVariant,
    TimestampGap,
)

POLICY = QualityPolicy()


@pytest.fixture(scope="module")
def clean() -> pd.DataFrame:
    return generate_aeb_scenario(AebScenarioConfig(variant=ScenarioVariant.NOMINAL, seed=5)).frame


def ingested(frame: pd.DataFrame) -> IngestedTelemetry:
    return ingest_frame(frame, source_path="memory", data_origin="synthetic")


def status_of(report_gates: tuple[object, ...], name: str) -> GateStatus:
    for g in report_gates:
        if g.gate == name:
            return GateStatus(g.status)
    raise KeyError(name)


# --- whole-report behaviour -------------------------------------------------------------


def test_clean_frame_passes_every_gate(clean: pd.DataFrame) -> None:
    report = evaluate_gates(ingested(clean))
    assert report.verdict is QualityVerdict.PASS
    assert report.analyzable
    assert report.quality_id.startswith("quality_")
    assert report.file_id.startswith("file_")
    assert len(report.gates) == len(ALL_GATES)
    assert all(g.status is GateStatus.PASS for g in report.gates), [
        (g.gate, g.message) for g in report.gates if g.status is not GateStatus.PASS
    ]
    require_analyzable(report)  # does not raise


def test_report_is_deterministic_apart_from_id(clean: pd.DataFrame) -> None:
    tel = ingested(clean)
    a, b = evaluate_gates(tel), evaluate_gates(tel)
    assert a.gates == b.gates
    assert a.quality_id != b.quality_id


def test_blocked_report_raises_with_reasons(clean: pd.DataFrame) -> None:
    frame = apply_faults(clean, FaultInjection(drop_columns=(COL_OBJECT_CONF,)))
    report = evaluate_gates(ingested(frame))
    assert report.verdict is QualityVerdict.BLOCKED
    assert not report.analyzable
    assert [g.gate for g in report.failed] == ["required_signals"]
    with pytest.raises(DataQualityError, match="required_signals.*object_confidence"):
        require_analyzable(report)


def test_optional_signal_missing_only_degrades(clean: pd.DataFrame) -> None:
    frame = apply_faults(clean, FaultInjection(drop_columns=(COL_WEATHER,)))
    report = evaluate_gates(ingested(frame))
    assert report.verdict is QualityVerdict.DEGRADED
    assert report.analyzable
    assert [g.gate for g in report.warned] == ["required_signals"]
    require_analyzable(report)


# --- individual gates -------------------------------------------------------------------


def test_required_signals_details(clean: pd.DataFrame) -> None:
    r = gate_required_signals(clean.drop(columns=[COL_BRAKE_CMD, COL_WEATHER]), POLICY)
    assert r.status is GateStatus.FAIL
    assert r.details["missing_critical"] == [COL_BRAKE_CMD]
    assert r.details["missing_optional"] == [COL_WEATHER]


def test_small_timestamp_gap_warns_with_exact_interval(clean: pd.DataFrame) -> None:
    frame = apply_faults(
        clean, FaultInjection(timestamp_gap=TimestampGap(start_s=2.0, duration_s=0.5))
    )
    r = gate_timestamps(frame, POLICY)
    assert r.status is GateStatus.WARN
    assert r.details["sample_rate_hz"] == pytest.approx(50.0)
    (gap,) = r.details["gaps"]
    assert gap["start_s"] == pytest.approx(1.98)
    assert gap["end_s"] == pytest.approx(2.5)
    assert gap["missing_samples"] == 25
    assert r.details["missing_time_s"] == pytest.approx(0.5)
    assert evaluate_gates(ingested(frame)).verdict is QualityVerdict.DEGRADED


def test_large_timestamp_gap_blocks(clean: pd.DataFrame) -> None:
    frame = apply_faults(
        clean, FaultInjection(timestamp_gap=TimestampGap(start_s=2.0, duration_s=4.0))
    )
    r = gate_timestamps(frame, POLICY)
    assert r.status is GateStatus.FAIL
    assert r.details["missing_time_fraction"] == pytest.approx(0.4, abs=0.01)
    assert evaluate_gates(ingested(frame)).verdict is QualityVerdict.BLOCKED


def test_non_monotonic_timestamps_block(clean: pd.DataFrame) -> None:
    frame = clean.copy()
    frame.loc[[100, 101], COL_TIMESTAMP] = frame.loc[[101, 100], COL_TIMESTAMP].to_numpy()
    r = gate_timestamps(frame, POLICY)
    assert r.status is GateStatus.FAIL
    assert r.details["non_increasing"] == 1
    assert r.details["first_rows"] == [101]

    dup = clean.copy()
    dup.loc[200, COL_TIMESTAMP] = dup.loc[199, COL_TIMESTAMP]
    assert gate_timestamps(dup, POLICY).status is GateStatus.FAIL


def test_nan_timestamps_block(clean: pd.DataFrame) -> None:
    frame = clean.copy()
    frame.loc[10, COL_TIMESTAMP] = float("nan")
    assert gate_timestamps(frame, POLICY).status is GateStatus.FAIL


def test_short_nan_burst_warns_with_run(clean: pd.DataFrame) -> None:
    frame = apply_faults(
        clean,
        FaultInjection(nan_burst=NanBurst(column=COL_REL_DISTANCE, start_s=5.0, duration_s=0.2)),
    )
    r = gate_missing_values(frame, POLICY)
    assert r.status is GateStatus.WARN
    info = r.details["per_signal"][COL_REL_DISTANCE]
    assert info["count"] == 10
    assert info["fraction"] == pytest.approx(0.02)
    (run,) = info["runs"]
    assert run == {"start_s": pytest.approx(5.0), "end_s": pytest.approx(5.18), "samples": 10}
    assert evaluate_gates(ingested(frame)).verdict is QualityVerdict.DEGRADED


def test_long_nan_burst_in_critical_signal_blocks(clean: pd.DataFrame) -> None:
    frame = apply_faults(
        clean,
        FaultInjection(nan_burst=NanBurst(column=COL_EGO_SPEED, start_s=1.0, duration_s=1.0)),
    )
    r = gate_missing_values(frame, POLICY)
    assert r.status is GateStatus.FAIL
    assert r.details["worst_critical_fraction"] == pytest.approx(0.10)
    assert evaluate_gates(ingested(frame)).verdict is QualityVerdict.BLOCKED


def test_nan_in_optional_signal_only_warns(clean: pd.DataFrame) -> None:
    frame = apply_faults(
        clean,
        FaultInjection(
            nan_burst=NanBurst(column="ego_acceleration_mps2", start_s=1.0, duration_s=3.0)
        ),
    )
    assert gate_missing_values(frame, POLICY).status is GateStatus.WARN


def test_value_ranges(clean: pd.DataFrame) -> None:
    assert gate_value_ranges(clean, POLICY).status is GateStatus.PASS
    bad = clean.copy()
    bad.loc[5, COL_BRAKE_CMD] = 2
    bad.loc[6, COL_OBJECT_CONF] = 1.5
    bad.loc[7, COL_EGO_SPEED] = -1.0
    r = gate_value_ranges(bad, POLICY)
    assert r.status is GateStatus.FAIL
    assert r.details["violations"] == {COL_EGO_SPEED: 1, COL_OBJECT_CONF: 1, COL_BRAKE_CMD: 1}


def test_unconverted_kmh_is_caught_by_plausibility(clean: pd.DataFrame) -> None:
    # Simulate an operator who mislabelled km/h values as m/s: 50 km/h → "50 m/s" is fine,
    # so push harder with a 400 km/h log that clearly cannot be m/s.
    frame = clean.copy()
    frame[COL_EGO_SPEED] = frame[COL_EGO_SPEED] * 3.6 * 8  # ≈ 400 "m/s"
    r = gate_unit_plausibility(frame, POLICY)
    assert r.status is GateStatus.FAIL
    assert "check units" in r.message
    assert r.details[COL_EGO_SPEED]["peak_abs"] > POLICY.max_plausible_speed_mps
    assert evaluate_gates(ingested(frame)).verdict is QualityVerdict.BLOCKED


def test_plausibility_soft_ceilings_only_warn(clean: pd.DataFrame) -> None:
    frame = clean.copy()
    frame[COL_REL_DISTANCE] = frame[COL_REL_DISTANCE] + 1000.0
    assert gate_unit_plausibility(frame, POLICY).status is GateStatus.WARN


def test_scenario_completeness_warns_without_brake_event() -> None:
    cfg = AebScenarioConfig(lead_decel_mps2=1e-9, lead_brake_onset_s=9.9, duration_s=10.0)
    frame = generate_aeb_scenario(cfg).frame
    r = gate_scenario_completeness(frame, POLICY)
    assert r.status is GateStatus.WARN
    assert r.details["brake_samples"] == 0
    assert "no brake command" in r.message
    report = evaluate_gates(ingested(frame))
    assert report.verdict is QualityVerdict.DEGRADED
    assert status_of(report.gates, "scenario_completeness") is GateStatus.WARN


def test_size_gate_blocks_tiny_logs(clean: pd.DataFrame) -> None:
    report = evaluate_gates(ingested(clean.head(5)))
    assert report.verdict is QualityVerdict.BLOCKED
    assert status_of(report.gates, "size") is GateStatus.FAIL


def test_policy_is_tunable(clean: pd.DataFrame) -> None:
    frame = apply_faults(
        clean, FaultInjection(timestamp_gap=TimestampGap(start_s=2.0, duration_s=0.5))
    )
    strict = QualityPolicy(max_missing_time_fraction=0.01)
    assert evaluate_gates(ingested(frame), strict).verdict is QualityVerdict.BLOCKED
    lenient = QualityPolicy(gap_factor=100.0)
    assert evaluate_gates(ingested(frame), lenient).verdict is QualityVerdict.PASS
