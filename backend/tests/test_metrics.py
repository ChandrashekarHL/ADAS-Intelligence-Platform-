"""M3 AEB metrics: validated against generator ground truth; windows, events, missing data."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.core.errors import DataQualityError
from app.core.signals import (
    COL_BRAKE_CMD,
    COL_EGO_ACCEL,
    COL_OBJECT_CONF,
    COL_REL_DISTANCE,
    COL_REL_VELOCITY,
    COL_TIMESTAMP,
)
from app.ingestion.csv_loader import ingest_frame
from app.ingestion.schemas import IngestedTelemetry
from app.metrics.aeb import (
    M_BRAKE_CMD,
    M_BRAKING_LATENCY,
    M_COLLISION,
    M_COLLISION_TIME,
    M_CONF_DROPOUT,
    M_FIRST_DETECTION,
    M_MAX_DECEL,
    M_MAX_JERK,
    M_MIN_CONF_IN_RISK,
    M_MIN_GAP,
    M_MIN_TTC,
    M_SPEED_AT_BRAKE,
    M_TTC_CROSSING,
    compute_aeb_metrics,
)
from app.metrics.cli import EXIT_BLOCKED
from app.metrics.cli import main as cli_main
from app.metrics.schemas import AebMetricsReport, AebThresholds, EventType
from app.metrics.windows import build_window, detect_events, primary_event, ttc_series
from app.quality.report import QualityReport, evaluate_gates
from app.synthetic.aeb_generator import SyntheticScenario, apply_faults, generate_aeb_scenario
from app.synthetic.io import CSV_NAME, write_scenario
from app.synthetic.schemas import AebScenarioConfig, FaultInjection, ScenarioVariant

THR = AebThresholds()
DT = 0.02


def pipeline(frame: pd.DataFrame) -> tuple[IngestedTelemetry, QualityReport]:
    tel = ingest_frame(frame, source_path="memory", data_origin="synthetic")
    return tel, evaluate_gates(tel)


def run(frame: pd.DataFrame, thresholds: AebThresholds = THR) -> AebMetricsReport:
    tel, q = pipeline(frame)
    return compute_aeb_metrics(tel, q, thresholds)


@pytest.fixture(scope="module")
def nominal() -> SyntheticScenario:
    return generate_aeb_scenario(AebScenarioConfig(variant=ScenarioVariant.NOMINAL, seed=11))


@pytest.fixture(scope="module")
def late() -> SyntheticScenario:
    return generate_aeb_scenario(AebScenarioConfig(variant=ScenarioVariant.LATE_BRAKING, seed=11))


# --- against ground truth ---------------------------------------------------------------


@pytest.mark.parametrize("which", ["nominal", "late"])
def test_metrics_match_ground_truth(
    which: str, nominal: SyntheticScenario, late: SyntheticScenario
) -> None:
    s = nominal if which == "nominal" else late
    gt = s.ground_truth
    r = run(s.frame)

    # Event times are read off the logged signals; noise may shift TTC crossing by a sample.
    assert r.metric(M_BRAKE_CMD).value == pytest.approx(gt.brake_command_s)
    assert r.metric(M_TTC_CROSSING).value == pytest.approx(gt.risk_threshold_crossing_s, abs=2 * DT)
    assert r.metric(M_BRAKING_LATENCY).value == pytest.approx(gt.braking_latency_s, abs=2 * DT)
    assert r.metric(M_MAX_DECEL).value == pytest.approx(gt.max_deceleration_mps2, abs=0.3)
    assert r.metric(M_COLLISION).value is gt.collision
    assert r.metric(M_MIN_GAP).value == pytest.approx(gt.min_gap_m, abs=0.2)
    min_ttc = r.metric(M_MIN_TTC).value
    assert isinstance(min_ttc, float) and gt.min_ttc_s is not None
    assert min_ttc == pytest.approx(gt.min_ttc_s, abs=0.1)
    if gt.collision:
        assert r.metric(M_COLLISION_TIME).value == pytest.approx(gt.collision_time_s, abs=3 * DT)
    else:
        with pytest.raises(KeyError):
            r.metric(M_COLLISION_TIME)


def test_nominal_passes_thresholds_and_late_fails(
    nominal: SyntheticScenario, late: SyntheticScenario
) -> None:
    n, lt = run(nominal.frame), run(late.frame)
    for name in (M_BRAKING_LATENCY, M_MIN_TTC, M_COLLISION, M_CONF_DROPOUT, M_MIN_CONF_IN_RISK):
        assert n.metric(name).passed is True, name
        assert lt.metric(name).passed is False, name
    assert n.metric(M_MAX_DECEL).passed is True
    assert n.metric(M_MAX_JERK).passed is True  # controller jerk limit 25 < 30 m/s³
    # the diagnostic signature of the late case
    dropout = lt.metric(M_CONF_DROPOUT)
    assert isinstance(dropout.value, float)
    assert dropout.value == pytest.approx(0.64, abs=3 * DT)  # 5.4 - 4.76
    assert dropout.details["risk_start_s"] == pytest.approx(4.76, abs=2 * DT)
    assert dropout.details["risk_end_s"] == pytest.approx(5.56)
    assert dropout.t_s == pytest.approx(4.76, abs=2 * DT)
    min_conf = lt.metric(M_MIN_CONF_IN_RISK).value
    assert isinstance(min_conf, float) and min_conf < 0.3
    assert n.metric(M_CONF_DROPOUT).value == 0.0


def test_every_metric_is_evidence(nominal: SyntheticScenario) -> None:
    r = run(nominal.frame)
    assert r.file_id.startswith("file_") and r.quality_id.startswith("quality_")
    assert r.primary_window_id is not None and r.primary_window_id.startswith("window_")
    for m in r.metrics:
        assert m.metric_id.startswith("metric_")
        assert m.window_id == r.primary_window_id
        assert m.method
        assert (m.value is None) == (m.missing_reason is not None)
    ids = r.evidence_ids
    assert len(ids) == len(set(ids))
    assert all(i.split("_")[0] in {"event", "window", "metric"} for i in ids)
    assert r.metric(M_SPEED_AT_BRAKE).value == pytest.approx(50 / 3.6, abs=0.2)
    assert r.metric(M_FIRST_DETECTION).value == pytest.approx(0.0)


def test_report_deterministic_apart_from_ids(nominal: SyntheticScenario) -> None:
    a, b = run(nominal.frame), run(nominal.frame)
    strip = {"metric_id", "window_id", "details"}
    assert [m.model_dump(exclude=strip) for m in a.metrics] == [
        m.model_dump(exclude=strip) for m in b.metrics
    ]
    for ma, mb in zip(a.metrics, b.metrics, strict=True):
        assert ma.details.keys() == mb.details.keys()
        for key, va in ma.details.items():
            if not key.endswith("event_id"):
                assert va == mb.details[key], (ma.name, key)
    assert a.metrics[0].metric_id != b.metrics[0].metric_id


# --- events and windows -----------------------------------------------------------------


def test_events_and_windows(late: SyntheticScenario) -> None:
    events = detect_events(late.frame, THR)
    types = [e.event_type for e in events]
    assert types == [
        EventType.TTC_THRESHOLD_CROSSING,
        EventType.AEB_BRAKE_COMMAND,
        EventType.COLLISION,
    ]
    assert [e.t_s for e in events] == sorted(e.t_s for e in events)
    assert all(e.event_id.startswith("event_") for e in events)
    anchor = primary_event(events)
    assert anchor is not None and anchor.event_type is EventType.AEB_BRAKE_COMMAND

    w = build_window(late.frame, anchor, THR)
    assert w.t_event_s == anchor.t_s
    assert w.start_s == pytest.approx(max(0.0, anchor.t_s - 5.0), abs=DT)
    assert w.end_s == pytest.approx(min(9.98, anchor.t_s + 5.0), abs=DT)
    assert w.sample_count == w.end_row - w.start_row + 1
    assert w.clipped_end is True  # 5.56 + 5 > 9.98
    assert w.clipped_start is False

    # a long log yields an unclipped window of exactly 10 s + 1 sample
    cfg = AebScenarioConfig(duration_s=30.0, lead_brake_onset_s=12.0, seed=1)
    long = generate_aeb_scenario(cfg).frame
    ev = primary_event(detect_events(long, THR))
    assert ev is not None
    w2 = build_window(long, ev, THR)
    assert not (w2.clipped_start or w2.clipped_end)
    assert w2.sample_count == int(10.0 / DT) + 1


def test_primary_falls_back_to_ttc_crossing_when_aeb_never_brakes(
    nominal: SyntheticScenario,
) -> None:
    frame = nominal.frame.copy()
    frame[COL_BRAKE_CMD] = 0  # AEB failed to act at all
    tel = ingest_frame(frame, source_path="memory")
    q = evaluate_gates(tel)  # scenario_completeness warns, still analyzable
    r = compute_aeb_metrics(tel, q, THR)
    assert [e.event_type for e in r.events][0] is EventType.TTC_THRESHOLD_CROSSING
    anchor_window = next(w for w in r.windows if w.window_id == r.primary_window_id)
    assert anchor_window.t_event_s == pytest.approx(4.76, abs=2 * DT)
    assert r.metric(M_BRAKE_CMD).missing_reason == "brake_command never asserted"
    assert r.metric(M_BRAKING_LATENCY).missing_reason == "no brake command"
    # with no brake command the risk phase runs to the end of the window
    dropout = r.metric(M_CONF_DROPOUT)
    assert dropout.details["risk_end_s"] == pytest.approx(anchor_window.end_s)
    # only the command signal was zeroed; the kinematics still show a stop without contact
    assert r.metric(M_COLLISION).value is False


def test_no_event_log_reports_missing_not_invented() -> None:
    cfg = AebScenarioConfig(lead_decel_mps2=1e-9, lead_brake_onset_s=9.9, duration_s=10.0)
    r = run(generate_aeb_scenario(cfg).frame)
    assert r.events == () and r.windows == () and r.primary_window_id is None
    assert r.metric(M_MIN_TTC).missing_reason == "ego never closes on the lead"
    assert r.metric(M_TTC_CROSSING).missing_reason == "TTC never fell to 2.0 s"
    assert r.metric(M_BRAKE_CMD).missing_reason == "brake_command never asserted"
    assert r.metric(M_CONF_DROPOUT).missing_reason == "no TTC threshold crossing"
    assert r.metric(M_COLLISION).value is False
    assert all(m.window_id is None for m in r.metrics)


def test_ttc_series_definition() -> None:
    frame = pd.DataFrame(
        {
            COL_REL_DISTANCE: [20.0, 20.0, 20.0, 0.0],
            COL_REL_VELOCITY: [-10.0, 0.0, 5.0, -10.0],  # closing, static, opening, contact
        }
    )
    ttc = ttc_series(frame, 0.5)
    assert ttc.iloc[0] == pytest.approx(2.0)
    assert math.isnan(ttc.iloc[1]) and math.isnan(ttc.iloc[2])
    assert ttc.iloc[3] == 0.0


# --- gates and missing signals ----------------------------------------------------------


def test_blocked_quality_prevents_metrics(nominal: SyntheticScenario) -> None:
    frame = apply_faults(nominal.frame, FaultInjection(drop_columns=(COL_OBJECT_CONF,)))
    tel, q = pipeline(frame)
    with pytest.raises(DataQualityError):
        compute_aeb_metrics(tel, q, THR)


def test_optional_signal_missing_uses_fallback_and_says_so(nominal: SyntheticScenario) -> None:
    frame = apply_faults(nominal.frame, FaultInjection(drop_columns=(COL_EGO_ACCEL,)))
    r = run(frame)  # DEGRADED, still analyzable
    decel = r.metric(M_MAX_DECEL)
    assert decel.available
    assert "d(ego_speed)/dt" in decel.method and "not logged" in decel.method
    assert decel.value == pytest.approx(nominal.ground_truth.max_deceleration_mps2, abs=1.5)


def test_thresholds_change_pass_fail_not_values(late: SyntheticScenario) -> None:
    strict = run(late.frame, AebThresholds(max_braking_latency_s=0.3))
    lenient = run(late.frame, AebThresholds(max_braking_latency_s=1.0))
    assert strict.metric(M_BRAKING_LATENCY).value == lenient.metric(M_BRAKING_LATENCY).value
    assert strict.metric(M_BRAKING_LATENCY).passed is False
    assert lenient.metric(M_BRAKING_LATENCY).passed is True
    # a different trigger threshold moves the crossing event and therefore the latency
    early = run(late.frame, AebThresholds(trigger_ttc_s=2.5))
    v_early, v_strict = (
        early.metric(M_BRAKING_LATENCY).value,
        strict.metric(M_BRAKING_LATENCY).value,
    )
    assert isinstance(v_early, float) and isinstance(v_strict, float) and v_early > v_strict


def test_timestamps_of_values_lie_inside_window(late: SyntheticScenario) -> None:
    r = run(late.frame)
    w = next(w for w in r.windows if w.window_id == r.primary_window_id)
    for m in r.metrics:
        if m.t_s is not None:
            assert w.start_s - 1e-9 <= m.t_s <= w.end_s + 1e-9, m.name
    t = late.frame[COL_TIMESTAMP].to_numpy()
    assert np.all(np.diff(t) > 0)


# --- CLI --------------------------------------------------------------------------------


def test_cli(tmp_path: Path, late: SyntheticScenario, capsys: pytest.CaptureFixture[str]) -> None:
    write_scenario(late, tmp_path)
    assert cli_main([str(tmp_path / CSV_NAME)]) == 0
    out = capsys.readouterr().out
    assert "braking_latency_s" in out and "FAIL" in out and "collision" in out
    assert cli_main([str(tmp_path / CSV_NAME), "--json", "--max-latency-s", "1.0"]) == 0
    assert '"name": "braking_latency_s"' in capsys.readouterr().out


def test_cli_blocked(
    tmp_path: Path, late: SyntheticScenario, capsys: pytest.CaptureFixture[str]
) -> None:
    write_scenario(late, tmp_path)
    csv = tmp_path / CSV_NAME
    pd.read_csv(csv).drop(columns=["brake_command"]).to_csv(csv, index=False)
    assert cli_main([str(csv)]) == EXIT_BLOCKED
    assert "BLOCKED" in capsys.readouterr().err


def test_evidence_ids_are_stable_for_the_same_file_and_thresholds(
    late: SyntheticScenario,
) -> None:
    """Same file_id + same thresholds → identical metric/event/window IDs on re-computation,
    so stored agent runs and reports keep resolving. Different thresholds → different IDs."""
    tel = ingest_frame(late.frame, source_path="memory", file_id="file_fixed0000001")
    q = evaluate_gates(tel)
    a = compute_aeb_metrics(tel, q, THR)
    b = compute_aeb_metrics(tel, q, THR)
    assert [m.metric_id for m in a.metrics] == [m.metric_id for m in b.metrics]
    assert [e.event_id for e in a.events] == [e.event_id for e in b.events]
    assert [w.window_id for w in a.windows] == [w.window_id for w in b.windows]
    assert a.quality_id == b.quality_id
    c = compute_aeb_metrics(tel, q, AebThresholds(trigger_ttc_s=2.5))
    assert c.metric(M_BRAKING_LATENCY).metric_id != a.metric(M_BRAKING_LATENCY).metric_id
    other = ingest_frame(late.frame, source_path="memory", file_id="file_fixed0000002")
    d = compute_aeb_metrics(other, evaluate_gates(other), THR)
    assert d.metric(M_BRAKING_LATENCY).metric_id != a.metric(M_BRAKING_LATENCY).metric_id
