"""M1 synthetic AEB generator: determinism, physics sanity, variants, faults, export."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.core.units import KMH_TO_MPS
from app.synthetic.aeb_generator import (
    GENERATOR_VERSION,
    apply_faults,
    generate_aeb_scenario,
)
from app.synthetic.cli import main as cli_main
from app.synthetic.io import (
    CSV_EGO_SPEED_KMH,
    CSV_NAME,
    CSV_REL_VELOCITY_KMH,
    METADATA_NAME,
    read_metadata,
    write_scenario,
)
from app.synthetic.schemas import (
    COL_AEB_STATE,
    COL_BRAKE_CMD,
    COL_EGO_SPEED,
    COL_OBJECT_CONF,
    COL_REL_DISTANCE,
    COL_TIMESTAMP,
    DATA_ORIGIN_SYNTHETIC,
    FRAME_COLUMNS,
    AebScenarioConfig,
    AebState,
    FaultInjection,
    NanBurst,
    ScenarioVariant,
    TimestampGap,
)


@pytest.fixture(scope="module")
def nominal() -> AebScenarioConfig:
    return AebScenarioConfig(variant=ScenarioVariant.NOMINAL, seed=7)


@pytest.fixture(scope="module")
def late() -> AebScenarioConfig:
    return AebScenarioConfig(variant=ScenarioVariant.LATE_BRAKING, seed=7)


# --- determinism -----------------------------------------------------------------------


def test_same_seed_is_bit_identical(nominal: AebScenarioConfig) -> None:
    a = generate_aeb_scenario(nominal)
    b = generate_aeb_scenario(nominal)
    pd.testing.assert_frame_equal(a.frame, b.frame)
    assert a.ground_truth == b.ground_truth


def test_different_seed_changes_noise_not_truth(nominal: AebScenarioConfig) -> None:
    a = generate_aeb_scenario(nominal)
    b = generate_aeb_scenario(nominal.model_copy(update={"seed": 8}))
    assert not a.frame[COL_EGO_SPEED].equals(b.frame[COL_EGO_SPEED])
    assert a.ground_truth == b.ground_truth  # truth is noise-free physics


# --- frame shape ------------------------------------------------------------------------


def test_frame_columns_and_sampling(nominal: AebScenarioConfig) -> None:
    f = generate_aeb_scenario(nominal).frame
    assert tuple(f.columns) == FRAME_COLUMNS
    assert len(f) == int(nominal.duration_s * nominal.sample_rate_hz)
    dts = np.diff(f[COL_TIMESTAMP].to_numpy())
    assert np.allclose(dts, nominal.dt_s)
    assert f[COL_TIMESTAMP].iloc[0] == 0.0
    assert set(f[COL_BRAKE_CMD].unique()) <= {0, 1}
    assert set(f[COL_AEB_STATE].unique()) <= {int(s) for s in AebState}
    assert f[COL_OBJECT_CONF].between(0.0, 1.0).all()
    assert (f[COL_REL_DISTANCE] >= 0.0).all()
    assert (f[COL_EGO_SPEED] >= 0.0).all()


# --- physics sanity ---------------------------------------------------------------------


def test_nominal_brakes_in_time(nominal: AebScenarioConfig) -> None:
    s = generate_aeb_scenario(nominal)
    gt = s.ground_truth
    assert gt.collision is False
    assert gt.collision_time_s is None
    assert gt.min_gap_m > 1.0
    assert gt.risk_threshold_crossing_s is not None
    assert gt.brake_command_s is not None
    assert gt.braking_latency_s is not None
    # command follows the request by system latency, quantised to at most one sample
    assert gt.braking_latency_s == pytest.approx(nominal.system_latency_s, abs=nominal.dt_s)
    assert gt.risk_threshold_crossing_s > nominal.lead_brake_onset_s
    assert gt.max_deceleration_mps2 == pytest.approx(nominal.ego_max_decel_mps2)
    assert gt.ego_stopped_s is not None
    assert gt.min_ttc_s is not None and gt.min_ttc_s > 0.5


def test_brake_signal_matches_ground_truth(nominal: AebScenarioConfig) -> None:
    s = generate_aeb_scenario(nominal)
    f = s.frame
    first_cmd = f.loc[f[COL_BRAKE_CMD] == 1, COL_TIMESTAMP].iloc[0]
    assert first_cmd == pytest.approx(s.ground_truth.brake_command_s)
    # latched: once on, stays on
    on = f[COL_BRAKE_CMD].to_numpy()
    assert (np.diff(on) >= 0).all()
    # state machine: IDLE -> WARNING -> BRAKING, monotonic
    states = f[COL_AEB_STATE].to_numpy()
    assert (np.diff(states) >= 0).all()
    assert states[0] == AebState.IDLE and states[-1] == AebState.BRAKING
    assert AebState.WARNING in states


def test_ego_speed_constant_before_brake_then_decreasing(nominal: AebScenarioConfig) -> None:
    s = generate_aeb_scenario(nominal)
    f = s.frame
    cmd_t = s.ground_truth.brake_command_s
    assert cmd_t is not None
    before = f.loc[f[COL_TIMESTAMP] < cmd_t, COL_EGO_SPEED]
    assert before.mean() == pytest.approx(nominal.ego_speed_mps, abs=0.05)
    after = f.loc[f[COL_TIMESTAMP] > cmd_t + 0.5, COL_EGO_SPEED]
    assert after.iloc[0] > after.iloc[len(after) // 2] >= after.iloc[-1]


def test_late_braking_variant_is_late_and_collides(
    nominal: AebScenarioConfig, late: AebScenarioConfig
) -> None:
    n, late_s = generate_aeb_scenario(nominal), generate_aeb_scenario(late)
    ngt, lgt = n.ground_truth, late_s.ground_truth
    # identical physics up to the controller: same risk-threshold crossing
    assert lgt.risk_threshold_crossing_s == ngt.risk_threshold_crossing_s
    # perception only becomes valid after the confidence window ends
    assert lgt.perception_valid_at_risk_s is not None
    assert ngt.perception_valid_at_risk_s is not None
    assert lgt.perception_valid_at_risk_s > ngt.perception_valid_at_risk_s
    assert lgt.braking_latency_s is not None and ngt.braking_latency_s is not None
    assert lgt.braking_latency_s > ngt.braking_latency_s + 0.4
    assert lgt.collision is True
    assert lgt.collision_time_s is not None
    assert lgt.min_gap_m == 0.0
    assert lgt.min_ttc_s is not None and lgt.min_ttc_s < 0.2
    # the confidence drop is visible in the reported signal
    f = late_s.frame
    drop_start = late.lead_brake_onset_s + late.confidence_drop_start_after_onset_s
    in_drop = f[COL_TIMESTAMP].between(
        drop_start, drop_start + late.confidence_drop_duration_s, inclusive="left"
    )
    assert (f.loc[in_drop, COL_OBJECT_CONF] < late.detection_confidence_threshold).all()
    assert f.loc[~in_drop, COL_OBJECT_CONF].min() > late.detection_confidence_threshold


def test_no_lead_braking_means_no_event() -> None:
    cfg = AebScenarioConfig(lead_decel_mps2=1e-9, lead_brake_onset_s=9.9, duration_s=10.0)
    gt = generate_aeb_scenario(cfg).ground_truth
    assert gt.risk_threshold_crossing_s is None
    assert gt.brake_command_s is None
    assert gt.braking_latency_s is None
    assert gt.collision is False
    assert gt.min_ttc_s is None or gt.min_ttc_s > 10.0


# --- config validation ------------------------------------------------------------------


def test_config_rejects_inconsistent_thresholds() -> None:
    with pytest.raises(ValueError):
        AebScenarioConfig(trigger_ttc_s=3.0, warning_ttc_s=2.0)
    with pytest.raises(ValueError):
        AebScenarioConfig(lead_brake_onset_s=20.0, duration_s=10.0)
    with pytest.raises(ValueError):
        FaultInjection(drop_columns=("nope",))
    with pytest.raises(ValueError):
        FaultInjection(drop_columns=(COL_TIMESTAMP,))


# --- fault injection --------------------------------------------------------------------


def test_fault_injection(nominal: AebScenarioConfig) -> None:
    clean = generate_aeb_scenario(nominal).frame
    faults = FaultInjection(
        drop_columns=(COL_OBJECT_CONF,),
        timestamp_gap=TimestampGap(start_s=2.0, duration_s=0.5),
        nan_burst=NanBurst(column=COL_REL_DISTANCE, start_s=5.0, duration_s=0.2),
    )
    faulted = apply_faults(clean, faults)
    assert COL_OBJECT_CONF in clean.columns  # input untouched
    assert COL_OBJECT_CONF not in faulted.columns
    t = faulted[COL_TIMESTAMP]
    assert not t.between(2.0, 2.5, inclusive="left").any()
    assert len(faulted) == len(clean) - int(0.5 * nominal.sample_rate_hz)
    assert np.diff(t.to_numpy()).max() == pytest.approx(0.5 + nominal.dt_s)
    burst = faulted.loc[t.between(5.0, 5.2, inclusive="left"), COL_REL_DISTANCE]
    assert burst.isna().all() and len(burst) == int(0.2 * nominal.sample_rate_hz)
    assert faulted[COL_REL_DISTANCE].notna().sum() == len(faulted) - len(burst)

    # faults declared in config are applied by the generator and recorded in config
    via_cfg = generate_aeb_scenario(nominal.model_copy(update={"faults": faults}))
    pd.testing.assert_frame_equal(via_cfg.frame, faulted)
    assert via_cfg.config.faults == faults


def test_nan_burst_on_int_column_widens_dtype(nominal: AebScenarioConfig) -> None:
    clean = generate_aeb_scenario(nominal).frame
    out = apply_faults(
        clean, FaultInjection(nan_burst=NanBurst(column=COL_BRAKE_CMD, start_s=1.0, duration_s=0.1))
    )
    assert out[COL_BRAKE_CMD].dtype == np.float64
    assert out[COL_BRAKE_CMD].isna().sum() == int(0.1 * nominal.sample_rate_hz)


# --- export -----------------------------------------------------------------------------


def test_write_and_read_back_kmh(tmp_path: Path, late: AebScenarioConfig) -> None:
    s = generate_aeb_scenario(late)
    meta = write_scenario(s, tmp_path / "late", speed_unit="kmh")
    csv = tmp_path / "late" / CSV_NAME
    assert csv.exists()
    assert meta.scenario_id.startswith("scenario_")
    assert meta.data_origin == DATA_ORIGIN_SYNTHETIC
    assert meta.generator_version == GENERATOR_VERSION
    assert meta.column_units[CSV_EGO_SPEED_KMH] == "km/h"
    assert meta.column_units[CSV_REL_VELOCITY_KMH] == "km/h"
    assert COL_EGO_SPEED not in meta.column_units
    assert any("Not real-world" in n for n in meta.notes)

    on_disk = pd.read_csv(csv)
    assert CSV_EGO_SPEED_KMH in on_disk.columns and COL_EGO_SPEED not in on_disk.columns
    assert len(on_disk) == len(s.frame)
    # converting back once at ingestion recovers SI to float precision
    recovered = on_disk[CSV_EGO_SPEED_KMH].to_numpy() * KMH_TO_MPS
    assert np.allclose(recovered, s.frame[COL_EGO_SPEED].to_numpy(), atol=1e-9)

    meta2 = read_metadata(tmp_path / "late" / METADATA_NAME)
    assert meta2 == meta
    # the sidecar is enough to regenerate the exact same scenario
    regenerated = generate_aeb_scenario(meta2.config)
    pd.testing.assert_frame_equal(regenerated.frame, s.frame)
    assert regenerated.ground_truth == meta2.ground_truth


def test_write_si(tmp_path: Path, nominal: AebScenarioConfig) -> None:
    s = generate_aeb_scenario(nominal)
    meta = write_scenario(s, tmp_path, speed_unit="mps", scenario_id="scenario_fixed000001")
    assert meta.scenario_id == "scenario_fixed000001"
    on_disk = pd.read_csv(tmp_path / CSV_NAME)
    assert list(on_disk.columns) == list(FRAME_COLUMNS)
    assert meta.column_units[COL_EGO_SPEED] == "m/s"
    assert math.isclose(on_disk[COL_EGO_SPEED].iloc[0], s.frame[COL_EGO_SPEED].iloc[0])


def test_cli_generates_all_variants(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli_main(["--all", "--seed", "3", "--out", str(tmp_path), "--duration-s", "8"])
    assert rc == 0
    for v in ScenarioVariant:
        d = tmp_path / f"aeb_{v.value}_seed3"
        assert (d / CSV_NAME).exists() and (d / METADATA_NAME).exists()
        assert read_metadata(d / METADATA_NAME).config.duration_s == 8.0
    out = capsys.readouterr().out
    assert "collision=False" in out and "collision=True" in out
