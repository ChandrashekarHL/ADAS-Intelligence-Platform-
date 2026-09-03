"""M2 ingestion: column resolution, one-time unit conversion, provenance, sidecar, CLI."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.core.errors import IngestionError
from app.core.signals import (
    COL_EGO_SPEED,
    COL_REL_DISTANCE,
    COL_REL_VELOCITY,
    COL_TIMESTAMP,
    FRAME_COLUMNS,
)
from app.core.units import KMH_TO_MPS
from app.ingestion.cli import EXIT_BLOCKED
from app.ingestion.cli import main as cli_main
from app.ingestion.csv_loader import (
    ingest_frame,
    load_telemetry_csv,
    resolve_column,
)
from app.ingestion.schemas import DATA_ORIGIN_UNKNOWN
from app.synthetic.aeb_generator import SyntheticScenario, generate_aeb_scenario
from app.synthetic.io import CSV_NAME, write_scenario
from app.synthetic.schemas import AebScenarioConfig, ScenarioVariant


@pytest.fixture(scope="module")
def late() -> SyntheticScenario:
    return generate_aeb_scenario(AebScenarioConfig(variant=ScenarioVariant.LATE_BRAKING, seed=3))


@pytest.fixture
def kmh_csv(tmp_path: Path, late: SyntheticScenario) -> Path:
    write_scenario(late, tmp_path, speed_unit="kmh")
    return tmp_path / CSV_NAME


def test_resolve_column() -> None:
    assert resolve_column("ego_speed_kmh", None) == (COL_EGO_SPEED, "km/h")
    assert resolve_column("ego_speed_mps", None) == (COL_EGO_SPEED, "m/s")
    assert resolve_column("ego_speed", None) == (COL_EGO_SPEED, "m/s")  # assumes SI
    assert resolve_column("ego_speed", "km/h") == (COL_EGO_SPEED, "km/h")  # declared wins
    assert resolve_column("timestamp_ms", None) == (COL_TIMESTAMP, "ms")
    assert resolve_column("relative_distance_m", None) == (COL_REL_DISTANCE, "m")
    assert resolve_column("steering_angle_deg", None) == ("steering_angle_deg", None)


def test_load_kmh_export_converts_once(kmh_csv: Path, late: SyntheticScenario) -> None:
    ing = load_telemetry_csv(kmh_csv)
    f, prov = ing.frame, ing.provenance

    assert list(f.columns) == list(FRAME_COLUMNS)
    assert len(f) == len(late.frame)
    for col in (COL_EGO_SPEED, COL_REL_VELOCITY, COL_REL_DISTANCE, COL_TIMESTAMP):
        assert np.allclose(f[col].to_numpy(), late.frame[col].to_numpy(), atol=1e-9), col

    assert prov.file_id.startswith("file_")
    assert prov.data_origin == "synthetic"
    assert prov.scenario_id is not None and prov.scenario_id.startswith("scenario_")
    assert prov.sidecar_path is not None and prov.sidecar_path.endswith("scenario.json")
    assert prov.sha256 is not None and len(prov.sha256) == 64
    assert prov.row_count == 500
    assert prov.duration_s == pytest.approx(9.98)
    assert prov.nominal_dt_s == pytest.approx(0.02)
    assert prov.renamed_columns == {
        "ego_speed_kmh": COL_EGO_SPEED,
        "relative_velocity_kmh": COL_REL_VELOCITY,
    }
    assert {(c.source_column, c.source_unit, c.target_unit) for c in prov.conversions} == {
        ("ego_speed_kmh", "km/h", "m/s"),
        ("relative_velocity_kmh", "km/h", "m/s"),
    }
    assert all(c.factor == KMH_TO_MPS for c in prov.conversions)
    assert prov.passthrough_columns == ()
    assert prov.coerced_values == {}


def test_load_si_export_has_no_conversions(tmp_path: Path, late: SyntheticScenario) -> None:
    write_scenario(late, tmp_path, speed_unit="mps")
    ing = load_telemetry_csv(tmp_path / CSV_NAME)
    assert ing.provenance.conversions == ()
    assert ing.provenance.renamed_columns == {}
    pd.testing.assert_frame_equal(
        ing.frame[[COL_EGO_SPEED]], late.frame[[COL_EGO_SPEED]], check_exact=False
    )


def test_caller_units_override_sidecar(kmh_csv: Path, late: SyntheticScenario) -> None:
    # The caller declares ego_speed_kmh is really m/s: the loader must obey and not convert
    # that column, while the sidecar's km/h entry for relative_velocity still applies.
    ing = load_telemetry_csv(kmh_csv, column_units={"ego_speed_kmh": "m/s"})
    assert [c.source_column for c in ing.provenance.conversions] == ["relative_velocity_kmh"]
    ratio = ing.frame[COL_EGO_SPEED].mean() / late.frame[COL_EGO_SPEED].mean()
    assert ratio == pytest.approx(3.6, rel=1e-3)  # raw km/h numbers kept as-is
    assert np.allclose(ing.frame[COL_REL_VELOCITY], late.frame[COL_REL_VELOCITY], atol=1e-9)


def test_no_sidecar_falls_back_to_suffix(tmp_path: Path, late: SyntheticScenario) -> None:
    write_scenario(late, tmp_path, speed_unit="kmh")
    (tmp_path / "scenario.json").unlink()
    ing = load_telemetry_csv(tmp_path / CSV_NAME)
    assert ing.provenance.data_origin == DATA_ORIGIN_UNKNOWN
    assert ing.provenance.scenario_id is None
    assert ing.provenance.sidecar_path is None
    assert len(ing.provenance.conversions) == 2  # from _kmh suffix
    assert np.allclose(ing.frame[COL_EGO_SPEED], late.frame[COL_EGO_SPEED], atol=1e-9)


def test_malformed_sidecar_raises(tmp_path: Path, late: SyntheticScenario) -> None:
    write_scenario(late, tmp_path)
    (tmp_path / "scenario.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(IngestionError):
        load_telemetry_csv(tmp_path / CSV_NAME)


def test_unknown_columns_pass_through_and_garbage_is_counted() -> None:
    raw = pd.DataFrame(
        {
            "timestamp_ms": [0, 100, 200, 300],
            "ego_speed_kmh": ["36.0", "36.0", "oops", "36.0"],
            "steering_angle_deg": [0.1, 0.2, 0.3, 0.4],
        }
    )
    ing = ingest_frame(raw, source_path="memory")
    f, prov = ing.frame, ing.provenance
    assert list(f.columns) == [COL_TIMESTAMP, COL_EGO_SPEED, "steering_angle_deg"]
    assert f[COL_TIMESTAMP].tolist() == pytest.approx([0.0, 0.1, 0.2, 0.3])
    assert f[COL_EGO_SPEED].iloc[0] == pytest.approx(10.0)
    assert np.isnan(f[COL_EGO_SPEED].iloc[2])
    assert prov.coerced_values == {COL_EGO_SPEED: 1}
    assert prov.passthrough_columns == ("steering_angle_deg",)
    assert prov.nominal_dt_s == pytest.approx(0.1)


@pytest.mark.parametrize(
    "headers",
    [
        ("ego_speed_kmh", "ego_speed_mps"),  # alias + canonical
        ("ego_speed", "ego_speed_kmh"),  # two aliases, neither is the canonical name
        ("timestamp_s", "time_ms"),  # different base names for the same signal
    ],
)
def test_duplicate_canonical_mapping_raises(headers: tuple[str, str]) -> None:
    dup = pd.DataFrame({h: [1.0, 2.0] for h in headers})
    with pytest.raises(IngestionError, match=f"{headers[0]!r} and {headers[1]!r} both map to"):
        ingest_frame(dup, source_path="memory")


def test_unknown_unit_raises() -> None:
    weird = pd.DataFrame({"ego_speed": [1.0]})
    with pytest.raises(IngestionError):
        ingest_frame(weird, source_path="memory", column_units={"ego_speed": "furlongs/fortnight"})


def test_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_telemetry_csv(Path("does/not/exist.csv"))


def test_cli(kmh_csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main([str(kmh_csv)]) == 0
    out = capsys.readouterr().out
    assert "verdict=PASS" in out
    assert "converted ego_speed_kmh [km/h] -> ego_speed_mps [m/s]" in out

    assert cli_main([str(kmh_csv), "--json"]) == 0
    assert '"verdict": "pass"' in capsys.readouterr().out


def test_cli_blocked(tmp_path: Path, late: SyntheticScenario) -> None:
    write_scenario(late, tmp_path)
    csv = tmp_path / CSV_NAME
    df = pd.read_csv(csv).drop(columns=["object_confidence"])
    df.to_csv(csv, index=False)
    assert cli_main([str(csv)]) == EXIT_BLOCKED
