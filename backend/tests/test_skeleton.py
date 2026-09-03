"""M0 skeleton tests: package imports, settings load without env, ID rules hold."""

import pytest

from app.core.config import Settings
from app.core.ids import new_id
from app.core.units import KMH_TO_MPS


def test_settings_load_without_env() -> None:
    s = Settings(_env_file=None)
    assert s.database_url.startswith("sqlite:///")
    assert s.openai_api_key is None


def test_new_id_prefix_enforced() -> None:
    assert new_id("metric").startswith("metric_")
    with pytest.raises(ValueError):
        new_id("bogus")


def test_units() -> None:
    speed_mps = 100 * KMH_TO_MPS
    assert speed_mps == pytest.approx(27.7778, abs=1e-3)
