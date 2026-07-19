from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from atlaslens_api.config import Settings


def test_phase6c_defaults_are_explicit_and_startup_safe() -> None:
    settings = Settings(_env_file=None)

    assert settings.phase6c_enabled is False
    assert settings.phase6c_pipeline_version == "phase6c-v1"
    assert settings.phase6c_ocr_config.name == "phase6c-v1.json"
    assert settings.phase6c_ocr_config.parent.name == "ocr"
    assert settings.geoclip_refinement_levels_km == (900.0, 300.0, 90.0)
    assert settings.megaloc_worker_host == "127.0.0.1"
    assert settings.megaloc_worker_port == 8794
    assert settings.reference_index_enabled is True
    assert settings.mapillary_enabled is False
    assert settings.mapillary_token_value is None
    assert settings.g3_enabled is False


def test_phase6c_paths_and_secret_are_backend_only(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        phase6c_enabled=True,
        reference_index_path=tmp_path / "index",
        mapillary_access_token="private-token",
    )

    assert settings.reference_index_path == tmp_path / "index"
    assert settings.mapillary_token_value == "private-token"
    assert "private-token" not in repr(settings)


@pytest.mark.parametrize(
    "levels",
    ("", "90,300", "900,0,90", "900,300,90,30,10", "not-a-number"),
)
def test_phase6c_refinement_levels_fail_closed(levels: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, geoclip_grid_refinement_levels_km=levels)
