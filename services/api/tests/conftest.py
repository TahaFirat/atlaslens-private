from __future__ import annotations

import io
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import piexif
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from atlaslens_api.config import Settings
from atlaslens_api.main import create_app


def image_bytes(
    image_format: str = "JPEG",
    *,
    size: tuple[int, int] = (64, 48),
    color: tuple[int, int, int] = (120, 140, 160),
    exif: bytes | None = None,
) -> bytes:
    buffer = io.BytesIO()
    image = Image.new("RGB", size, color)
    kwargs: dict[str, Any] = {}
    if exif is not None:
        kwargs["exif"] = exif
    image.save(buffer, format=image_format, **kwargs)
    return buffer.getvalue()


def _dms(value: float) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    absolute = abs(value)
    degrees = int(absolute)
    minutes_float = (absolute - degrees) * 60
    minutes = int(minutes_float)
    seconds = (minutes_float - minutes) * 60
    return (degrees, 1), (minutes, 1), (round(seconds * 100_000), 100_000)


def gps_jpeg(
    latitude: float,
    longitude: float,
    *,
    altitude: float = 12.5,
    orientation: int = 1,
    size: tuple[int, int] = (64, 48),
) -> bytes:
    exif = {
        "0th": {piexif.ImageIFD.Orientation: orientation},
        "GPS": {
            piexif.GPSIFD.GPSLatitudeRef: b"N" if latitude >= 0 else b"S",
            piexif.GPSIFD.GPSLatitude: _dms(latitude),
            piexif.GPSIFD.GPSLongitudeRef: b"E" if longitude >= 0 else b"W",
            piexif.GPSIFD.GPSLongitude: _dms(longitude),
            piexif.GPSIFD.GPSAltitudeRef: 0,
            piexif.GPSIFD.GPSAltitude: (round(altitude * 10), 10),
        },
    }
    return image_bytes("JPEG", size=size, exif=piexif.dump(exif))


def upload(
    client: TestClient,
    payload: bytes,
    *,
    filename: str = "fixture.jpg",
    content_type: str = "image/jpeg",
    mode: str = "local_only",
    consent: str = "false",
    authorization: str = "true",
    headers: dict[str, str] | None = None,
) -> Any:
    return client.post(
        "/api/v1/analyses",
        files={"image": (filename, payload, content_type)},
        data={
            "analysis_mode": mode,
            "cloud_processing_consent": consent,
            "authorization_acknowledged": authorization,
        },
        headers=headers or {},
    )


def wait_for_terminal(client: TestClient, analysis_id: str, timeout: float = 5.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/analyses/{analysis_id}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] in {"completed", "failed"}:
            return body
        time.sleep(0.01)
    raise AssertionError("analysis did not reach a terminal state")


@pytest.fixture
def client_factory(tmp_path: Path) -> Any:
    with ExitStack() as stack:
        counter = 0

        def factory(**overrides: Any) -> TestClient:
            nonlocal counter
            counter += 1
            root = tmp_path / f"instance-{counter}"
            app_overrides = {
                key: settings_value
                for key, settings_value in list(overrides.items())
                if key.endswith("_provider")
            }
            for key in app_overrides:
                overrides.pop(key, None)
            setting_values: dict[str, Any] = {
                "database_url": f"sqlite:///{(root / 'atlaslens.db').as_posix()}",
                "temp_storage_dir": root / "tmp",
                "openai_api_key": None,
                "ocr_enabled": False,
                "global_model_enabled": False,
                "phase6b_enabled": False,
                "sse_heartbeat_seconds": 0.05,
                "max_sse_lifetime_seconds": 2,
            }
            setting_values.update(overrides)
            settings = Settings(_env_file=None, **setting_values)
            app = create_app(settings, **app_overrides)
            return stack.enter_context(TestClient(app, raise_server_exceptions=False))

        yield factory


@pytest.fixture
def client(client_factory: Any) -> TestClient:
    return client_factory()
