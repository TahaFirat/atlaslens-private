from __future__ import annotations

from pathlib import Path

import pytest

from atlaslens_api.model_management.errors import ModelManagementError
from atlaslens_api.model_management.rapidocr import RapidOCRModelManagementService


def _package(tmp_path: Path) -> Path:
    root = tmp_path / "package" / "rapidocr"
    models = root / "models"
    models.mkdir(parents=True)
    (root / "__init__.py").write_text("", encoding="utf-8")
    (models / "PP-OCRv6_det_small.onnx").write_bytes(b"detector-model")
    (models / "PP-OCRv6_rec_small.onnx").write_bytes(b"recognizer-model")
    return root


def test_explicit_install_receipts_and_detects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _package(tmp_path)
    service = RapidOCRModelManagementService(tmp_path / "cache")
    monkeypatch.setattr(service, "_package_root", lambda: package)
    monkeypatch.setattr(service, "_runtime", lambda: ("onnxruntime-gpu", "1.27.0"))

    installed = service.install()
    assert installed.status == "ready"
    assert installed.verified is True
    assert installed.profiles == 1

    model = (
        tmp_path
        / "cache"
        / "rapidocr-3.9.1"
        / "models"
        / "PP-OCRv6_rec_small.onnx"
    )
    model.write_bytes(b"tampered")
    invalid = service.info()
    assert invalid.status == "invalid"
    assert invalid.verified is False
    with pytest.raises(ModelManagementError):
        service.verify()


def test_missing_install_and_confirmed_remove(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _package(tmp_path)
    service = RapidOCRModelManagementService(tmp_path / "cache")
    assert service.info().status == "not_installed"
    monkeypatch.setattr(service, "_package_root", lambda: package)
    monkeypatch.setattr(service, "_runtime", lambda: ("onnxruntime-cpu", "1.27.0"))
    service.install()
    with pytest.raises(ModelManagementError):
        service.remove(confirmed=False)
    removed = service.remove(confirmed=True)
    assert removed.status == "not_installed"
