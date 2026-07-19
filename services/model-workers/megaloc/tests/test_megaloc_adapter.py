from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest
from PIL import Image


_MODEL_WORKERS_ROOT = Path(__file__).resolve().parents[2]
if str(_MODEL_WORKERS_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_WORKERS_ROOT))
_ADAPTER_PATH = Path(__file__).resolve().parents[1] / "adapter.py"
_SPEC = importlib.util.spec_from_file_location("atlaslens_megaloc_worker_adapter", _ADAPTER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

MegaLocAdapter = _MODULE.MegaLocAdapter
MegaLocAdapterError = _MODULE.MegaLocAdapterError
MegaLocArtifacts = _MODULE.MegaLocArtifacts
verify_weight = _MODULE.verify_weight


def _image_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (31, 29), "white").save(output, format="PNG")
    return output.getvalue()


class _FakeScalar:
    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return self._value


class _FakeDescriptor:
    ndim = 1

    def __init__(self) -> None:
        self._values = [1.0 / math.sqrt(8448)] * 8448

    def float(self) -> _FakeDescriptor:
        return self

    def cpu(self) -> _FakeDescriptor:
        return self

    def norm(self, *, p: int) -> _FakeScalar:
        assert p == 2
        return _FakeScalar(math.sqrt(sum(value * value for value in self._values)))

    def numel(self) -> int:
        return len(self._values)

    def tolist(self) -> list[float]:
        return list(self._values)


class _FakeBatch:
    def unsqueeze(self, _: int) -> _FakeBatch:
        return self

    def to(self, _: str) -> _FakeBatch:
        return self


class _FakeModel:
    def __call__(self, _: object) -> list[_FakeDescriptor]:
        return [_FakeDescriptor()]


class _FakeTorch:
    @staticmethod
    def inference_mode() -> object:
        return nullcontext()


def _artifacts(tmp_path: Path, *, digest: str = "0" * 64) -> MegaLocArtifacts:
    return MegaLocArtifacts(
        source_dir=tmp_path / "source",
        model_dir=tmp_path / "model",
        source_revision="source-revision",
        model_revision="model-revision",
        weight_sha256=digest,
    )


def test_artifact_readiness_requires_exact_receipt_and_non_symlink_files(tmp_path: Path) -> None:
    payload = b"real-reviewed-weight-bytes"
    digest = hashlib.sha256(payload).hexdigest()
    artifacts = _artifacts(tmp_path, digest=digest)
    artifacts.source_dir.mkdir()
    artifacts.model_dir.mkdir()
    (artifacts.source_dir / "megaloc_model.py").write_text("class MegaLoc: pass\n")
    (artifacts.source_dir / "LICENSE").write_text("MIT\n")
    artifacts.weights_path.write_bytes(payload)
    artifacts.receipt_path.write_text(
        json.dumps(
            {
                "source_revision": "source-revision",
                "model_revision": "model-revision",
                "weight_sha256": digest,
                "weight_size": len(payload),
            }
        )
    )

    adapter = MegaLocAdapter(artifacts)

    assert adapter.health()["weights_available"] is True
    artifacts.receipt_path.write_text("{}")
    assert adapter.health()["weights_available"] is False


def test_weight_verification_fails_closed_on_checksum_mismatch(tmp_path: Path) -> None:
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"weight")

    with pytest.raises(MegaLocAdapterError) as error:
        verify_weight(weight, "0" * 64)

    assert error.value.code == "checksum_mismatch"


def test_infer_returns_only_a_normalized_descriptor_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = MegaLocAdapter(_artifacts(tmp_path))
    adapter._model = _FakeModel()
    adapter._torch = _FakeTorch()
    adapter._device = "cpu"
    monkeypatch.setattr(_MODULE, "_preprocess", lambda *_args, **_kwargs: _FakeBatch())

    result = adapter.infer(_image_bytes(), {"device": "cpu"})

    descriptor = result["descriptor"]
    assert isinstance(descriptor, list)
    assert len(descriptor) == 8448
    assert math.isclose(math.sqrt(sum(value * value for value in descriptor)), 1.0)
    assert result["descriptor_semantics"] == "visual_place_descriptor_not_confidence"
    assert "latitude" not in result and "longitude" not in result and "confidence" not in result


def test_infer_rejects_device_mismatch_before_model_execution(tmp_path: Path) -> None:
    adapter = MegaLocAdapter(_artifacts(tmp_path))
    adapter._model = _FakeModel()
    adapter._torch = _FakeTorch()
    adapter._device = "cpu"

    with pytest.raises(MegaLocAdapterError) as error:
        adapter.infer(_image_bytes(), {"device": "cuda"})

    assert error.value.code == "invalid_device_mismatch"
