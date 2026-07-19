from __future__ import annotations

import importlib.util
import io
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
_SPEC = importlib.util.spec_from_file_location("atlaslens_osv5m_worker_adapter", _ADAPTER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

ModelAdapterError = _MODULE.ModelAdapterError
OSV5MAdapter = _MODULE.OSV5MAdapter
OSV5MArtifacts = _MODULE.OSV5MArtifacts
adapt_osv5m_state_dict = _MODULE.adapt_osv5m_state_dict
normalize_radian_prediction = _MODULE.normalize_radian_prediction
patch_offline_config = _MODULE.patch_offline_config


def _config() -> dict[str, object]:
    return {
        "model": {
            "backbone": {
                "instance": {
                    "_target_": "models.networks.backbones.CLIP",
                    "path": "remote/model",
                }
            }
        }
    }


def _image_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (32, 24), "white").save(output, format="PNG")
    return output.getvalue()


def test_config_is_copied_and_remote_backbone_resolution_is_disabled() -> None:
    source = _config()
    patched = patch_offline_config(source)

    assert source["model"]["backbone"]["instance"]["path"] == "remote/model"  # type: ignore[index]
    assert (
        patched["model"]["backbone"]["instance"]["path"]  # type: ignore[index]
        == "atlaslens-offline-complete-checkpoint"
    )


def test_checkpoint_adapter_renames_only_clip_layout_and_drops_legacy_buffers() -> None:
    marker = object()
    adapted = adapt_osv5m_state_dict(
        {
            "model.backbone.clip.vision_model.encoder.weight": marker,
            "model.head.cell_center": object(),
            "head.cell_size_up": object(),
            "model.mid.weight": marker,
        }
    )

    assert adapted == {
        "model.backbone.clip.encoder.weight": marker,
        "model.mid.weight": marker,
    }


@pytest.mark.parametrize("value", ([[float("nan"), 0]], [[math.pi, 0]], [0.0]))
def test_invalid_radian_outputs_fail_closed(value: object) -> None:
    with pytest.raises(ModelAdapterError) as error:
        normalize_radian_prediction(value)
    assert error.value.code == "invalid_model_output"


class _FakeTensor:
    def unsqueeze(self, _: int) -> _FakeTensor:
        return self

    def to(self, _: str) -> _FakeTensor:
        return self


class _FakeModel:
    def transform(self, _: object) -> _FakeTensor:
        return _FakeTensor()

    def __call__(self, _: object) -> list[list[float]]:
        return [[0.25, -0.5]]


class _FakeTorch:
    @staticmethod
    def inference_mode() -> object:
        return nullcontext()


def test_infer_returns_one_real_contract_pair_in_radians(tmp_path: Path) -> None:
    adapter = OSV5MAdapter(
        OSV5MArtifacts(
            source_dir=tmp_path / "source",
            model_dir=tmp_path / "model",
            source_revision="source-revision",
            model_revision="model-revision",
        )
    )
    adapter._model = _FakeModel()
    adapter._torch = _FakeTorch()
    adapter._device = "cpu"
    adapter._load_ms = 12

    result = adapter.infer(
        _image_bytes(),
        {"device": "cpu", "model_id": "osv5m/baseline"},
    )

    assert result["coordinates_radians"] == [0.25, -0.5]
    assert result["coordinate_order"] == "latitude_longitude"
    assert result["units"] == "radians"
    assert result["model_revision"] == "model-revision"


def test_infer_rejects_loaded_device_mismatch(tmp_path: Path) -> None:
    adapter = OSV5MAdapter(
        OSV5MArtifacts(
            source_dir=tmp_path,
            model_dir=tmp_path,
            source_revision="source",
            model_revision="model",
        )
    )
    adapter._model = _FakeModel()
    adapter._torch = _FakeTorch()
    adapter._device = "cpu"

    with pytest.raises(ModelAdapterError) as error:
        adapter.infer(_image_bytes(), {"device": "cuda"})
    assert error.value.code == "device_mismatch"
