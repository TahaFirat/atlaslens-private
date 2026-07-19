from __future__ import annotations

import importlib.util
import io
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest
from PIL import Image


_MODEL_WORKERS_ROOT = Path(__file__).resolve().parents[2]
if str(_MODEL_WORKERS_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_WORKERS_ROOT))
_ADAPTER_PATH = Path(__file__).resolve().parents[1] / "adapter.py"
_SPEC = importlib.util.spec_from_file_location("atlaslens_plonk_worker_adapter", _ADAPTER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

ModelAdapterError = _MODULE.ModelAdapterError
PlonkAdapter = _MODULE.PlonkAdapter
PlonkArtifacts = _MODULE.PlonkArtifacts
PlonkModelArtifact = _MODULE.PlonkModelArtifact
normalize_degree_samples = _MODULE.normalize_degree_samples
unwrap_state_dict = _MODULE._unwrap_state_dict


MODEL_ID = "nicolas-dufour/PLONK_YFCC"


def _artifacts(tmp_path: Path) -> tuple[PlonkArtifacts, PlonkModelArtifact]:
    artifact = PlonkModelArtifact(
        model_id=MODEL_ID,
        revision="model-revision",
        model_dir=tmp_path / "yfcc",
        conditioning="dinov2",
    )
    return (
        PlonkArtifacts(
            models={MODEL_ID: artifact},
            source_revision="source-revision",
            streetclip_dir=tmp_path / "streetclip",
            dinov2_repo_dir=tmp_path / "dinov2",
            dinov2_weights_path=tmp_path / "dinov2.pth",
        ),
        artifact,
    )


def _image_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (32, 24), "white").save(output, format="PNG")
    return output.getvalue()


def test_degree_samples_preserve_latitude_longitude_order() -> None:
    samples = normalize_degree_samples([[41.0, 29.0], [-33.9, 151.2]], maximum=8)
    assert samples == [[41.0, 29.0], [-33.9, 151.2]]


@pytest.mark.parametrize(
    "value",
    ([[91.0, 0.0]], [[0.0, 181.0]], [[float("nan"), 0.0]], [[1.0]]),
)
def test_invalid_degree_samples_fail_closed(value: object) -> None:
    with pytest.raises(ModelAdapterError) as error:
        normalize_degree_samples(value, maximum=8)
    assert error.value.code == "invalid_model_output"


def test_state_dict_unwrap_accepts_only_a_nonempty_mapping() -> None:
    marker = object()
    assert unwrap_state_dict({"model": {"layer": marker}}) == {"layer": marker}
    with pytest.raises(ModelAdapterError):
        unwrap_state_dict({})


class _FakePipeline:
    def __call__(self, *_: object, **__: object) -> list[list[float]]:
        return [[41.0, 29.0], [40.9, 28.9]]


class _FakeTorch:
    @staticmethod
    def inference_mode() -> object:
        return nullcontext()


def test_infer_returns_bounded_degree_samples_and_exact_revision(tmp_path: Path) -> None:
    artifacts, artifact = _artifacts(tmp_path)
    adapter = PlonkAdapter(artifacts)
    adapter._pipeline = _FakePipeline()
    adapter._torch = _FakeTorch()
    adapter._model = artifact
    adapter._device = "cpu"
    adapter._load_ms = 9

    result = adapter.infer(
        _image_bytes(),
        {"model_id": MODEL_ID, "device": "cpu", "sample_count": 2},
    )

    assert result["samples_degrees"] == [[41.0, 29.0], [40.9, 28.9]]
    assert result["model_revision"] == "model-revision"
    assert result["localizability"] is None
    assert result["localizability_semantics"] == "raw_model_diagnostic_not_confidence"


def test_infer_rejects_model_mismatch_without_running_pipeline(tmp_path: Path) -> None:
    artifacts, artifact = _artifacts(tmp_path)
    adapter = PlonkAdapter(artifacts)
    adapter._pipeline = _FakePipeline()
    adapter._torch = _FakeTorch()
    adapter._model = artifact
    adapter._device = "cpu"

    with pytest.raises(ModelAdapterError) as error:
        adapter.infer(
            _image_bytes(),
            {"model_id": "nicolas-dufour/PLONK_OSV_5M", "sample_count": 2},
        )
    assert error.value.code == "model_mismatch"
