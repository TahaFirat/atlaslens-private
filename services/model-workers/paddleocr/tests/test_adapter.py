from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import sys
import threading
from collections.abc import Iterator
from http.client import HTTPConnection
from pathlib import Path

import pytest
from PIL import Image

_ADAPTER_PATH = Path(__file__).resolve().parents[1] / "adapter.py"
_MODEL_WORKERS_ROOT = _ADAPTER_PATH.parents[1]
if str(_MODEL_WORKERS_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_WORKERS_ROOT))
from common.protocol import PROTOCOL_VERSION  # noqa: E402
from common.server import create_worker_server  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "atlaslens_paddleocr_worker_adapter", _ADAPTER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

DETECTION_MODEL_ID = _MODULE.DETECTION_MODEL_ID
RECOGNITION_MODEL_ID = _MODULE.RECOGNITION_MODEL_ID
PaddleOCRAdapter = _MODULE.PaddleOCRAdapter
PaddleOCRAdapterError = _MODULE.PaddleOCRAdapterError
PaddleOCRConfig = _MODULE.PaddleOCRConfig

_WORKER_ENVIRONMENT = (
    "USERPROFILE",
    "HOME",
    "XDG_CACHE_HOME",
    "PADDLE_PDX_CACHE_HOME",
    "HF_HOME",
    "MODELSCOPE_CACHE",
    "TEMP",
    "TMP",
    "PADDLEOCR_DISABLE_AUTO_LOGGING_CONFIG",
    "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK",
    "HF_HUB_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "FLAGS_use_mkldnn",
    "FLAGS_enable_pir_api",
    "FLAGS_minloglevel",
    "GLOG_minloglevel",
)


@pytest.fixture(autouse=True)
def _restore_worker_environment() -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in _WORKER_ENVIRONMENT}
    yield
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


class _FakeEngine:
    def __init__(self) -> None:
        self.closed = False
        self.predict_kwargs: dict[str, object] | None = None

    def predict(self, image: object, **kwargs: object) -> list[dict[str, object]]:
        self.predict_kwargs = kwargs
        assert getattr(image, "shape") == (64, 128, 3)
        return [
            {
                "rec_texts": ["ERCİYES ÜNİVERSİTESİ"],
                "rec_scores": [0.91],
                "rec_polys": [[[2, 4], [100, 4], [100, 40], [2, 40]]],
            }
        ]

    def close(self) -> None:
        self.closed = True


def _write_model(directory: Path) -> None:
    directory.mkdir(parents=True)
    (directory / "inference.pdiparams").write_bytes(b"params")
    (directory / "inference.json").write_text("{}", encoding="utf-8")
    (directory / "config.json").write_text("{}", encoding="utf-8")


def _config(tmp_path: Path) -> PaddleOCRConfig:
    model_root = tmp_path / "models"
    detector = model_root / "det"
    recognizer = model_root / "rec" / RECOGNITION_MODEL_ID
    _write_model(detector)
    _write_model(recognizer)
    return PaddleOCRConfig(
        model_root=model_root,
        detection_model_dir=detector,
        recognition_model_dir=recognizer,
        private_home=tmp_path / "private-home",
        max_decoded_pixels=100_000,
        max_side=512,
    )


def _image_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (64, 32), "white").save(output, format="PNG")
    return output.getvalue()


def test_load_uses_exact_local_models_and_safe_cpu_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    engine = _FakeEngine()

    def factory(**kwargs: object) -> _FakeEngine:
        captured.update(kwargs)
        return engine

    for name in (
        "USERPROFILE",
        "HOME",
        "PADDLE_PDX_CACHE_HOME",
        "FLAGS_use_mkldnn",
        "FLAGS_enable_pir_api",
    ):
        monkeypatch.delenv(name, raising=False)
    config = _config(tmp_path)
    adapter = PaddleOCRAdapter(
        config, ocr_factory=factory, enforce_versions=False
    )

    loaded = adapter.load({"device": "cpu"})

    assert loaded["loaded"] is True
    assert captured["text_detection_model_name"] == DETECTION_MODEL_ID
    assert captured["text_detection_model_dir"] == str(
        config.detection_model_dir.resolve()
    )
    assert captured["text_recognition_model_name"] == RECOGNITION_MODEL_ID
    assert captured["text_recognition_model_dir"] == str(
        config.recognition_model_dir.resolve()
    )
    assert captured["use_doc_orientation_classify"] is False
    assert captured["use_doc_unwarping"] is False
    assert captured["use_textline_orientation"] is False
    assert captured["enable_mkldnn"] is False
    assert os.environ["FLAGS_use_mkldnn"] == "0"
    assert os.environ["FLAGS_enable_pir_api"] == "0"
    assert Path(os.environ["USERPROFILE"]) == config.private_home
    assert (config.private_home / ".cache" / "paddle" / "dataset").is_dir()


def test_infer_normalizes_unicode_scores_and_scaled_polygons(tmp_path: Path) -> None:
    engine = _FakeEngine()
    adapter = PaddleOCRAdapter(
        _config(tmp_path), ocr_factory=lambda **_: engine, enforce_versions=False
    )

    result = adapter.infer(
        _image_bytes(),
        {"device": "cpu", "scales": [2.0], "minimum_confidence": 0.45},
    )

    assert result["device"] == "cpu"
    assert result["model_ids"] == [DETECTION_MODEL_ID, RECOGNITION_MODEL_ID]
    assert result["lines"] == [
        {
            "text": "ERCİYES ÜNİVERSİTESİ",
            "confidence": 0.91,
            "polygon": [[1.0, 2.0], [50.0, 2.0], [50.0, 20.0], [1.0, 20.0]],
            "crop_id": "scale-2",
        }
    ]
    assert engine.predict_kwargs == {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "text_rec_score_thresh": 0.45,
        "return_word_box": False,
    }
    assert adapter.health()["real_inference_verified"] is True
    assert adapter.unload({})["unloaded"] is True
    assert engine.closed is True
    assert adapter.health()["model_loaded"] is False


def test_missing_or_unsafe_weights_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "models"
    adapter = PaddleOCRAdapter(
        PaddleOCRConfig(
            model_root=root,
            detection_model_dir=root / "det",
            recognition_model_dir=root / "rec" / RECOGNITION_MODEL_ID,
            private_home=tmp_path / "home",
        ),
        ocr_factory=lambda **_: _FakeEngine(),
        enforce_versions=False,
    )

    assert adapter.health()["weights_available"] is False
    with pytest.raises(PaddleOCRAdapterError) as error:
        adapter.load({})
    assert error.value.code == "pretrained_weights_missing"


def test_invalid_image_and_device_return_only_safe_codes(tmp_path: Path) -> None:
    adapter = PaddleOCRAdapter(
        _config(tmp_path),
        ocr_factory=lambda **_: _FakeEngine(),
        enforce_versions=False,
    )
    with pytest.raises(PaddleOCRAdapterError) as device_error:
        adapter.load({"device": "cuda"})
    assert device_error.value.code == "unsupported_device"
    with pytest.raises(PaddleOCRAdapterError) as image_error:
        adapter.infer(b"not-an-image", {"device": "cpu"})
    assert image_error.value.code == "invalid_image"


def test_common_worker_transport_returns_normalized_real_shape(tmp_path: Path) -> None:
    adapter = PaddleOCRAdapter(
        _config(tmp_path),
        ocr_factory=lambda **_: _FakeEngine(),
        enforce_versions=False,
    )
    server = create_worker_server(
        adapter,
        provider="paddleocr",
        provider_revision="3.7.0",
        model_revision=_MODULE.MODEL_REVISION,
        port=0,
        max_decoded_pixels=100_000,
        max_image_dimension=512,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        health_connection = HTTPConnection("127.0.0.1", port, timeout=5)
        health_connection.request("GET", "/health")
        health_response = health_connection.getresponse()
        health_payload = json.loads(health_response.read())
        health_connection.close()
        assert health_response.status == 200
        assert health_payload["weights_available"] is True
        assert health_payload["real_inference_verified"] is False

        infer_payload = {
            "schema_version": PROTOCOL_VERSION,
            "request_id": "paddle-test-0001",
            "parameters": {
                "device": "cpu",
                "scales": [2.0],
                "minimum_confidence": 0.45,
            },
            "image_base64": base64.b64encode(_image_bytes()).decode("ascii"),
        }
        body = json.dumps(infer_payload).encode("utf-8")
        infer_connection = HTTPConnection("127.0.0.1", port, timeout=5)
        infer_connection.request(
            "POST",
            "/v1/infer",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        infer_response = infer_connection.getresponse()
        response_payload = json.loads(infer_response.read())
        infer_connection.close()

        assert infer_response.status == 200
        assert response_payload["ok"] is True
        assert response_payload["provider"] == "paddleocr"
        assert response_payload["device"] == "cpu"
        assert response_payload["result"]["lines"][0]["text"] == (
            "ERCİYES ÜNİVERSİTESİ"
        )
        assert adapter.health()["real_inference_verified"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        adapter.unload({})
