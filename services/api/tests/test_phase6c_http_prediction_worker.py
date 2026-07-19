from __future__ import annotations

import io
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import pytest

from atlaslens_api.evaluation.isolation import IsolatedPredictionRequest
from atlaslens_api.evaluation.phase6c_http import (
    Phase6CHTTPPredictionWorker,
    Phase6CHTTPWorkerConfig,
    Phase6CHTTPWorkerError,
    read_isolated_request,
)

_ANALYSIS_ID = UUID("12345678-1234-5678-9234-567812345678")


def _image(path: Path) -> None:
    path.write_bytes(b"\xff\xd8\xff\xe0" + b"safe-image-bytes")


def _completed(
    *,
    classification: str = "real",
    candidates: list[dict[str, Any]] | None = None,
    abstained: bool = False,
    leakage_status: str = "passed",
    reference_index_version: str | None = "index-v1",
) -> dict[str, Any]:
    selected = candidates
    if selected is None and not abstained:
        selected = [
            {
                "rank": 1,
                "center": {"latitude": 38.72, "longitude": 35.48},
                "radius_km": 18.5,
                "phase5b_assessment": {
                    "relative_rank_score": 0.73,
                    "reranker_version": "phase6c-v1",
                },
                "reverse_geocode": {
                    "country_code": "TR",
                    "region": "Evaluation Region",
                    "city": "Evaluation City",
                },
            }
        ]
    return {
        "id": str(_ANALYSIS_ID),
        "status": "completed",
        "pipeline_version": "phase6c-v1",
        "result_classification": classification,
        "candidates": selected or [],
        "abstention": {"abstained": True} if abstained else None,
        "phase6c": {
            "pipeline_version": "phase6c-v1",
            "fusion_version": "phase6c-v1",
            "reference_index_version": reference_index_version,
            "leakage_audit": {"status": leakage_status},
            "cache_fingerprint": f"phase6c-v1:{'a' * 64}",
        },
    }


def _transport(
    terminal: dict[str, Any], *, captured_upload: list[bytes] | None = None
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            body = request.read()
            if captured_upload is not None:
                captured_upload.append(body)
            return httpx.Response(
                202,
                json={"id": str(_ANALYSIS_ID), "status": "queued"},
            )
        assert request.url.path == f"/api/v1/analyses/{_ANALYSIS_ID}"
        return httpx.Response(200, json=terminal)

    return httpx.MockTransport(handler)


def _worker(
    terminal: dict[str, Any], *, captured_upload: list[bytes] | None = None
) -> Phase6CHTTPPredictionWorker:
    return Phase6CHTTPPredictionWorker(
        Phase6CHTTPWorkerConfig(api_base_url="http://127.0.0.1:8123"),
        transport=_transport(terminal, captured_upload=captured_upload),
    )


def test_maps_real_phase6c_public_candidates_and_sanitizes_upload(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    private_path = tmp_path / "private-city-label-super-secret.jpg"
    _image(private_path)
    uploads: list[bytes] = []
    request = IsolatedPredictionRequest(
        request_id="prediction-opaque-1", image_path=str(private_path)
    )
    terminal = _completed()
    terminal["ocr"] = {"detections": [{"redacted_text": "private-ocr-secret"}]}

    response = _worker(terminal, captured_upload=uploads).predict(request)

    candidate = response.prediction.candidates[0]
    assert response.provider_id == "atlaslens-phase6c-http"
    assert response.model_revision == "phase6c-v1"
    assert candidate.raw_score == 0.73
    assert candidate.score_type == (
        "phase6c-v1:uncalibrated_relative_rank_not_probability"
    )
    assert candidate.uncertainty_radius_km == 18.5
    assert candidate.country_code == "TR"
    assert candidate.region == "Evaluation Region"
    assert candidate.city_or_area == "Evaluation City"
    multipart = uploads[0]
    assert b'evaluation-input.jpg' in multipart
    assert str(private_path).encode() not in multipart
    assert b"private-city-label" not in multipart
    assert b"super-secret" not in multipart
    assert b'name="analysis_mode"' in multipart and b"local_only" in multipart
    assert b'name="cloud_processing_consent"' in multipart and b"false" in multipart
    assert b'name="allow_cloud_assist"' in multipart and b"false" in multipart
    output = response.model_dump_json()
    assert str(private_path) not in output
    assert "super-secret" not in output
    assert "private-ocr-secret" not in output
    assert capsys.readouterr() == ("", "")


def test_isolated_payload_rejects_ground_truth_or_labels() -> None:
    payload = {
        "protocol_version": "atlaslens-isolated-prediction-v1",
        "request_id": "prediction-opaque-2",
        "image_path": "input.jpg",
        "ground_truth_city": "must-not-cross-boundary",
    }

    with pytest.raises(Phase6CHTTPWorkerError, match="request_invalid"):
        read_isolated_request(io.BytesIO(json.dumps(payload).encode()))


@pytest.mark.parametrize(
    "api_base_url",
    (
        "https://127.0.0.1:8000",
        "http://192.0.2.1:8000",
        "http://user:password@127.0.0.1:8000",
        "http://127.0.0.1:8000/?token=secret",
    ),
)
def test_configuration_accepts_only_plain_loopback_origins(api_base_url: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        Phase6CHTTPWorkerConfig(api_base_url=api_base_url)


def test_completed_public_abstention_is_preserved(tmp_path: Path) -> None:
    image = tmp_path / "input.jpg"
    _image(image)

    response = _worker(_completed(candidates=[], abstained=True)).predict(
        IsolatedPredictionRequest(request_id="prediction-3", image_path=str(image))
    )

    assert response.prediction.abstained
    assert response.prediction.failure_code is None
    assert response.prediction.candidates == ()


def test_place_fields_are_not_inferred_without_reverse_geocode(tmp_path: Path) -> None:
    image = tmp_path / "input.jpg"
    _image(image)
    terminal = _completed()
    terminal["candidates"][0]["country_code"] = "TR"
    terminal["candidates"][0]["label"] = "Do Not Parse This City"
    terminal["candidates"][0]["reverse_geocode"] = None

    response = _worker(terminal).predict(
        IsolatedPredictionRequest(request_id="prediction-no-place", image_path=str(image))
    )

    candidate = response.prediction.candidates[0]
    assert candidate.country_code is None
    assert candidate.region is None
    assert candidate.city_or_area is None
    assert "Do Not Parse" not in response.model_dump_json()


@pytest.mark.parametrize(
    ("terminal", "failure_code"),
    (
        (
            {"id": str(_ANALYSIS_ID), "status": "failed"},
            "analysis_failed",
        ),
        (_completed(classification="simulated"), "phase6c_output_invalid"),
        (
            _completed(leakage_status="failed"),
            "reference_index_leakage_not_passed",
        ),
        (
            _completed(candidates=[{
                "rank": 1,
                "center": {"latitude": 0, "longitude": 0},
                "radius_km": 1,
                "reverse_geocode": None,
            }]),
            "candidate_relative_score_unavailable",
        ),
    ),
)
def test_terminal_or_protocol_failures_are_safe(
    tmp_path: Path, terminal: dict[str, Any], failure_code: str
) -> None:
    image = tmp_path / "do-not-log-this-path.jpg"
    _image(image)

    response = _worker(terminal).predict(
        IsolatedPredictionRequest(request_id="prediction-4", image_path=str(image))
    )

    assert response.prediction.failure_code == failure_code
    encoded = response.model_dump_json()
    assert str(image) not in encoded
    assert "do-not-log" not in encoded


def test_poll_timeout_returns_failure_without_exception_details(tmp_path: Path) -> None:
    image = tmp_path / "timeout-private.jpg"
    _image(image)
    now = [0.0]

    def clock() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            request.read()
            return httpx.Response(
                202, json={"id": str(_ANALYSIS_ID), "status": "queued"}
            )
        return httpx.Response(
            200,
            json={"id": str(_ANALYSIS_ID), "status": "processing"},
        )

    worker = Phase6CHTTPPredictionWorker(
        Phase6CHTTPWorkerConfig(
            api_base_url="http://localhost:8123",
            timeout_seconds=0.2,
            request_timeout_seconds=0.1,
            poll_interval_seconds=0.1,
        ),
        transport=httpx.MockTransport(handler),
        clock=clock,
        sleeper=sleep,
    )

    response = worker.predict(
        IsolatedPredictionRequest(request_id="prediction-5", image_path=str(image))
    )

    assert response.prediction.failure_code == "atlaslens_http_timeout"
    assert response.prediction.latency_ms == 200
    assert str(image) not in response.model_dump_json()


def test_http_transport_failure_uses_stable_code(tmp_path: Path) -> None:
    image = tmp_path / "transport-private.jpg"
    _image(image)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("sensitive transport detail", request=request)

    worker = Phase6CHTTPPredictionWorker(
        Phase6CHTTPWorkerConfig(api_base_url="http://127.0.0.1:8123"),
        transport=httpx.MockTransport(handler),
    )
    response = worker.predict(
        IsolatedPredictionRequest(request_id="prediction-6", image_path=str(image))
    )

    assert response.prediction.failure_code == "atlaslens_http_unavailable"
    assert "sensitive" not in response.model_dump_json()
    assert str(image) not in response.model_dump_json()


def test_unexpected_failure_cannot_escape_to_logs(tmp_path: Path) -> None:
    image = tmp_path / "unexpected-private-path.jpg"
    _image(image)

    def handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError(f"must-not-log:{image}")

    worker = Phase6CHTTPPredictionWorker(
        Phase6CHTTPWorkerConfig(api_base_url="http://127.0.0.1:8123"),
        transport=httpx.MockTransport(handler),
    )
    response = worker.predict(
        IsolatedPredictionRequest(request_id="prediction-7", image_path=str(image))
    )

    assert response.prediction.failure_code == "worker_internal_failure"
    assert str(image) not in response.model_dump_json()


def test_worker_cli_help_is_truth_free() -> None:
    root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts" / "phase6c_http_prediction_worker.py"), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0
    assert "--api-base-url" in completed.stdout
    combined = (completed.stdout + completed.stderr).casefold()
    assert "manifest" not in combined
    assert "ground truth" not in combined
    assert "api key" not in combined


def test_worker_cli_writes_exactly_one_protocol_response(tmp_path: Path) -> None:
    image = tmp_path / "private-cli-secret.jpg"
    _image(image)
    uploads: list[bytes] = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler contract
            length = int(self.headers["Content-Length"])
            uploads.append(self.rfile.read(length))
            self._send({"id": str(_ANALYSIS_ID), "status": "queued"}, status=202)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
            assert self.path == f"/api/v1/analyses/{_ANALYSIS_ID}"
            self._send(_completed())

        def _send(self, payload: dict[str, Any], *, status: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = Path(__file__).resolve().parents[3]
    request = IsolatedPredictionRequest(
        request_id="prediction-cli", image_path=str(image)
    )
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "phase6c_http_prediction_worker.py"),
                "--api-base-url",
                f"http://127.0.0.1:{server.server_port}",
            ],
            cwd=root,
            input=request.model_dump_json(),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert completed.returncode == 0
    decoded, end = json.JSONDecoder().raw_decode(completed.stdout)
    assert not completed.stdout[end:]
    assert decoded["request_id"] == "prediction-cli"
    assert len(decoded["prediction"]["candidates"]) == 1
    assert completed.stderr == ""
    assert b"evaluation-input.jpg" in uploads[0]
    assert str(image).encode() not in uploads[0]
    assert "private-cli-secret" not in completed.stdout


def test_worker_module_has_no_truth_or_environment_dependency() -> None:
    root = Path(__file__).resolve().parents[3]
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            root / "services/api/src/atlaslens_api/evaluation/phase6c_http.py",
            root / "scripts/phase6c_http_prediction_worker.py",
        )
    )

    assert "evaluation.holdout" not in sources
    assert "holdoutmanifest" not in sources.casefold()
    assert "os.environ" not in sources
    assert "getenv(" not in sources
