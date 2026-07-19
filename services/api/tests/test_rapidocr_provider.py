from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image

from atlaslens_api.providers.base import InvocationContext, OutcomeStatus
from atlaslens_api.providers.rapidocr import (
    RapidOCRProfileConfig,
    RapidOCRProvider,
    RapidOCRRuntimeConfig,
)
from atlaslens_api.providers.rapidocr_worker import (
    RapidOCRWorkerFailure,
    RawOCRBatch,
    RawOCRLine,
)
from atlaslens_api.schemas import AnalysisMode
from atlaslens_api.storage import LocalImageHandle


def invocation_context() -> InvocationContext:
    return InvocationContext(
        analysis_id=uuid4(),
        request_id="rapidocr-test",
        mode=AnalysisMode.LOCAL_ONLY,
        cloud_consent=False,
        deadline=datetime.now(UTC) + timedelta(seconds=5),
        cancellation=asyncio.Event(),
    )


def runtime(tmp_path: Path, *, verified: bool = True) -> RapidOCRRuntimeConfig:
    root = tmp_path / "models"
    root.mkdir()
    for name in ("det.onnx", "rec.onnx", "cls.onnx"):
        (root / name).write_bytes(b"verified-test-artifact")
    return RapidOCRRuntimeConfig(
        model_root=root,
        profiles=(
            RapidOCRProfileConfig(
                profile_id="ppocrv6-multi",
                detector_file="det.onnx",
                recognizer_file="rec.onnx",
                classifier_file="cls.onnx",
            ),
        ),
        device="cuda",
        verified=verified,
    )


def image_handle(tmp_path: Path, *, size: tuple[int, int] = (200, 100)) -> LocalImageHandle:
    path = tmp_path / "input.png"
    Image.new("RGB", size, "white").save(path)
    return LocalImageHandle(key="fixture.upload", path=path)


class StubWorker:
    def __init__(self, lines: tuple[RawOCRLine, ...], *, failure: str | None = None) -> None:
        self.lines = lines
        self.failure = failure
        self.closed = False

    async def infer(
        self,
        image_bytes: bytes,
        *,
        timeout_seconds: float,
        cancellation: asyncio.Event,
    ) -> RawOCRBatch:
        assert image_bytes.startswith(b"\x89PNG")
        assert timeout_seconds > 0
        assert not cancellation.is_set()
        if self.failure:
            raise RapidOCRWorkerFailure(self.failure)
        return RawOCRBatch(lines=self.lines, device="cuda", duration_ms=12)

    async def close(self) -> None:
        self.closed = True


def raw_line(
    text: str,
    confidence: float = 0.91,
    polygon: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]
    ] = ((10, 10), (190, 10), (190, 40), (10, 40)),
) -> RawOCRLine:
    return RawOCRLine(
        text=text,
        confidence=confidence,
        polygon=polygon,
        profile_id="ppocrv6-multi",
    )


@pytest.mark.asyncio
async def test_real_adapter_normalizes_boxes_scripts_and_redacts_before_result(
    tmp_path: Path,
) -> None:
    worker = StubWorker(
        (
            raw_line("İstanbul Merkez"),
            raw_line("Contact jane@example.com +90 555 123 45 67 AB123CD"),
            raw_line("Москва"),
        )
    )
    provider = RapidOCRProvider(
        enabled=True,
        runtime=runtime(tmp_path),
        worker=worker,
        dependency_probe=lambda: True,
    )
    outcome = await provider.extract(image_handle(tmp_path), invocation_context())
    assert outcome.status == OutcomeStatus.SUCCEEDED
    assert outcome.value is not None
    assert len(outcome.value.blocks) == 3
    istanbul = next(block for block in outcome.value.blocks if "stanbul" in block.redacted_text)
    assert istanbul.script == "latin"
    assert istanbul.bounding_polygon[0].x == pytest.approx(0.05)
    rendered = repr(outcome.value)
    assert "jane@example.com" not in rendered
    assert "555 123" not in rendered
    assert "AB123CD" not in rendered
    sensitive = next(block for block in outcome.value.blocks if block.sensitive_content)
    assert "[redacted-email]" in sensitive.redacted_text
    assert "[redacted-phone]" in sensitive.redacted_text
    assert "[redacted-identifier]" in sensitive.redacted_text
    assert "Москва" in outcome.value.redacted_snippets
    await provider.close()
    assert worker.closed


@pytest.mark.asyncio
async def test_one_malformed_line_does_not_erase_valid_ocr(tmp_path: Path) -> None:
    worker = StubWorker(
        (
            raw_line("Ankara"),
            raw_line("Outside", polygon=((-1, 0), (1, 0), (1, 1), (0, 1))),
            raw_line("NaN", confidence=float("nan")),
        )
    )
    provider = RapidOCRProvider(
        enabled=True,
        runtime=runtime(tmp_path),
        worker=worker,
        dependency_probe=lambda: True,
    )
    outcome = await provider.extract(image_handle(tmp_path), invocation_context())
    assert outcome.status == OutcomeStatus.SUCCEEDED
    assert outcome.value is not None
    assert [block.redacted_text for block in outcome.value.blocks] == ["Ankara"]


@pytest.mark.asyncio
async def test_unavailable_provider_never_invokes_or_downloads(tmp_path: Path) -> None:
    provider = RapidOCRProvider(enabled=True, runtime=None, dependency_probe=lambda: True)
    outcome = await provider.extract(image_handle(tmp_path), invocation_context())
    assert outcome.status == OutcomeStatus.SKIPPED
    assert outcome.failure is not None
    assert outcome.failure.code == "model_not_installed"
    assert provider.safe_status()["offline"] is True
    assert "private" not in repr(provider.safe_status())


@pytest.mark.asyncio
async def test_worker_timeout_is_safe_and_path_free(tmp_path: Path, caplog) -> None:
    provider = RapidOCRProvider(
        enabled=True,
        runtime=runtime(tmp_path),
        worker=StubWorker((), failure="timeout"),
        dependency_probe=lambda: True,
    )
    outcome = await provider.extract(image_handle(tmp_path), invocation_context())
    assert outcome.status == OutcomeStatus.FAILED
    assert outcome.failure is not None
    assert outcome.failure.code == "inference_timeout"
    assert outcome.failure.subreason_code == "timeout"
    assert str(tmp_path) not in caplog.text


def test_runtime_rejects_traversal_and_unverified_artifacts(tmp_path: Path) -> None:
    unverified = runtime(tmp_path, verified=False)
    with pytest.raises(ValueError, match="not verified"):
        unverified.resolve_profiles()
    unsafe = RapidOCRRuntimeConfig(
        model_root=unverified.model_root,
        profiles=(
            RapidOCRProfileConfig(
                profile_id="unsafe-profile",
                detector_file="../det.onnx",
                recognizer_file="rec.onnx",
            ),
        ),
        verified=True,
    )
    with pytest.raises(ValueError, match="unsafe"):
        unsafe.resolve_profiles()


def test_raw_line_repr_never_contains_text() -> None:
    line = raw_line("private OCR sentinel")
    assert "private OCR sentinel" not in repr(line)
    assert "<redacted>" in repr(line)
