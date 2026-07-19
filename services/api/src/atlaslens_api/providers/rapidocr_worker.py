from __future__ import annotations

import asyncio
import contextlib
import importlib
import math
import multiprocessing
import time
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from typing import Protocol


@dataclass(frozen=True, slots=True, repr=False)
class ResolvedRapidOCRProfile:
    profile_id: str
    detector_path: str
    recognizer_path: str
    classifier_path: str | None
    detector_language: str
    recognizer_language: str
    ocr_version: str
    model_type: str

    def __repr__(self) -> str:
        return (
            "ResolvedRapidOCRProfile(paths=<private>, "
            f"profile_id={self.profile_id!r}, ocr_version={self.ocr_version!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class RawOCRLine:
    text: str
    confidence: float
    polygon: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]
    ]
    profile_id: str

    def __repr__(self) -> str:
        return (
            "RawOCRLine(text=<redacted>, "
            f"confidence={self.confidence!r}, profile_id={self.profile_id!r})"
        )


@dataclass(frozen=True, slots=True)
class RawOCRBatch:
    lines: tuple[RawOCRLine, ...]
    device: str
    duration_ms: int


class RapidOCRWorkerFailure(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class RapidOCRWorker(Protocol):
    async def infer(
        self,
        image_bytes: bytes,
        *,
        timeout_seconds: float,
        cancellation: asyncio.Event,
    ) -> RawOCRBatch: ...

    async def close(self) -> None: ...


class _WorkerConnection(Protocol):
    def send(self, obj: object) -> None: ...

    def recv(self) -> object: ...

    def poll(self, timeout: float = 0.0) -> bool: ...

    def close(self) -> None: ...


def _engine_params(profile: ResolvedRapidOCRProfile, *, device: str) -> dict[str, object]:
    # RapidOCR 3.9 validates these fields as enum instances. Keep the import
    # inside the isolated worker path so an optional missing dependency does
    # not make the main API module unimportable.
    typings = importlib.import_module("rapidocr.utils.typings")
    params: dict[str, object] = {
        "Global.log_level": "error",
        "Global.max_side_len": 2560,
        "Global.return_word_box": False,
        "EngineConfig.onnxruntime.use_cuda": device == "cuda",
        "EngineConfig.onnxruntime.cuda_ep_cfg.device_id": 0,
        "EngineConfig.onnxruntime.cuda_ep_cfg.gpu_mem_limit": 1_610_612_736,
        "EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search": "DEFAULT",
        "Det.engine_type": typings.EngineType.ONNXRUNTIME,
        "Det.lang_type": profile.detector_language,
        "Det.model_type": typings.ModelType(profile.model_type),
        "Det.ocr_version": typings.OCRVersion(profile.ocr_version),
        "Det.model_path": profile.detector_path,
        "Det.limit_side_len": 2560,
        "Det.limit_type": "max",
        "Det.max_candidates": 256,
        "Rec.engine_type": typings.EngineType.ONNXRUNTIME,
        "Rec.lang_type": profile.recognizer_language,
        "Rec.model_type": typings.ModelType(profile.model_type),
        "Rec.ocr_version": typings.OCRVersion(profile.ocr_version),
        "Rec.model_path": profile.recognizer_path,
    }
    if profile.classifier_path is None:
        params["Global.use_cls"] = False
    else:
        params.update(
            {
                "Cls.engine_type": typings.EngineType.ONNXRUNTIME,
                "Cls.model_path": profile.classifier_path,
            }
        )
    return params


def _worker_main(
    connection: _WorkerConnection,
    profiles: tuple[ResolvedRapidOCRProfile, ...],
    device: str,
) -> None:
    try:
        import cv2
        import numpy as np
        RapidOCR = importlib.import_module("rapidocr").RapidOCR

        engines = tuple(
            (profile.profile_id, RapidOCR(params=_engine_params(profile, device=device)))
            for profile in profiles
        )
        connection.send({"type": "ready"})
    except Exception:  # noqa: BLE001 - child reports only a safe code
        with contextlib.suppress(OSError, EOFError):
            connection.send({"type": "error", "code": "worker_initialization_failed"})
        connection.close()
        return
    while True:
        try:
            request = connection.recv()
        except (EOFError, OSError):
            break
        if not isinstance(request, dict) or request.get("type") == "close":
            break
        if request.get("type") != "infer" or not isinstance(request.get("image"), bytes):
            with contextlib.suppress(OSError, EOFError):
                connection.send({"type": "error", "code": "invalid_worker_request"})
            continue
        started = time.monotonic()
        try:
            encoded = np.frombuffer(request["image"], dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            request["image"] = b""
            if image is None:
                raise ValueError("decode")
            lines: list[dict[str, object]] = []
            for profile_id, engine in engines:
                result = engine(image)
                boxes = getattr(result, "boxes", None)
                texts = getattr(result, "txts", None)
                scores = getattr(result, "scores", None)
                if boxes is None or texts is None or scores is None:
                    continue
                for box, text, score in zip(boxes, texts, scores, strict=True):
                    points = tuple(
                        (float(point[0]), float(point[1])) for point in box
                    )
                    if len(points) != 4:
                        continue
                    lines.append(
                        {
                            "text": str(text)[:512],
                            "confidence": float(score),
                            "polygon": points,
                            "profile_id": profile_id,
                        }
                    )
                    if len(lines) >= 128:
                        break
                if len(lines) >= 128:
                    break
            del image, encoded
            connection.send(
                {
                    "type": "result",
                    "lines": lines,
                    "duration_ms": int((time.monotonic() - started) * 1000),
                }
            )
        except Exception:  # noqa: BLE001 - never expose OCR text/path/stack details
            with contextlib.suppress(OSError, EOFError):
                connection.send({"type": "error", "code": "worker_inference_failed"})
    connection.close()


class ProcessRapidOCRWorker:
    """Lazy single-flight worker whose native call can be killed on timeout."""

    def __init__(
        self, profiles: tuple[ResolvedRapidOCRProfile, ...], *, device: str
    ) -> None:
        if not profiles or len(profiles) > 4:
            raise ValueError("RapidOCR requires between one and four profiles")
        if device not in {"cpu", "cuda"}:
            raise ValueError("RapidOCR device must be cpu or cuda")
        self._profiles = profiles
        self._device = device
        self._process: BaseProcess | None = None
        self._connection: _WorkerConnection | None = None
        self._lock = asyncio.Lock()

    async def infer(
        self,
        image_bytes: bytes,
        *,
        timeout_seconds: float,
        cancellation: asyncio.Event,
    ) -> RawOCRBatch:
        if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
            raise ValueError("RapidOCR timeout must be positive and finite")
        async with self._lock:
            started = time.monotonic()
            await self._ensure_started(timeout_seconds, cancellation)
            if cancellation.is_set():
                await self._terminate()
                raise RapidOCRWorkerFailure("cancelled")
            connection = self._connection
            if connection is None:
                raise RapidOCRWorkerFailure("worker_unavailable")
            remaining = max(0.05, timeout_seconds - (time.monotonic() - started))
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(connection.send, {"type": "infer", "image": image_bytes}),
                    timeout=remaining,
                )
                response = await self._receive(
                    timeout_seconds=max(
                        0.05, timeout_seconds - (time.monotonic() - started)
                    ),
                    cancellation=cancellation,
                )
            except (TimeoutError, EOFError, OSError):
                await self._terminate()
                raise RapidOCRWorkerFailure("timeout") from None
            if response.get("type") != "result":
                await self._terminate()
                raise RapidOCRWorkerFailure(str(response.get("code", "worker_failed")))
            duration_value = response.get("duration_ms", 0)
            duration_ms = duration_value if isinstance(duration_value, int) else 0
            return RawOCRBatch(
                lines=self._parse_lines(response.get("lines")),
                device=self._device,
                duration_ms=max(0, duration_ms),
            )

    async def close(self) -> None:
        async with self._lock:
            connection = self._connection
            if connection is not None:
                with contextlib.suppress(OSError, EOFError):
                    await asyncio.to_thread(connection.send, {"type": "close"})
            await self._terminate()

    async def _ensure_started(
        self, timeout_seconds: float, cancellation: asyncio.Event
    ) -> None:
        if self._process is not None and self._process.is_alive():
            return
        await self._terminate()
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        process = context.Process(
            target=_worker_main,
            args=(child, self._profiles, self._device),
            daemon=True,
            name="atlaslens-rapidocr-worker",
        )
        process.start()
        child.close()
        self._process = process
        self._connection = parent
        try:
            response = await self._receive(
                timeout_seconds=timeout_seconds, cancellation=cancellation
            )
        except (TimeoutError, EOFError, OSError):
            await self._terminate()
            raise RapidOCRWorkerFailure("worker_initialization_failed") from None
        if response.get("type") != "ready":
            await self._terminate()
            raise RapidOCRWorkerFailure(str(response.get("code", "worker_initialization_failed")))

    async def _receive(
        self, *, timeout_seconds: float, cancellation: asyncio.Event
    ) -> dict[str, object]:
        connection = self._connection
        if connection is None:
            raise RapidOCRWorkerFailure("worker_unavailable")
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if cancellation.is_set():
                await self._terminate()
                raise RapidOCRWorkerFailure("cancelled")
            if await asyncio.to_thread(connection.poll, 0.05):
                value = await asyncio.to_thread(connection.recv)
                if isinstance(value, dict):
                    return value
                raise RapidOCRWorkerFailure("invalid_worker_output")
            await asyncio.sleep(0)
        raise TimeoutError

    async def _terminate(self) -> None:
        connection, process = self._connection, self._process
        self._connection = None
        self._process = None
        if connection is not None:
            with contextlib.suppress(OSError):
                connection.close()
        if process is not None and process.is_alive():
            process.terminate()
            await asyncio.to_thread(process.join, 2.0)
            if process.is_alive():
                process.kill()
                await asyncio.to_thread(process.join, 2.0)
        if process is not None:
            with contextlib.suppress(ValueError):
                process.close()

    @staticmethod
    def _parse_lines(value: object) -> tuple[RawOCRLine, ...]:
        if not isinstance(value, list):
            raise RapidOCRWorkerFailure("invalid_worker_output")
        lines: list[RawOCRLine] = []
        for item in value[:128]:
            if not isinstance(item, dict):
                continue
            polygon = item.get("polygon")
            if not isinstance(polygon, tuple) or len(polygon) != 4:
                continue
            try:
                points = tuple((float(point[0]), float(point[1])) for point in polygon)
                line = RawOCRLine(
                    text=str(item.get("text", ""))[:512],
                    confidence=float(item.get("confidence", -1)),
                    polygon=(points[0], points[1], points[2], points[3]),
                    profile_id=str(item.get("profile_id", "unknown"))[:80],
                )
            except (TypeError, ValueError, IndexError):
                continue
            lines.append(line)
        return tuple(lines)
