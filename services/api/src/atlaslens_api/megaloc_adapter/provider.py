"""Offline, rights-gated MegaLoc descriptor provider for corpus construction."""

from __future__ import annotations

import gc
import json
import os
import sys
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, cast

import numpy as np

from atlaslens_api.corpus_index.artifacts import read_json, sha256_path
from atlaslens_api.corpus_index.models import (
    DescriptorSpec,
    FloatMatrix,
    RuntimeKind,
)
from atlaslens_api.megaloc_adapter.errors import MegaLocAdapterError
from atlaslens_api.megaloc_adapter.models import (
    MEGALOC_DESCRIPTOR_DIMENSION,
    MEGALOC_PROVIDER_ID,
    MegaLocAdapterConfig,
    MegaLocReadiness,
    MegaLocSmokeReceipt,
    ResolvedMegaLocDevice,
    load_megaloc_config,
)


class MegaLocBackend(Protocol):
    """Injectable inference boundary; injected instances are permanently test-only."""

    @property
    def device(self) -> ResolvedMegaLocDevice: ...

    def describe_batch(self, locators: Sequence[Path]) -> FloatMatrix: ...

    def recover_cuda_oom(self) -> None: ...

    def close(self) -> None: ...


type MegaLocBackendFactory = Callable[
    [MegaLocAdapterConfig, ResolvedMegaLocDevice], MegaLocBackend
]


@dataclass(frozen=True, slots=True)
class PrivateDemoExecutionMetrics:
    device: ResolvedMegaLocDevice | None
    image_count: int
    batch_call_count: int
    elapsed_seconds: float
    peak_vram_bytes: int | None


class MegaLocDescriptorProvider:
    """Production adapter that cannot activate before receipts and a real smoke pass."""

    def __init__(
        self,
        config: MegaLocAdapterConfig,
        *,
        backend_factory: MegaLocBackendFactory | None = None,
    ) -> None:
        _validate_descriptor_version_binding(config)
        self._config = config
        self._backend_factory = backend_factory
        self._injected_backend = backend_factory is not None
        self._backend: MegaLocBackend | None = None
        self._smoke_verified_in_process = False
        self._full_artifact_verified = False
        self._private_demo_image_count = 0
        self._private_demo_batch_call_count = 0
        self._private_demo_elapsed_seconds = 0.0
        self._spec = DescriptorSpec(
            provider_id=MEGALOC_PROVIDER_ID,
            version=config.descriptor_version,
            dimension=MEGALOC_DESCRIPTOR_DIMENSION,
            artifact_sha256=config.artifact_sha256,
            preprocessing_version=config.preprocessing_id,
        )

    @property
    def spec(self) -> DescriptorSpec:
        return self._spec

    @property
    def runtime_kind(self) -> RuntimeKind:
        if self._injected_backend:
            return "test_only"
        self._ensure_production_ready()
        return "production"

    @property
    def readiness(self) -> MegaLocReadiness:
        try:
            _verify_integrity(self._config, verify_artifact_hash=True)
            self._full_artifact_verified = True
            smoke_verified = (
                self._smoke_verified_in_process or _verify_smoke_receipt(self._config)
            )
            _verify_approval_flags(self._config)
            if not smoke_verified:
                raise MegaLocAdapterError("MEGALOC_SMOKE_NOT_VERIFIED")
        except MegaLocAdapterError as exc:
            return _readiness(
                self._config,
                reason_code=exc.code,
                smoke_verified=self._smoke_verified_in_process
                or _verify_smoke_receipt(self._config),
            )
        return _readiness(self._config, reason_code=None, smoke_verified=True)

    def run_smoke(self, locator: Path) -> MegaLocSmokeReceipt:
        """Run one real descriptor without persisting the image or descriptor."""

        # A caller-authorized smoke proves technical integrity only. It must not
        # convert missing legal or production approvals into readiness.
        _verify_integrity(self._config, verify_artifact_hash=True)
        self._full_artifact_verified = True
        matrix = self._describe_with_reduction((locator,))
        validated = _validate_descriptor_matrix(matrix, rows=1)
        input_hash = sha256_path(_regular_input(locator, self._config.max_input_bytes))
        norm = float(np.linalg.norm(validated[0]))
        self._smoke_verified_in_process = True
        return MegaLocSmokeReceipt(
            artifact_sha256=self._config.artifact_sha256,
            source_revision=self._config.source_revision,
            model_revision=self._config.model_revision,
            descriptor_version=self._config.descriptor_version,
            descriptor_dimension=8448,
            preprocessing_id=self._config.preprocessing_id,
            device=self._backend_instance().device,
            smoke_input_sha256=input_hash,
            descriptor_l2_norm=norm,
        )

    def describe_batch(self, locators: Sequence[Path]) -> FloatMatrix:
        if self._injected_backend:
            if not self._smoke_verified_in_process:
                raise MegaLocAdapterError("MEGALOC_SMOKE_NOT_VERIFIED")
        else:
            self._ensure_production_ready()
        return self._describe_verified_batches(locators)

    def describe_batch_for_private_demo(
        self,
        locators: Sequence[Path],
        *,
        authorization_id: str,
        source_policy_sha256: str,
    ) -> FloatMatrix:
        """Describe a Phase 3B3 private demo batch without promoting production readiness."""

        if authorization_id != "phase3b3-mapillary-private-demo-v1":
            raise MegaLocAdapterError("MEGALOC_PRIVATE_DEMO_NOT_AUTHORIZED")
        if (
            len(source_policy_sha256) != 64
            or source_policy_sha256 != source_policy_sha256.lower()
            or any(character not in "0123456789abcdef" for character in source_policy_sha256)
        ):
            raise MegaLocAdapterError("MEGALOC_PRIVATE_DEMO_POLICY_INVALID")
        if self._injected_backend:
            raise MegaLocAdapterError("MEGALOC_PRIVATE_DEMO_REAL_BACKEND_REQUIRED")
        if (
            self._config.artifact_sha256
            != "d4f9f2bcb60018f91eb6a8e061ed054fd55654e10c2569cf13841ea986ffb4f8"
        ):
            raise MegaLocAdapterError("MEGALOC_PRIVATE_DEMO_ARTIFACT_MISMATCH")
        _verify_integrity(self._config, verify_artifact_hash=True)
        self._full_artifact_verified = True
        if not self._smoke_verified_in_process and not _verify_smoke_receipt(self._config):
            raise MegaLocAdapterError("MEGALOC_SMOKE_NOT_VERIFIED")
        started = perf_counter()
        result = self._describe_verified_batches(locators)
        self._private_demo_elapsed_seconds += perf_counter() - started
        self._private_demo_image_count += len(locators)
        self._private_demo_batch_call_count += 1
        return result

    @property
    def private_demo_metrics(self) -> PrivateDemoExecutionMetrics:
        backend = self._backend
        peak = getattr(backend, "peak_vram_bytes", None) if backend is not None else None
        return PrivateDemoExecutionMetrics(
            device=backend.device if backend is not None else None,
            image_count=self._private_demo_image_count,
            batch_call_count=self._private_demo_batch_call_count,
            elapsed_seconds=self._private_demo_elapsed_seconds,
            peak_vram_bytes=peak if isinstance(peak, int) and not isinstance(peak, bool) else None,
        )

    def reset_private_demo_metrics(self) -> None:
        self._private_demo_image_count = 0
        self._private_demo_batch_call_count = 0
        self._private_demo_elapsed_seconds = 0.0
        backend = self._backend
        reset_peak = getattr(backend, "reset_peak_vram", None) if backend is not None else None
        if callable(reset_peak):
            reset_peak()

    def _describe_verified_batches(self, locators: Sequence[Path]) -> FloatMatrix:
        if not locators:
            return np.empty((0, MEGALOC_DESCRIPTOR_DIMENSION), dtype=np.float32)
        rows: list[FloatMatrix] = []
        for offset in range(0, len(locators), self._config.max_batch_size):
            batch = tuple(locators[offset : offset + self._config.max_batch_size])
            rows.append(
                _validate_descriptor_matrix(self._describe_with_reduction(batch), len(batch))
            )
        return np.ascontiguousarray(np.concatenate(rows, axis=0), dtype=np.float32)

    def close(self) -> None:
        backend, self._backend = self._backend, None
        if backend is not None:
            backend.close()

    def _ensure_production_ready(self) -> None:
        _verify_approval_flags(self._config)
        if not self._full_artifact_verified:
            _verify_integrity(self._config, verify_artifact_hash=True)
            self._full_artifact_verified = True
        if not self._smoke_verified_in_process and not _verify_smoke_receipt(self._config):
            raise MegaLocAdapterError("MEGALOC_SMOKE_NOT_VERIFIED")

    def _backend_instance(self) -> MegaLocBackend:
        if self._backend is None:
            device = _resolve_device(self._config, injected=self._injected_backend)
            factory = self._backend_factory or _create_torch_backend
            self._backend = factory(self._config, device)
        return self._backend

    def _describe_with_reduction(self, locators: Sequence[Path]) -> FloatMatrix:
        for locator in locators:
            _regular_input(locator, self._config.max_input_bytes)
        try:
            return self._backend_instance().describe_batch(locators)
        except Exception as exc:
            if not _is_cuda_oom(exc):
                if isinstance(exc, MegaLocAdapterError):
                    raise
                raise MegaLocAdapterError("MEGALOC_INFERENCE_FAILED") from exc
            self._backend_instance().recover_cuda_oom()
            if not self._config.reduce_batch_on_cuda_oom or len(locators) <= 1:
                raise MegaLocAdapterError("MEGALOC_CUDA_OOM") from exc
            midpoint = len(locators) // 2
            left = self._describe_with_reduction(locators[:midpoint])
            right = self._describe_with_reduction(locators[midpoint:])
            return np.ascontiguousarray(np.concatenate((left, right), axis=0), dtype=np.float32)


def inspect_megaloc_approval(
    config_path: Path | None,
    *,
    project_root: Path,
) -> MegaLocReadiness:
    """Exact offline readiness: no inference; verify the checkpoint hash."""

    if config_path is None:
        return _empty_readiness("MEGALOC_CONFIG_MISSING")
    try:
        config = load_megaloc_config(config_path, project_root=project_root)
    except MegaLocAdapterError as exc:
        return _empty_readiness(exc.code)
    smoke_verified = False
    try:
        _verify_integrity(config, verify_artifact_hash=True)
        smoke_verified = _verify_smoke_receipt(config)
        _verify_approval_flags(config)
        if not smoke_verified:
            raise MegaLocAdapterError("MEGALOC_SMOKE_NOT_VERIFIED")
    except MegaLocAdapterError as exc:
        return _readiness(
            config,
            reason_code=exc.code,
            smoke_verified=smoke_verified or _verify_smoke_receipt(config),
        )
    return _readiness(config, reason_code=None, smoke_verified=True)


class _TorchMegaLocBackend:
    def __init__(
        self,
        config: MegaLocAdapterConfig,
        device: ResolvedMegaLocDevice,
    ) -> None:
        _force_offline_mode()
        _configure_deterministic_cuda(config, device)
        try:
            import torch
            from safetensors.torch import load_file
        except ImportError as exc:
            raise MegaLocAdapterError("MEGALOC_RUNTIME_MISSING") from exc
        try:
            source = str(config.source_dir)
            if source not in sys.path:
                sys.path.insert(0, source)
            from megaloc_model import MegaLoc  # type: ignore[import-not-found]

            if device == "cuda" and not bool(torch.cuda.is_available()):
                raise MegaLocAdapterError("MEGALOC_CUDA_UNAVAILABLE")
            if config.deterministic_algorithms:
                torch.use_deterministic_algorithms(True)
                if hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
            model = MegaLoc()
            state = load_file(str(config.artifact_path), device="cpu")
            model.load_state_dict(state, strict=True)
            del state
            model.requires_grad_(False).eval().to(device)
        except MegaLocAdapterError:
            raise
        except Exception as exc:
            code = "MEGALOC_CUDA_OOM" if _is_cuda_oom(exc) else "MEGALOC_MODEL_LOAD_FAILED"
            raise MegaLocAdapterError(code) from exc
        self._config = config
        self._device = device
        self._torch = torch
        self._model = model

    @property
    def device(self) -> ResolvedMegaLocDevice:
        return self._device

    @property
    def peak_vram_bytes(self) -> int:
        if self._device != "cuda":
            return 0
        return int(self._torch.cuda.max_memory_allocated())

    def reset_peak_vram(self) -> None:
        if self._device == "cuda":
            self._torch.cuda.reset_peak_memory_stats()

    def describe_batch(self, locators: Sequence[Path]) -> FloatMatrix:
        if not locators:
            return np.empty((0, MEGALOC_DESCRIPTOR_DIMENSION), dtype=np.float32)
        tensors: list[Any] = []
        images: list[Any] = []
        try:
            for locator in locators:
                image = _decode_image(locator, self._config)
                images.append(image)
                tensors.append(_preprocess(image, self._config.maximum_edge))
            grouped: dict[tuple[int, int], list[tuple[int, Any]]] = {}
            for position, tensor in enumerate(tensors):
                grouped.setdefault((int(tensor.shape[1]), int(tensor.shape[2])), []).append(
                    (position, tensor)
                )
            output: FloatMatrix = np.empty(
                (len(locators), MEGALOC_DESCRIPTOR_DIMENSION), dtype=np.float32
            )
            with self._torch.inference_mode():
                for group in grouped.values():
                    batch = self._torch.stack([tensor for _, tensor in group]).to(self._device)
                    descriptors = self._model(batch).float().cpu().numpy()
                    for row, (position, _) in enumerate(group):
                        output[position] = descriptors[row]
            return output
        except Exception as exc:
            if _is_cuda_oom(exc):
                raise MegaLocAdapterError("MEGALOC_CUDA_OOM") from exc
            if isinstance(exc, MegaLocAdapterError):
                raise
            raise MegaLocAdapterError("MEGALOC_INFERENCE_FAILED") from exc
        finally:
            for image in images:
                image.close()

    def recover_cuda_oom(self) -> None:
        gc.collect()
        if self._device == "cuda":
            self._torch.cuda.empty_cache()

    def close(self) -> None:
        model, self._model = self._model, None
        del model
        gc.collect()
        if self._device == "cuda":
            self._torch.cuda.empty_cache()


def _create_torch_backend(
    config: MegaLocAdapterConfig,
    device: ResolvedMegaLocDevice,
) -> MegaLocBackend:
    return _TorchMegaLocBackend(config, device)


def _verify_integrity(
    config: MegaLocAdapterConfig,
    *,
    verify_artifact_hash: bool,
) -> None:
    artifact = config.artifact_path
    if not artifact.is_file() or artifact.is_symlink():
        raise MegaLocAdapterError("MEGALOC_ARTIFACT_MISSING")
    if (
        artifact.suffix.casefold() != ".safetensors"
        or artifact.stat().st_size != config.artifact_size
    ):
        raise MegaLocAdapterError("MEGALOC_ARTIFACT_MISMATCH")
    if verify_artifact_hash and sha256_path(artifact) != config.artifact_sha256:
        raise MegaLocAdapterError("MEGALOC_ARTIFACT_HASH_MISMATCH")
    for path, expected_hash, missing_code, mismatch_code in (
        (
            config.source_code_path,
            config.source_code_sha256,
            "MEGALOC_SOURCE_MISSING",
            "MEGALOC_SOURCE_HASH_MISMATCH",
        ),
        (
            config.source_license_path,
            config.source_license_sha256,
            "MEGALOC_LICENSE_MISSING",
            "MEGALOC_LICENSE_HASH_MISMATCH",
        ),
        (
            config.receipt_path,
            config.receipt_sha256,
            "MEGALOC_RECEIPT_MISSING",
            "MEGALOC_RECEIPT_INVALID",
        ),
    ):
        if not path.is_file() or path.is_symlink():
            raise MegaLocAdapterError(missing_code)
        if sha256_path(path) != expected_hash:
            raise MegaLocAdapterError(mismatch_code)
    _verify_install_receipt(config)


def _verify_approval_flags(config: MegaLocAdapterConfig) -> None:
    if not config.approval.source_license_approved or not config.approval.weight_license_approved:
        raise MegaLocAdapterError("MEGALOC_LICENSE_APPROVAL_MISSING")
    if not config.approval.fully_approved:
        raise MegaLocAdapterError("MEGALOC_PRODUCTION_APPROVAL_MISSING")


def _verify_install_receipt(config: MegaLocAdapterConfig) -> None:
    try:
        value = read_json(config.receipt_path)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MegaLocAdapterError("MEGALOC_RECEIPT_INVALID") from exc
    if not isinstance(value, dict):
        raise MegaLocAdapterError("MEGALOC_RECEIPT_INVALID")
    receipt = cast(dict[str, Any], value)
    expected: dict[str, object] = {
        "schema_version": "atlaslens-model-receipt-v1",
        "provider": MEGALOC_PROVIDER_ID,
        "source_revision": config.source_revision,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "weight_file": config.artifact_path.name,
        "weight_size": config.artifact_size,
        "weight_sha256": config.artifact_sha256,
        "license": config.weight_license_spdx,
        "datasets_downloaded": False,
    }
    if any(receipt.get(key) != expected_value for key, expected_value in expected.items()):
        raise MegaLocAdapterError("MEGALOC_RECEIPT_INVALID")


def _verify_smoke_receipt(config: MegaLocAdapterConfig) -> bool:
    path = config.smoke_receipt_path
    if not path.is_file() or path.is_symlink():
        return False
    try:
        value = read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    try:
        receipt = MegaLocSmokeReceipt.from_json(value)
    except ValueError:
        return False
    return bool(
        receipt.artifact_sha256 == config.artifact_sha256
        and receipt.source_revision == config.source_revision
        and receipt.model_revision == config.model_revision
        and receipt.descriptor_version == config.descriptor_version
        and receipt.descriptor_dimension == config.descriptor_dimension
        and receipt.preprocessing_id == config.preprocessing_id
    )


def _readiness(
    config: MegaLocAdapterConfig,
    *,
    reason_code: str | None,
    smoke_verified: bool,
) -> MegaLocReadiness:
    return MegaLocReadiness(
        state="ready" if reason_code is None else "not_ready",
        reason_code=reason_code,
        artifact_present=config.artifact_path.is_file() and not config.artifact_path.is_symlink(),
        artifact_receipt_verified=reason_code not in {
            "MEGALOC_ARTIFACT_MISSING",
            "MEGALOC_ARTIFACT_MISMATCH",
            "MEGALOC_ARTIFACT_HASH_MISMATCH",
            "MEGALOC_RECEIPT_MISSING",
            "MEGALOC_RECEIPT_INVALID",
        },
        source_verified=reason_code not in {
            "MEGALOC_SOURCE_MISSING",
            "MEGALOC_SOURCE_HASH_MISMATCH",
        },
        license_records_verified=reason_code not in {
            "MEGALOC_LICENSE_MISSING",
            "MEGALOC_LICENSE_HASH_MISMATCH",
        },
        approvals_verified=config.approval.fully_approved,
        smoke_verified=smoke_verified,
        descriptor_version=config.descriptor_version,
        descriptor_dimension=config.descriptor_dimension,
        preprocessing_id=config.preprocessing_id,
        artifact_sha256=config.artifact_sha256,
    )


def _empty_readiness(reason_code: str) -> MegaLocReadiness:
    return MegaLocReadiness(
        state="not_ready",
        reason_code=reason_code,
        artifact_present=False,
        artifact_receipt_verified=False,
        source_verified=False,
        license_records_verified=False,
        approvals_verified=False,
        smoke_verified=False,
        descriptor_version=None,
        descriptor_dimension=MEGALOC_DESCRIPTOR_DIMENSION,
        preprocessing_id=None,
        artifact_sha256=None,
    )


def _validate_descriptor_version_binding(config: MegaLocAdapterConfig) -> None:
    required = (
        config.model_revision[:8],
        config.artifact_sha256[:12],
        f"max{config.maximum_edge}",
        "imagenet",
    )
    if any(fragment not in config.descriptor_version for fragment in required):
        raise MegaLocAdapterError("MEGALOC_DESCRIPTOR_VERSION_UNBOUND")


def _validate_descriptor_matrix(value: object, rows: int) -> FloatMatrix:
    try:
        matrix = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise MegaLocAdapterError("MEGALOC_INVALID_DESCRIPTOR") from exc
    if matrix.shape != (rows, MEGALOC_DESCRIPTOR_DIMENSION) or not np.isfinite(matrix).all():
        raise MegaLocAdapterError("MEGALOC_INVALID_DESCRIPTOR")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(norms).all() or not np.allclose(norms, 1.0, atol=1e-3, rtol=0.0):
        raise MegaLocAdapterError("MEGALOC_INVALID_DESCRIPTOR")
    return np.ascontiguousarray(matrix, dtype=np.float32)


def _regular_input(locator: Path, maximum_bytes: int) -> Path:
    if (
        not locator.is_file()
        or locator.is_symlink()
        or locator.stat().st_size <= 0
        or locator.stat().st_size > maximum_bytes
    ):
        raise MegaLocAdapterError("MEGALOC_INVALID_IMAGE")
    return locator


def _resolve_device(
    config: MegaLocAdapterConfig,
    *,
    injected: bool,
) -> ResolvedMegaLocDevice:
    if config.device == "cpu":
        return "cpu"
    if config.device == "cuda":
        return "cuda"
    if injected:
        return "cpu"
    try:
        import torch
    except ImportError as exc:
        raise MegaLocAdapterError("MEGALOC_RUNTIME_MISSING") from exc
    return "cuda" if bool(torch.cuda.is_available()) else "cpu"


def _decode_image(locator: Path, config: MegaLocAdapterConfig) -> Any:
    try:
        from PIL import Image, ImageOps, UnidentifiedImageError

        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            source = Image.open(locator)
            source.load()
        if (
            source.width <= 0
            or source.height <= 0
            or source.width * source.height > config.max_image_pixels
        ):
            source.close()
            raise MegaLocAdapterError("MEGALOC_INVALID_IMAGE")
        normalized = ImageOps.exif_transpose(source).convert("RGB")
        source.close()
        return normalized
    except MegaLocAdapterError:
        raise
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise MegaLocAdapterError("MEGALOC_INVALID_IMAGE") from exc


def _preprocess(image: Any, maximum_edge: int) -> Any:
    from torchvision.transforms import functional as transform  # type: ignore[import-untyped]

    width, height = image.size
    scale = min(1.0, maximum_edge / max(width, height))
    target_width = max(14, round(width * scale / 14) * 14)
    target_height = max(14, round(height * scale / 14) * 14)
    if (target_width, target_height) != (width, height):
        image = transform.resize(image, [target_height, target_width], antialias=True)
    tensor = transform.to_tensor(image)
    return transform.normalize(
        tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )


def _force_offline_mode() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _configure_deterministic_cuda(
    config: MegaLocAdapterConfig,
    device: ResolvedMegaLocDevice,
) -> None:
    if not config.deterministic_algorithms or device != "cuda":
        return
    configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if configured is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    elif configured not in {":4096:8", ":16:8"}:
        raise MegaLocAdapterError("MEGALOC_DETERMINISM_CONFIG_INVALID")


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, MegaLocAdapterError) and exc.code == "MEGALOC_CUDA_OOM":
        return True
    text = f"{type(exc).__name__} {exc}".casefold()
    return "outofmemory" in text or "out of memory" in text
