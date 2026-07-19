from __future__ import annotations

import asyncio
import hashlib
import math
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import Field

from atlaslens_api.phase6b.models import (
    GeographicCandidate,
    GeographicProviderCapability,
    GeographicProviderResult,
    Phase6BModel,
    normalize_longitude,
)
from atlaslens_api.phase6b.scheduler import (
    DeviceName,
    HeavyModelScheduler,
    ScheduledExecution,
)
from atlaslens_api.reranking.geo import geodesic_km, percentile, spherical_center
from atlaslens_api.schemas import SceneSegmentationSummary

type PlonkModelKind = Literal["osv5m", "yfcc", "inat"]


class PlonkRoutingConfig(Phase6BModel):
    road_surface_min: float = Field(default=0.08, ge=0, le=1)
    street_total_min: float = Field(default=0.18, ge=0, le=1)
    nature_total_min: float = Field(default=0.55, ge=0, le=1)
    nature_street_max: float = Field(default=0.10, ge=0, le=1)
    nature_built_max: float = Field(default=0.15, ge=0, le=1)


class PlonkRoute(Phase6BModel):
    kind: PlonkModelKind
    model_id: str = Field(min_length=1, max_length=160)
    source_family: Literal["osv5m_family", "yfcc_family", "inat_family"]
    reason_code: str = Field(min_length=1, max_length=120, pattern=r"^[a-z0-9._-]+$")
    street_score: float = Field(ge=0, le=1)
    nature_score: float = Field(ge=0, le=1)
    built_score: float = Field(ge=0, le=1)
    explicit_override: bool = False


class PlonkModelRouter:
    """Deterministically select exactly one PLONK specialization."""

    def __init__(
        self,
        config: PlonkRoutingConfig | None = None,
        *,
        osv_model_id: str = "nicolas-dufour/PLONK_OSV_5M",
        yfcc_model_id: str = "nicolas-dufour/PLONK_YFCC",
        inat_model_id: str = "nicolas-dufour/PLONK_iNaturalist",
    ) -> None:
        self._config = config or PlonkRoutingConfig()
        self._ids: dict[PlonkModelKind, str] = {
            "osv5m": osv_model_id,
            "yfcc": yfcc_model_id,
            "inat": inat_model_id,
        }
        if any(not value or len(value) > 160 for value in self._ids.values()):
            raise ValueError("PLONK model ids must be bounded")

    def route(
        self,
        scene: SceneSegmentationSummary | None,
        *,
        override: PlonkModelKind | None = None,
    ) -> PlonkRoute:
        scores = self._scores(scene)
        if override is not None:
            return self._route(
                override,
                reason="phase6b.plonk.explicit_diagnostic_override",
                scores=scores,
                explicit=True,
            )
        if scene is None or not scene.semantic_label_names_available:
            return self._route(
                "yfcc",
                reason="phase6b.plonk.semantic_scene_unavailable",
                scores=scores,
            )
        street, nature, built, road = scores
        if (
            nature >= self._config.nature_total_min
            and street <= self._config.nature_street_max
            and built <= self._config.nature_built_max
        ):
            return self._route(
                "inat",
                reason="phase6b.plonk.strong_nature_low_built_scene",
                scores=scores,
            )
        if road >= self._config.road_surface_min and street >= self._config.street_total_min:
            return self._route(
                "osv5m",
                reason="phase6b.plonk.street_oriented_scene",
                scores=scores,
            )
        return self._route(
            "yfcc",
            reason="phase6b.plonk.mixed_or_ambiguous_scene",
            scores=scores,
        )

    @staticmethod
    def _scores(
        scene: SceneSegmentationSummary | None,
    ) -> tuple[float, float, float, float]:
        if scene is None:
            return 0.0, 0.0, 0.0, 0.0
        groups = {item.name: item.pixel_ratio for item in scene.scene_groups}
        road = groups.get("road_surface", 0.0)
        built = groups.get("built_environment", 0.0)
        street = min(
            1.0,
            road
            + groups.get("sidewalk", 0.0)
            + groups.get("vehicles", 0.0)
            + groups.get("traffic_infrastructure", 0.0),
        )
        nature = min(1.0, groups.get("vegetation", 0.0) + groups.get("terrain", 0.0))
        return street, nature, built, road

    def _route(
        self,
        kind: PlonkModelKind,
        *,
        reason: str,
        scores: tuple[float, float, float, float],
        explicit: bool = False,
    ) -> PlonkRoute:
        street, nature, built, _ = scores
        family: dict[PlonkModelKind, Literal["osv5m_family", "yfcc_family", "inat_family"]] = {
            "osv5m": "osv5m_family",
            "yfcc": "yfcc_family",
            "inat": "inat_family",
        }
        return PlonkRoute(
            kind=kind,
            model_id=self._ids[kind],
            source_family=family[kind],
            reason_code=reason,
            street_score=street,
            nature_score=nature,
            built_score=built,
            explicit_override=explicit,
        )


@dataclass(frozen=True, slots=True)
class PlonkWorkerOutput:
    samples_degrees: object
    localizability: float | None = None


class PlonkWorker(Protocol):
    """Isolated ``diff-plonk`` 0.4 worker seam using prepared local snapshots."""

    async def load(self, model_id: str, device: DeviceName) -> None: ...

    async def sample(
        self,
        image_bytes: bytes,
        *,
        model_id: str,
        sample_count: int,
        device: DeviceName,
    ) -> PlonkWorkerOutput | object: ...

    async def unload(self, model_id: str, device: DeviceName) -> None: ...


class InvalidPlonkOutput(ValueError):
    pass


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except (ImportError, RuntimeError):
        return False


def _python_samples(value: object) -> object:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return value


def normalize_plonk_samples(
    value: object, *, maximum: int = 256
) -> tuple[tuple[float, float], ...]:
    raw = _python_samples(value)
    if not isinstance(raw, tuple | list):
        raise InvalidPlonkOutput("PLONK samples must be a sequence")
    valid: list[tuple[float, float]] = []
    for item in raw[:maximum]:
        if not isinstance(item, tuple | list) or len(item) != 2:
            continue
        latitude_raw, longitude_raw = item
        if (
            isinstance(latitude_raw, bool)
            or isinstance(longitude_raw, bool)
            or not isinstance(latitude_raw, int | float)
            or not isinstance(longitude_raw, int | float)
        ):
            continue
        latitude, longitude = float(latitude_raw), float(longitude_raw)
        if not math.isfinite(latitude) or not math.isfinite(longitude) or not -90 <= latitude <= 90:
            continue
        valid.append((latitude, normalize_longitude(longitude)))
    if not valid:
        raise InvalidPlonkOutput("PLONK did not return a valid WGS84 sample")
    return tuple(valid)


def cluster_plonk_samples(
    samples: Sequence[tuple[float, float]],
    *,
    radius_km: float = 100.0,
    max_clusters: int = 20,
) -> tuple[GeographicCandidate, ...]:
    if not math.isfinite(radius_km) or not 0 < radius_km <= 2_000:
        raise ValueError("PLONK cluster radius must be positive and finite")
    if not 1 <= max_clusters <= 100:
        raise ValueError("PLONK max clusters must be between 1 and 100")
    if not samples:
        return ()
    parents = list(range(len(samples)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for left in range(len(samples)):
        for right in range(left + 1, len(samples)):
            if geodesic_km(samples[left], samples[right]) <= radius_km:
                union(left, right)
    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for index, sample in enumerate(samples):
        grouped[find(index)].append(sample)
    clusters: list[tuple[int, float, tuple[float, float], list[tuple[float, float]]]] = []
    for members in grouped.values():
        center = spherical_center(members)
        dispersion = percentile([geodesic_km(center, item) for item in members], 0.80)
        clusters.append((len(members), dispersion, center, members))
    clusters.sort(key=lambda item: (-item[0], item[1], item[2][0], item[2][1]))
    candidates: list[GeographicCandidate] = []
    total = len(samples)
    for rank, (support, dispersion, center, members) in enumerate(clusters[:max_clusters], start=1):
        stable_payload = "|".join(
            f"{latitude:.6f}:{longitude:.6f}" for latitude, longitude in sorted(members)
        )
        stable = hashlib.sha256(stable_payload.encode()).hexdigest()[:24]
        candidates.append(
            GeographicCandidate(
                candidate_id=f"plonk-{stable}",
                latitude=center[0],
                longitude=center[1],
                raw_score=support / total,
                provider_rank=rank,
                sample_support=support,
                metadata={
                    "dispersion_km": dispersion,
                    "sample_density": support / total,
                    "score_is_calibrated": False,
                },
            )
        )
    return tuple(candidates)


class PlonkProvider:
    provider_id = "plonk"

    def __init__(
        self,
        *,
        enabled: bool,
        worker: PlonkWorker | None,
        scheduler: HeavyModelScheduler,
        router: PlonkModelRouter | None = None,
        model_revisions: Mapping[str, str] | None = None,
        source_revision: str = "unprepared",
        device: Literal["auto", "cuda", "cpu"] = "auto",
        cuda_available: Callable[[], bool] = _cuda_available,
        sample_count: int = 32,
        timeout_seconds: float = 60.0,
        max_input_bytes: int = 20 * 1024 * 1024,
        cluster_radius_km: float = 100.0,
        estimated_vram_mb: int = 4_500,
        allow_cpu_fallback: bool = True,
    ) -> None:
        if not 1 <= sample_count <= 256:
            raise ValueError("PLONK sample count must be between 1 and 256")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("PLONK timeout must be positive")
        if max_input_bytes <= 0:
            raise ValueError("PLONK input bound must be positive")
        self._enabled = enabled
        self._worker = worker
        self._scheduler = scheduler
        self._router = router or PlonkModelRouter()
        self._model_revisions = dict(model_revisions or {})
        self._source_revision = source_revision
        self._device: DeviceName = (
            "cuda" if device in {"auto", "cuda"} and cuda_available() else "cpu"
        )
        self._sample_count = sample_count
        self._timeout = timeout_seconds
        self._max_input_bytes = max_input_bytes
        self._cluster_radius = cluster_radius_km
        self._estimated_vram_mb = estimated_vram_mb
        self._allow_cpu_fallback = allow_cpu_fallback

    def status(self, route: PlonkRoute | None = None) -> GeographicProviderCapability:
        selected = route or self._router.route(None)
        revision = self._model_revisions.get(selected.model_id, "unprepared")
        if not self._enabled:
            state: Literal["ready", "disabled", "not_installed"] = "disabled"
            reason = "disabled"
        elif self._worker is None:
            state = "not_installed"
            reason = "isolated_worker_not_installed"
        elif revision == "unprepared" or self._source_revision == "unprepared":
            state = "not_installed"
            reason = "model_artifact_not_verified"
        else:
            state = "ready"
            reason = None
        return GeographicProviderCapability(
            provider=self.provider_id,
            available=state == "ready",
            state=state,
            reason_code=reason,
            model_id=selected.model_id,
            model_revision=revision,
            source_revision=self._source_revision,
            device=self._device,
        )

    async def predict(
        self,
        image_bytes: bytes,
        *,
        scene: SceneSegmentationSummary | None,
        override: PlonkModelKind | None = None,
        cancellation: asyncio.Event | None = None,
    ) -> GeographicProviderResult:
        route = self._router.route(scene, override=override)
        capability = self.status(route)
        if not capability.available or self._worker is None:
            return self._outcome(
                route,
                status="disabled" if capability.state == "disabled" else "skipped",
                device=self._device,
                duration_ms=0,
                reason=capability.reason_code or "unavailable",
            )
        worker = self._worker
        if type(image_bytes) is not bytes or not 0 < len(image_bytes) <= self._max_input_bytes:
            return self._outcome(
                route,
                status="failed",
                device=self._device,
                duration_ms=0,
                reason="unsupported_input",
            )
        if cancellation is not None and cancellation.is_set():
            return self._outcome(
                route,
                status="skipped",
                device=self._device,
                duration_ms=0,
                reason="cancelled",
            )
        started = time.monotonic()
        try:
            async with asyncio.timeout(self._timeout):
                execution: ScheduledExecution[object] = await self._scheduler.execute(
                    model_name=route.model_id,
                    requested_device=self._device,
                    estimated_vram_mb=self._estimated_vram_mb,
                    load=lambda device: worker.load(route.model_id, device),
                    infer=lambda device: worker.sample(
                        image_bytes,
                        model_id=route.model_id,
                        sample_count=self._sample_count,
                        device=device,
                    ),
                    unload=lambda device: worker.unload(route.model_id, device),
                    allow_cpu_fallback=self._allow_cpu_fallback,
                )
            raw = execution.value
            batch = raw if isinstance(raw, PlonkWorkerOutput) else PlonkWorkerOutput(raw)
            samples = normalize_plonk_samples(batch.samples_degrees, maximum=self._sample_count)
            candidates = cluster_plonk_samples(samples, radius_km=self._cluster_radius)
            if not candidates:
                raise InvalidPlonkOutput("PLONK clustering produced no candidates")
            if batch.localizability is not None and not math.isfinite(batch.localizability):
                raise InvalidPlonkOutput("PLONK localizability must be finite")
        except TimeoutError:
            return self._outcome(
                route,
                status="timeout",
                device=self._device,
                duration_ms=_elapsed_ms(started),
                reason="inference_timeout",
            )
        except asyncio.CancelledError:
            raise
        except InvalidPlonkOutput:
            return self._outcome(
                route,
                status="failed",
                device=self._device,
                duration_ms=_elapsed_ms(started),
                reason="invalid_model_output",
            )
        except Exception as exc:
            reason = (
                "cuda_out_of_memory"
                if "out of memory" in str(exc).casefold()
                else "worker_unavailable"
            )
            return self._outcome(
                route,
                status="failed",
                device=self._device,
                duration_ms=_elapsed_ms(started),
                reason=reason,
            )
        warnings = ["warning.plonk.sample_density_not_calibrated"]
        discarded = self._sample_count - len(samples)
        if discarded > 0:
            warnings.append("warning.plonk.invalid_samples_discarded")
        if execution.diagnostic.cpu_fallback_used:
            warnings.append("warning.plonk.cuda_oom_cpu_fallback")
        diagnostics: dict[str, str | int | float | bool | None] = {
            "routing_reason": route.reason_code,
            "requested_samples": self._sample_count,
            "valid_samples": len(samples),
            "source_revision": self._source_revision,
            "cpu_fallback_used": execution.diagnostic.cpu_fallback_used,
        }
        if batch.localizability is not None:
            diagnostics["raw_localizability"] = batch.localizability
        return GeographicProviderResult(
            provider=self.provider_id,
            model_id=route.model_id,
            model_revision=self._model_revisions[route.model_id],
            source_family=route.source_family,
            status="completed",
            device=execution.diagnostic.actual_device,
            duration_ms=_elapsed_ms(started),
            score_semantics="sample_density",
            candidates=candidates,
            warnings=tuple(warnings),
            diagnostics=diagnostics,
        )

    def _outcome(
        self,
        route: PlonkRoute,
        *,
        status: Literal["skipped", "disabled", "failed", "timeout"],
        device: DeviceName,
        duration_ms: int,
        reason: str,
    ) -> GeographicProviderResult:
        return GeographicProviderResult(
            provider=self.provider_id,
            model_id=route.model_id,
            model_revision=self._model_revisions.get(route.model_id, "unprepared"),
            source_family=route.source_family,
            status=status,
            device=device,
            duration_ms=duration_ms,
            score_semantics="sample_density",
            reason_code=reason,
            diagnostics={
                "routing_reason": route.reason_code,
                "source_revision": self._source_revision,
            },
        )


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))
