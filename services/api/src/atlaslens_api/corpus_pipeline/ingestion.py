from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic import ValidationError

from .checkpoint import (
    CheckpointStore,
    PipelineCheckpoint,
    PipelineState,
    state_at_or_after,
)
from .errors import (
    AssetSafetyError,
    CheckpointError,
    CorpusPipelineError,
    ManifestValidationError,
    PipelineInterrupted,
    RightsValidationError,
)
from .hashing import inspect_image
from .leakage import build_split_lock, filter_leakage
from .models import (
    AssetRejection,
    IngestedAsset,
    IngestedCorpus,
    IngestionConfig,
    LeakageReport,
    ManifestAsset,
    RightsAdmission,
    SplitLock,
)
from .policy import SourcePolicy
from .safety import (
    atomic_write_model,
    read_bounded_bytes,
    resolve_contained_file,
    resolve_output_file,
    sha256_bytes,
)

INGESTED_FILENAME = "ingested-assets.json"
CORPUS_FILENAME = "corpus.json"
LEAKAGE_FILENAME = "leakage-report.json"
SPLIT_LOCK_FILENAME = "split-lock.json"

_EXTENSION_BY_MIME = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/tiff": ".tiff",
}
_OPAQUE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")


@dataclass(frozen=True, slots=True)
class ManifestReadResult:
    assets: tuple[ManifestAsset, ...]
    rejections: tuple[AssetRejection, ...]
    manifest_sha256: str
    manifest_schema_sha256: str


@dataclass(frozen=True, slots=True)
class CorpusPreparation:
    corpus: IngestedCorpus
    leakage_report: LeakageReport
    split_lock: SplitLock


@dataclass(frozen=True, slots=True)
class IngestionResult:
    corpus: IngestedCorpus
    leakage_report: LeakageReport
    split_lock: SplitLock
    checkpoint: PipelineCheckpoint
    artifact_sha256: Mapping[str, str]


def _load_manifest_schema(path: Path) -> tuple[dict[str, object], str]:
    try:
        if path.is_symlink() or not path.is_file():
            raise ManifestValidationError("manifest_schema_path_rejected")
        payload = read_bounded_bytes(
            path,
            max_bytes=2 * 1024 * 1024,
            error_prefix="manifest_schema",
        )
        raw = json.loads(payload)
    except ManifestValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ManifestValidationError("manifest_schema_invalid") from exc
    if not isinstance(raw, dict):
        raise ManifestValidationError("manifest_schema_invalid")
    expected_id = "https://atlaslens.local/schemas/corpus/manifest-schema-v1.json"
    required_by_model = {
        name for name, field in ManifestAsset.model_fields.items() if field.is_required()
    }
    properties = raw.get("properties")
    required = raw.get("required")
    try:
        schema_properties = set(properties) if isinstance(properties, dict) else set()
        schema_required = set(required) if isinstance(required, list) else set()
        decision_values = set(raw["$defs"]["sourceDecision"]["enum"])
    except (KeyError, TypeError) as exc:
        raise ManifestValidationError("manifest_schema_contract_mismatch") from exc
    expected_decisions = {
        "GO",
        "GO_WITH_ATTRIBUTION",
        "FIRST_PARTY_ONLY",
        "WRITTEN_PERMISSION_REQUIRED",
        "LEGAL_REVIEW_REQUIRED",
        "RESEARCH_ONLY",
        "BLOCKED",
        "UNKNOWN",
    }
    if (
        raw.get("$id") != expected_id
        or raw.get("type") != "object"
        or raw.get("additionalProperties") is not False
        or schema_properties != set(ManifestAsset.model_fields)
        or schema_required != required_by_model
        or decision_values != expected_decisions
    ):
        raise ManifestValidationError("manifest_schema_contract_mismatch")
    return raw, sha256_bytes(payload)


def _parse_manifest_rows(payload: bytes, *, max_rows: int) -> list[object]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeError as exc:
        raise ManifestValidationError("manifest_encoding_invalid") from exc
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        rows = []
        try:
            for line in text.splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ManifestValidationError("manifest_json_invalid") from exc
    else:
        if isinstance(decoded, list):
            rows = decoded
        elif isinstance(decoded, dict):
            rows = [decoded]
        else:
            raise ManifestValidationError("manifest_root_invalid")
    if not rows:
        raise ManifestValidationError("manifest_empty")
    if len(rows) > max_rows:
        raise ManifestValidationError("manifest_row_limit_exceeded")
    return rows


def _safe_rejection_id(row: object, position: int) -> str:
    if isinstance(row, dict):
        candidate = row.get("asset_id")
        if isinstance(candidate, str) and _OPAQUE_ID.fullmatch(candidate):
            return candidate
    return f"row-{position}"


def read_manifest(
    corpus_root: Path,
    manifest_path: Path | str,
    manifest_schema_path: Path,
    *,
    config: IngestionConfig,
) -> ManifestReadResult:
    _, schema_sha256 = _load_manifest_schema(manifest_schema_path)
    manifest = resolve_contained_file(corpus_root, manifest_path)
    payload = read_bounded_bytes(
        manifest,
        max_bytes=config.max_manifest_bytes,
        error_prefix="manifest",
    )
    rows = _parse_manifest_rows(payload, max_rows=config.max_manifest_rows)
    assets: list[ManifestAsset] = []
    rejections: list[AssetRejection] = []
    seen_asset_ids: set[str] = set()
    for position, row in enumerate(rows, start=1):
        asset_id = _safe_rejection_id(row, position)
        try:
            asset = ManifestAsset.model_validate(row)
        except (ValidationError, ValueError):
            rejections.append(
                AssetRejection(
                    asset_id=asset_id,
                    stage="manifest",
                    reason_code="manifest_row_invalid",
                )
            )
            continue
        if asset.asset_id in seen_asset_ids:
            rejections.append(
                AssetRejection(
                    asset_id=asset.asset_id,
                    stage="manifest",
                    reason_code="manifest_asset_id_duplicate",
                )
            )
            continue
        seen_asset_ids.add(asset.asset_id)
        assets.append(asset)
    return ManifestReadResult(
        assets=tuple(sorted(assets, key=lambda item: item.asset_id)),
        rejections=tuple(sorted(rejections, key=lambda item: (item.asset_id, item.reason_code))),
        manifest_sha256=sha256_bytes(payload),
        manifest_schema_sha256=schema_sha256,
    )


def _validate_rights(
    assets: Sequence[ManifestAsset],
    policy: SourcePolicy,
) -> tuple[tuple[tuple[ManifestAsset, RightsAdmission], ...], tuple[AssetRejection, ...]]:
    admitted: list[tuple[ManifestAsset, RightsAdmission]] = []
    rejected: list[AssetRejection] = []
    for asset in assets:
        try:
            admission = policy.enforce(asset)
        except RightsValidationError as exc:
            rejected.append(
                AssetRejection(
                    asset_id=asset.asset_id,
                    stage="rights",
                    reason_code=exc.code,
                )
            )
        else:
            admitted.append((asset, admission))
    return tuple(admitted), tuple(rejected)


def _file_locator(asset: ManifestAsset, config: IngestionConfig) -> str:
    return f"{config.asset_subdirectory}/{asset.asset_id}{_EXTENSION_BY_MIME[asset.mime_type]}"


def _inspect_assets(
    corpus_root: Path,
    admitted: Sequence[tuple[ManifestAsset, RightsAdmission]],
    config: IngestionConfig,
) -> tuple[tuple[IngestedAsset, ...], tuple[AssetRejection, ...]]:
    ingested: list[IngestedAsset] = []
    rejected: list[AssetRejection] = []
    for asset, admission in admitted:
        try:
            if asset.perceptual_hash_algorithm != config.perceptual_hash_algorithm:
                raise AssetSafetyError("perceptual_hash_algorithm_unsupported")
            locator = _file_locator(asset, config)
            path = resolve_contained_file(corpus_root, locator)
            identity = inspect_image(
                path,
                max_bytes=config.max_asset_bytes,
                max_pixels=config.max_asset_pixels,
            )
            if identity.sha256 != asset.image_sha256:
                raise AssetSafetyError("asset_sha256_mismatch")
            if identity.perceptual_hash != asset.perceptual_hash:
                raise AssetSafetyError("asset_perceptual_hash_mismatch")
            if (identity.width_px, identity.height_px) != (asset.width_px, asset.height_px):
                raise AssetSafetyError("asset_dimensions_mismatch")
            if identity.mime_type != asset.mime_type:
                raise AssetSafetyError("asset_mime_type_mismatch")
            ingested.append(
                IngestedAsset(
                    asset_id=asset.asset_id,
                    source_asset_id=asset.source_asset_id,
                    source_tile_id=asset.source_tile_id,
                    parent_source_asset_id=asset.parent_source_asset_id,
                    source_id=admission.source_id,
                    source_name=asset.source_name,
                    file_locator=locator,
                    image_sha256=identity.sha256,
                    perceptual_hash=identity.perceptual_hash,
                    rights_decision=admission.decision,
                    license_identifier=asset.license_identifier,
                    license_url=asset.license_url,
                    attribution_text=asset.attribution_text,
                    provenance_receipt_id=asset.provenance_receipt.receipt_id,
                    provenance_receipt_sha256=asset.provenance_receipt.receipt_sha256,
                    privacy_review_state=cast(
                        Literal[
                            "NOT_DETECTED",
                            "BLURRED_AT_SOURCE",
                            "BLURRED_BY_ATLASLENS",
                        ],
                        asset.personal_data_blur_state,
                    ),
                    role=asset.role,
                    tier=asset.tier,
                    spatial_split=asset.spatial_split,
                    sampling_cell=asset.spatial_split,
                    province_code=asset.province_code,
                    latitude=asset.latitude,
                    longitude=asset.longitude,
                    coordinate_accuracy_m=asset.coordinate_accuracy_m,
                    contributor_or_owner=asset.contributor_or_owner,
                    capture_timestamp=asset.capture_timestamp,
                    capture_run_id=asset.capture_run_id,
                    sequence_id=asset.sequence_id,
                    descriptor_version=config.descriptor_version,
                    index_version=config.index_version,
                )
            )
        except AssetSafetyError as exc:
            rejected.append(
                AssetRejection(
                    asset_id=asset.asset_id,
                    stage="identity",
                    reason_code=exc.code,
                )
            )
    return (
        tuple(sorted(ingested, key=lambda item: item.asset_id)),
        tuple(sorted(rejected, key=lambda item: (item.asset_id, item.reason_code))),
    )


def _raise_if_strict(config: IngestionConfig, rejections: Sequence[AssetRejection]) -> None:
    if config.strict_rejections and rejections:
        raise CorpusPipelineError(rejections[0].reason_code)


def _build_preparation(
    *,
    manifest: ManifestReadResult,
    policy: SourcePolicy,
    config: IngestionConfig,
    corpus_root: Path,
) -> CorpusPreparation:
    all_rejections = list(manifest.rejections)
    _raise_if_strict(config, all_rejections)
    rights_admitted, rights_rejections = _validate_rights(manifest.assets, policy)
    all_rejections.extend(rights_rejections)
    _raise_if_strict(config, rights_rejections)
    ingested, identity_rejections = _inspect_assets(corpus_root, rights_admitted, config)
    all_rejections.extend(identity_rejections)
    _raise_if_strict(config, identity_rejections)
    if not ingested:
        raise CorpusPipelineError("no_assets_accepted")
    leakage = filter_leakage(
        ingested,
        near_duplicate_hamming_threshold=config.near_duplicate_hamming_threshold,
        spatial_leakage_radius_m=config.spatial_leakage_radius_m,
    )
    leakage_rejections = tuple(
        AssetRejection(
            asset_id=asset_id,
            stage=(
                "deduplication"
                if reason in {"exact_duplicate", "near_duplicate"}
                else "leakage"
            ),
            reason_code=reason,
        )
        for asset_id, reason in leakage.excluded_reason_by_asset.items()
    )
    all_rejections.extend(leakage_rejections)
    _raise_if_strict(config, leakage_rejections)
    if not leakage.accepted:
        raise CorpusPipelineError("no_assets_after_leakage_filter")
    corpus = IngestedCorpus(
        manifest_sha256=manifest.manifest_sha256,
        manifest_schema_sha256=manifest.manifest_schema_sha256,
        source_policy_sha256=policy.sha256,
        config_sha256=config.fingerprint,
        assets=leakage.accepted,
        rejections=tuple(
            sorted(all_rejections, key=lambda item: (item.asset_id, item.stage, item.reason_code))
        ),
    )
    return CorpusPreparation(
        corpus=corpus,
        leakage_report=leakage.report,
        split_lock=build_split_lock(
            corpus.assets,
            manifest_sha256=manifest.manifest_sha256,
            config_sha256=config.fingerprint,
        ),
    )


class CorpusIngestor:
    """Rights-aware, idempotent Phase 3B corpus admission through ``SPLIT_LOCKED``."""

    def __init__(
        self,
        *,
        corpus_root: Path,
        work_root: Path,
        checkpoint_root: Path | None = None,
        manifest_schema_path: Path,
        source_policy_path: Path,
        config: IngestionConfig | None = None,
    ) -> None:
        self.corpus_root = corpus_root
        self.work_root = work_root
        self.checkpoint_root = checkpoint_root or work_root
        self.manifest_schema_path = manifest_schema_path
        self.source_policy_path = source_policy_path
        self.config = config or IngestionConfig()

    def prepare(self, manifest_path: Path | str) -> CorpusPreparation:
        """Perform a no-write validation/dedup/split preview for ``--dry-run``."""

        manifest = read_manifest(
            self.corpus_root,
            manifest_path,
            self.manifest_schema_path,
            config=self.config,
        )
        policy = SourcePolicy.load(self.source_policy_path)
        return _build_preparation(
            manifest=manifest,
            policy=policy,
            config=self.config,
            corpus_root=self.corpus_root,
        )

    def ingest(
        self,
        manifest_path: Path | str,
        *,
        checkpoint_path: Path | str,
        run_id: str | None = None,
        resume: bool = False,
        interrupt_before_complete: PipelineState | None = None,
    ) -> IngestionResult:
        manifest = read_manifest(
            self.corpus_root,
            manifest_path,
            self.manifest_schema_path,
            config=self.config,
        )
        policy = SourcePolicy.load(self.source_policy_path)
        resolved_run_id = run_id or f"phase3b1-{manifest.manifest_sha256[:16]}"
        store = CheckpointStore(self.checkpoint_root, checkpoint_path)
        checkpoint_initialized = False
        try:
            checkpoint = store.initialize(
                run_id=resolved_run_id,
                manifest_sha256=manifest.manifest_sha256,
                manifest_schema_sha256=manifest.manifest_schema_sha256,
                source_policy_sha256=policy.sha256,
                config_sha256=self.config.fingerprint,
                resume=resume,
            )
            checkpoint_initialized = True
            self._complete_stage(
                store,
                PipelineState.MANIFEST_VALIDATED,
                interrupt_before_complete=interrupt_before_complete,
            )
            _raise_if_strict(self.config, manifest.rejections)

            rights_admitted, rights_rejections = _validate_rights(manifest.assets, policy)
            self._complete_stage(
                store,
                PipelineState.RIGHTS_VALIDATED,
                interrupt_before_complete=interrupt_before_complete,
            )
            _raise_if_strict(self.config, rights_rejections)

            ingested, identity_rejections = _inspect_assets(
                self.corpus_root,
                rights_admitted,
                self.config,
            )
            _raise_if_strict(self.config, identity_rejections)
            if not ingested:
                raise CorpusPipelineError("no_assets_accepted")
            initial_rejections = tuple(
                sorted(
                    manifest.rejections + rights_rejections + identity_rejections,
                    key=lambda item: (item.asset_id, item.stage, item.reason_code),
                )
            )
            ingested_envelope = IngestedCorpus(
                manifest_sha256=manifest.manifest_sha256,
                manifest_schema_sha256=manifest.manifest_schema_sha256,
                source_policy_sha256=policy.sha256,
                config_sha256=self.config.fingerprint,
                assets=ingested,
                rejections=initial_rejections,
            )
            ingested_path = resolve_output_file(self.work_root, INGESTED_FILENAME)
            ingested_sha256 = atomic_write_model(ingested_path, ingested_envelope)
            self._complete_stage(
                store,
                PipelineState.INGESTED,
                artifact_sha256={INGESTED_FILENAME: ingested_sha256},
                interrupt_before_complete=interrupt_before_complete,
            )

            leakage = filter_leakage(
                ingested,
                near_duplicate_hamming_threshold=self.config.near_duplicate_hamming_threshold,
                spatial_leakage_radius_m=self.config.spatial_leakage_radius_m,
            )
            leakage_rejections = tuple(
                AssetRejection(
                    asset_id=asset_id,
                    stage=(
                        "deduplication"
                        if reason in {"exact_duplicate", "near_duplicate"}
                        else "leakage"
                    ),
                    reason_code=reason,
                )
                for asset_id, reason in leakage.excluded_reason_by_asset.items()
            )
            _raise_if_strict(self.config, leakage_rejections)
            if not leakage.accepted:
                raise CorpusPipelineError("no_assets_after_leakage_filter")
            corpus = ingested_envelope.model_copy(
                update={
                    "assets": leakage.accepted,
                    "rejections": tuple(
                        sorted(
                            initial_rejections + leakage_rejections,
                            key=lambda item: (
                                item.asset_id,
                                item.stage,
                                item.reason_code,
                            ),
                        )
                    ),
                }
            )
            corpus_path = resolve_output_file(self.work_root, CORPUS_FILENAME)
            leakage_path = resolve_output_file(self.work_root, LEAKAGE_FILENAME)
            corpus_sha256 = atomic_write_model(corpus_path, corpus)
            leakage_sha256 = atomic_write_model(leakage_path, leakage.report)
            self._complete_stage(
                store,
                PipelineState.DEDUPLICATED,
                artifact_sha256={
                    CORPUS_FILENAME: corpus_sha256,
                    LEAKAGE_FILENAME: leakage_sha256,
                },
                interrupt_before_complete=interrupt_before_complete,
            )

            split_lock = build_split_lock(
                corpus.assets,
                manifest_sha256=manifest.manifest_sha256,
                config_sha256=self.config.fingerprint,
            )
            split_path = resolve_output_file(self.work_root, SPLIT_LOCK_FILENAME)
            split_sha256 = atomic_write_model(split_path, split_lock)
            checkpoint = self._complete_stage(
                store,
                PipelineState.SPLIT_LOCKED,
                artifact_sha256={SPLIT_LOCK_FILENAME: split_sha256},
                interrupt_before_complete=interrupt_before_complete,
            )
            return IngestionResult(
                corpus=corpus,
                leakage_report=leakage.report,
                split_lock=split_lock,
                checkpoint=checkpoint,
                artifact_sha256={
                    INGESTED_FILENAME: ingested_sha256,
                    CORPUS_FILENAME: corpus_sha256,
                    LEAKAGE_FILENAME: leakage_sha256,
                    SPLIT_LOCK_FILENAME: split_sha256,
                },
            )
        except PipelineInterrupted:
            raise
        except CorpusPipelineError as exc:
            if checkpoint_initialized:
                with suppress(CheckpointError):
                    store.fail(exc.code)
            raise

    @staticmethod
    def _complete_stage(
        store: CheckpointStore,
        target: PipelineState,
        *,
        artifact_sha256: Mapping[str, str] | None = None,
        interrupt_before_complete: PipelineState | None,
    ) -> PipelineCheckpoint:
        checkpoint = store.load()
        additions = dict(artifact_sha256 or {})
        if state_at_or_after(checkpoint.state, target):
            for key, digest in additions.items():
                if checkpoint.artifact_sha256.get(key) != digest:
                    raise CheckpointError("checkpoint_idempotent_artifact_mismatch")
            return checkpoint
        store.begin(target)
        if interrupt_before_complete == target:
            raise PipelineInterrupted("pipeline_interrupted")
        return store.complete(target, artifact_sha256=additions)


def load_ingested_corpus(
    work_root: Path,
    path: Path | str = CORPUS_FILENAME,
) -> IngestedCorpus:
    selected = resolve_contained_file(work_root, path)
    try:
        payload = read_bounded_bytes(
            selected,
            max_bytes=128 * 1024 * 1024,
            error_prefix="ingested_corpus",
        )
        return IngestedCorpus.model_validate(json.loads(payload))
    except (UnicodeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise ManifestValidationError("ingested_corpus_invalid") from exc


def load_split_lock(
    work_root: Path,
    path: Path | str = SPLIT_LOCK_FILENAME,
) -> SplitLock:
    selected = resolve_contained_file(work_root, path)
    try:
        payload = read_bounded_bytes(
            selected,
            max_bytes=128 * 1024 * 1024,
            error_prefix="split_lock",
        )
        return SplitLock.model_validate(json.loads(payload))
    except (UnicodeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise ManifestValidationError("split_lock_invalid") from exc


__all__ = [
    "CORPUS_FILENAME",
    "INGESTED_FILENAME",
    "LEAKAGE_FILENAME",
    "SPLIT_LOCK_FILENAME",
    "CorpusIngestor",
    "CorpusPreparation",
    "IngestionResult",
    "ManifestReadResult",
    "load_ingested_corpus",
    "load_split_lock",
    "read_manifest",
]
