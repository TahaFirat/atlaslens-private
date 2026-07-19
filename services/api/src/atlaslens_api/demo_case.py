from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Collection, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Literal, Protocol
from urllib.parse import parse_qsl, urlsplit
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import ValidationError

from atlaslens_api.cases import (
    DEFAULT_WORKSPACE_ID,
    CaseConflictError,
    CaseInvestigationService,
    CaseNotFoundError,
    CasePurpose,
    CaseRecord,
    CaseSensitivity,
    CaseStatus,
    MediaSourceType,
    MediaStorageState,
    MediaType,
)
from atlaslens_api.mapillary_demo.acquisition import (
    _contained_path,
    _safe_root,
    sha256_file,
)
from atlaslens_api.mapillary_demo.api import (
    MAPILLARY_DEMO_BENCHMARK_VERSION,
    MAPILLARY_DEMO_COVERAGE_LABEL,
    MAPILLARY_DEMO_EVIDENCE_VERSION,
    MAPILLARY_DEMO_RESULT_SEMANTICS,
    MAPILLARY_DEMO_RETRIEVAL_PROVIDER,
    MAPILLARY_DEMO_RETRIEVAL_SCOPE,
    MAPILLARY_DEMO_SIMILARITY_SEMANTICS,
    MAPILLARY_DEMO_SUPPORTED_REGION,
    MapillaryDemoQueryResponse,
    MapillaryDemoStatusResponse,
)
from atlaslens_api.mapillary_demo.errors import MapillarySafetyError
from atlaslens_api.mapillary_demo.models import AcquiredImage
from atlaslens_api.repository import (
    AnalysisRepository,
    DuplicateIdempotencyError,
    StoredAnalysis,
)
from atlaslens_api.schemas import (
    Abstention,
    Analysis,
    AnalysisMode,
    AnalysisStatus,
    Candidate,
    Evidence,
    GeoJsonPoint,
    GeoPoint,
    Progress,
    Provenance,
    RetrievalCoverageContext,
)

INVESTOR_DEMO_CASE_TITLE = "AtlasLens Ankara Pilot — Yatırımcı Özel Demo"
INVESTOR_DEMO_PIPELINE_VERSION = "investor-private-demo-case-v1"
INVESTOR_DEMO_FUSION_POLICY = "mapillary-private-demo-isolated-no-fusion-v1"
INVESTOR_DEMO_ACTOR_ID = "atlaslens-private-demo-system"
INVESTOR_DEMO_RETENTION_POLICY = "runtime-private-temporary-4h"
INVESTOR_DEMO_CASE_ID = uuid5(
    NAMESPACE_URL, "atlaslens:investor-private-demo:canonical-case:v1"
)
INVESTOR_DEMO_ANALYSIS_ID = uuid5(
    NAMESPACE_URL, "atlaslens:investor-private-demo:canonical-analysis:v1"
)
INVESTOR_DEMO_MEDIA_ID = uuid5(
    NAMESPACE_URL, "atlaslens:investor-private-demo:canonical-media:v1"
)
_MAX_QUERY_BYTES = 20 * 1024 * 1024
_MAX_SIDECAR_BYTES = 128 * 1024
_CANONICAL_UNVERIFIED_RADIUS_KM = 25.0
_CASE_DESCRIPTION = (
    "Attribution doğrulaması yapılmış Mapillary Ankara özel pilot materyaliyle, "
    "yalnız yerel MegaLoc referans erişimini gösteren geçici yatırımcı vakası. "
    "Sonuçlar doğrulanmamış ve kalibre edilmemiştir; Türkiye geneli doğruluk, "
    "hukuken sertifikalı delil veya üretim uygunluğu iddiası yoktur."
)
_SOURCE_CONTEXT = (
    "Retained, attribution-verified Mapillary private-pilot query used with explicit "
    "demo authorization. Processing is local-only. Query ground-truth coordinates are "
    "never supplied to retrieval or persisted as inferred evidence. Ankara names pilot "
    "corpus scope only, not inferred truth or Türkiye-wide accuracy."
)


class InvestorDemoCaseError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class InvestorDemoQueryRuntime(Protocol):
    async def status(self) -> MapillaryDemoStatusResponse: ...

    async def query(
        self, image_bytes: bytes, *, analysis_scope: Literal["ankara_reference_pilot"]
    ) -> MapillaryDemoQueryResponse: ...


@dataclass(frozen=True, slots=True)
class RetainedPilotAsset:
    """Rights-cleared query metadata; deliberately excludes ground-truth geometry."""

    path: Path
    source_asset_id: str
    normalized_sha256: str
    byte_size: int
    captured_at: datetime
    source_page_url: str
    attribution_text: str
    creator_id: str


@dataclass(frozen=True, slots=True)
class LockedPilotAssetIdentity:
    """Non-geographic identity fields from the checksum-bound split publication."""

    source_asset_id: str
    raw_sha256: str
    normalized_sha256: str
    perceptual_hash: str
    sequence_id: str | None
    creator_id: str | None


@dataclass(frozen=True, slots=True)
class InvestorDemoPrepareResult:
    status: Literal["created", "existing", "repaired"]
    case_id: UUID
    analysis_id: UUID
    media_id: UUID
    evidence_count: int
    hypothesis_count: int
    abstained: bool
    audit_integrity_valid: bool

    def safe_payload(self) -> dict[str, object]:
        return {
            "status": self.status,
            "case_id": str(self.case_id),
            "analysis_id": str(self.analysis_id),
            "media_id": str(self.media_id),
            "evidence_count": self.evidence_count,
            "hypothesis_count": self.hypothesis_count,
            "abstained": self.abstained,
            "audit_integrity_valid": self.audit_integrity_valid,
            "analysis_mode": "local_only",
            "analysis_scope": "ankara_reference_pilot",
            "coverage_label": MAPILLARY_DEMO_COVERAGE_LABEL,
            "retrieval_scope": MAPILLARY_DEMO_RETRIEVAL_SCOPE,
            "result_semantics": MAPILLARY_DEMO_RESULT_SEMANTICS,
            "confidence_semantics": "uncalibrated_not_probability",
            "uncertainty_radius_semantics": "positive_conservative_not_accuracy",
            "query_bytes_retained": False,
        }


def _safe_time(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    if value.tzinfo is None or value.utcoffset() is None:
        raise InvestorDemoCaseError("demo_clock_invalid")
    return value.astimezone(UTC)


def _sidecar_model(path: Path) -> AcquiredImage:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_SIDECAR_BYTES:
        raise InvestorDemoCaseError("demo_attribution_sidecar_invalid")
    try:
        return AcquiredImage.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError) as exc:
        raise InvestorDemoCaseError("demo_attribution_sidecar_invalid") from exc


def load_retained_pilot_asset(
    pilot_root: Path,
    *,
    approved_root: Path,
    reference_image_ids: Collection[str],
    reference_assets: Collection[LockedPilotAssetIdentity] = (),
    locked_holdout_assets: Mapping[str, LockedPilotAssetIdentity] | None = None,
) -> RetainedPilotAsset:
    """Select one hash-verified retained non-reference query without decoding pixels."""

    try:
        root = _safe_root(pilot_root, approved_root)
        sidecar_root = _contained_path(root, PurePosixPath("retained-sidecars"))
    except MapillarySafetyError as exc:
        raise InvestorDemoCaseError("demo_pilot_boundary_invalid") from exc
    if sidecar_root.is_symlink() or not sidecar_root.is_dir():
        raise InvestorDemoCaseError("demo_retained_assets_missing")
    sidecars = sorted(sidecar_root.glob("*.json"), key=lambda item: item.name.casefold())
    if not 1 <= len(sidecars) <= 20:
        raise InvestorDemoCaseError("demo_retained_assets_missing")

    reference_by_id = {item.source_asset_id: item for item in reference_assets}
    if len(reference_by_id) != len(reference_assets):
        raise InvestorDemoCaseError("demo_reference_identity_invalid")
    if locked_holdout_assets is not None and (
        not reference_by_id or not locked_holdout_assets
    ):
        raise InvestorDemoCaseError("demo_reference_identity_invalid")
    selected: list[RetainedPilotAsset] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for sidecar in sidecars:
        source = _sidecar_model(sidecar)
        if sidecar.stem != source.mapillary_image_id or source.mapillary_image_id in seen_ids:
            raise InvestorDemoCaseError("demo_attribution_sidecar_invalid")
        if source.normalized_sha256 in seen_hashes:
            raise InvestorDemoCaseError("demo_retained_asset_duplicate")
        seen_ids.add(source.mapillary_image_id)
        seen_hashes.add(source.normalized_sha256)
        if (
            source.mapillary_image_id in reference_image_ids
            or source.mapillary_image_id in reference_by_id
            or any(
                source.raw_sha256 == reference.raw_sha256
                or source.normalized_sha256 == reference.normalized_sha256
                or (
                    source.sequence_id is not None
                    and source.sequence_id == reference.sequence_id
                )
                for reference in reference_by_id.values()
            )
        ):
            raise InvestorDemoCaseError("demo_query_reference_leakage")
        if locked_holdout_assets is not None:
            locked = locked_holdout_assets.get(source.mapillary_image_id)
            if locked is None:
                raise InvestorDemoCaseError("demo_query_not_locked_holdout")
            if (
                source.raw_sha256 != locked.raw_sha256
                or source.normalized_sha256 != locked.normalized_sha256
                or source.sequence_id != locked.sequence_id
                or source.creator_id != locked.creator_id
            ):
                raise InvestorDemoCaseError("demo_query_holdout_identity_mismatch")
            if any(
                _perceptual_hamming_distance(
                    locked.perceptual_hash, reference.perceptual_hash
                )
                <= 4
                for reference in reference_by_id.values()
            ):
                raise InvestorDemoCaseError("demo_query_reference_leakage")
        if source.reconciliation_state != "active" or not source.creator_id:
            raise InvestorDemoCaseError("demo_attribution_incomplete")
        if not source.attribution_text.strip():
            raise InvestorDemoCaseError("demo_attribution_incomplete")
        try:
            asset_path = _contained_path(root, source.relative_path)
        except MapillarySafetyError as exc:
            raise InvestorDemoCaseError("demo_retained_asset_invalid") from exc
        if (
            asset_path.is_symlink()
            or not asset_path.is_file()
            or asset_path.suffix.casefold() not in {".jpg", ".jpeg"}
            or asset_path.stat().st_size != source.normalized_byte_size
            or source.normalized_byte_size > _MAX_QUERY_BYTES
        ):
            raise InvestorDemoCaseError("demo_retained_asset_invalid")
        try:
            digest = sha256_file(asset_path, max_bytes=_MAX_QUERY_BYTES)
        except (MapillarySafetyError, OSError) as exc:
            raise InvestorDemoCaseError("demo_retained_asset_invalid") from exc
        if digest != source.normalized_sha256:
            raise InvestorDemoCaseError("demo_retained_asset_hash_mismatch")
        selected.append(
            RetainedPilotAsset(
                path=asset_path,
                source_asset_id=source.mapillary_image_id,
                normalized_sha256=source.normalized_sha256,
                byte_size=source.normalized_byte_size,
                captured_at=source.captured_at.astimezone(UTC),
                source_page_url=source.source_page_url,
                attribution_text=source.attribution_text,
                creator_id=source.creator_id,
            )
        )
    if not selected:
        raise InvestorDemoCaseError("demo_retained_assets_missing")
    result = min(selected, key=lambda item: (item.normalized_sha256, item.source_asset_id))
    if any(
        result.creator_id == reference.creator_id
        for reference in reference_by_id.values()
        if reference.creator_id is not None
    ):
        raise InvestorDemoCaseError("demo_query_reference_lineage_leakage")
    return result


def _perceptual_hamming_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        raise InvestorDemoCaseError("demo_reference_identity_invalid")
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError as exc:
        raise InvestorDemoCaseError("demo_reference_identity_invalid") from exc


def _read_query_bytes(asset: RetainedPilotAsset) -> bytes:
    try:
        if (
            asset.path.is_symlink()
            or not asset.path.is_file()
            or asset.path.stat().st_size != asset.byte_size
        ):
            raise InvestorDemoCaseError("demo_retained_asset_changed")
        with asset.path.open("rb") as stream:
            payload = stream.read(_MAX_QUERY_BYTES + 1)
    except OSError as exc:
        raise InvestorDemoCaseError("demo_retained_asset_changed") from exc
    if not payload or len(payload) != asset.byte_size or len(payload) > _MAX_QUERY_BYTES:
        raise InvestorDemoCaseError("demo_retained_asset_changed")
    if hashlib.sha256(payload).hexdigest() != asset.normalized_sha256:
        raise InvestorDemoCaseError("demo_retained_asset_changed")
    return payload


def _idempotency_hash(asset: RetainedPilotAsset, publication_sha256: str) -> str:
    return hashlib.sha256(
        (
            f"{INVESTOR_DEMO_PIPELINE_VERSION}\n"
            f"{asset.normalized_sha256}\n{publication_sha256}\n"
        ).encode("ascii")
    ).hexdigest()


def _provenance(status: MapillaryDemoStatusResponse) -> Provenance:
    if not status.model_version or not status.index_version:
        raise InvestorDemoCaseError("demo_runtime_identity_incomplete")
    return Provenance(
        provider_id="mapillary-private-demo-retrieval",
        provider_kind="megaloc-attributed-reference-retrieval",
        provider_version=status.index_version,
        execution_boundary="local",
        model_name="gberton/MegaLoc",
        output_schema_version="mapillary-private-demo-query-v1",
    )


def _analysis_from_query(
    query: MapillaryDemoQueryResponse,
    *,
    runtime_status: MapillaryDemoStatusResponse,
    created_at: datetime,
    runtime_ms: int,
) -> Analysis:
    if (
        query.analysis_scope != "ankara_reference_pilot"
        or query.coverage_status != "pilot_eligible"
    ):
        raise InvestorDemoCaseError("demo_query_coverage_invalid")
    retrieval_context = RetrievalCoverageContext(
        analysis_scope=query.analysis_scope,
        coverage_status=query.coverage_status,
        coverage_label=query.coverage_label,
        retrieval_provider=query.retrieval_provider,
        retrieval_scope=query.retrieval_scope,
        result_semantics=query.result_semantics,
        abstained=query.abstained,
        abstention_reason=query.abstention_reason,
        similarity_semantics=query.similarity_semantics,
        supported_region=query.supported_region,
        evidence_version=query.evidence_version,
        benchmark_version=query.benchmark_version,
    )
    if query.status != "completed":
        if query.status == "abstained" and query.reason_code == "no_reference_match":
            provider = _provenance(runtime_status)
            return Analysis(
                id=INVESTOR_DEMO_ANALYSIS_ID,
                status=AnalysisStatus.COMPLETED,
                analysis_mode=AnalysisMode.LOCAL_ONLY,
                created_at=created_at,
                expires_at=created_at + timedelta(hours=4),
                progress=Progress(
                    stage="completed",
                    percent=100,
                    message_key="analysis.completed_with_abstention",
                ),
                image=None,
                evidence=[
                    Evidence(
                        id="mapillary-retrieval-coverage",
                        type="retrieval_coverage",
                        label="Ankara reference collection coverage assessment",
                        display_value=None,
                        confidence=None,
                        confidence_basis=(
                            "Pilot collection eligibility was explicit; no reference "
                            "match was emitted. Coverage is not confidence."
                        ),
                        source="local retrieval coverage gate",
                        sensitive=False,
                        provenance=provider,
                        retrieval_context=retrieval_context,
                    )
                ],
                candidates=[],
                abstention=Abstention(
                    reason_code="insufficient_attributed_reference_evidence",
                    message_key="analysis.abstained.insufficient_evidence",
                ),
                warnings=[
                    "Private Ankara pilot scope only; no Türkiye-wide accuracy claim.",
                    "No attributed reference match was sufficient; abstention is preferred.",
                ],
                timings_ms={"mapillary_private_demo_retrieval": runtime_ms},
                fusion_policy_version=INVESTOR_DEMO_FUSION_POLICY,
                pipeline_version=INVESTOR_DEMO_PIPELINE_VERSION,
                result_classification="real",
            )
        raise InvestorDemoCaseError("demo_query_not_ready")

    provider = _provenance(runtime_status)
    evidence: list[Evidence] = []
    candidates: list[Candidate] = []
    for item in query.candidates:
        if item.confidence is not None or item.confidence_semantics != (
            "uncalibrated_unavailable"
        ):
            raise InvestorDemoCaseError("demo_query_confidence_invalid")
        if not math.isfinite(item.uncertainty_radius_m) or item.uncertainty_radius_m <= 0:
            raise InvestorDemoCaseError("demo_query_uncertainty_invalid")
        evidence_id = f"mapillary-reference-match-{item.rank}"
        contributor = item.contributor or "contributor unavailable in source metadata"
        evidence.append(
            Evidence(
                id=evidence_id,
                type="retrieval_match",
                label=f"Attributed Mapillary reference retrieval match, rank {item.rank}",
                display_value=(
                    f"{contributor}; {item.license_identifier}; {item.source_url}"
                ),
                confidence=None,
                confidence_basis=(
                    f"Raw cosine similarity {item.cosine_similarity:.6f}; uncalibrated "
                    "relative retrieval score, not a probability."
                ),
                source=item.source_url,
                sensitive=False,
                provenance=provider,
                retrieval_context=retrieval_context,
            )
        )
        radius_km = max(
            _CANONICAL_UNVERIFIED_RADIUS_KM, item.uncertainty_radius_m / 1_000.0
        )
        candidates.append(
            Candidate(
                id=f"mapillary-reference-hypothesis-{item.rank}",
                rank=item.rank,
                center=GeoPoint(latitude=item.latitude, longitude=item.longitude),
                geometry=GeoJsonPoint(coordinates=(item.longitude, item.latitude)),
                radius_km=radius_km,
                uncertainty_basis=(
                    "Conservative canonical minimum for an unverified retrieval-model "
                    "hypothesis; the radius is not calibrated accuracy or probability."
                ),
                confidence=None,
                confidence_kind="uncalibrated_score",
                confidence_basis=(
                    f"Raw cosine similarity {item.cosine_similarity:.6f}; ranking signal only."
                ),
                granularity="region",
                country_code=None,
                label=None,
                source="local attributed Mapillary pilot reference retrieval",
                evidence_ids=[evidence_id],
                evidence_summary=(
                    "One attributed reference retrieval match; reference proximity is not "
                    "proof of the query location."
                ),
                provenance=[provider],
                verification_status="unverified_model",
                verified=False,
            )
        )
    if not candidates:
        raise InvestorDemoCaseError("demo_query_empty_result")
    return Analysis(
        id=INVESTOR_DEMO_ANALYSIS_ID,
        status=AnalysisStatus.COMPLETED,
        analysis_mode=AnalysisMode.LOCAL_ONLY,
        created_at=created_at,
        expires_at=created_at + timedelta(hours=4),
        progress=Progress(
            stage="completed", percent=100, message_key="analysis.completed"
        ),
        image=None,
        evidence=evidence,
        candidates=candidates,
        abstention=None,
        warnings=[
            "Private Ankara pilot scope only; no Türkiye-wide accuracy claim.",
            "Raw cosine similarity and candidate ordering are uncalibrated.",
            "Reference proximity is not geographic proof; abstention remains valid.",
            "The audit chain is tamper-evident application history, not certified evidence.",
        ],
        timings_ms={"mapillary_private_demo_retrieval": runtime_ms},
        fusion_policy_version=INVESTOR_DEMO_FUSION_POLICY,
        pipeline_version=INVESTOR_DEMO_PIPELINE_VERSION,
        result_classification="real",
    )


def _validate_existing_analysis(
    stored: StoredAnalysis,
    *,
    expected_idempotency_hash: str,
    runtime_status: MapillaryDemoStatusResponse,
) -> None:
    analysis = stored.analysis
    expected_provider_version = runtime_status.index_version

    def coverage_context_valid(
        value: RetrievalCoverageContext | None, *, abstained: bool
    ) -> bool:
        return bool(
            value is not None
            and value.analysis_scope == "ankara_reference_pilot"
            and value.coverage_status == "pilot_eligible"
            and value.coverage_label == MAPILLARY_DEMO_COVERAGE_LABEL
            and value.retrieval_provider == MAPILLARY_DEMO_RETRIEVAL_PROVIDER
            and value.retrieval_scope == MAPILLARY_DEMO_RETRIEVAL_SCOPE
            and value.result_semantics == MAPILLARY_DEMO_RESULT_SEMANTICS
            and value.similarity_semantics == MAPILLARY_DEMO_SIMILARITY_SEMANTICS
            and value.supported_region == MAPILLARY_DEMO_SUPPORTED_REGION
            and value.evidence_version == MAPILLARY_DEMO_EVIDENCE_VERSION
            and value.benchmark_version == MAPILLARY_DEMO_BENCHMARK_VERSION
            and value.abstained is abstained
            and (value.abstention_reason is not None) is abstained
        )

    def private_provenance(value: Provenance) -> bool:
        return bool(
            value.provider_id == "mapillary-private-demo-retrieval"
            and value.provider_kind == "megaloc-attributed-reference-retrieval"
            and value.provider_version == expected_provider_version
            and value.execution_boundary == "local"
            and value.model_name == "gberton/MegaLoc"
            and value.output_schema_version == "mapillary-private-demo-query-v1"
        )

    evidence_by_id = {item.id: item for item in analysis.evidence}
    candidate_invariants = all(
        candidate.id == f"mapillary-reference-hypothesis-{candidate.rank}"
        and candidate.evidence_ids == [f"mapillary-reference-match-{candidate.rank}"]
        and candidate.label is None
        and candidate.country_code is None
        and candidate.confidence is None
        and candidate.confidence_kind == "uncalibrated_score"
        and candidate.radius_km >= _CANONICAL_UNVERIFIED_RADIUS_KM
        and candidate.verification_status == "unverified_model"
        and not candidate.verified
        and candidate.source == "local attributed Mapillary pilot reference retrieval"
        and candidate.phase4_assessment is None
        and candidate.phase5b_assessment is None
        and candidate.model_prediction is None
        and candidate.geoclip_cluster is None
        and candidate.confidence_assessment is None
        and candidate.reverse_geocode is None
        and len(candidate.provenance) == 1
        and private_provenance(candidate.provenance[0])
        and candidate.evidence_ids[0] in evidence_by_id
        for candidate in analysis.candidates
    )
    evidence_invariants = all(
        item.id == f"mapillary-reference-match-{position}"
        and item.type == "retrieval_match"
        and item.confidence is None
        and not item.sensitive
        and private_provenance(item.provenance)
        and coverage_context_valid(item.retrieval_context, abstained=False)
        and _is_safe_mapillary_source(item.source)
        for position, item in enumerate(analysis.evidence, start=1)
    )
    result_shape_valid = (
        bool(analysis.candidates)
        and analysis.abstention is None
        and len(analysis.evidence) == len(analysis.candidates)
        and candidate_invariants
        and evidence_invariants
    ) or (
        not analysis.candidates
        and len(analysis.evidence) == 1
        and analysis.evidence[0].id == "mapillary-retrieval-coverage"
        and coverage_context_valid(
            analysis.evidence[0].retrieval_context, abstained=True
        )
        and analysis.abstention is not None
        and analysis.abstention.reason_code
        == "insufficient_attributed_reference_evidence"
    )
    if (
        stored.idempotency_hash != expected_idempotency_hash
        or stored.storage_key is not None
        or analysis.id != INVESTOR_DEMO_ANALYSIS_ID
        or analysis.status is not AnalysisStatus.COMPLETED
        or analysis.analysis_mode is not AnalysisMode.LOCAL_ONLY
        or analysis.pipeline_version != INVESTOR_DEMO_PIPELINE_VERSION
        or analysis.fusion_policy_version != INVESTOR_DEMO_FUSION_POLICY
        or analysis.result_classification != "real"
        or analysis.simulation is not None
        or analysis.image is not None
        or analysis.quality is not None
        or analysis.phase5b_diagnostics is not None
        or analysis.scene_analysis is not None
        or analysis.model_predictions is not None
        or analysis.fusion is not None
        or analysis.ocr is not None
        or analysis.cloud_assist is not None
        or analysis.phase6c is not None
        or analysis.provider_comparisons
        or analysis.failure is not None
        or not result_shape_valid
    ):
        raise InvestorDemoCaseError("demo_existing_analysis_conflict")


def _is_safe_mapillary_source(value: str) -> bool:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "www.mapillary.com"
        or parsed.username
        or parsed.password
        or parsed.path != "/app/"
    ):
        return False
    sensitive = {"access_token", "authorization", "signature", "token"}
    return not any(
        key.casefold() in sensitive for key, _ in parse_qsl(parsed.query, keep_blank_values=True)
    )


async def _case_or_none(service: CaseInvestigationService) -> CaseRecord | None:
    try:
        return await service.get_case(INVESTOR_DEMO_CASE_ID)
    except CaseNotFoundError:
        return None


def _validate_case_header(case: CaseRecord) -> None:
    if (
        case.id != INVESTOR_DEMO_CASE_ID
        or case.workspace_id != DEFAULT_WORKSPACE_ID
        or case.title != INVESTOR_DEMO_CASE_TITLE
        or case.description != _CASE_DESCRIPTION
        or case.purpose is not CasePurpose.AUTHORIZED_SECURITY_RESEARCH
        or case.purpose_detail is not None
        or case.source_context != _SOURCE_CONTEXT
        or not case.authorization_attested
        or case.status is not CaseStatus.OPEN
        or case.sensitivity is not CaseSensitivity.STANDARD
        or case.created_by_actor_id != INVESTOR_DEMO_ACTOR_ID
        or case.retention_policy != INVESTOR_DEMO_RETENTION_POLICY
        or case.adjudication_count != 0
        or case.media_count > 1
    ):
        raise InvestorDemoCaseError("demo_existing_case_conflict")


async def _validate_complete_baseline(
    service: CaseInvestigationService,
    *,
    case: CaseRecord,
    stored: StoredAnalysis,
    asset: RetainedPilotAsset,
) -> InvestorDemoPrepareResult | None:
    _validate_case_header(case)
    media, media_total = await service.list_media(INVESTOR_DEMO_CASE_ID)
    if media_total == 0:
        return None
    if media_total != 1:
        raise InvestorDemoCaseError("demo_existing_case_conflict")
    item = media[0]
    if (
        item.id != INVESTOR_DEMO_MEDIA_ID
        or item.analysis_id != INVESTOR_DEMO_ANALYSIS_ID
        or item.sha256 != asset.normalized_sha256
        or item.byte_size != asset.byte_size
        or item.source_url != asset.source_page_url
        or item.storage_state is not MediaStorageState.DELETED_AFTER_ANALYSIS
        or item.materialized_at is None
    ):
        raise InvestorDemoCaseError("demo_existing_media_conflict")
    analysis = stored.analysis
    evidence, evidence_total = await service.list_evidence(INVESTOR_DEMO_CASE_ID)
    hypotheses, hypothesis_total = await service.list_hypotheses(INVESTOR_DEMO_CASE_ID)
    if (
        evidence_total != len(analysis.evidence)
        or hypothesis_total != len(analysis.candidates)
        or len(evidence) != evidence_total
        or len(hypotheses) != hypothesis_total
        or case.evidence_count != evidence_total
        or case.hypothesis_count != hypothesis_total
        or any(
            item.calibration_state.value != "uncalibrated"
            or item.uncertainty_radius_m <= 0
            for item in hypotheses
        )
    ):
        raise InvestorDemoCaseError("demo_existing_materialization_conflict")
    integrity = await service.verify_audit_integrity(INVESTOR_DEMO_CASE_ID)
    if not integrity.valid:
        raise InvestorDemoCaseError("demo_audit_integrity_invalid")
    return InvestorDemoPrepareResult(
        status="existing",
        case_id=INVESTOR_DEMO_CASE_ID,
        analysis_id=INVESTOR_DEMO_ANALYSIS_ID,
        media_id=INVESTOR_DEMO_MEDIA_ID,
        evidence_count=evidence_total,
        hypothesis_count=hypothesis_total,
        abstained=analysis.abstention is not None,
        audit_integrity_valid=True,
    )


async def prepare_investor_demo_case(
    *,
    service: CaseInvestigationService,
    analysis_repository: AnalysisRepository,
    runtime: InvestorDemoQueryRuntime,
    asset: RetainedPilotAsset,
    publication_sha256: str,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> InvestorDemoPrepareResult:
    """Create or verify one canonical, real, local-only investor demo case."""

    if len(publication_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in publication_sha256
    ):
        raise InvestorDemoCaseError("demo_publication_hash_invalid")
    runtime_status = await runtime.status()
    if runtime_status.state != "active" or not runtime_status.available:
        raise InvestorDemoCaseError("demo_runtime_not_ready")

    expected_hash = _idempotency_hash(asset, publication_sha256)
    stored = await analysis_repository.get(INVESTOR_DEMO_ANALYSIS_ID)
    case = await _case_or_none(service)
    if stored is not None:
        _validate_existing_analysis(
            stored,
            expected_idempotency_hash=expected_hash,
            runtime_status=runtime_status,
        )
    if case is not None:
        _validate_case_header(case)
    if stored is not None and case is not None:
        complete = await _validate_complete_baseline(
            service, case=case, stored=stored, asset=asset
        )
        if complete is not None:
            return complete

    created_analysis = stored is None
    if stored is None:
        payload = _read_query_bytes(asset)
        started = perf_counter()
        try:
            query = await runtime.query(
                payload, analysis_scope="ankara_reference_pilot"
            )
        finally:
            del payload
        elapsed_ms = max(0, round((perf_counter() - started) * 1_000))
        analysis = _analysis_from_query(
            query,
            runtime_status=runtime_status,
            created_at=_safe_time(clock),
            runtime_ms=elapsed_ms,
        )
        try:
            await analysis_repository.create(
                analysis, storage_key=None, idempotency_hash=expected_hash
            )
        except DuplicateIdempotencyError:
            stored = await analysis_repository.get(INVESTOR_DEMO_ANALYSIS_ID)
            if stored is None:
                raise InvestorDemoCaseError("demo_analysis_create_conflict") from None
            _validate_existing_analysis(
                stored,
                expected_idempotency_hash=expected_hash,
                runtime_status=runtime_status,
            )
        else:
            stored = await analysis_repository.get(INVESTOR_DEMO_ANALYSIS_ID)
    if stored is None:
        raise InvestorDemoCaseError("demo_analysis_unavailable")
    _validate_existing_analysis(
        stored,
        expected_idempotency_hash=expected_hash,
        runtime_status=runtime_status,
    )

    created_case = case is None
    if case is None:
        try:
            case = await service.create_case(
                record_id=INVESTOR_DEMO_CASE_ID,
                title=INVESTOR_DEMO_CASE_TITLE,
                description=_CASE_DESCRIPTION,
                purpose=CasePurpose.AUTHORIZED_SECURITY_RESEARCH,
                purpose_detail=None,
                sensitivity=CaseSensitivity.STANDARD,
                source_context=_SOURCE_CONTEXT,
                authorization_attested=True,
                created_by_actor_id=INVESTOR_DEMO_ACTOR_ID,
                retention_policy=INVESTOR_DEMO_RETENTION_POLICY,
            )
        except CaseConflictError:
            case = await _case_or_none(service)
            if case is None:
                raise InvestorDemoCaseError("demo_case_create_conflict") from None
    _validate_case_header(case)

    media, media_total = await service.list_media(INVESTOR_DEMO_CASE_ID)
    if media_total == 0:
        source_description = (
            f"Mapillary private-pilot query attribution: {asset.attribution_text}; "
            "CC-BY-SA-4.0; case processing copy is not retained. The governed pilot "
            "source remains outside case storage."
        )
        with suppress(CaseConflictError):
            await service.create_media(
                INVESTOR_DEMO_CASE_ID,
                record_id=INVESTOR_DEMO_MEDIA_ID,
                actor_id=INVESTOR_DEMO_ACTOR_ID,
                media_type=MediaType.IMAGE,
                source_type=MediaSourceType.SOURCE_URL,
                original_filename_display="mapillary-pilot-query.jpg",
                mime_type="image/jpeg",
                byte_size=asset.byte_size,
                sha256=asset.normalized_sha256,
                captured_at=asset.captured_at,
                received_at=_safe_time(clock),
                source_url=asset.source_page_url,
                archive_url=None,
                source_description=source_description,
                authorization_attested=True,
                storage_state=MediaStorageState.DELETED_AFTER_ANALYSIS,
                analysis_id=INVESTOR_DEMO_ANALYSIS_ID,
            )
    elif media_total != 1:
        raise InvestorDemoCaseError("demo_existing_media_conflict")

    materialized = await service.materialize_analysis(
        INVESTOR_DEMO_CASE_ID,
        INVESTOR_DEMO_ANALYSIS_ID,
        media_id=INVESTOR_DEMO_MEDIA_ID,
        actor_id=INVESTOR_DEMO_ACTOR_ID,
    )
    refreshed = await service.get_case(INVESTOR_DEMO_CASE_ID)
    complete = await _validate_complete_baseline(
        service, case=refreshed, stored=stored, asset=asset
    )
    if complete is None:
        raise InvestorDemoCaseError("demo_case_materialization_incomplete")
    status: Literal["created", "existing", "repaired"] = (
        "created" if created_analysis and created_case and materialized.created else "repaired"
    )
    return InvestorDemoPrepareResult(
        status=status,
        case_id=complete.case_id,
        analysis_id=complete.analysis_id,
        media_id=complete.media_id,
        evidence_count=complete.evidence_count,
        hypothesis_count=complete.hypothesis_count,
        abstained=complete.abstained,
        audit_integrity_valid=complete.audit_integrity_valid,
    )


__all__ = [
    "INVESTOR_DEMO_ANALYSIS_ID",
    "INVESTOR_DEMO_CASE_ID",
    "INVESTOR_DEMO_CASE_TITLE",
    "INVESTOR_DEMO_MEDIA_ID",
    "InvestorDemoCaseError",
    "InvestorDemoPrepareResult",
    "LockedPilotAssetIdentity",
    "RetainedPilotAsset",
    "load_retained_pilot_asset",
    "prepare_investor_demo_case",
]
