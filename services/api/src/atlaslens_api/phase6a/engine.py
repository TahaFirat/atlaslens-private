from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from atlaslens_api.fusion import CandidateBatch, CandidateDraft, provenance_for
from atlaslens_api.global_prediction.reverse_geocoding import ReverseGeocodedCluster
from atlaslens_api.phase6a.config import Phase6AHybridConfig
from atlaslens_api.providers.base import OCRResult, ProviderDescriptor
from atlaslens_api.reranking.geo import geodesic_km
from atlaslens_api.schemas import (
    Evidence,
    GeoPoint,
    Phase5BAssessment,
    Phase5BDiagnostics,
    Phase5BScoreContribution,
    PlaceEvidenceSummary,
    ProviderRunDiagnostic,
    QualitySummary,
    ReverseGeocodeSummary,
    SceneSegmentationSummary,
    UncalibratedConfidenceSummary,
)

_PLACE_RADIUS_KM = {
    "country": 1_500.0,
    "domain_suffix": 1_500.0,
    "region": 350.0,
    "city": 100.0,
    "district": 50.0,
    "road": 50.0,
    "airport": 50.0,
    "station": 50.0,
    "public_landmark": 25.0,
    "public_institution": 50.0,
}


@dataclass(frozen=True, slots=True)
class Phase6AHybridResult:
    batch: CandidateBatch
    evidence: tuple[Evidence, ...]
    diagnostics: Phase5BDiagnostics
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ScoredCluster:
    enriched: ReverseGeocodedCluster
    score: float
    assessment: Phase5BAssessment
    matched_places: tuple[PlaceEvidenceSummary, ...]
    evidence_ids: tuple[str, ...]
    evidence: tuple[Evidence, ...]


class Phase6AHybridEvidenceEngine:
    """Transparent GeoCLIP-first cluster ranking with real optional evidence only."""

    provider_id = "phase6a-hybrid-evidence-engine"

    def __init__(self, config: Phase6AHybridConfig) -> None:
        self._config = config
        self._descriptor = ProviderDescriptor(
            id=self.provider_id,
            kind="candidate_reranking",
            version=config.version,
            execution_boundary="local",
            criticality="optional",
            available=True,
        )

    @property
    def version(self) -> str:
        return str(self._config.version)

    def rank(
        self,
        clusters: Sequence[ReverseGeocodedCluster],
        *,
        geoclip_descriptor: ProviderDescriptor,
        ocr_descriptor: ProviderDescriptor,
        ocr: OCRResult | None,
        quality: QualitySummary | None,
        segmentation: SceneSegmentationSummary | None,
        providers: Sequence[ProviderRunDiagnostic],
        partial_failures: Sequence[str] = (),
    ) -> Phase6AHybridResult:
        ocr_result = ocr or OCRResult(redacted_snippets=[], blocks=[], place_matches=[])
        scored = [
            self._score_cluster(
                item,
                geoclip_descriptor=geoclip_descriptor,
                ocr_descriptor=ocr_descriptor,
                ocr=ocr_result,
                segmentation=segmentation,
            )
            for item in clusters
        ]
        scored.sort(
            key=lambda item: (
                -item.score,
                min(item.enriched.cluster.summary.member_ranks),
                item.enriched.cluster.summary.cluster_id,
            )
        )
        selected = [item for item in scored if item.assessment.classification != "contradicted"][
            : self._config.max_candidates
        ]
        failed_provider_count = sum(
            item.status == "failed" and item.provider_type != "reverse_geocoding"
            for item in providers
        )
        drafts: list[CandidateDraft] = []
        evidence: dict[str, Evidence] = {}
        for index, ranked in enumerate(selected):
            for item in ranked.evidence:
                evidence[item.id] = item
            confidence = self._confidence(
                ranked,
                margin=(
                    ranked.score - selected[index + 1].score if index + 1 < len(selected) else 0.0
                ),
                quality=quality,
                failed_provider_count=failed_provider_count,
            )
            drafts.append(
                self._candidate(
                    ranked,
                    confidence=confidence,
                    geoclip_descriptor=geoclip_descriptor,
                    ocr_descriptor=ocr_descriptor,
                )
            )
        warning_values = [
            *(item.warning for item in clusters if item.warning is not None),
            *partial_failures,
        ]
        if selected:
            warning_values.insert(0, "warning.confidence_not_calibrated")
        warnings = list(dict.fromkeys(warning_values))
        diagnostics = Phase5BDiagnostics(
            reranker_version="phase6a-v1",
            providers=list(providers)[:16],
            partial_failures=list(dict.fromkeys(partial_failures))[:24],
        )
        return Phase6AHybridResult(
            batch=CandidateBatch(provider_id=self.provider_id, candidates=tuple(drafts)),
            evidence=tuple(sorted(evidence.values(), key=lambda item: item.id)),
            diagnostics=diagnostics,
            warnings=tuple(warnings[:24]),
        )

    def _score_cluster(
        self,
        enriched: ReverseGeocodedCluster,
        *,
        geoclip_descriptor: ProviderDescriptor,
        ocr_descriptor: ProviderDescriptor,
        ocr: OCRResult,
        segmentation: SceneSegmentationSummary | None,
    ) -> _ScoredCluster:
        cluster = enriched.cluster
        matches: list[tuple[PlaceEvidenceSummary, float]] = []
        contradictions: list[str] = []
        for match in ocr.place_matches:
            if match.evidence_strength < self._config.minimum_ocr_evidence:
                continue
            radius = _PLACE_RADIUS_KM[match.match_type]
            distance = geodesic_km(
                (cluster.latitude, cluster.longitude),
                (match.center.latitude, match.center.longitude),
            )
            if distance <= radius:
                proximity = max(0.5, 1.0 - distance / max(radius, 1e-9))
                matches.append((match, match.evidence_strength * proximity))
            elif (
                match.evidence_strength >= self._config.strong_ocr_threshold
                and match.ambiguity_count <= 3
                and distance > radius * 2
            ):
                contradictions.append("phase6a.contradiction.strong_ocr_place_elsewhere")
        matches.sort(
            key=lambda item: (
                -item[1],
                item[0].ambiguity_count,
                item[0].normalized_name.casefold(),
            )
        )
        best_ocr = matches[0][1] if matches else 0.0
        language_value, language_reason = self._language_consistency(ocr, enriched)
        if language_reason == "phase6a.language.specific_contradiction":
            contradictions.append(language_reason)
        features = {
            "geoclip_cluster": cluster.summary.cluster_support,
            "ocr_place_match": best_ocr,
            "language_consistency": max(0.0, language_value),
        }
        breakdown = [
            Phase5BScoreContribution(
                feature=name,
                raw_value=value,
                weight=self._config.weights[name],
                contribution=value * self._config.weights[name],
                reason_code={
                    "geoclip_cluster": "phase6a.feature.geoclip_cluster_support",
                    "ocr_place_match": "phase6a.feature.real_ocr_place_proximity",
                    "language_consistency": "phase6a.feature.specific_language_consistency",
                }[name],
            )
            for name, value in features.items()
        ]
        # Segmentation is intentionally represented as zero-weight descriptive
        # evidence until a reviewed geographic reference database exists.
        breakdown.append(
            Phase5BScoreContribution(
                feature="segmentation_descriptive_only",
                raw_value=1.0 if segmentation is not None else 0.0,
                weight=0.0,
                contribution=0.0,
                reason_code="phase6a.feature.segmentation_has_no_geographic_weight",
            )
        )
        penalty_strength = min(
            1.0,
            (0.75 if contradictions else 0.0) + (max(0.0, -language_value) * 0.25),
        )
        if penalty_strength:
            breakdown.append(
                Phase5BScoreContribution(
                    feature="contradiction_penalty",
                    raw_value=penalty_strength,
                    weight=-self._config.contradiction_penalty,
                    contribution=-penalty_strength * self._config.contradiction_penalty,
                    reason_code="phase6a.feature.explicit_contradiction_penalty",
                )
            )
        score = min(1.0, max(0.0, sum(item.contribution for item in breakdown)))
        matched_places = tuple(item[0] for item in matches[:4])
        supports = ["phase6a.support.geoclip_cluster"]
        if best_ocr > 0:
            supports.append("phase6a.support.ocr_place_match")
        if language_value > 0:
            supports.append("phase6a.support.specific_language")
        classification = (
            "contradicted"
            if penalty_strength >= 0.75 and not matches
            else "place_supported"
            if matches
            else "model_only"
        )
        evidence = [self._cluster_evidence(cluster, geoclip_descriptor)]
        evidence.extend(self._place_evidence(match, ocr_descriptor) for match in matched_places)
        assessment = Phase5BAssessment(
            classification=classification,
            relative_rank_score=score,
            reranker_version="phase6a-v1",
            score_breakdown=breakdown,
            provider_diversity=1 + int(bool(matches)),
            source_diversity=1 + int(bool(matches)),
            place_matches=list(matched_places),
            supports=supports,
            contradictions=list(dict.fromkeys(contradictions)),
            limitations=[
                "phase6a.relative_rank_is_not_probability",
                "phase6a.reverse_geocoding_names_coordinates_only",
                "phase6a.segmentation_is_descriptive_only",
            ],
        )
        return _ScoredCluster(
            enriched=enriched,
            score=score,
            assessment=assessment,
            matched_places=matched_places,
            evidence_ids=tuple(item.id for item in evidence),
            evidence=tuple(evidence),
        )

    def _candidate(
        self,
        scored: _ScoredCluster,
        *,
        confidence: UncalibratedConfidenceSummary,
        geoclip_descriptor: ProviderDescriptor,
        ocr_descriptor: ProviderDescriptor,
    ) -> CandidateDraft:
        cluster = scored.enriched.cluster
        strongest = scored.matched_places[0] if scored.matched_places else None
        reverse = scored.enriched.place
        has_ocr_support = strongest is not None
        if has_ocr_support:
            assert strongest is not None
            radius = max(
                self._config.ocr_supported_radius_floor_km,
                min(_PLACE_RADIUS_KM[strongest.match_type], cluster.dispersion_km * 1.5),
            )
        else:
            radius = max(
                self._config.model_only_radius_floor_km,
                cluster.dispersion_km * 2.0,
            )
        label = (
            strongest.normalized_name
            if strongest is not None
            else reverse.label
            if reverse
            else None
        )
        country_code = (
            strongest.country_code
            if strongest is not None
            else reverse.country_code
            if reverse
            else None
        )
        provenance = [provenance_for(geoclip_descriptor), provenance_for(self._descriptor)]
        if has_ocr_support:
            provenance.append(provenance_for(ocr_descriptor))
        stable = hashlib.sha256(cluster.summary.cluster_id.encode("utf-8")).hexdigest()[:20]
        return CandidateDraft(
            id=f"candidate-phase6a-{stable}",
            center=GeoPoint(latitude=cluster.latitude, longitude=cluster.longitude),
            radius_km=radius,
            uncertainty_basis=(
                "phase6a.cluster_dispersion_and_real_ocr_place_radius"
                if has_ocr_support
                else "phase6a.uncalibrated_geoclip_cluster_floor"
            ),
            confidence=None,
            confidence_kind="uncalibrated_score",
            confidence_basis="phase6a.honest_label_without_probability",
            granularity=self._granularity(strongest, reverse),
            country_code=country_code,
            label=label,
            source=self.provider_id,
            evidence_ids=list(scored.evidence_ids),
            evidence_summary="evidence.phase6a_hybrid_cluster",
            provenance=provenance,
            verification_status="corroborated" if has_ocr_support else "unverified_model",
            verified=False,
            phase5b_assessment=scored.assessment,
            geoclip_cluster=cluster.summary,
            confidence_assessment=confidence,
            reverse_geocode=(
                ReverseGeocodeSummary(
                    country=reverse.country,
                    country_code=reverse.country_code,
                    region=reverse.region,
                    city=reverse.city,
                    district=None,
                    display_name=reverse.label,
                    provider=reverse.source,
                    dataset_version=reverse.dataset_version,
                    license=reverse.license,
                )
                if reverse is not None
                else None
            ),
        )

    def _confidence(
        self,
        scored: _ScoredCluster,
        *,
        margin: float,
        quality: QualitySummary | None,
        failed_provider_count: int,
    ) -> UncalibratedConfidenceSummary:
        points = 0
        basis: list[str] = []
        cluster = scored.enriched.cluster
        if cluster.summary.member_count >= 3:
            points += 1
            basis.append("dense_candidate_cluster")
        else:
            basis.append("limited_candidate_consensus")
        if margin >= 0.08:
            points += 1
            basis.append("candidate_margin")
        else:
            basis.append("small_candidate_margin")
        strong_ocr = any(
            item.evidence_strength >= self._config.strong_ocr_threshold
            for item in scored.matched_places
        )
        if strong_ocr:
            points += 2
            basis.append("strong_ocr_match")
        elif scored.matched_places:
            points += 1
            basis.append("weak_ocr_support")
        else:
            basis.append("no_strong_ocr_match")
        if not scored.assessment.contradictions:
            points += 1
            basis.append("no_contradictions")
        else:
            points = max(0, points - 2)
            basis.append("contradiction_count")
        if quality is not None:
            mean_quality = (
                quality.blur_score
                + quality.brightness_score
                + quality.contrast_score
                + quality.resolution_score
            ) / 4.0
            if mean_quality >= 0.55:
                points += 1
                basis.append("image_quality")
            elif mean_quality < 0.35:
                points = max(0, points - 1)
                basis.append("low_image_quality")
        if failed_provider_count:
            points = max(0, points - min(2, failed_provider_count))
            basis.append("provider_failures")
        if len(scored.matched_places) == 0:
            basis.append("only_one_geographic_model")
        label = (
            "very_high"
            if points >= 6 and strong_ocr and not scored.assessment.contradictions
            else "high"
            if points >= 5
            else "medium"
            if points >= 3
            else "low"
        )
        return UncalibratedConfidenceSummary(
            label=label,
            basis=list(dict.fromkeys(basis))[:16],
        )

    def _language_consistency(
        self, ocr: OCRResult, enriched: ReverseGeocodedCluster
    ) -> tuple[float, str | None]:
        place = enriched.place
        if place is None or place.country_code is None:
            return 0.0, None
        specific = {
            hint.casefold()
            for block in ocr.blocks
            if block.confidence >= self._config.strong_ocr_threshold
            for hint in block.language_hints
            if hint.casefold() in self._config.language_country_support
        }
        if not specific:
            return 0.0, None
        supported = {
            country
            for language in specific
            for country in self._config.language_country_support[language]
        }
        if place.country_code in supported:
            return 1.0, "phase6a.language.specific_consistency"
        return -1.0, "phase6a.language.specific_contradiction"

    @staticmethod
    def _cluster_evidence(cluster: object, descriptor: ProviderDescriptor) -> Evidence:
        from atlaslens_api.global_prediction.clustering import GeoClipCandidateCluster

        if not isinstance(cluster, GeoClipCandidateCluster):
            raise TypeError("GeoCLIP cluster expected")
        return Evidence(
            id=f"evidence-{cluster.summary.cluster_id}",
            type="global_prediction_cluster",
            label="evidence.geoclip_candidate_cluster",
            display_value=f"{cluster.summary.member_count} nearby raw candidates",
            confidence=None,
            confidence_basis="phase6a.raw_similarity_is_not_confidence",
            source=descriptor.id,
            sensitive=False,
            provenance=provenance_for(descriptor),
        )

    @staticmethod
    def _place_evidence(match: PlaceEvidenceSummary, descriptor: ProviderDescriptor) -> Evidence:
        stable = hashlib.sha256(
            (
                f"{match.normalized_name}|{match.country_code}|"
                f"{match.center.latitude:.6f}|{match.center.longitude:.6f}"
            ).encode()
        ).hexdigest()[:24]
        return Evidence(
            id=f"evidence-phase6a-ocr-place-{stable}",
            type="ocr_place_match",
            label="evidence.ocr_public_place_match",
            display_value=match.normalized_name,
            confidence=match.evidence_strength,
            confidence_basis="phase6a.ocr_evidence_strength_not_probability",
            source=descriptor.id,
            sensitive=False,
            provenance=provenance_for(descriptor),
        )

    @staticmethod
    def _granularity(match: PlaceEvidenceSummary | None, reverse: object | None) -> str:
        if match is not None:
            if match.match_type == "country":
                return "country"
            if match.match_type == "region":
                return "region"
            return "city"
        if reverse is not None:
            from atlaslens_api.gazetteer import ResolvedPlace

            if isinstance(reverse, ResolvedPlace):
                if reverse.city:
                    return "city"
                if reverse.region:
                    return "region"
                if reverse.country:
                    return "country"
        return "broad_area"
