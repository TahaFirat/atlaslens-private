from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atlaslens_api.phase6c.fusion import (
    OCRPlaceMatchEvidence,
    Phase6CEvidenceCandidate,
)
from atlaslens_api.place_evidence.normalization import sanitize_unicode
from atlaslens_api.providers.base import OCRBlock, OCRResult
from atlaslens_api.schemas import PlaceEvidenceSummary

type OCRTargetLabel = Literal[
    "Signage - Advertisement",
    "Signage - Information",
    "Signage - Store",
    "Traffic Sign - Direction",
    "Lane Marking - Text",
    "Building",
]
type OCRMatchSemantics = Literal["exact", "alias", "fuzzy"]

_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,159}$")
_SAFE_TEXT = re.compile(r"[^\w.\- '\u0600-\u06ff\u3040-\u30ff\u3400-\u9fff]+")
_TARGET_LABELS: tuple[OCRTargetLabel, ...] = (
    "Signage - Advertisement",
    "Signage - Information",
    "Signage - Store",
    "Traffic Sign - Direction",
    "Lane Marking - Text",
    "Building",
)
_SPECIFIC_TYPES = {
    "city",
    "district",
    "airport",
    "station",
    "public_landmark",
    "public_institution",
}
_UNCERTAINTY_RADIUS_KM = {
    "city": 25.0,
    "district": 12.0,
    "airport": 8.0,
    "station": 5.0,
    "public_landmark": 3.0,
    "public_institution": 5.0,
}


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class Phase6COCRConfig(_FrozenModel):
    version: Literal["phase6c-v1"] = "phase6c-v1"
    total_timeout_seconds: float = Field(default=45.0, gt=0, le=60)
    paddle_attempt_timeout_seconds: float = Field(default=28.0, gt=0, le=60)
    rapidocr_reserved_seconds: float = Field(default=16.0, gt=0, le=60)
    full_image_scales: tuple[float, ...] = (1.0, 1.5)
    targeted_crop_scales: tuple[float, ...] = (1.5, 2.0)
    targeted_segmentation_labels: tuple[OCRTargetLabel, ...] = _TARGET_LABELS
    max_targeted_crops: int = Field(default=6, ge=0, le=12)
    max_rotated_crops: int = Field(default=2, ge=0, le=4)
    rotated_crop_degrees: tuple[Literal[-90, 90], ...] = (-90, 90)
    crop_padding_fraction: float = Field(default=0.08, ge=0, le=0.25)
    minimum_region_score: float = Field(default=0.02, ge=0, le=1)
    minimum_region_area_fraction: float = Field(default=0.0005, gt=0, le=0.25)
    maximum_region_area_fraction: float = Field(default=0.65, gt=0, le=1)
    fuzzy_minimum_similarity: float = Field(default=0.86, ge=0.75, le=1)
    alias_minimum_similarity: float = Field(default=0.9, ge=0.8, le=1)
    max_place_token_evidence: int = Field(default=12, ge=1, le=24)

    @field_validator("full_image_scales", "targeted_crop_scales")
    @classmethod
    def bounded_scales(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if (
            not value
            or len(value) > 3
            or len(value) != len(set(value))
            or any(not math.isfinite(item) or not 0.5 <= item <= 3.0 for item in value)
        ):
            raise ValueError("OCR scales must be unique, finite and bounded")
        return value

    @field_validator("targeted_segmentation_labels")
    @classmethod
    def unique_target_labels(
        cls, value: tuple[OCRTargetLabel, ...]
    ) -> tuple[OCRTargetLabel, ...]:
        if not value or len(value) != len(set(value)):
            raise ValueError("OCR target labels must be non-empty and unique")
        return value

    @field_validator("rotated_crop_degrees")
    @classmethod
    def unique_rotations(
        cls, value: tuple[Literal[-90, 90], ...]
    ) -> tuple[Literal[-90, 90], ...]:
        if not value or len(value) != len(set(value)):
            raise ValueError("OCR rotations must be non-empty and unique")
        return value

    @model_validator(mode="after")
    def coherent_bounds(self) -> Phase6COCRConfig:
        if (
            self.paddle_attempt_timeout_seconds + self.rapidocr_reserved_seconds
            > self.total_timeout_seconds
        ):
            raise ValueError("PaddleOCR must leave the configured RapidOCR reserve")
        if self.max_rotated_crops > self.max_targeted_crops:
            raise ValueError("rotated OCR crops cannot exceed targeted crops")
        if 1 + self.max_targeted_crops + self.max_rotated_crops > 17:
            raise ValueError("OCR crop plan exceeds the hard pass bound")
        if self.minimum_region_area_fraction >= self.maximum_region_area_fraction:
            raise ValueError("OCR region area bounds are incoherent")
        if self.alias_minimum_similarity < self.fuzzy_minimum_similarity:
            raise ValueError("OCR alias threshold cannot be weaker than fuzzy matching")
        return self


def load_phase6c_ocr_config(path: Path) -> Phase6COCRConfig:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024:
        raise ValueError("Phase 6C OCR config is missing or unsafe")
    return Phase6COCRConfig.model_validate_json(path.read_text(encoding="utf-8"))


class NormalizedImageBox(_FrozenModel):
    x_min: float = Field(ge=0, le=1)
    y_min: float = Field(ge=0, le=1)
    x_max: float = Field(ge=0, le=1)
    y_max: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def positive_area(self) -> NormalizedImageBox:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("OCR image box must have positive area")
        return self

    @property
    def area(self) -> float:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)


class OCRSegmentationRegion(_FrozenModel):
    label: OCRTargetLabel
    bounds: NormalizedImageBox
    segmentation_score: float = Field(ge=0, le=1)
    score_semantics: Literal["segmentation_region_priority_not_probability"] = (
        "segmentation_region_priority_not_probability"
    )


class OCRCropPass(_FrozenModel):
    crop_id: str = Field(pattern=_SAFE_ID.pattern)
    crop_kind: Literal["full_image", "segmentation_region"]
    bounds: NormalizedImageBox
    scales: tuple[float, ...] = Field(min_length=1, max_length=3)
    rotation_degrees: Literal[-90, 0, 90]
    segmentation_label: OCRTargetLabel | None = None
    provenance: str = Field(min_length=1, max_length=160)

    @model_validator(mode="after")
    def coherent_crop(self) -> OCRCropPass:
        if self.crop_kind == "full_image" and (
            self.segmentation_label is not None
            or self.rotation_degrees != 0
            or self.bounds != NormalizedImageBox(x_min=0, y_min=0, x_max=1, y_max=1)
        ):
            raise ValueError("full-image OCR pass must cover the unrotated image")
        if self.crop_kind == "segmentation_region" and self.segmentation_label is None:
            raise ValueError("targeted OCR crop requires segmentation provenance")
        return self


class Phase6COCRCropPlan(_FrozenModel):
    version: Literal["phase6c-v1"] = "phase6c-v1"
    passes: tuple[OCRCropPass, ...] = Field(min_length=1, max_length=17)

    @model_validator(mode="after")
    def exactly_one_full_image(self) -> Phase6COCRCropPlan:
        if sum(item.crop_kind == "full_image" for item in self.passes) != 1:
            raise ValueError("OCR crop plan requires exactly one full-image pass")
        if len({item.crop_id for item in self.passes}) != len(self.passes):
            raise ValueError("OCR crop identifiers must be unique")
        return self


def build_ocr_crop_plan(
    regions: Sequence[OCRSegmentationRegion], config: Phase6COCRConfig
) -> Phase6COCRCropPlan:
    """Select deterministic, bounded text-region passes; image pixels stay outside the plan."""

    passes = [
        OCRCropPass(
            crop_id="full-0",
            crop_kind="full_image",
            bounds=NormalizedImageBox(x_min=0, y_min=0, x_max=1, y_max=1),
            scales=config.full_image_scales,
            rotation_degrees=0,
            provenance="full-image",
        )
    ]
    label_priority = {
        label: index for index, label in enumerate(config.targeted_segmentation_labels)
    }
    eligible = [
        region
        for region in regions
        if region.label in label_priority
        and region.segmentation_score >= config.minimum_region_score
        and config.minimum_region_area_fraction
        <= region.bounds.area
        <= config.maximum_region_area_fraction
    ]
    eligible.sort(
        key=lambda item: (
            label_priority[item.label],
            -item.segmentation_score,
            -item.bounds.area,
            item.bounds.y_min,
            item.bounds.x_min,
        )
    )
    selected: list[tuple[OCRSegmentationRegion, NormalizedImageBox]] = []
    for region in eligible:
        expanded = _padded_box(region.bounds, config.crop_padding_fraction)
        if any(_box_iou(expanded, existing) >= 0.85 for _, existing in selected):
            continue
        selected.append((region, expanded))
        if len(selected) >= config.max_targeted_crops:
            break
    for index, (region, bounds) in enumerate(selected, start=1):
        passes.append(
            OCRCropPass(
                crop_id=f"crop-{index}",
                crop_kind="segmentation_region",
                bounds=bounds,
                scales=config.targeted_crop_scales,
                rotation_degrees=0,
                segmentation_label=region.label,
                provenance=f"segformer-region:{_slug(region.label)}",
            )
        )
    for index, (region, bounds) in enumerate(selected[: config.max_rotated_crops], start=1):
        rotation = config.rotated_crop_degrees[(index - 1) % len(config.rotated_crop_degrees)]
        suffix = "ccw" if rotation < 0 else "cw"
        passes.append(
            OCRCropPass(
                crop_id=f"crop-{index}-rot-{suffix}",
                crop_kind="segmentation_region",
                bounds=bounds,
                scales=config.targeted_crop_scales,
                rotation_degrees=rotation,
                segmentation_label=region.label,
                provenance=f"segformer-region:{_slug(region.label)}:rotated",
            )
        )
    return Phase6COCRCropPlan(passes=tuple(passes))


def normalize_turkish_latin(text: str) -> str:
    """Make a search key while preserving Turkish dotted/dotless-I semantics."""

    safe = sanitize_unicode(text, max_length=240).replace("I", "ı").replace("İ", "i")
    lowered = safe.lower().replace("i\u0307", "i")
    cleaned = _SAFE_TEXT.sub(" ", lowered)
    return " ".join(cleaned.split()).strip(".-' ")


def _alias_key(text: str) -> str:
    exact = normalize_turkish_latin(text).replace("ı", "i")
    decomposed = unicodedata.normalize("NFKD", exact)
    return "".join(item for item in decomposed if unicodedata.category(item) != "Mn")


class OCRCropProvenance(_FrozenModel):
    crop_id: str = Field(pattern=_SAFE_ID.pattern)
    crop_kind: Literal["full_image", "segmentation_region"]
    scales: tuple[float, ...] = Field(min_length=1, max_length=3)
    rotation_degrees: Literal[-90, 0, 90]
    segmentation_label: OCRTargetLabel | None = None
    provenance: str = Field(min_length=1, max_length=160)


class OCRPlaceSourceProvenance(_FrozenModel):
    ocr_provider: str = Field(min_length=1, max_length=80)
    ocr_profile: str = Field(min_length=1, max_length=80)
    crop: OCRCropProvenance
    gazetteer_source: str = Field(min_length=1, max_length=500)
    gazetteer_version: str = Field(min_length=1, max_length=120)
    gazetteer_license: str = Field(min_length=1, max_length=500)


class OCRPlaceTokenEvidence(_FrozenModel):
    evidence_id: str = Field(pattern=_SAFE_ID.pattern)
    detected_redacted_text: str = Field(min_length=1, max_length=240, repr=False)
    normalized_detected_text: str = Field(min_length=1, max_length=240, repr=False)
    matched_entity: str = Field(min_length=1, max_length=200)
    match_semantics: OCRMatchSemantics
    match_type: str = Field(min_length=1, max_length=40)
    country_code: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    text_similarity: float = Field(ge=0, le=1)
    ambiguity_count: int = Field(ge=1)
    place_match_strength: float = Field(ge=0, le=1)
    score_semantics: Literal["place_match_strength_not_location_confidence"] = (
        "place_match_strength_not_location_confidence"
    )
    engine_confidence: float = Field(ge=0, le=1)
    engine_confidence_semantics: Literal["uncalibrated_ocr_engine_score"] = (
        "uncalibrated_ocr_engine_score"
    )
    location_confidence: None = None
    calibrated: Literal[False] = False
    specific_and_unambiguous: bool
    independent_support_eligible: bool
    independence_basis: Literal[
        "exact_or_alias_specific_unique_public_place",
        "fuzzy_match_not_independent",
        "ambiguous_match_not_independent",
        "broad_or_countryless_match_not_independent",
    ]
    source: OCRPlaceSourceProvenance

    @model_validator(mode="after")
    def coherent_independence(self) -> OCRPlaceTokenEvidence:
        expected = (
            self.specific_and_unambiguous
            and self.match_semantics in {"exact", "alias"}
        )
        if self.independent_support_eligible != expected:
            raise ValueError("OCR independent-support eligibility is inconsistent")
        return self


class OCRCandidateCountryAssessment(_FrozenModel):
    candidate_country_code: str = Field(pattern=r"^[A-Z]{2}$")
    supporting_evidence_ids: tuple[str, ...] = Field(max_length=24)
    contradiction_evidence_ids: tuple[str, ...] = Field(max_length=24)
    neutral_evidence_ids: tuple[str, ...] = Field(max_length=24)
    independent_support: bool
    contradiction_present: bool


def build_place_token_evidence(
    result: OCRResult,
    config: Phase6COCRConfig,
    *,
    crop_passes_by_profile: Mapping[str, OCRCropPass] | None = None,
) -> tuple[OCRPlaceTokenEvidence, ...]:
    """Associate typed, already-redacted OCR blocks with real gazetteer matches."""

    safe_blocks = [block for block in result.blocks[:32] if not block.sensitive_content]
    evidence: list[OCRPlaceTokenEvidence] = []
    for match_index, match in enumerate(result.place_matches[:12], start=1):
        association = _best_block_association(safe_blocks, match, config)
        if association is None:
            continue
        block, semantics = association
        specific = (
            match.match_type in _SPECIFIC_TYPES
            and match.ambiguity_count == 1
            and match.country_code is not None
        )
        independent = specific and semantics in {"exact", "alias"}
        if independent:
            basis = "exact_or_alias_specific_unique_public_place"
        elif semantics == "fuzzy":
            basis = "fuzzy_match_not_independent"
        elif match.ambiguity_count > 1:
            basis = "ambiguous_match_not_independent"
        else:
            basis = "broad_or_countryless_match_not_independent"
        crop = _crop_provenance(block, crop_passes_by_profile or {})
        evidence.append(
            OCRPlaceTokenEvidence(
                evidence_id=f"ocr-place-{match_index}",
                detected_redacted_text=block.redacted_text,
                normalized_detected_text=normalize_turkish_latin(block.redacted_text),
                matched_entity=match.matched_entity,
                match_semantics=semantics,
                match_type=match.match_type,
                country_code=match.country_code,
                latitude=match.center.latitude,
                longitude=match.center.longitude,
                text_similarity=match.text_similarity,
                ambiguity_count=match.ambiguity_count,
                place_match_strength=match.evidence_strength,
                engine_confidence=block.confidence,
                specific_and_unambiguous=specific,
                independent_support_eligible=independent,
                independence_basis=basis,
                source=OCRPlaceSourceProvenance(
                    ocr_provider=block.provider,
                    ocr_profile=block.profile,
                    crop=crop,
                    gazetteer_source=match.source,
                    gazetteer_version=match.dataset_version,
                    gazetteer_license=match.license,
                ),
            )
        )
    evidence.sort(
        key=lambda item: (
            not item.independent_support_eligible,
            {"exact": 0, "alias": 1, "fuzzy": 2}[item.match_semantics],
            -item.place_match_strength,
            item.ambiguity_count,
            item.evidence_id,
        )
    )
    return tuple(evidence[: config.max_place_token_evidence])


def assess_ocr_country_support(
    evidence: Sequence[OCRPlaceTokenEvidence], candidate_country_code: str
) -> OCRCandidateCountryAssessment:
    country_code = candidate_country_code.strip().upper()
    if not re.fullmatch(r"[A-Z]{2}", country_code):
        raise ValueError("candidate country code must be ISO alpha-2")
    supporting: list[str] = []
    contradictions: list[str] = []
    neutral: list[str] = []
    for item in evidence[:24]:
        if not item.independent_support_eligible or item.country_code is None:
            neutral.append(item.evidence_id)
        elif item.country_code == country_code:
            supporting.append(item.evidence_id)
        else:
            contradictions.append(item.evidence_id)
    return OCRCandidateCountryAssessment(
        candidate_country_code=country_code,
        supporting_evidence_ids=tuple(supporting),
        contradiction_evidence_ids=tuple(contradictions),
        neutral_evidence_ids=tuple(neutral),
        independent_support=bool(supporting),
        contradiction_present=bool(contradictions),
    )


def to_phase6c_fusion_evidence(
    evidence: Sequence[OCRPlaceTokenEvidence],
    *,
    provider: str,
    model_id: str,
    model_revision: str,
    duration_ms: int,
) -> OCRPlaceMatchEvidence:
    """Admit only specific, unique exact/alias place tokens as an OCR source vote."""

    eligible = [item for item in evidence if item.independent_support_eligible]
    if not eligible:
        return OCRPlaceMatchEvidence(
            provider=provider,
            model_id=model_id,
            model_revision=model_revision,
            status="abstained",
            duration_ms=duration_ms,
            reason_code="no_specific_unambiguous_place_token",
        )
    unique: dict[tuple[object, ...], OCRPlaceTokenEvidence] = {}
    for item in eligible:
        key = (
            item.match_type,
            item.matched_entity.casefold(),
            item.country_code,
            round(item.latitude, 6),
            round(item.longitude, 6),
        )
        previous = unique.get(key)
        if previous is None or item.place_match_strength > previous.place_match_strength:
            unique[key] = item
    ranked = sorted(
        unique.values(),
        key=lambda item: (
            -item.place_match_strength,
            item.ambiguity_count,
            item.matched_entity.casefold(),
        ),
    )[:12]
    candidates = tuple(
        Phase6CEvidenceCandidate(
            candidate_id=_candidate_id(item, rank),
            latitude=item.latitude,
            longitude=item.longitude,
            raw_value=item.place_match_strength,
            provider_rank=rank,
            uncertainty_radius_km=_UNCERTAINTY_RADIUS_KM[item.match_type],
            provenance=(
                f"ocr-place:{item.source.gazetteer_source}:"
                f"{item.source.gazetteer_version}:{item.source.crop.crop_id}"
            )[:240],
            country_code=item.country_code,
        )
        for rank, item in enumerate(ranked, start=1)
    )
    return OCRPlaceMatchEvidence(
        provider=provider,
        model_id=model_id,
        model_revision=model_revision,
        duration_ms=duration_ms,
        candidates=candidates,
    )


def _best_block_association(
    blocks: Sequence[OCRBlock],
    match: PlaceEvidenceSummary,
    config: Phase6COCRConfig,
) -> tuple[OCRBlock, OCRMatchSemantics] | None:
    names = tuple(
        dict.fromkeys(
            name
            for name in (
                normalize_turkish_latin(match.normalized_name),
                normalize_turkish_latin(match.matched_entity),
            )
            if name
        )
    )
    best: tuple[int, float, OCRBlock, OCRMatchSemantics] | None = None
    for block in blocks:
        observed = normalize_turkish_latin(block.redacted_text)
        if not observed:
            continue
        semantics: OCRMatchSemantics | None = None
        quality = 0.0
        if any(_contains_phrase(observed, name) for name in names):
            semantics = "exact"
            quality = 1.0
        elif any(_contains_phrase(_alias_key(observed), _alias_key(name)) for name in names):
            if match.text_similarity >= config.alias_minimum_similarity:
                semantics = "alias"
                quality = match.text_similarity
        else:
            quality = max((_phrase_similarity(observed, name) for name in names), default=0.0)
            if (
                quality >= config.fuzzy_minimum_similarity
                and match.text_similarity >= config.fuzzy_minimum_similarity
            ):
                semantics = "fuzzy"
        if semantics is None:
            continue
        candidate = ({"exact": 0, "alias": 1, "fuzzy": 2}[semantics], -quality, block, semantics)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    return (best[2], best[3]) if best is not None else None


def _contains_phrase(text: str, phrase: str) -> bool:
    if not text or not phrase:
        return False
    words = text.split()
    target = phrase.split()
    if not target or len(target) > len(words):
        return False
    return any(
        words[index : index + len(target)] == target
        for index in range(len(words) - len(target) + 1)
    )


def _phrase_similarity(text: str, phrase: str) -> float:
    words = text.split()
    target = phrase.split()
    if not words or not target:
        return 0.0
    widths = {max(1, len(target) - 1), len(target), len(target) + 1}
    values = [
        SequenceMatcher(None, " ".join(words[start : start + width]), phrase).ratio()
        for width in widths
        for start in range(max(0, len(words) - width + 1))
    ]
    return max(values, default=0.0)


def _crop_provenance(
    block: OCRBlock, crop_passes_by_profile: Mapping[str, OCRCropPass]
) -> OCRCropProvenance:
    selected = crop_passes_by_profile.get(block.profile)
    if selected is None and block.profile.startswith("paddle-"):
        selected = crop_passes_by_profile.get(block.profile.removeprefix("paddle-"))
    if selected is None:
        return OCRCropProvenance(
            crop_id="full-0",
            crop_kind="full_image",
            scales=(1.0,),
            rotation_degrees=0,
            provenance="full-image-unspecified-scale",
        )
    return OCRCropProvenance(
        crop_id=selected.crop_id,
        crop_kind=selected.crop_kind,
        scales=selected.scales,
        rotation_degrees=selected.rotation_degrees,
        segmentation_label=selected.segmentation_label,
        provenance=selected.provenance,
    )


def _candidate_id(item: OCRPlaceTokenEvidence, rank: int) -> str:
    digest = hashlib.sha256(
        (
            f"{item.matched_entity}|{item.country_code}|{item.latitude:.6f}|"
            f"{item.longitude:.6f}|{item.source.gazetteer_version}"
        ).encode()
    ).hexdigest()[:20]
    return f"ocr-{rank}-{digest}"


def _padded_box(bounds: NormalizedImageBox, padding: float) -> NormalizedImageBox:
    width = bounds.x_max - bounds.x_min
    height = bounds.y_max - bounds.y_min
    return NormalizedImageBox(
        x_min=max(0.0, bounds.x_min - width * padding),
        y_min=max(0.0, bounds.y_min - height * padding),
        x_max=min(1.0, bounds.x_max + width * padding),
        y_max=min(1.0, bounds.y_max + height * padding),
    )


def _box_iou(left: NormalizedImageBox, right: NormalizedImageBox) -> float:
    width = max(0.0, min(left.x_max, right.x_max) - max(left.x_min, right.x_min))
    height = max(0.0, min(left.y_max, right.y_max) - max(left.y_min, right.y_min))
    intersection = width * height
    union = left.area + right.area - intersection
    return intersection / union if union > 0 else 0.0


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug[:80] or "region"
