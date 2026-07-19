from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence

from atlaslens_api.gazetteer.models import GazetteerMetadata
from atlaslens_api.gazetteer.resolvers import ForwardGazetteerResolver
from atlaslens_api.place_evidence.models import ForwardPlaceMatch, OCRTextObservation
from atlaslens_api.place_evidence.normalization import normalize_search_text
from atlaslens_api.schemas import GeoPoint, PlaceEvidenceSummary

_TOKEN = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*", re.UNICODE)
_COUNTRY_SUFFIX = re.compile(r"(?<!\w)\.([a-z]{2})(?!\w)", re.IGNORECASE)
_GENERIC_TLDS = {"ai", "app", "biz", "com", "dev", "edu", "gov", "info", "net", "org"}
_STOPWORDS = {
    "and",
    "the",
    "for",
    "from",
    "with",
    "ve",
    "bir",
    "bu",
    "ile",
    "de",
    "da",
    "hotel",
    "cafe",
    "restaurant",
    "station",
    "airport",
}
_ALLOWED_MATCH_TYPES = {
    "country",
    "region",
    "city",
    "district",
    "road",
    "airport",
    "station",
    "public_landmark",
    "public_institution",
    "domain_suffix",
}
_TYPE_WEIGHT = {
    "country": 0.72,
    "region": 0.82,
    "city": 1.0,
    "district": 0.92,
    "road": 0.55,
    "airport": 0.95,
    "station": 0.75,
    "public_landmark": 0.9,
    "public_institution": 0.68,
    "domain_suffix": 0.5,
}


class PlaceEvidenceService:
    """Turns ephemeral OCR text into bounded, attributed public-place matches."""

    def __init__(
        self,
        resolver: ForwardGazetteerResolver,
        metadata: GazetteerMetadata,
        *,
        max_queries: int = 48,
        max_results: int = 12,
        min_similarity: float = 0.82,
    ) -> None:
        if not 1 <= max_queries <= 128 or not 1 <= max_results <= 24:
            raise ValueError("place evidence bounds are invalid")
        if not 0 <= min_similarity <= 1 or not math.isfinite(min_similarity):
            raise ValueError("place evidence similarity must be finite and bounded")
        self._resolver = resolver
        self._metadata = metadata
        self._max_queries = max_queries
        self._max_results = max_results
        self._min_similarity = min_similarity

    def resolve(
        self, observations: Sequence[OCRTextObservation]
    ) -> tuple[PlaceEvidenceSummary, ...]:
        best: dict[tuple[object, ...], PlaceEvidenceSummary] = {}
        queries = 0
        for observation in observations[:32]:
            if not math.isfinite(observation.confidence) or not 0 <= observation.confidence <= 1:
                continue
            for phrase in self._candidate_phrases(observation.text, observation.script):
                if queries >= self._max_queries:
                    break
                queries += 1
                for match in self._resolver.search(
                    phrase, limit=4, min_similarity=self._min_similarity
                ):
                    summary = self._to_summary(observation, match)
                    if summary is None:
                        continue
                    key = (
                        summary.match_type,
                        summary.matched_entity.casefold(),
                        summary.country_code,
                        round(summary.center.latitude, 6),
                        round(summary.center.longitude, 6),
                    )
                    previous = best.get(key)
                    if previous is None or summary.evidence_strength > previous.evidence_strength:
                        best[key] = summary
            if queries >= self._max_queries:
                break
        ranked = sorted(
            best.values(),
            key=lambda match: (
                -match.evidence_strength,
                -match.text_similarity,
                match.ambiguity_count,
                match.matched_entity.casefold(),
                match.center.latitude,
                match.center.longitude,
            ),
        )
        return tuple(ranked[: self._max_results])

    @classmethod
    def _candidate_phrases(cls, text: str, script: str) -> Iterable[str]:
        normalized = normalize_search_text(text)
        if not normalized:
            return ()
        phrases: list[str] = []
        for suffix in _COUNTRY_SUFFIX.findall(normalized):
            if suffix not in _GENERIC_TLDS:
                phrases.append(f".{suffix}")
        tokens = [token for token in _TOKEN.findall(normalized) if token not in _STOPWORDS]
        if script in {"han", "japanese", "hangul"} and len(tokens) <= 1:
            compact = "".join(tokens)
            if 2 <= len(compact) <= 32:
                phrases.append(compact)
                for size in range(min(8, len(compact)), 1, -1):
                    for start in range(0, len(compact) - size + 1):
                        phrases.append(compact[start : start + size])
                        if len(phrases) >= 32:
                            break
                    if len(phrases) >= 32:
                        break
        else:
            tokens = tokens[:16]
            for width in range(min(5, len(tokens)), 0, -1):
                for start in range(len(tokens) - width + 1):
                    phrase = " ".join(tokens[start : start + width])
                    if width > 1 or len(phrase) >= 4:
                        phrases.append(phrase)
                    if len(phrases) >= 32:
                        break
                if len(phrases) >= 32:
                    break
        return tuple(dict.fromkeys(phrases))

    def _to_summary(
        self, observation: OCRTextObservation, match: ForwardPlaceMatch
    ) -> PlaceEvidenceSummary | None:
        if match.match_type not in _ALLOWED_MATCH_TYPES:
            return None
        script_factor = self._script_factor(observation.script, match.script)
        ambiguity_factor = 1.0 / math.sqrt(max(1, match.ambiguity_count))
        strength = (
            observation.confidence
            * match.text_similarity
            * script_factor
            * ambiguity_factor
            * _TYPE_WEIGHT[match.match_type]
        )
        if strength < 0.12:
            return None
        return PlaceEvidenceSummary(
            matched_entity=match.matched_entity,
            normalized_name=match.normalized_name,
            country_code=match.country_code,
            region=match.region,
            center=GeoPoint(latitude=match.latitude, longitude=match.longitude),
            match_type=match.match_type,
            text_similarity=match.text_similarity,
            ambiguity_count=match.ambiguity_count,
            evidence_strength=min(1.0, strength),
            source=self._metadata.source,
            dataset_version=self._metadata.version,
            license=self._metadata.license,
        )

    @staticmethod
    def _script_factor(observed: str, matched: str) -> float:
        if observed == matched:
            return 1.0
        if observed in {"unknown", "mixed"} or matched in {"unknown", "mixed"}:
            return 0.82
        return 0.55
