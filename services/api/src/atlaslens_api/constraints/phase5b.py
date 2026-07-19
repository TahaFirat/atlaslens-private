from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence

from atlaslens_api.constraints.models import MapClue
from atlaslens_api.constraints.providers import MapConstraintProvider
from atlaslens_api.schemas import MapConstraintSummary


class BoundedMapEvidenceEvaluator:
    """Adapts an explicitly injected map provider to frozen public summaries.

    The evaluator performs no discovery or network access of its own. Callers must
    select and configure the concrete provider and pass structured clues rather than
    raw OCR text or arbitrary query instructions.
    """

    def __init__(
        self,
        provider: MapConstraintProvider,
        *,
        timeout_seconds: float = 5.0,
        max_clues: int = 12,
        max_query_radius_km: float = 25.0,
    ) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("map evidence timeout must be positive and finite")
        if not 1 <= max_clues <= 32:
            raise ValueError("map clue bound is invalid")
        if not math.isfinite(max_query_radius_km) or not 0 < max_query_radius_km <= 100:
            raise ValueError("map query radius bound is invalid")
        self._provider = provider
        self._timeout = timeout_seconds
        self._max_clues = max_clues
        self._max_radius = max_query_radius_km

    async def evaluate(
        self,
        clues: Sequence[MapClue],
        *,
        latitude: float,
        longitude: float,
        radius_km: float,
    ) -> tuple[MapConstraintSummary, ...]:
        if not math.isfinite(latitude) or not -90 <= latitude <= 90:
            raise ValueError("invalid map evidence latitude")
        if not math.isfinite(longitude) or not -180 <= longitude <= 180:
            raise ValueError("invalid map evidence longitude")
        if not math.isfinite(radius_km) or radius_km <= 0:
            raise ValueError("invalid map evidence radius")
        bounded = self._bounded_clues(clues)
        if not bounded:
            return ()
        query_radius = min(radius_km, self._max_radius)
        try:
            observations = await asyncio.wait_for(
                self._provider.evaluate(
                    bounded,
                    latitude=latitude,
                    longitude=longitude,
                    radius_km=query_radius,
                ),
                timeout=self._timeout,
            )
        except TimeoutError:
            return self._unavailable(bounded, query_radius, "map evidence timed out safely")
        except Exception:
            return self._unavailable(bounded, query_radius, "map evidence failed safely")
        allowed = {(item.clue, item.map_feature) for item in bounded}
        summaries = [
            MapConstraintSummary(
                clue=item.clue[:120],
                map_feature=item.map_feature[:120],
                status=item.status,
                reliability=item.reliability,
                query_radius_km=item.query_radius_km,
                provider=item.provider,
                limitation=item.limitation,
            )
            for item in observations
            if (item.clue, item.map_feature) in allowed
        ]
        return tuple(summaries[: self._max_clues])

    def _bounded_clues(self, clues: Sequence[MapClue]) -> tuple[MapClue, ...]:
        unique: dict[tuple[str, str, bool], MapClue] = {}
        for clue in clues:
            key = (clue.clue.casefold(), clue.map_feature.casefold(), clue.expected_present)
            unique.setdefault(key, clue)
        return tuple(
            sorted(
                unique.values(),
                key=lambda item: (
                    item.map_feature.casefold(),
                    item.clue.casefold(),
                    item.expected_present,
                ),
            )[: self._max_clues]
        )

    @staticmethod
    def _unavailable(
        clues: Sequence[MapClue], radius_km: float, limitation: str
    ) -> tuple[MapConstraintSummary, ...]:
        return tuple(
            MapConstraintSummary(
                clue=item.clue[:120],
                map_feature=item.map_feature[:120],
                status="unknown",
                reliability=0,
                query_radius_km=radius_km,
                provider="phase5b-map-evidence",
                limitation=limitation,
            )
            for item in clues
        )
