from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from atlaslens_api.constraints.models import MapClue, MapConstraintObservation


class MapConstraintProvider(Protocol):
    provider_id: str
    available: bool

    async def evaluate(
        self,
        clues: Sequence[MapClue],
        *,
        latitude: float,
        longitude: float,
        radius_km: float,
    ) -> list[MapConstraintObservation]: ...


def _unavailable(
    provider: str, clues: Sequence[MapClue], radius_km: float, limitation: str
) -> list[MapConstraintObservation]:
    return [
        MapConstraintObservation(
            clue=clue.clue,
            map_feature=clue.map_feature,
            status="unknown",
            reliability=0,
            query_radius_km=radius_km,
            provider=provider,
            limitation=limitation,
        )
        for clue in clues
    ]


class DisabledMapConstraintProvider:
    provider_id = "disabled"
    available = False

    async def evaluate(
        self,
        clues: Sequence[MapClue],
        *,
        latitude: float,
        longitude: float,
        radius_km: float,
    ) -> list[MapConstraintObservation]:
        del latitude, longitude
        return _unavailable(self.provider_id, clues, radius_km, "map constraints are disabled")


class LocalPbfMapConstraintProvider:
    provider_id = "local_pbf"
    available = False

    async def evaluate(
        self,
        clues: Sequence[MapClue],
        *,
        latitude: float,
        longitude: float,
        radius_km: float,
    ) -> list[MapConstraintObservation]:
        del latitude, longitude
        return _unavailable(self.provider_id, clues, radius_km, "local PBF adapter unavailable")


class FixtureMapConstraintProvider:
    provider_id = "offline_fixture"
    available = True

    def __init__(
        self,
        features: set[str],
        *,
        assessed_features: set[str] | None = None,
        reliability: float = 0.9,
    ) -> None:
        if not 0 <= reliability <= 1:
            raise ValueError("reliability must be between zero and one")
        self._features = frozenset(features)
        self._assessed = frozenset(assessed_features or features)
        self._reliability = reliability

    async def evaluate(
        self,
        clues: Sequence[MapClue],
        *,
        latitude: float,
        longitude: float,
        radius_km: float,
    ) -> list[MapConstraintObservation]:
        del latitude, longitude
        observations: list[MapConstraintObservation] = []
        for clue in clues:
            if clue.map_feature not in self._assessed:
                status = "neutral"
                limitation = "fixture does not assess this map feature"
                reliability = 0.0
            else:
                actual = clue.map_feature in self._features
                status = "supported" if actual == clue.expected_present else "contradicted"
                limitation = "offline fixture; not evidence of real-world map completeness"
                reliability = self._reliability
            observations.append(
                MapConstraintObservation(
                    clue=clue.clue,
                    map_feature=clue.map_feature,
                    status=status,
                    reliability=reliability,
                    query_radius_km=radius_km,
                    provider=self.provider_id,
                    limitation=limitation,
                )
            )
        return observations
