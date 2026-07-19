from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest
import torch

from atlaslens_api.providers.geoclip import (
    GeoClipOutputNormalizationError,
    NormalizedGeoClipOutput,
    _normalize_predictions,
    _safe_output_structure,
)


def _normalize(
    gps: Any,
    scores: Any,
    *,
    top_k: int = 5,
    minimum_hypotheses: int = 3,
    deduplication_radius_km: float = 25.0,
) -> NormalizedGeoClipOutput:
    return _normalize_predictions(
        (gps, scores),
        device="cpu",
        dtype="float32",
        inference_ms=7,
        deduplication_radius_km=deduplication_radius_km,
        top_k=top_k,
        minimum_hypotheses=minimum_hypotheses,
    )


def _spread_coordinates() -> list[list[float]]:
    return [[0.0, 0.0], [10.0, 20.0], [20.0, 40.0], [30.0, 60.0], [40.0, 80.0]]


def test_official_cpu_tensor_tuple_is_detached_and_schema_compatible() -> None:
    gps = torch.tensor(_spread_coordinates(), dtype=torch.float64, requires_grad=True)
    scores = torch.tensor([0.014, 0.012, 0.009, 0.006, 0.003], dtype=torch.float64)

    normalized = _normalize(gps, scores)

    assert len(normalized.result.hypotheses) == 5
    assert [item.original_rank for item in normalized.result.hypotheses] == [1, 2, 3, 4, 5]
    assert all(type(item.latitude) is float for item in normalized.result.hypotheses)
    assert all(type(item.raw_score) is float for item in normalized.result.hypotheses)
    payload = json.loads(normalized.result.model_dump_json())
    assert len(payload["hypotheses"]) == 5
    assert payload["hypotheses"][0]["calibration_state"] == "uncalibrated"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_official_cuda_tensor_tuple_is_moved_to_cpu_before_conversion() -> None:
    gps = torch.tensor(_spread_coordinates(), device="cuda", dtype=torch.float32)
    scores = torch.tensor([0.014, 0.012, 0.009, 0.006, 0.003], device="cuda")

    normalized = _normalize(gps, scores)

    assert len(normalized.result.hypotheses) == 5
    assert all(type(item.longitude) is float for item in normalized.result.hypotheses)


@pytest.mark.parametrize(
    ("gps", "scores"),
    [
        (
            np.asarray(_spread_coordinates(), dtype=np.float32),
            np.asarray([0.014, 0.012, 0.009, 0.006, 0.003], dtype=np.float32),
        ),
        (_spread_coordinates(), [0.014, 0.012, 0.009, 0.006, 0.003]),
    ],
)
def test_numpy_and_python_list_outputs_use_native_scalar_values(gps: Any, scores: Any) -> None:
    normalized = _normalize(gps, scores)

    assert len(normalized.result.hypotheses) == 5
    assert all(type(item.latitude) is float for item in normalized.result.hypotheses)
    assert all(type(item.raw_score) is float for item in normalized.result.hypotheses)


def test_singleton_batch_dimensions_are_removed_only_when_valid() -> None:
    normalized = _normalize(
        np.asarray([_spread_coordinates()], dtype=np.float32),
        np.asarray([[0.014, 0.012, 0.009, 0.006, 0.003]], dtype=np.float32),
    )

    assert len(normalized.result.hypotheses) == 5


@pytest.mark.parametrize(
    ("gps", "score"),
    [
        ([41.0, 29.0], 0.001),
        (np.asarray([41.0, 29.0], dtype=np.float32), np.float32(0.001)),
        (torch.tensor([41.0, 29.0]), torch.tensor(0.001)),
    ],
)
def test_top_one_coordinate_and_scalar_score_shapes(gps: Any, score: Any) -> None:
    normalized = _normalize(gps, score, top_k=1, minimum_hypotheses=1)

    assert len(normalized.result.hypotheses) == 1
    hypothesis = normalized.result.hypotheses[0]
    assert type(hypothesis.latitude) is float
    assert type(hypothesis.raw_score) is float


@pytest.mark.parametrize(
    ("prediction", "subreason"),
    [
        ({"gps": [], "scores": []}, "unsupported_return_type"),
        (([],), "invalid_return_arity"),
        (([[[1.0, 2.0]], [[3.0, 4.0]]], [0.5, 0.4]), "gps_shape_invalid"),
        ((_spread_coordinates(), [[[0.5, 0.4]]]), "score_shape_invalid"),
        ((_spread_coordinates(), [0.5, 0.4]), "candidate_count_mismatch"),
    ],
)
def test_structural_errors_have_exact_safe_subreasons(
    prediction: Any, subreason: str
) -> None:
    with pytest.raises(GeoClipOutputNormalizationError) as raised:
        _normalize_predictions(
            prediction,
            device="cpu",
            dtype="float32",
            inference_ms=1,
            deduplication_radius_km=25.0,
            top_k=5,
        )

    assert raised.value.subreason_code == subreason
    assert str(raised.value) == subreason


def test_one_invalid_coordinate_does_not_erase_unrelated_candidates() -> None:
    gps = [[0.0, 0.0], [math.nan, 1.0], [10.0, 20.0], [20.0, 40.0], [30.0, 60.0]]
    normalized = _normalize(gps, [0.5, 0.4, 0.3, 0.2, 0.1])

    assert [item.original_rank for item in normalized.result.hypotheses] == [1, 3, 4, 5]
    assert normalized.diagnostics.rejected_coordinate_non_finite == 1


def test_one_nan_score_does_not_erase_unrelated_candidates() -> None:
    normalized = _normalize(_spread_coordinates(), [0.5, math.nan, 0.3, 0.2, 0.1])

    assert [item.original_rank for item in normalized.result.hypotheses] == [1, 3, 4, 5]
    assert normalized.diagnostics.rejected_score_non_finite == 1


@pytest.mark.parametrize(
    ("gps", "scores", "subreason"),
    [
        (
            [[math.nan, 0.0], [math.inf, 1.0], [-math.inf, 2.0]],
            [0.3, 0.2, 0.1],
            "all_coordinates_non_finite",
        ),
        (
            [[91.0, 0.0], [-91.0, 1.0], [180.0, 2.0]],
            [0.3, 0.2, 0.1],
            "all_coordinates_out_of_range",
        ),
        (_spread_coordinates()[:3], [math.nan, math.inf, -math.inf], "all_scores_non_finite"),
    ],
)
def test_no_valid_candidate_has_a_precise_subreason(
    gps: Any, scores: Any, subreason: str
) -> None:
    with pytest.raises(GeoClipOutputNormalizationError) as raised:
        _normalize(gps, scores)

    assert raised.value.subreason_code == subreason


def test_small_scores_need_not_sum_to_one_and_official_ranking_is_stable() -> None:
    scores = [0.014545, 0.012, 0.009, 0.008, 0.007839]
    normalized = _normalize(_spread_coordinates(), scores)

    assert sum(scores) != pytest.approx(1.0)
    assert [item.original_rank for item in normalized.result.hypotheses] == [1, 2, 3, 4, 5]
    assert [item.raw_score for item in normalized.result.hypotheses] == pytest.approx(scores)


def test_geographic_deduplication_preserves_the_highest_ranked_nearby_candidate() -> None:
    gps = [
        [0.0, 0.0],
        [0.01, 0.01],
        [10.0, 20.0],
        [20.0, 40.0],
        [30.0, 60.0],
        [40.0, 80.0],
    ]
    normalized = _normalize(gps, [0.6, 0.5, 0.4, 0.3, 0.2, 0.1])

    assert [item.original_rank for item in normalized.result.hypotheses] == [1, 3, 4, 5, 6]
    assert normalized.diagnostics.configured_dedup_survivors == 5


def test_clustered_valid_output_is_backfilled_without_fabricating_candidates() -> None:
    gps = [[41.0 + index * 0.01, 29.0] for index in range(5)]
    normalized = _normalize(gps, [0.014, 0.012, 0.009, 0.006, 0.003])

    assert [item.original_rank for item in normalized.result.hypotheses] == [1, 2, 3, 4, 5]
    assert normalized.diagnostics.configured_dedup_survivors == 1
    assert normalized.diagnostics.minimum_diversity_backfill == 4


def test_exact_duplicates_cannot_be_fabricated_into_minimum_hypotheses() -> None:
    with pytest.raises(GeoClipOutputNormalizationError) as raised:
        _normalize([[41.0, 29.0]] * 5, [0.5, 0.4, 0.3, 0.2, 0.1])

    assert raised.value.subreason_code == "post_deduplication_insufficient"


def test_longitude_is_normalized_only_when_outside_the_antimeridian() -> None:
    gps = [[0.0, 181.0], [10.0, -181.0], [20.0, 180.0]]
    normalized = _normalize(gps, [0.3, 0.2, 0.1], top_k=3)

    assert [item.longitude for item in normalized.result.hypotheses] == [-179.0, 179.0, 180.0]


def test_safe_structure_and_errors_never_contain_raw_values_or_paths(caplog) -> None:
    gps = torch.tensor(_spread_coordinates(), dtype=torch.float32)
    scores = torch.tensor([0.014, 0.012, 0.009, 0.006, 0.003])
    structure = _safe_output_structure((gps, scores))
    serialized = json.dumps(structure, sort_keys=True)

    assert structure["tuple_length"] == 2
    assert structure["gps"] == {
        "type": "torch.Tensor",
        "shape": [5, 2],
        "dtype": "torch.float32",
        "device": "cpu",
    }
    assert "41.0" not in serialized
    assert "private" not in serialized.lower()
    assert caplog.text == ""
