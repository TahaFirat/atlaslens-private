from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from atlaslens_api.config import Settings
from atlaslens_api.constraints.models import MapConstraintObservation
from atlaslens_api.constraints.phase5b import BoundedMapEvidenceEvaluator
from atlaslens_api.main import create_app
from atlaslens_api.providers.base import (
    GlobalPredictionHypothesis,
    GlobalPredictionResult,
    GlobalProviderStatus,
    InvocationContext,
    OCRResult,
    ProviderDescriptor,
    ProviderOutcome,
)
from atlaslens_api.retrieval.models import (
    EmbeddingSpec,
    ImageMetadata,
    RetrievalHit,
)
from atlaslens_api.schemas import AnalysisMode, GeoPoint, PlaceEvidenceSummary
from atlaslens_api.storage import LocalImageHandle
from conftest import image_bytes, upload, wait_for_terminal


class FakeGlobalProvider:
    descriptor = ProviderDescriptor(
        id="phase5b-test-global",
        kind="global_geolocation",
        version="test",
        execution_boundary="local",
        criticality="optional",
        available=True,
        model_name="test-only-global",
    )

    def status(self) -> GlobalProviderStatus:
        return GlobalProviderStatus(
            status="ready",
            installed=True,
            verified=True,
            model_name="test-only-global",
            model_revision="test",
            device="cpu",
            calibration_state="uncalibrated",
        )

    async def predict(
        self, handle: LocalImageHandle, context: InvocationContext
    ) -> ProviderOutcome[GlobalPredictionResult]:
        assert handle.path.is_file()
        assert context.mode == AnalysisMode.LOCAL_ONLY
        hypotheses = [
            GlobalPredictionHypothesis(
                rank=rank,
                original_rank=rank,
                latitude=latitude,
                longitude=longitude,
                raw_score=score,
                score_type="uncalibrated_gallery_softmax",
                normalization_method="softmax_over_fixed_gallery",
                calibration_state="uncalibrated",
                limitations=["test.fixture_only"],
            )
            for rank, (latitude, longitude, score) in enumerate(
                ((0.0, 0.0, 0.9), (40.0, 29.0, 0.1)), 1
            )
        ]
        return ProviderOutcome.succeeded(
            GlobalPredictionResult(
                provider_id=self.descriptor.id,
                model_name="test-only-global",
                model_revision="test",
                implementation_revision="test",
                device="cpu",
                dtype="float32",
                external_transfer=False,
                inference_ms=1,
                hypotheses=hypotheses,
            )
        )


class FakeOCRProvider:
    descriptor = ProviderDescriptor(
        id="phase5b-test-ocr",
        kind="ocr",
        version="test",
        execution_boundary="local",
        criticality="optional",
        available=True,
        model_name="test-only-ocr",
    )

    async def extract(
        self, handle: LocalImageHandle, context: InvocationContext
    ) -> ProviderOutcome[OCRResult]:
        del context
        assert handle.path.is_file()
        return ProviderOutcome.succeeded(
            OCRResult(
                redacted_snippets=["[REDACTED]"],
                place_matches=[
                    PlaceEvidenceSummary(
                        matched_entity="Public Test Landmark",
                        normalized_name="public test landmark",
                        country_code="TR",
                        region="Istanbul",
                        center=GeoPoint(latitude=40.0, longitude=29.0),
                        match_type="public_landmark",
                        text_similarity=0.95,
                        ambiguity_count=1,
                        evidence_strength=0.9,
                        source="GeoNames",
                        dataset_version="test",
                        license="CC BY 4.0",
                    )
                ],
            )
        )


class FakeRetrievalProvider:
    descriptor = ProviderDescriptor(
        id="phase5b-test-retrieval",
        kind="image_retrieval",
        version="test",
        execution_boundary="local",
        criticality="optional",
        available=True,
        model_name="test-only-embedding",
    )

    def __init__(self) -> None:
        self.calls = 0

    async def search(
        self, handle: LocalImageHandle, context: InvocationContext
    ) -> ProviderOutcome[object]:
        del context
        assert handle.path.is_file()
        self.calls += 1
        spec = EmbeddingSpec(provider="siglip2-b16-384", version="test", dimension=4)
        metadata = ImageMetadata(
            image_id=uuid4(),
            index_id=1,
            latitude=40.01,
            longitude=29.01,
            country="Türkiye",
            region="Istanbul",
            city="Istanbul",
            source="Wikimedia Commons",
            license="CC BY 4.0",
            capture_type="camera_raw",
            hash="a" * 64,
            embedding_provider=spec.provider,
            embedding_version=spec.version,
            asset_key="commons-test-reference",
            attribution="Test Author / Wikimedia Commons",
            capture_family_id="test-capture-family",
            coordinate_kind="camera_raw",
            geographic_cell="test-cell",
        )
        return ProviderOutcome.succeeded(
            [RetrievalHit(distance=0.1, provider=spec, metadata=metadata)]
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "status": "ready",
            "index_size": 1,
            "embedding_provider": "siglip2-b16-384",
            "embedding_version": "test",
            "dimension": 4,
            "checksum": "b" * 64,
            "index_id": "test-index",
        }


class FakeMapProvider:
    async def evaluate(
        self,
        clues: object,
        *,
        latitude: float,
        longitude: float,
        radius_km: float,
    ) -> list[MapConstraintObservation]:
        del clues, latitude, longitude
        return [
            MapConstraintObservation(
                clue="public_place_type_public_landmark",
                map_feature="buildings",
                status="supported",
                reliability=0.7,
                query_radius_km=min(radius_km, 25.0),
                provider="test-map",
                limitation="test fixture",
            )
        ]


def test_normal_upload_uses_ocr_retrieval_and_map_to_change_rank(tmp_path: Path) -> None:
    retrieval = FakeRetrievalProvider()
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite:///{(tmp_path / 'atlaslens.db').as_posix()}",
        temp_storage_dir=tmp_path / "uploads",
        global_model_enabled=False,
        ocr_enabled=True,
        phase5b_enabled=True,
        retrieval_enabled=True,
    )
    app = create_app(
        settings,
        global_provider=FakeGlobalProvider(),
        ocr_provider=FakeOCRProvider(),
        retrieval_provider=retrieval,
        map_evaluator=BoundedMapEvidenceEvaluator(FakeMapProvider()),  # type: ignore[arg-type]
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        accepted = upload(client, image_bytes())
        assert accepted.status_code == 202
        result = wait_for_terminal(client, accepted.json()["id"])

    assert result["status"] == "completed"
    assert retrieval.calls == 1
    assert result["fusion_policy_version"] == "phase5b-v1"
    assert result["candidates"]
    primary = result["candidates"][0]
    assert primary["center"]["latitude"] > 39.0
    assert primary["center"]["longitude"] > 28.0
    assessment = primary["phase5b_assessment"]
    assert assessment["classification"] == "multi_source_supported"
    assert assessment["retrieval_matches"][0]["reference_id"] == "commons-test-reference"
    assert assessment["map_observations"][0]["status"] == "supported"
    assert result["phase5b_diagnostics"]["reference_index"]["image_count"] == 1
    serialized = str(result)
    assert "raw-private-ocr-sentinel" not in serialized
    assert "C:\\" not in serialized
