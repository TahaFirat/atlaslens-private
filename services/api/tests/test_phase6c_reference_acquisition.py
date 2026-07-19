from __future__ import annotations

import io
import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image

from atlaslens_api.gazetteer import ResolvedPlace
from atlaslens_api.phase6c.reference_acquisition import (
    KARTAVIEW_REVIEWED_TERMS_VERSION,
    AcquisitionPolicy,
    HttpResponse,
    KartaViewApiAdapter,
    LocalAdministrativeResolver,
    MapillaryGraphAdapter,
    ProvinceAcquisitionTarget,
    ReferenceAcquisitionError,
    ReferenceCorpusAcquirer,
    RemoteReferenceCandidate,
    ResolvedReferenceCandidate,
    RetryingHttpClient,
    build_acquisition_plan,
    load_mapillary_backend_settings,
    stratified_select,
)
from atlaslens_api.phase6c.reference_index import ReferenceBuildInput


class QueueTransport:
    def __init__(self, responses: list[HttpResponse]) -> None:
        self.responses = responses
        self.requests: list[tuple[str, Mapping[str, str]]] = []

    def request(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
        max_bytes: int,
        validate_url: Callable[[str], None],
    ) -> HttpResponse:
        del timeout_seconds, max_bytes
        validate_url(url)
        self.requests.append((url, headers))
        return self.responses.pop(0)


def _json_response(url: str, payload: object, *, status: int = 200) -> HttpResponse:
    return HttpResponse(
        status_code=status,
        headers={},
        body=json.dumps(payload).encode(),
        final_url=url,
    )


def _mapillary_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in (
        "MAPILLARY_ENABLED",
        "MAPILLARY_ACCESS_TOKEN",
        "MAPILLARY_MAX_IMAGES",
        "MAPILLARY_IMAGE_WIDTH",
    ):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "MAPILLARY_ENABLED=true\n"
        "MAPILLARY_ACCESS_TOKEN=private-test-token\n"
        "MAPILLARY_MAX_IMAGES=2025\n"
        "MAPILLARY_IMAGE_WIDTH=1024\n",
        encoding="utf-8",
    )
    return load_mapillary_backend_settings(env)


def _target(name: str = "Ankara", code: str = "6") -> ProvinceAcquisitionTarget:
    return ProvinceAcquisitionTarget(
        admin1_code=code,
        province=name,
        center_latitude=39.93,
        center_longitude=32.86,
        requested_images=2,
    )


def _candidate(
    identifier: str,
    *,
    source: str = "mapillary",
    sequence: str = "sequence-a",
    heading: float | None = None,
    captured_at: datetime | None = None,
) -> RemoteReferenceCandidate:
    return RemoteReferenceCandidate(
        reference_id=f"{source}-{identifier}",
        source=source,
        source_image_id=f"image-{identifier}",
        source_sequence_id=sequence,
        source_url=f"https://example.test/metadata/{identifier}",
        download_url="https://private.example.test/signed-thumbnail",
        latitude=39.93,
        longitude=32.86,
        coordinate_uncertainty_m=50,
        heading_degrees=heading,
        captured_at=captured_at,
        license="CC BY-SA 4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        attribution="Test fixture contributor",
    )


def test_default_plan_represents_all_provinces_without_special_preference() -> None:
    plan = build_acquisition_plan()

    assert len(plan.provinces) == 81
    assert len({item.admin1_code for item in plan.provinces}) == 81
    assert plan.estimated_image_count == 2_025
    assert {item.requested_images for item in plan.provinces} == {25}
    quota = {item.province: item.requested_images for item in plan.provinces}
    assert quota["Kayseri"] == quota["Ankara"]
    assert quota["Kayseri"] == next(
        value for name, value in quota.items() if "stanbul" in name.casefold()
    )
    assert all(target.heading_bins == tuple(range(8)) for target in plan.provinces)
    assert all("rural" in target.urbanicity_strata for target in plan.provinces)


def test_plan_refuses_estimate_above_disk_limit() -> None:
    with pytest.raises(ReferenceAcquisitionError, match="estimated_storage"):
        build_acquisition_plan(
            policy=AcquisitionPolicy(
                max_images=81,
                max_per_province=1,
                estimated_image_bytes=1_024,
                max_disk_bytes=81 * 1_024 - 1,
            )
        )


def test_mapillary_secret_comes_from_backend_env_and_uses_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _mapillary_settings(tmp_path, monkeypatch)
    endpoint = "https://graph.mapillary.com/images"
    transport = QueueTransport(
        [
            _json_response(
                endpoint,
                {
                    "data": [
                        {
                            "id": "12345",
                            "computed_geometry": {
                                "type": "Point",
                                "coordinates": [32.86, 39.93],
                            },
                            "computed_compass_angle": 181.5,
                            "captured_at": 1_717_243_200_000,
                            "sequence": "sequence-123",
                            "creator": {"username": "fixture-user"},
                            "thumb_1024_url": (
                                "https://scontent.fixture.fbcdn.net/image.jpg?oh=abc&oe=def"
                            ),
                        }
                    ]
                },
            )
        ]
    )
    adapter = MapillaryGraphAdapter(
        settings,
        http=RetryingHttpClient(transport=transport, minimum_interval_seconds=0),
    )

    candidates = adapter.discover(_target(), limit=5)

    request_url, headers = transport.requests[0]
    assert "private-test-token" not in request_url
    assert "access_token" not in request_url
    assert headers["Authorization"] == "OAuth private-test-token"
    assert "private-test-token" not in repr(settings)
    assert len(candidates) == 1
    assert candidates[0].source == "mapillary"
    assert candidates[0].source_url == "https://graph.mapillary.com/12345"
    assert candidates[0].license == "CC BY-SA 4.0"
    assert candidates[0].attribution.startswith("Mapillary image by fixture-user")
    assert candidates[0].coordinate_uncertainty_m > 0


def test_enabled_mapillary_without_backend_token_fails_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MAPILLARY_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("MAPILLARY_ENABLED", raising=False)
    env = tmp_path / ".env"
    env.write_text("MAPILLARY_ENABLED=true\nMAPILLARY_ACCESS_TOKEN=\n", encoding="utf-8")

    with pytest.raises(ReferenceAcquisitionError, match="configuration"):
        load_mapillary_backend_settings(env)


def test_kartaview_adapter_requires_reviewed_terms_and_preserves_attribution() -> None:
    with pytest.raises(ReferenceAcquisitionError, match="terms_acceptance"):
        KartaViewApiAdapter(accepted_terms_version="")

    endpoint = "https://api.openstreetcam.org/2.0/photo/"
    transport = QueueTransport(
        [
            _json_response(
                endpoint,
                {
                    "result": {
                        "data": [
                            {
                                "id": "photo-1",
                                "sequenceId": "sequence-1",
                                "sequenceIndex": 7,
                                "lat": 39.93,
                                "lng": 32.86,
                                "heading": 45,
                                "shotDate": "2024-04-01T10:30:00Z",
                                "imageLthUrl": (
                                    "https://storage1.openstreetcam.org/files/photo-1.jpg"
                                ),
                            }
                        ]
                    }
                },
            )
        ]
    )
    adapter = KartaViewApiAdapter(
        accepted_terms_version=KARTAVIEW_REVIEWED_TERMS_VERSION,
        http=RetryingHttpClient(transport=transport, minimum_interval_seconds=0),
    )

    candidates = adapter.discover(_target(), limit=5)

    request_url, _ = transport.requests[0]
    assert "radius=2000" in request_url
    assert "zoomLevel" not in request_url
    assert len(candidates) == 1
    assert candidates[0].source == "kartaview"
    assert candidates[0].license == "CC BY-SA 4.0"
    assert candidates[0].attribution == "© Grab and KartaView Contributors"
    assert candidates[0].source_url.endswith("/sequence-1/7/track-info")
    assert candidates[0].captured_at == datetime(2024, 4, 1, 10, 30, tzinfo=UTC)


def test_kartaview_adapter_treats_documented_empty_response_as_no_coverage() -> None:
    endpoint = "https://api.openstreetcam.org/2.0/photo/"
    transport = QueueTransport(
        [
            _json_response(
                endpoint,
                {
                    "status": {
                        "apiCode": 601,
                        "apiMessage": "The request has an empty response",
                        "httpCode": 200,
                        "httpMessage": "Success",
                    }
                },
            )
        ]
    )
    adapter = KartaViewApiAdapter(
        accepted_terms_version=KARTAVIEW_REVIEWED_TERMS_VERSION,
        http=RetryingHttpClient(transport=transport, minimum_interval_seconds=0),
    )

    assert adapter.discover(_target(), limit=5) == ()


def test_http_client_retries_rate_limit_and_caches_metadata() -> None:
    url = "https://api.openstreetcam.org/test"
    transport = QueueTransport(
        [
            HttpResponse(429, {"Retry-After": "0.5"}, b"{}", url),
            _json_response(url, {"result": {"data": []}}),
        ]
    )
    sleeps: list[float] = []
    client = RetryingHttpClient(
        transport=transport,
        minimum_interval_seconds=0,
        sleep=sleeps.append,
    )

    first = client.get_json(url, headers={}, allowed_hosts=frozenset({"openstreetcam.org"}))
    second = client.get_json(url, headers={}, allowed_hosts=frozenset({"openstreetcam.org"}))

    assert first == second
    assert len(transport.requests) == 2
    assert sleeps == [0.5]


def test_stratified_selection_caps_sequences_and_spreads_headings_and_seasons() -> None:
    values = [
        ResolvedReferenceCandidate(
            candidate=_candidate(
                str(index),
                sequence="sequence-a" if index < 4 else f"sequence-{index}",
                heading=float(index * 45),
                captured_at=datetime(2024, (index % 4) * 3 + 1, 1, tzinfo=UTC),
                source="mapillary" if index % 2 == 0 else "kartaview",
            ),
            province="Ankara",
            city="Ankara",
        )
        for index in range(8)
    ]

    selected = stratified_select(values, limit=6, max_per_sequence=2)

    sequence_counts: dict[tuple[str, str], int] = {}
    for item in selected:
        key = (item.candidate.source, item.candidate.source_sequence_id)
        sequence_counts[key] = sequence_counts.get(key, 0) + 1
    assert len(selected) == 6
    assert max(sequence_counts.values()) <= 2
    assert len({item.candidate.heading_bin for item in selected}) >= 4
    assert len({item.candidate.season for item in selected}) >= 3
    assert {item.candidate.source for item in selected} == {"mapillary", "kartaview"}


class StaticGazetteer:
    def resolve(self, latitude: float, longitude: float) -> ResolvedPlace:
        del latitude, longitude
        return ResolvedPlace(
            label="Ankara",
            country_code="TR",
            country="Türkiye",
            region="Ankara",
            city="Ankara",
            distance_km=1,
            source="GeoNames",
            dataset_version="fixture-v1",
            license="CC BY 4.0",
        )


class FakeMapillarySource:
    source = "mapillary"
    terms_version = "mapillary-cc-by-sa-reviewed-2025-05-12"

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.download_calls = 0

    def discover(
        self,
        target: ProvinceAcquisitionTarget,
        *,
        limit: int,
    ) -> tuple[RemoteReferenceCandidate, ...]:
        del limit
        return (_candidate("one"),) if target.province == "Ankara" else ()

    def download(self, candidate: RemoteReferenceCandidate, *, max_bytes: int) -> bytes:
        del candidate
        assert len(self.payload) <= max_bytes
        self.download_calls += 1
        return self.payload


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (32, 24), color=(20, 80, 160)).save(output, format="PNG")
    return output.getvalue()


def test_acquisition_writes_exact_builder_input_and_resumes_without_signed_url(
    tmp_path: Path,
) -> None:
    source = FakeMapillarySource(_png_bytes())
    acquirer = ReferenceCorpusAcquirer(
        sources=[source],
        administrative_resolver=LocalAdministrativeResolver(StaticGazetteer()),
    )
    plan = build_acquisition_plan(
        policy=AcquisitionPolicy(
            max_images=81,
            max_per_province=1,
            estimated_image_bytes=1,
            max_disk_bytes=1024 * 1024,
        )
    )

    first = acquirer.acquire(plan, tmp_path)
    receipt_path = tmp_path / "acquisition-receipt.json"
    interrupted_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    interrupted_receipt["entries"] = []
    interrupted_receipt["completed"] = False
    receipt_path.write_text(json.dumps(interrupted_receipt), encoding="utf-8")
    second = acquirer.acquire(plan, tmp_path)

    manifest_text = (tmp_path / "reference-input.json").read_text(encoding="utf-8")
    manifest = ReferenceBuildInput.model_validate_json(manifest_text)
    receipt_text = (tmp_path / "acquisition-receipt.json").read_text(encoding="utf-8")
    assert first.status == "partial"
    assert first.total_images == 1
    assert second.resumed_images == 1
    assert second.newly_downloaded_images == 0
    assert source.download_calls == 1
    assert len(manifest.records) == 1
    assert manifest.records[0].province == "Ankara"
    assert manifest.records[0].city == "Ankara"
    assert manifest.records[0].coordinate_uncertainty_m > 0
    assert manifest.records[0].expected_sha256 is not None
    assert manifest.records[0].expected_perceptual_hash is not None
    assert "signed-thumbnail" not in manifest_text
    assert "signed-thumbnail" not in receipt_text
    assert "access_token" not in receipt_text


def test_acquisition_rejects_near_identical_downloads_before_manifest(
    tmp_path: Path,
) -> None:
    class DuplicateSource(FakeMapillarySource):
        def discover(
            self,
            target: ProvinceAcquisitionTarget,
            *,
            limit: int,
        ) -> tuple[RemoteReferenceCandidate, ...]:
            del limit
            if target.province != "Ankara":
                return ()
            return (
                _candidate("duplicate-a", sequence="sequence-a"),
                _candidate("duplicate-b", sequence="sequence-b"),
            )

    source = DuplicateSource(_png_bytes())
    plan = build_acquisition_plan(
        policy=AcquisitionPolicy(
            max_images=162,
            max_per_province=2,
            estimated_image_bytes=1,
            max_disk_bytes=1024 * 1024,
        )
    )
    result = ReferenceCorpusAcquirer(
        sources=[source],
        administrative_resolver=LocalAdministrativeResolver(StaticGazetteer()),
    ).acquire(plan, tmp_path)

    manifest = ReferenceBuildInput.model_validate_json(
        (tmp_path / "reference-input.json").read_text(encoding="utf-8")
    )
    assert result.selected_images == 2
    assert result.newly_downloaded_images == 1
    assert source.download_calls == 2
    assert len(manifest.records) == 1


def test_local_admin_resolution_rejects_wrong_country_or_province() -> None:
    class WrongGazetteer:
        def resolve(self, latitude: float, longitude: float) -> ResolvedPlace:
            del latitude, longitude
            return ResolvedPlace(
                label="Sofia",
                country_code="BG",
                country="Bulgaria",
                region="Sofia-Capital",
                city="Sofia",
                distance_km=1,
                source="GeoNames",
                dataset_version="fixture-v1",
                license="CC BY 4.0",
            )

    resolved = LocalAdministrativeResolver(WrongGazetteer()).resolve(
        _target(),
        _candidate("outside"),
    )

    assert resolved is None
