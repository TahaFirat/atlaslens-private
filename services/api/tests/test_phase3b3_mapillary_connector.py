from __future__ import annotations

import argparse
import gzip
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest
from PIL import Image
from pydantic import SecretStr, ValidationError

import atlaslens_api.corpus_cli as corpus_cli
import atlaslens_api.mapillary_demo.cli as mapillary_cli_module
from atlaslens_api.mapillary_demo.acquisition import (
    MapillaryAcquisition,
    canonical_sha256,
    load_aoi_catalog,
    select_acquisition_candidates,
    select_pilot_aoi,
    sha256_file,
)
from atlaslens_api.mapillary_demo.cli import (
    MAPILLARY_COMMANDS,
    MAPILLARY_TOKEN_ENV_NAME,
    add_mapillary_subparsers,
    dispatch_mapillary_command,
    load_mapillary_access_token,
)
from atlaslens_api.mapillary_demo.client import (
    MapillaryClient,
    RemoteImage,
    redact_headers,
    redact_url,
    validate_access_token,
    validated_next_url,
)
from atlaslens_api.mapillary_demo.errors import (
    MapillaryDemoError,
    MapillaryLimitError,
    MapillarySafetyError,
    MapillaryTokenError,
)
from atlaslens_api.mapillary_demo.models import (
    MAPILLARY_API_FIELDS,
    AcquiredImage,
    AcquisitionPlan,
    ClientLimits,
    CoverageAudit,
    CoverageSummary,
    GeoPoint,
    ImageMetadata,
)

FAKE_TOKEN = "fake-mapillary-token-for-tests"
HASH_A = "a" * 64
HASH_B = "b" * 64


def _row(
    image_id: str,
    *,
    thumbnail: str | None = None,
    sequence_id: str = "sequence-1",
) -> dict[str, object]:
    row: dict[str, object] = {
        "id": image_id,
        "computed_geometry": {"type": "Point", "coordinates": [35.45, 38.73]},
        "captured_at": 1_700_000_000_000,
        "computed_compass_angle": 91.0,
        "sequence": {"id": sequence_id},
        "creator": {"id": "creator-1"},
        "width": 2048,
        "height": 1536,
    }
    if thumbnail is not None:
        row["thumb_1024_url"] = thumbnail
    return row


def _image_payload(color: tuple[int, int, int] = (20, 40, 60)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (32, 24), color).save(output, format="PNG")
    return output.getvalue()


def _catalog_path() -> Path:
    return Path(__file__).parents[3] / "config" / "phase3b3" / "mapillary-aoi-v1.json"


def _coverage_summary(
    aoi_id: str,
    *,
    image_count: int,
    cell_count: int,
    sequence_count: int,
    contributor_count: int,
    city: str | None,
    region_kind: str = "urban",
) -> CoverageSummary:
    return CoverageSummary(
        aoi_id=aoi_id,
        aoi_version="v1",
        display_name=aoi_id,
        region_kind=region_kind,
        province=city or aoi_id,
        city=city,
        image_count=image_count,
        rejected_item_count=0,
        sequence_count=sequence_count,
        contributor_count=contributor_count,
        capture_year_distribution={"2023": image_count} if image_count else {},
        spatial_cell_count=cell_count,
        compass_direction_bins={"E": image_count} if image_count else {},
        approximate_road_km=100.0,
        approximate_images_per_road_km=image_count / 100.0,
        approximate_sequence_density=sequence_count / 100.0,
        estimated_selected_download_bytes=image_count * 100,
        expected_descriptor_bytes=image_count * 8448 * 4,
        expected_index_bytes=image_count * 8448 * 4,
        eligible_for_selection=image_count > 0 and cell_count > 0,
        metadata_sha256=HASH_A,
    )


def _audit() -> CoverageAudit:
    regions = (
        _coverage_summary(
            "kayseri-urban-v1",
            image_count=1_000,
            cell_count=20,
            sequence_count=25,
            contributor_count=5,
            city="Kayseri",
        ),
        _coverage_summary(
            "ankara-urban-v1",
            image_count=1_200,
            cell_count=30,
            sequence_count=30,
            contributor_count=6,
            city="Ankara",
        ),
        _coverage_summary(
            "sivas-urban-v1",
            image_count=500,
            cell_count=10,
            sequence_count=10,
            contributor_count=3,
            city="Sivas",
        ),
        _coverage_summary(
            "kayseri-ankara-corridor-v1",
            image_count=100,
            cell_count=4,
            sequence_count=5,
            contributor_count=2,
            city=None,
            region_kind="corridor",
        ),
        _coverage_summary(
            "kayseri-sivas-corridor-v1",
            image_count=80,
            cell_count=3,
            sequence_count=4,
            contributor_count=2,
            city=None,
            region_kind="corridor",
        ),
    )
    selected, reason = select_pilot_aoi(regions)
    return CoverageAudit(
        catalog_version="phase3b3-aoi-v1",
        audited_at=datetime(2026, 7, 17, tzinfo=UTC),
        request_count=20,
        page_count=20,
        regions=regions,
        selected_aoi_id=selected,
        selection_reason=reason,
    )


def test_kayseri_dense_in_one_cell_does_not_auto_win() -> None:
    regions = list(_audit().regions)
    regions[0] = _coverage_summary(
        "kayseri-urban-v1",
        image_count=2_000,
        cell_count=1,
        sequence_count=1,
        contributor_count=1,
        city="Kayseri",
    )
    selected, reason = select_pilot_aoi(regions)
    assert selected == "ankara-urban-v1"
    assert reason == "best_covered_eligible_urban_aoi"


def test_candidate_pool_is_deterministic_round_robin_and_sequence_spaced() -> None:
    catalog = load_aoi_catalog(_catalog_path())
    aoi = catalog.by_id("kayseri-urban-v1")
    base = datetime(2024, 1, 1, tzinfo=UTC)

    def item(image_id: str, longitude: float, latitude: float, sequence: str) -> RemoteImage:
        return RemoteImage(
            metadata=ImageMetadata(
                mapillary_image_id=image_id,
                computed_geometry=GeoPoint(coordinates=(longitude, latitude)),
                captured_at=base,
                sequence_id=sequence,
                creator_id="c",
                width_px=20,
                height_px=20,
            ),
            thumbnail_url=SecretStr(
                f"https://scontent.example.test/{image_id}?signature=temporary"
            ),
        )

    rows = [
        item("a-near", 35.45000, 38.73000, "seq-a"),
        item("a-duplicate-near", 35.45001, 38.73001, "seq-a"),
        item("a-far", 35.45100, 38.73000, "seq-a"),
        item("b", 35.47000, 38.75000, "seq-b"),
        item("c", 35.42000, 38.70000, "seq-c"),
    ]
    first = select_acquisition_candidates(rows, aoi, cap=4)
    second = select_acquisition_candidates(tuple(reversed(rows)), aoi, cap=4)
    first_ids = [row.metadata.mapillary_image_id for row in first]
    assert first_ids == [row.metadata.mapillary_image_id for row in second]
    assert len(first) == 4
    assert not {"a-near", "a-duplicate-near"}.issubset(first_ids)


def test_token_validation_and_redaction_never_return_fragments() -> None:
    with pytest.raises(MapillaryTokenError, match="mapillary_token_not_configured"):
        validate_access_token(None)
    with pytest.raises(MapillaryTokenError, match="mapillary_token_not_configured"):
        validate_access_token("aldigin_token")
    secret = validate_access_token(FAKE_TOKEN)
    assert secret.get_secret_value() == FAKE_TOKEN
    assert FAKE_TOKEN not in repr(secret)

    unsafe = f"https://graph.mapillary.com/images?access_token={FAKE_TOKEN}&limit=1"
    redacted = redact_url(unsafe)
    assert FAKE_TOKEN not in redacted
    assert "%5BREDACTED%5D" in redacted
    headers = redact_headers({"Authorization": f"OAuth {FAKE_TOKEN}", "Accept": "x"})
    assert headers == {"Authorization": "[REDACTED]", "Accept": "x"}
    assert redact_url("https://example.test:invalid/?access_token=x") == "[invalid-url]"


def test_named_only_token_loader_is_bounded_and_environment_wins(tmp_path: Path) -> None:
    source = tmp_path / ".env"
    source.write_text(
        "UNRELATED_SECRET=must-not-be-returned\n"
        f"{MAPILLARY_TOKEN_ENV_NAME}='{FAKE_TOKEN}'\n"
        f"{MAPILLARY_TOKEN_ENV_NAME}=second-fake-token\n",
        encoding="utf-8",
    )
    loaded = load_mapillary_access_token(source, environ={})
    assert loaded.get_secret_value() == FAKE_TOKEN
    assert "must-not-be-returned" not in repr(loaded)
    assert "second-fake-token" not in repr(loaded)
    environment_loaded = load_mapillary_access_token(
        tmp_path / "missing.env",
        environ={MAPILLARY_TOKEN_ENV_NAME: FAKE_TOKEN},
    )
    assert environment_loaded.get_secret_value() == FAKE_TOKEN
    with pytest.raises(MapillaryTokenError, match="mapillary_token_not_configured"):
        load_mapillary_access_token(
            source,
            environ={MAPILLARY_TOKEN_ENV_NAME: "aldigin_token"},
        )
    with pytest.raises(MapillaryTokenError, match="mapillary_token_not_configured"):
        load_mapillary_access_token(tmp_path / "missing.env", environ={})


def test_token_check_uses_authorization_header_and_returns_status_only() -> None:
    observed: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return httpx.Response(200, json={"data": []})

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as http_client:
        client = MapillaryClient(FAKE_TOKEN, http_client=http_client)
        assert client.check_token() == "configured"
    assert len(observed) == 1
    assert observed[0].url.host == "graph.mapillary.com"
    assert parse_qs(observed[0].url.query.decode())["bbox"] == ["35.4400,38.7200,35.4410,38.7210"]
    assert "access_token" not in parse_qs(observed[0].url.query.decode())
    assert observed[0].headers["Authorization"] == f"OAuth {FAKE_TOKEN}"


def test_pagination_strips_embedded_token_and_rejects_nonofficial_next_host() -> None:
    observed: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        if len(observed) == 1:
            return httpx.Response(
                200,
                json={
                    "data": [_row("image-1")],
                    "paging": {
                        "next": (
                            "https://graph.mapillary.com/images?after=cursor-1"
                            f"&access_token={FAKE_TOKEN}"
                        )
                    },
                },
            )
        return httpx.Response(200, json={"data": [_row("image-2")]})

    limits = ClientLimits(request_cap=4, page_cap=4, metadata_item_cap=4)
    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as http_client:
        client = MapillaryClient(FAKE_TOKEN, limits=limits, http_client=http_client)
        images = list(
            client.iter_images(
                [load_aoi_catalog(_catalog_path()).aois[0].tiles[0]],
                include_thumbnail=False,
            )
        )
    assert [item.metadata.mapillary_image_id for item in images] == ["image-1", "image-2"]
    assert "access_token" not in parse_qs(observed[1].url.query.decode())
    assert observed[1].headers["Authorization"] == f"OAuth {FAKE_TOKEN}"
    with pytest.raises(MapillarySafetyError, match="mapillary_paging_url_invalid"):
        validated_next_url("https://evil.example/images?after=cursor")


def test_relative_pagination_is_canonical_and_never_duplicates_first_page_params() -> None:
    observed: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        if len(observed) == 1:
            return httpx.Response(
                200,
                json={
                    "data": [_row("image-1")],
                    "paging": {"next": "/images?limit=100&after=cursor-1"},
                },
            )
        return httpx.Response(200, json={"data": [_row("image-2")]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = MapillaryClient(FAKE_TOKEN, http_client=http_client)
        images = list(
            client.iter_images(
                [load_aoi_catalog(_catalog_path()).aois[0].tiles[0]],
                include_thumbnail=False,
            )
        )

    assert [item.metadata.mapillary_image_id for item in images] == ["image-1", "image-2"]
    second_query = parse_qs(observed[1].url.query.decode())
    assert second_query == {"after": ["cursor-1"], "limit": ["100"]}
    assert "bbox" not in second_query
    assert "fields" not in second_query


def test_pagination_loop_and_duplicate_query_key_fail_closed() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        next_url = (
            "https://graph.mapillary.com/images?limit=100&after=cursor-1"
            if calls == 1
            else "/images?after=cursor-1&limit=100"
        )
        return httpx.Response(200, json={"data": [], "paging": {"next": next_url}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = MapillaryClient(FAKE_TOKEN, http_client=http_client)
        with pytest.raises(MapillarySafetyError, match="mapillary_paging_loop_detected"):
            list(
                client.iter_images(
                    [load_aoi_catalog(_catalog_path()).aois[0].tiles[0]],
                    include_thumbnail=False,
                )
            )
    assert calls == 2
    assert validated_next_url(
        "https://graph.mapillary.com/images?limit=100&after=cursor-1"
    ) == validated_next_url("/images?after=cursor-1&limit=100")
    with pytest.raises(MapillarySafetyError, match="mapillary_paging_query_duplicate"):
        validated_next_url("/images?after=one&after=two")


@pytest.mark.parametrize(
    ("status", "code", "expected_calls"),
    [
        (400, "mapillary_api_bad_request", 1),
        (401, "mapillary_token_rejected", 1),
        (403, "mapillary_permission_denied", 1),
        (429, "mapillary_api_rate_limit_retry_exhausted", 2),
        (503, "mapillary_api_server_retry_exhausted", 2),
    ],
)
def test_graph_http_failures_have_typed_secret_free_codes(
    status: int,
    code: str,
    expected_calls: int,
) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, text=f"provider body {FAKE_TOKEN}")

    limits = ClientLimits(request_cap=2, retry_cap=1, backoff_base_seconds=0)
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = MapillaryClient(
            FAKE_TOKEN,
            limits=limits,
            http_client=http_client,
            sleep=lambda _seconds: None,
        )
        with pytest.raises(MapillaryDemoError, match=code) as error:
            list(
                client.iter_images(
                    [load_aoi_catalog(_catalog_path()).aois[0].tiles[0]],
                    include_thumbnail=False,
                )
            )
    assert FAKE_TOKEN not in str(error.value)
    assert "provider body" not in str(error.value)
    assert calls == expected_calls


def test_graph_timeout_is_bounded_and_secret_free() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout(f"unsafe {FAKE_TOKEN}", request=request)

    limits = ClientLimits(request_cap=2, retry_cap=1, backoff_base_seconds=0)
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = MapillaryClient(
            FAKE_TOKEN,
            limits=limits,
            http_client=http_client,
            sleep=lambda _seconds: None,
        )
        with pytest.raises(MapillaryDemoError, match="mapillary_api_timeout") as error:
            list(
                client.iter_images(
                    [load_aoi_catalog(_catalog_path()).aois[0].tiles[0]],
                    include_thumbnail=False,
                )
            )
    assert calls == 2
    assert FAKE_TOKEN not in str(error.value)


def test_retry_after_and_bounded_exponential_retry() -> None:
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        if calls == 2:
            return httpx.Response(503)
        return httpx.Response(200, json={"data": []})

    limits = ClientLimits(
        request_cap=3,
        retry_cap=2,
        backoff_base_seconds=0.5,
        backoff_cap_seconds=5.0,
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = MapillaryClient(
            FAKE_TOKEN,
            limits=limits,
            http_client=http_client,
            sleep=sleeps.append,
            jitter=lambda _start, _end: 0.0,
        )
        assert client.check_token() == "configured"
    assert sleeps == [2.0, 1.0]
    assert calls == limits.request_cap


def test_item_and_request_caps_fail_closed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [_row("one"), _row("two")]})

    limits = ClientLimits(request_cap=1, page_cap=1, metadata_item_cap=1)
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = MapillaryClient(FAKE_TOKEN, limits=limits, http_client=http_client)
        with pytest.raises(MapillaryLimitError, match="mapillary_item_cap_reached"):
            list(
                client.iter_images(
                    [load_aoi_catalog(_catalog_path()).aois[0].tiles[0]],
                    include_thumbnail=False,
                )
            )
        assert client.check_token() == "unavailable"


def test_media_download_never_sends_graph_authorization() -> None:
    observed: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return httpx.Response(
            200,
            headers={"Content-Type": "image/png"},
            content=_image_payload(),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = MapillaryClient(
            FAKE_TOKEN,
            http_client=http_client,
            media_host_suffixes=(".example.test",),
        )
        payload, media_type = client.download_thumbnail(
            "https://scontent.example.test/photo?signature=temporary", max_bytes=1_000_000
        )
    assert payload
    assert media_type == "image/png"
    assert "Authorization" not in observed[0].headers


def test_media_byte_cap_is_enforced_before_persistence() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "image/png", "Content-Length": "2000"},
            content=b"x" * 2_000,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = MapillaryClient(
            FAKE_TOKEN,
            http_client=http_client,
            media_host_suffixes=(".example.test",),
        )
        with pytest.raises(MapillaryLimitError, match="mapillary_image_byte_cap_reached"):
            client.download_thumbnail(
                "https://scontent.example.test/photo?signature=temporary",
                max_bytes=1_024,
            )


def test_graph_response_byte_cap_is_enforced_during_streaming() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "application/json"},
            content=b"x" * 2_000,
        )

    limits = ClientLimits(response_byte_cap=1_024, retry_cap=0)
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = MapillaryClient(FAKE_TOKEN, limits=limits, http_client=http_client)
        with pytest.raises(MapillaryLimitError, match="mapillary_response_byte_cap_reached"):
            list(
                client.iter_images(
                    [load_aoi_catalog(_catalog_path()).aois[0].tiles[0]],
                    include_thumbnail=False,
                )
            )


def test_gzip_graph_response_is_decoded_exactly_once() -> None:
    payload = json.dumps({"data": [_row("gzip-image")]}, separators=(",", ":")).encode()
    compressed = gzip.compress(payload)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
                "Content-Length": str(len(compressed)),
            },
            content=compressed,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = MapillaryClient(FAKE_TOKEN, http_client=http_client)
        images = list(
            client.iter_images(
                [load_aoi_catalog(_catalog_path()).aois[0].tiles[0]],
                include_thumbnail=False,
            )
        )
    assert [item.metadata.mapillary_image_id for item in images] == ["gzip-image"]


def test_incomplete_graph_item_is_rejected_without_fabricating_geometry() -> None:
    invalid = _row("missing-geometry")
    invalid.pop("computed_geometry")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [invalid, _row("complete-image")]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = MapillaryClient(FAKE_TOKEN, http_client=http_client)
        images = list(
            client.iter_images(
                [load_aoi_catalog(_catalog_path()).aois[0].tiles[0]],
                include_thumbnail=False,
            )
        )
    assert [item.metadata.mapillary_image_id for item in images] == ["complete-image"]
    assert client.rejected_item_count == 1


def test_aoi_catalog_is_versioned_complete_and_each_tile_is_under_current_cap() -> None:
    catalog = load_aoi_catalog(_catalog_path())
    assert "approximate_road_km" not in _catalog_path().read_text(encoding="utf-8")
    assert len(catalog.aois) == 5
    assert {aoi.region_kind for aoi in catalog.aois} == {"urban", "corridor"}
    assert all(tile.area_square_degrees < 0.01 for aoi in catalog.aois for tile in aoi.tiles)


class _FakeAcquisitionClient:
    def __init__(self, rows: list[RemoteImage], payloads: dict[str, bytes]) -> None:
        self.limits = ClientLimits(
            request_cap=10,
            page_cap=10,
            metadata_item_cap=10,
            image_cap=2,
            concurrency=2,
        )
        self.request_count = 0
        self.page_count = 0
        self._rows = rows
        self._payloads = payloads

    def iter_images(self, *_args: object, **_kwargs: object) -> Any:
        self.request_count += 1
        self.page_count += 1
        yield from self._rows

    def download_thumbnail(self, signed_url: str, *, max_bytes: int) -> tuple[bytes, str]:
        self.request_count += 1
        payload = self._payloads[signed_url]
        assert len(payload) <= max_bytes
        return payload, "image/png"

    def image_exists(self, image_id: str) -> bool:
        self.request_count += 1
        return image_id != "missing"


def _remote(image_id: str, signed_url: str, sequence_id: str) -> RemoteImage:
    return RemoteImage(
        metadata=ImageMetadata(
            mapillary_image_id=image_id,
            computed_geometry=GeoPoint(coordinates=(35.45, 38.73)),
            captured_at=datetime(2024, 1, 2, tzinfo=UTC),
            compass_angle=90.0,
            sequence_id=sequence_id,
            creator_id="creator-test",
            width_px=32,
            height_px=24,
        ),
        thumbnail_url=SecretStr(signed_url),
    )


def test_acquisition_is_attributed_idempotent_and_excludes_signed_urls(tmp_path: Path) -> None:
    audit = _audit()
    assert audit.selected_aoi_id == "kayseri-urban-v1"
    signed_one = f"https://scontent.example.test/one?access_token={FAKE_TOKEN}"
    signed_two = "https://scontent.example.test/two?signature=temporary"
    rows = [
        _remote("mapillary-1", signed_one, "sequence-1"),
        _remote("mapillary-2", signed_two, "sequence-2"),
    ]
    client = _FakeAcquisitionClient(
        rows,
        {signed_one: _image_payload(), signed_two: _image_payload((70, 80, 90))},
    )
    plan = AcquisitionPlan(
        aoi_id="kayseri-urban-v1",
        aoi_version="v1",
        coverage_audit_sha256=canonical_sha256(audit),
        source_policy_receipt_sha256=HASH_B,
        target_reference_images=1,
        target_holdout_images=1,
        hard_image_cap=2,
        hard_raw_byte_cap=2_000_000,
        hard_request_cap=10,
        concurrency=1,
        created_at=datetime(2026, 7, 17, tzinfo=UTC),
    )
    acquisition = MapillaryAcquisition(
        client,  # type: ignore[arg-type]
        tmp_path,
        approved_root=tmp_path,
        disk_free_bytes=lambda _drive: 100 * 1024**3,
    )
    manifest = acquisition.acquire(load_aoi_catalog(_catalog_path()), audit, plan)
    assert len(manifest.assets) == 2
    assert manifest.downloaded_bytes == sum(asset.byte_size for asset in manifest.assets)
    assert all(asset.attribution_text.startswith("Mapillary image") for asset in manifest.assets)
    assert all(asset.license_identifier == "CC-BY-SA-4.0" for asset in manifest.assets)
    persisted = (tmp_path / "metadata" / "acquisition-manifest.json").read_text()
    assert FAKE_TOKEN not in persisted
    assert "thumb_1024_url" not in persisted
    assert "signature=temporary" not in persisted
    resumed = acquisition.acquire(load_aoi_catalog(_catalog_path()), audit, plan)
    assert resumed == manifest


def test_partial_batch_failure_preserves_checkpoint_and_resumes(tmp_path: Path) -> None:
    audit = _audit()
    signed_one = "https://scontent.example.test/one?signature=first"
    signed_two = "https://scontent.example.test/two?signature=second"
    client = _FakeAcquisitionClient(
        [
            _remote("mapillary-1", signed_one, "sequence-1"),
            _remote("mapillary-2", signed_two, "sequence-2"),
        ],
        {signed_one: _image_payload(), signed_two: b"not-an-image"},
    )
    plan = AcquisitionPlan(
        aoi_id="kayseri-urban-v1",
        aoi_version="v1",
        coverage_audit_sha256=canonical_sha256(audit),
        source_policy_receipt_sha256=HASH_B,
        target_reference_images=1,
        target_holdout_images=1,
        hard_image_cap=2,
        hard_raw_byte_cap=2_000_000,
        hard_request_cap=10,
        concurrency=2,
        created_at=datetime(2026, 7, 17, tzinfo=UTC),
    )
    acquisition = MapillaryAcquisition(
        client,  # type: ignore[arg-type]
        tmp_path,
        approved_root=tmp_path,
        disk_free_bytes=lambda _drive: 100 * 1024**3,
    )
    with pytest.raises(MapillarySafetyError, match="mapillary_image_decode_invalid"):
        acquisition.acquire(load_aoi_catalog(_catalog_path()), audit, plan)

    manifest_path = tmp_path / "metadata" / "acquisition-manifest.json"
    checkpoint_path = tmp_path / "checkpoints" / "acquisition.json"
    persisted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    persisted_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert [asset["mapillary_image_id"] for asset in persisted_manifest["assets"]] == [
        "mapillary-1"
    ]
    assert persisted_checkpoint["completed_image_ids"] == ["mapillary-1"]
    assert persisted_checkpoint["downloaded_bytes"] == persisted_manifest["downloaded_bytes"]

    client._payloads[signed_two] = _image_payload((90, 80, 70))
    resumed = acquisition.acquire(load_aoi_catalog(_catalog_path()), audit, plan, resume=True)
    assert len(resumed.assets) == 2
    completed = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert completed["status"] == "completed"
    assert completed["completed_image_ids"] == ["mapillary-1", "mapillary-2"]


def test_persistent_source_url_rejects_credentials() -> None:
    with pytest.raises(ValidationError):
        AcquiredImage(
            mapillary_image_id="image-1",
            computed_geometry=GeoPoint(coordinates=(35.0, 39.0)),
            captured_at=datetime(2024, 1, 1, tzinfo=UTC),
            width_px=10,
            height_px=10,
            source_page_url=(
                f"https://www.mapillary.com/app/?pKey=image-1&access_token={FAKE_TOKEN}"
            ),
            attribution_text="Mapillary test attribution",
            acquired_at=datetime(2026, 7, 17, tzinfo=UTC),
            raw_sha256=HASH_A,
            normalized_sha256=HASH_B,
            relative_path="imagery/image-1.jpg",
            byte_size=100,
            normalized_byte_size=90,
        )


def test_cleanup_is_dry_run_first_receipt_gated_and_root_bounded(tmp_path: Path) -> None:
    audit = _audit()
    signed_one = "https://scontent.example.test/one?signature=a"
    signed_two = "https://scontent.example.test/two?signature=b"
    rows = [
        _remote("mapillary-1", signed_one, "sequence-1"),
        _remote("mapillary-2", signed_two, "sequence-2"),
    ]
    client = _FakeAcquisitionClient(
        rows,
        {signed_one: _image_payload(), signed_two: _image_payload((1, 2, 3))},
    )
    plan = AcquisitionPlan(
        aoi_id="kayseri-urban-v1",
        aoi_version="v1",
        coverage_audit_sha256=canonical_sha256(audit),
        source_policy_receipt_sha256=HASH_B,
        target_reference_images=1,
        target_holdout_images=1,
        hard_image_cap=2,
        hard_raw_byte_cap=2_000_000,
        hard_request_cap=10,
        concurrency=1,
        created_at=datetime(2026, 7, 17, tzinfo=UTC),
    )
    acquisition = MapillaryAcquisition(
        client,  # type: ignore[arg-type]
        tmp_path,
        approved_root=tmp_path,
        disk_free_bytes=lambda _drive: 100 * 1024**3,
    )
    acquisition.acquire(load_aoi_catalog(_catalog_path()), audit, plan)
    materialized_root = tmp_path / "phase3b1-corpus" / "assets"
    materialized_root.mkdir(parents=True)
    for number in (1, 2):
        source = tmp_path / "imagery" / f"mapillary-{number}.jpg"
        (materialized_root / f"mapillary-mapillary-{number}.jpg").write_bytes(source.read_bytes())
    materialized_one = materialized_root / "mapillary-mapillary-1.jpg"
    materialized_one.write_bytes(b"tampered")
    with pytest.raises(MapillarySafetyError, match="mapillary_cleanup_path_invalid"):
        acquisition.cleanup(retain_image_ids=["mapillary-1"])
    assert all((tmp_path / "imagery" / f"mapillary-{number}.jpg").exists() for number in (1, 2))
    materialized_one.write_bytes((tmp_path / "imagery" / "mapillary-1.jpg").read_bytes())
    outside = tmp_path.parent / "must-remain.txt"
    outside.write_text("untouched", encoding="utf-8")
    dry_run = acquisition.cleanup(retain_image_ids=["mapillary-1"])
    assert dry_run.mode == "dry_run"
    assert dry_run.candidate_count == 1
    assert dry_run.materialized_candidate_count == 2
    assert all((tmp_path / "imagery" / f"mapillary-{number}.jpg").exists() for number in (1, 2))
    with pytest.raises(MapillarySafetyError, match="mapillary_cleanup_validation_required"):
        acquisition.cleanup(retain_image_ids=["mapillary-1"], execute=True)
    manifest_path = tmp_path / "metadata" / "acquisition-manifest.json"
    receipt = {
        "index_validated": True,
        "attribution_validated": True,
        "manifest_sha256": sha256_file(manifest_path),
        "index_sha256": "c" * 64,
    }
    report = acquisition.cleanup(
        retain_image_ids=["mapillary-1"],
        execute=True,
        validation_receipt=receipt,
    )
    assert report.deleted_count == 1
    assert report.materialized_deleted_count == 2
    assert (tmp_path / "imagery" / "mapillary-1.jpg").exists()
    assert not (tmp_path / "imagery" / "mapillary-2.jpg").exists()
    assert (tmp_path / "retained-sidecars" / "mapillary-1.json").exists()
    assert not any(materialized_root.iterdir())
    assert outside.read_text(encoding="utf-8") == "untouched"


def test_exact_seven_cli_commands_and_token_is_not_a_parser_argument() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_mapillary_subparsers(subparsers)
    assert tuple(name for name in subparsers.choices if name.startswith("mapillary-")) == (
        MAPILLARY_COMMANDS
    )
    assert "--token" not in parser.format_help()
    integrated = corpus_cli.build_parser()
    integrated_help = integrated.format_help()
    assert all(command in integrated_help for command in MAPILLARY_COMMANDS)


def test_corpus_cli_projects_exact_missing_token_marker(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def missing(_path: Path) -> SecretStr:
        raise MapillaryTokenError("mapillary_token_not_configured")

    monkeypatch.setattr(corpus_cli, "load_mapillary_access_token", missing)
    assert corpus_cli.main(["mapillary-check-token"]) == 4
    payload = json.loads(capsys.readouterr().err)
    assert payload == {"status": "error", "code": "MAPILLARY_TOKEN_NOT_CONFIGURED"}


def test_cli_check_token_projection_cannot_expose_runtime_value() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_mapillary_subparsers(subparsers)
    args = parser.parse_args(["mapillary-check-token", "--request-cap", "2"])

    def factory(token: str, limits: ClientLimits) -> MapillaryClient:
        assert token == FAKE_TOKEN
        http_client = httpx.Client(
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={"data": []}))
        )
        return MapillaryClient(token, limits=limits, http_client=http_client)

    result = dispatch_mapillary_command(
        args,
        access_token=FAKE_TOKEN,
        client_factory=factory,
    )
    assert result == {"token_status": "configured"}
    assert FAKE_TOKEN not in json.dumps(result)


def test_cli_coverage_projects_rejected_item_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = _audit()
    regions = list(audit.regions)
    regions[2] = regions[2].model_copy(update={"rejected_item_count": 7})
    audit = audit.model_copy(update={"regions": tuple(regions)})

    class _Auditor:
        def __init__(self, _client: MapillaryClient) -> None:
            pass

        def audit(self, _catalog: object) -> CoverageAudit:
            return audit

    monkeypatch.setattr(mapillary_cli_module, "MapillaryCoverageAuditor", _Auditor)
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_mapillary_subparsers(subparsers)
    args = parser.parse_args(
        [
            "mapillary-coverage",
            "--pilot-root",
            str(tmp_path),
            "--aoi-config",
            str(_catalog_path()),
        ]
    )

    def factory(token: str, limits: ClientLimits) -> MapillaryClient:
        assert token == FAKE_TOKEN
        client = httpx.Client(
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={"data": []}))
        )
        return MapillaryClient(token, limits=limits, http_client=client)

    result = dispatch_mapillary_command(
        args,
        access_token=FAKE_TOKEN,
        client_factory=factory,
        approved_root=tmp_path,
    )
    assert result["region_count"] == 5
    assert result["rejected_item_count"] == 7
    assert result["imagery_downloaded"] is False
    assert (tmp_path / "metadata" / "coverage-audit.json").is_file()
    assert FAKE_TOKEN not in json.dumps(result)


def test_api_field_receipt_excludes_ephemeral_thumbnail() -> None:
    audit = _audit()
    assert audit.api_fields == MAPILLARY_API_FIELDS[:-1]
    serialized = audit.model_dump_json()
    assert "thumb_1024_url" not in serialized
    assert "https://graph.mapillary.com" in serialized
    assert all(
        region.road_length_semantics == "bounded_sequence_track_sum_not_unique_road_length"
        for region in audit.regions
    )
    assert all(
        region.selected_download_size_semantics == "configured_per_image_hard_cap_upper_bound"
        and region.index_size_semantics
        == "faiss_float32_vector_payload_only_excludes_metadata_overhead"
        for region in audit.regions
    )
