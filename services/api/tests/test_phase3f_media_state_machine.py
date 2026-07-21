from __future__ import annotations

import io
import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from PIL import Image
from pydantic import SecretStr

from atlaslens_api.mapillary_demo.client import MapillaryClient, RemoteImage
from atlaslens_api.mapillary_demo.errors import (
    MapillaryApiError,
    MapillaryLimitError,
    MapillaryTokenError,
)
from atlaslens_api.mapillary_demo.models import GeoPoint, ImageMetadata
from atlaslens_api.phase3f import cloud_job as worker
from atlaslens_api.phase3f.acquisition import AcquisitionGuard
from atlaslens_api.phase3f.cloud_job import (
    MEDIA_CIRCUIT_FAILURE_THRESHOLD,
    MEDIA_PROVIDER_COOLDOWN_SECONDS,
    MEDIA_RATE_LIMIT_COOLDOWN_SECONDS,
    MEDIA_RESOLVER_CONTRACT_VERSION,
    CityArea,
    MetadataAsset,
    PlannedAsset,
    acquire_planned_assets,
)

ROOT = Path(__file__).parents[3]
END_TO_END_LAUNCHER = ROOT / "scripts" / "phase3f-end-to-end.ps1"


def _payload(seed: int) -> bytes:
    pixels = np.random.default_rng(seed).integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    output = io.BytesIO()
    Image.fromarray(pixels, mode="RGB").save(output, format="PNG")
    return output.getvalue()


def _solid_payload(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(output, format="PNG")
    return output.getvalue()


def _planned(primary: int, reserve: int) -> tuple[PlannedAsset, ...]:
    rows: list[PlannedAsset] = []
    for index in range(primary + reserve):
        image_id = f"candidate-{index:03d}"
        metadata = ImageMetadata(
            mapillary_image_id=image_id,
            computed_geometry=GeoPoint(coordinates=(30.0 + index * 0.02, 40.0)),
            captured_at=datetime(2024, 1, 1, tzinfo=UTC),
            sequence_id=f"sequence-{index:03d}",
            creator_id=f"contributor-{index:03d}",
            width_px=1024,
            height_px=768,
        )
        rows.append(PlannedAsset(MetadataAsset("TestCity", RemoteImage(metadata)), "reference"))
    return tuple(rows)


class _FakeClient:
    def __init__(
        self,
        planned: Sequence[PlannedAsset],
        *,
        events: dict[str, list[bytes | BaseException]] | None = None,
        resolution_events: dict[str, list[BaseException]] | None = None,
    ) -> None:
        self.request_count = 0
        self.page_count = 0
        self.rejected_item_count = 0
        self.downloaded: list[str] = []
        self.resolved: list[tuple[str, str]] = []
        self._events = events or {}
        self._resolution_events = resolution_events or {}
        self._image_ids = {item.metadata.image_id for item in planned}

    def add_historical_counts(
        self, *, request_count: int, page_count: int, rejected_item_count: int
    ) -> None:
        self.request_count += request_count
        self.page_count += page_count
        self.rejected_item_count += rejected_item_count

    def iter_images(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("media acquisition must not call the /images collection endpoint")

    def resolve_image_thumbnail(
        self,
        image_id: str,
        quality: str = "thumb_1024_url",
    ) -> SecretStr:
        assert image_id in self._image_ids
        assert quality == "thumb_1024_url"
        self.request_count += 1
        self.resolved.append((image_id, quality))
        events = self._resolution_events.get(image_id)
        if events:
            raise events.pop(0)
        return SecretStr(f"https://scontent.example/{image_id}")

    def download_thumbnail(self, signed_url: str, *, max_bytes: int) -> tuple[bytes, str]:
        assert max_bytes > 0
        image_id = signed_url.rsplit("/", 1)[-1]
        self.request_count += 1
        self.downloaded.append(image_id)
        events = self._events.get(image_id)
        event = events.pop(0) if events else _payload(int(image_id.rsplit("-", 1)[-1]) + 1)
        if isinstance(event, BaseException):
            raise event
        return event, "image/png"


def _acquire(
    root: Path,
    planned: Sequence[PlannedAsset],
    primary_count: int,
    client: _FakeClient,
    *,
    now: datetime | None = None,
    observer: Callable[[], None] | None = None,
) -> tuple[tuple[object, ...], dict[str, object]]:
    work = root / "work"
    return cast(
        tuple[tuple[object, ...], dict[str, object]],
        acquire_planned_assets(
            cast(MapillaryClient, client),
            (CityArea("TestCity", 30.0, 40.0, 0.01),),
            planned,
            work / "private-media",
            "a" * 32,
            AcquisitionGuard(),
            work / "acquisition-checkpoint.json",
            restore_client_counts=False,
            checkpoint_observer=observer,
            allow_item_failures=True,
            primary_count=primary_count,
            now_utc=(lambda: now) if now is not None else None,
        ),
    )


@pytest.mark.parametrize("failure_index", [0, 2, 5])
def test_item_5xx_at_first_middle_or_last_uses_reserve_without_global_error(
    tmp_path: Path, failure_index: int
) -> None:
    planned = _planned(6, 1)
    failed_id = planned[failure_index].metadata.image_id
    client = _FakeClient(
        planned,
        events={failed_id: [MapillaryApiError("mapillary_media_server_retry_exhausted")]},
    )

    assets, provenance = _acquire(tmp_path, planned, 6, client)

    assert len(assets) == 6
    assert provenance["media_status"] == "MEDIA_READY"
    assert provenance["media_task_counts"] == {
        "accepted": 6,
        "rejected": 0,
        "quarantined": 1,
        "pending": 0,
        "reserve": 0,
        "media_bytes": pytest.approx(provenance["media_task_counts"]["media_bytes"]),
        "retry_not_before": None,
    }
    assert client.downloaded.count(failed_id) == 1


def test_interleaved_5xx_never_opens_global_circuit(tmp_path: Path) -> None:
    planned = _planned(6, 2)
    failed = (planned[2].metadata.image_id, planned[5].metadata.image_id)
    client = _FakeClient(
        planned,
        events={
            image_id: [MapillaryApiError("mapillary_media_server_retry_exhausted")]
            for image_id in failed
        },
    )

    assets, provenance = _acquire(tmp_path, planned, 6, client)

    assert len(assets) == 6
    assert provenance["media_status"] == "MEDIA_READY"
    assert provenance["provider_circuit"]["open_count"] == 0
    assert all(client.downloaded.count(image_id) == 1 for image_id in failed)


@pytest.mark.parametrize(
    ("error", "terminal_state"),
    [
        pytest.param(MapillaryApiError("mapillary_media_unavailable"), "REJECTED", id="404"),
        pytest.param(MapillaryApiError("mapillary_media_unavailable"), "REJECTED", id="410"),
        (MapillaryApiError("mapillary_media_timeout"), "QUARANTINED"),
        (
            MapillaryApiError("mapillary_media_transport_retry_exhausted"),
            "QUARANTINED",
        ),
        (MapillaryApiError("mapillary_media_transport_failed"), "QUARANTINED"),
        (MapillaryLimitError("mapillary_image_byte_cap_reached"), "REJECTED"),
    ],
)
def test_typed_media_failures_are_item_local(
    tmp_path: Path, error: BaseException, terminal_state: str
) -> None:
    planned = _planned(1, 1)
    failed_id = planned[0].metadata.image_id
    client = _FakeClient(planned, events={failed_id: [error]})

    assets, provenance = _acquire(tmp_path, planned, 1, client)

    assert len(assets) == 1
    assert provenance["media_status"] == "MEDIA_READY"
    ledger = json.loads((tmp_path / "work" / "media-task-ledger.json").read_text())
    failed_task = next(row for row in ledger["tasks"] if row["image_id"] == failed_id)
    assert failed_task["state"] == terminal_state
    assert client.downloaded.count(failed_id) == 1


@pytest.mark.parametrize(
    "error",
    [
        MapillaryTokenError("mapillary_token_rejected"),
        MapillaryApiError("mapillary_permission_denied"),
    ],
)
def test_graph_401_or_403_is_terminal_before_media_requests(
    tmp_path: Path, error: BaseException
) -> None:
    planned = _planned(1, 1)
    client = _FakeClient(
        planned,
        resolution_events={planned[0].metadata.image_id: [error]},
    )

    with pytest.raises(type(error)):
        _acquire(tmp_path, planned, 1, client)

    assert client.downloaded == []


def test_direct_image_resolution_5xx_is_item_local_and_uses_reserve(
    tmp_path: Path,
) -> None:
    planned = _planned(1, 1)
    failed_id = planned[0].metadata.image_id
    client = _FakeClient(
        planned,
        resolution_events={failed_id: [MapillaryApiError("mapillary_api_server_retry_exhausted")]},
    )

    assets, provenance = _acquire(tmp_path, planned, 1, client)

    assert len(assets) == 1
    assert provenance["media_status"] == "MEDIA_READY"
    assert provenance["media_task_counts"]["quarantined"] == 1
    assert failed_id not in client.downloaded
    assert client.resolved[0] == (failed_id, "thumb_1024_url")


@pytest.mark.parametrize(
    ("error", "expected_state"),
    [
        (MapillaryApiError("mapillary_image_not_found"), "REJECTED"),
        (MapillaryApiError("mapillary_thumbnail_url_missing"), "REJECTED"),
        (MapillaryApiError("mapillary_image_id_mismatch"), "REJECTED"),
        (MapillaryApiError("mapillary_api_timeout"), "QUARANTINED"),
        (
            MapillaryApiError("mapillary_api_transport_retry_exhausted"),
            "QUARANTINED",
        ),
    ],
)
def test_direct_resolver_failures_are_typed_item_results(
    tmp_path: Path,
    error: BaseException,
    expected_state: str,
) -> None:
    planned = _planned(1, 1)
    failed_id = planned[0].metadata.image_id
    client = _FakeClient(planned, resolution_events={failed_id: [error]})

    assets, provenance = _acquire(tmp_path, planned, 1, client)

    assert len(assets) == 1
    assert provenance["media_status"] == "MEDIA_READY"
    ledger = json.loads((tmp_path / "work" / "media-task-ledger.json").read_text())
    task = next(row for row in ledger["tasks"] if row["image_id"] == failed_id)
    assert task["state"] == expected_state
    assert client.downloaded.count(failed_id) == 0


def test_direct_resolver_429_pauses_with_bounded_retry_after(tmp_path: Path) -> None:
    planned = _planned(1, 1)
    failed_id = planned[0].metadata.image_id
    client = _FakeClient(
        planned,
        resolution_events={
            failed_id: [
                MapillaryApiError(
                    "mapillary_api_rate_limit_retry_exhausted",
                    retry_after_seconds=71,
                )
            ]
        },
    )
    now = datetime(2026, 1, 1, tzinfo=UTC)

    assets, provenance = _acquire(tmp_path, planned, 1, client, now=now)

    assert assets == ()
    assert provenance["media_status"] == "PAUSED_RATE_LIMIT"
    assert provenance["retry_not_before"] == (now + timedelta(seconds=71)).isoformat()
    assert client.downloaded == []


def test_rate_limit_cooldown_prevents_early_network_retry(tmp_path: Path) -> None:
    planned = _planned(2, 0)
    client = _FakeClient(
        planned,
        events={
            planned[0].metadata.image_id: [
                MapillaryApiError(
                    "mapillary_media_rate_limit_retry_exhausted",
                    retry_after_seconds=123,
                )
            ]
        },
    )
    now = datetime(2026, 1, 1, tzinfo=UTC)

    _assets, first = _acquire(tmp_path, planned, 2, client, now=now)
    calls_at_pause = len(client.downloaded)
    _assets, second = _acquire(
        tmp_path,
        planned,
        2,
        client,
        now=now + timedelta(seconds=MEDIA_RATE_LIMIT_COOLDOWN_SECONDS - 1),
    )

    assert first["media_status"] == "PAUSED_RATE_LIMIT"
    assert first["retry_not_before"] == (now + timedelta(seconds=123)).isoformat()
    assert second["media_status"] == "PAUSED_RATE_LIMIT"
    assert len(client.downloaded) == calls_at_pause


def test_signed_url_is_resolved_once_then_item_is_rejected(tmp_path: Path) -> None:
    planned = _planned(1, 1)
    failed_id = planned[0].metadata.image_id
    client = _FakeClient(
        planned,
        events={failed_id: [MapillaryTokenError("mapillary_media_authorization_rejected")] * 2},
    )

    assets, provenance = _acquire(tmp_path, planned, 1, client)

    assert len(assets) == 1
    assert provenance["media_status"] == "MEDIA_READY"
    assert client.downloaded.count(failed_id) == 2
    assert [image_id for image_id, _quality in client.resolved].count(failed_id) == 2
    ledger = json.loads((tmp_path / "work" / "media-task-ledger.json").read_text())
    task = next(row for row in ledger["tasks"] if row["image_id"] == failed_id)
    assert task["state"] == "REJECTED"
    assert task["url_refresh_count"] == 1


@pytest.mark.parametrize("invalid_payload", [b"truncated", b""])
def test_invalid_image_is_rejected_and_replaced(tmp_path: Path, invalid_payload: bytes) -> None:
    planned = _planned(1, 1)
    client = _FakeClient(planned, events={planned[0].metadata.image_id: [invalid_payload]})

    assets, provenance = _acquire(tmp_path, planned, 1, client)

    assert len(assets) == 1
    assert provenance["media_status"] == "MEDIA_READY"
    assert provenance["media_task_counts"]["rejected"] == 1


def test_exact_and_perceptual_duplicates_use_reserves(tmp_path: Path) -> None:
    planned = _planned(3, 2)
    exact = _payload(50)
    client = _FakeClient(
        planned,
        events={
            planned[0].metadata.image_id: [exact],
            planned[1].metadata.image_id: [exact],
            planned[2].metadata.image_id: [_solid_payload((255, 0, 0))],
            planned[3].metadata.image_id: [_solid_payload((0, 0, 255))],
        },
    )

    assets, provenance = _acquire(tmp_path, planned, 3, client)

    assert len(assets) == 3
    assert provenance["media_status"] == "MEDIA_READY"
    assert provenance["media_task_counts"]["rejected"] == 2


def test_partial_and_downloading_state_recover_after_crash(tmp_path: Path) -> None:
    planned = _planned(1, 0)
    client = _FakeClient(planned)
    part = tmp_path / "work" / "private-media" / "assets" / ".orphan.jpg.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(b"partial")

    class Crash(RuntimeError):
        pass

    def crash_while_downloading() -> None:
        ledger_path = tmp_path / "work" / "media-task-ledger.json"
        if ledger_path.exists():
            ledger = json.loads(ledger_path.read_text())
            if any(row["state"] == "DOWNLOADING" for row in ledger["tasks"]):
                raise Crash

    with pytest.raises(Crash):
        _acquire(tmp_path, planned, 1, client, observer=crash_while_downloading)
    assert not part.exists()

    assets, provenance = _acquire(tmp_path, planned, 1, client)

    assert len(assets) == 1
    assert provenance["media_status"] == "MEDIA_READY"


def test_checkpoint_wins_over_preaccepted_ledger_crash(tmp_path: Path) -> None:
    planned = _planned(1, 0)
    client = _FakeClient(planned)

    class Crash(RuntimeError):
        pass

    def crash_after_asset_checkpoint() -> None:
        checkpoint = tmp_path / "work" / "acquisition-checkpoint.json"
        ledger_path = tmp_path / "work" / "media-task-ledger.json"
        if checkpoint.exists() and ledger_path.exists():
            ledger = json.loads(ledger_path.read_text())
            if any(row["state"] == "VERIFYING" for row in ledger["tasks"]):
                raise Crash

    with pytest.raises(Crash):
        _acquire(tmp_path, planned, 1, client, observer=crash_after_asset_checkpoint)
    calls_after_crash = len(client.downloaded)

    assets, provenance = _acquire(tmp_path, planned, 1, client)

    assert len(assets) == 1
    assert provenance["media_status"] == "MEDIA_READY"
    assert len(client.downloaded) == calls_after_crash


def test_v1_checkpoint_migration_preserves_accepted_reserve_without_overfill(
    tmp_path: Path,
) -> None:
    planned = _planned(2, 1)
    first = _FakeClient(
        planned,
        events={
            planned[1].metadata.image_id: [
                MapillaryApiError("mapillary_media_server_retry_exhausted")
            ]
        },
    )
    assets, provenance = _acquire(tmp_path, planned, 2, first)
    assert len(assets) == 2
    assert provenance["media_status"] == "MEDIA_READY"
    (tmp_path / "work" / "media-task-ledger.json").unlink()

    resumed = _FakeClient(planned)
    assets, provenance = _acquire(tmp_path, planned, 2, resumed)

    assert len(assets) == 2
    assert provenance["media_status"] == "MEDIA_READY"
    assert resumed.downloaded == []


def test_direct_resolver_migration_preserves_33_accepted_and_plan_hash(
    tmp_path: Path,
) -> None:
    planned = _planned(33, 3)
    first = _FakeClient(planned)
    assets, provenance = _acquire(tmp_path, planned, 33, first)
    assert len(assets) == 33
    assert provenance["media_status"] == "MEDIA_READY"

    ledger_path = tmp_path / "work" / "media-task-ledger.json"
    checkpoint_path = tmp_path / "work" / "acquisition-checkpoint.json"
    checkpoint_before = checkpoint_path.read_bytes()
    ledger = json.loads(ledger_path.read_text())
    plan_sha256 = ledger["planned_sha256"]
    ledger.pop("resolver_contract_version")
    ledger.pop("resolver_migration")
    reserves = [row for row in ledger["tasks"] if row["priority"] == "RESERVE"]
    reserves[0]["state"] = "RETRYABLE"
    reserves[0]["reason_code"] = "mapillary_api_server_retry_exhausted"
    reserves[1]["state"] = "REJECTED"
    reserves[1]["reason_code"] = "mapillary_image_not_found"
    reserves[2]["state"] = "QUARANTINED"
    reserves[2]["reason_code"] = "mapillary_media_server_retry_exhausted"
    ledger["provider_circuit"]["consecutive_failures"] = 5
    ledger["provider_circuit"]["open_count"] = 1
    ledger["pause_state"] = "PAUSED_PROVIDER_UNAVAILABLE"
    ledger["retry_not_before"] = "2099-01-01T00:00:00+00:00"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

    resumed = _FakeClient(planned)
    resumed_assets, resumed_provenance = _acquire(tmp_path, planned, 33, resumed)
    migrated = json.loads(ledger_path.read_text())

    assert len(resumed_assets) == 33
    assert resumed_provenance["media_status"] == "MEDIA_READY"
    assert resumed.downloaded == []
    assert checkpoint_path.read_bytes() == checkpoint_before
    assert migrated["planned_sha256"] == plan_sha256
    assert migrated["resolver_contract_version"] == MEDIA_RESOLVER_CONTRACT_VERSION
    assert migrated["resolver_migration"] == {
        "source": "collection-bbox-v1",
        "reset_retryable_count": 1,
    }
    migrated_reserves = [row for row in migrated["tasks"] if row["priority"] == "RESERVE"]
    assert migrated_reserves[0]["state"] == "PENDING"
    assert migrated_reserves[0]["reason_code"] is None
    assert migrated_reserves[1]["state"] == "REJECTED"
    assert migrated_reserves[1]["reason_code"] == "mapillary_image_not_found"
    assert migrated_reserves[2]["state"] == "QUARANTINED"
    assert migrated_reserves[2]["reason_code"] == "mapillary_media_server_retry_exhausted"
    assert sum(row["state"] == "ACCEPTED" for row in migrated["tasks"]) == 33


def test_additive_plan_download_loss_uses_next_reserve_and_resume_is_idempotent(
    tmp_path: Path,
) -> None:
    base = _planned(3, 0)
    first_client = _FakeClient(base)
    first_assets, first_provenance = _acquire(tmp_path, base, 3, first_client)
    assert len(first_assets) == 3
    assert first_provenance["media_status"] == "MEDIA_READY"
    checkpoint_path = tmp_path / "work" / "acquisition-checkpoint.json"
    original_checkpoint = json.loads(checkpoint_path.read_text())
    original_completed = tuple(original_checkpoint["completed"])

    expanded = _planned(3, 3)
    failed_new_id = expanded[3].metadata.image_id
    replacement_id = expanded[4].metadata.image_id
    second_client = _FakeClient(
        expanded,
        events={failed_new_id: [b"decode-failure"]},
    )
    second_assets, second_provenance = acquire_planned_assets(
        cast(MapillaryClient, second_client),
        (CityArea("TestCity", 30.0, 40.0, 0.01),),
        expanded,
        tmp_path / "work" / "private-media",
        "a" * 32,
        AcquisitionGuard(),
        checkpoint_path,
        restore_client_counts=False,
        allow_item_failures=True,
        primary_count=3,
        compatible_plan_sha256=(worker._planned_acquisition_sha256(base),),  # noqa: SLF001
        target_count_overrides={("TestCity", "reference"): 4},
    )

    assert len(second_assets) == 4
    assert second_provenance["media_status"] == "MEDIA_READY"
    assert second_client.downloaded == [failed_new_id, replacement_id]
    assert not ({item.metadata.image_id for item in base} & set(second_client.downloaded))
    migrated_checkpoint = json.loads(checkpoint_path.read_text())
    assert tuple(migrated_checkpoint["completed"][:3]) == original_completed

    resumed_client = _FakeClient(expanded)
    resumed_assets, resumed_provenance = acquire_planned_assets(
        cast(MapillaryClient, resumed_client),
        (CityArea("TestCity", 30.0, 40.0, 0.01),),
        expanded,
        tmp_path / "work" / "private-media",
        "a" * 32,
        AcquisitionGuard(),
        checkpoint_path,
        restore_client_counts=False,
        allow_item_failures=True,
        primary_count=3,
        compatible_plan_sha256=(worker._planned_acquisition_sha256(base),),  # noqa: SLF001
        target_count_overrides={("TestCity", "reference"): 4},
    )
    assert len(resumed_assets) == 4
    assert resumed_provenance["media_status"] == "MEDIA_READY"
    assert resumed_client.downloaded == []


def test_reserve_exhaustion_is_structured_not_generic(tmp_path: Path) -> None:
    planned = _planned(1, 0)
    client = _FakeClient(
        planned,
        events={
            planned[0].metadata.image_id: [
                MapillaryApiError("mapillary_media_server_retry_exhausted")
            ]
        },
    )

    assets, provenance = _acquire(tmp_path, planned, 1, client)

    assert assets == ()
    assert provenance["media_status"] == "MEDIA_SPLIT_MINIMUM_UNAVAILABLE"
    assert provenance["media_task_counts"]["quarantined"] == 1


def test_five_consecutive_server_failures_open_bounded_circuit(tmp_path: Path) -> None:
    planned = _planned(MEDIA_CIRCUIT_FAILURE_THRESHOLD + 1, 6)
    resolution_events = {
        item.metadata.image_id: [MapillaryApiError("mapillary_api_server_retry_exhausted")]
        for item in planned[:MEDIA_CIRCUIT_FAILURE_THRESHOLD]
    }
    client = _FakeClient(planned, resolution_events=resolution_events)
    now = datetime(2026, 1, 1, tzinfo=UTC)

    assets, provenance = _acquire(
        tmp_path, planned, MEDIA_CIRCUIT_FAILURE_THRESHOLD + 1, client, now=now
    )

    assert assets == ()
    assert provenance["media_status"] == "PAUSED_PROVIDER_UNAVAILABLE"
    assert provenance["provider_circuit"]["open_count"] == 1
    assert provenance["media_task_counts"]["quarantined"] == 5
    assert (
        provenance["retry_not_before"]
        == (now + timedelta(seconds=MEDIA_PROVIDER_COOLDOWN_SECONDS)).isoformat()
    )


def test_twenty_resumes_rotate_rate_limited_candidate(tmp_path: Path) -> None:
    planned = _planned(20, 0)
    events = {
        item.metadata.image_id: [
            MapillaryApiError("mapillary_media_rate_limit_retry_exhausted"),
            MapillaryApiError("mapillary_media_rate_limit_retry_exhausted"),
        ]
        for item in planned
    }
    client = _FakeClient(planned, events=events)
    started = datetime(2026, 1, 1, tzinfo=UTC)

    for resume_index in range(20):
        _assets, provenance = _acquire(
            tmp_path,
            planned,
            20,
            client,
            now=started + timedelta(seconds=(MEDIA_RATE_LIMIT_COOLDOWN_SECONDS + 1) * resume_index),
        )
        assert provenance["media_status"] == "PAUSED_RATE_LIMIT"

    assert len(client.downloaded) == 20
    assert len(set(client.downloaded)) == 20

    for resume_index in range(20, 40):
        _assets, provenance = _acquire(
            tmp_path,
            planned,
            20,
            client,
            now=started + timedelta(seconds=(MEDIA_RATE_LIMIT_COOLDOWN_SECONDS + 1) * resume_index),
        )
        assert provenance["media_status"] == "PAUSED_RATE_LIMIT"

    calls_after_bounded_attempts = len(client.downloaded)
    _assets, final = _acquire(
        tmp_path,
        planned,
        20,
        client,
        now=started + timedelta(seconds=(MEDIA_RATE_LIMIT_COOLDOWN_SECONDS + 1) * 40),
    )
    assert final["media_status"] == "MEDIA_SPLIT_MINIMUM_UNAVAILABLE"
    assert len(client.downloaded) == calls_after_bounded_attempts
    assert max(client.downloaded.count(image_id) for image_id in set(client.downloaded)) == 2


def test_runpod_launcher_is_after_media_pause_gate() -> None:
    source = END_TO_END_LAUNCHER.read_text(encoding="utf-8")

    gate = source.index('$localResult.stage -like "PAUSED_*"')
    cloud_call = source.index("Invoke-CloudTraining -RunId")
    assert gate < cloud_call
    assert '"MEDIA_SPLIT_MINIMUM_UNAVAILABLE"' in source
    assert '"MEDIA_CORPUS_EXHAUSTED"' in source
    assert '"MEDIA_REPLENISHMENT_LIMIT_REACHED"' in source
