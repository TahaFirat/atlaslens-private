from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest

from atlaslens_api.mapillary_demo.client import MapillaryClient, RemoteImage
from atlaslens_api.mapillary_demo.errors import MapillaryApiError, MapillarySafetyError
from atlaslens_api.mapillary_demo.models import ClientLimits, ImageMetadata
from atlaslens_api.phase3f.cloud_job import (
    LEGACY_METADATA_PAGE_CHECKPOINT_SCHEMA,
    LEGACY_V2_METADATA_PAGE_CHECKPOINT_SCHEMA,
    MAX_ADAPTIVE_PARTITION_DEPTH,
    METADATA_PAGE_CHECKPOINT_SCHEMA,
    CityArea,
    Phase3FCloudJobError,
    _load_metadata_page_checkpoint,
    _new_metadata_page_checkpoint,
    _subdivide_failed_metadata_cell,
    _write_metadata_page_checkpoint,
    audit_metadata,
    migrate_and_subdivide_failed_pagination_checkpoint,
    quarter_bbox,
)
from atlaslens_api.phase3f.coverage import CoverageInsufficient

ROOT = Path(__file__).parents[3]
MANUAL_SCRIPT = ROOT / "scripts" / "phase3f" / "existing_pod_job.py"
SECRET_STATUS_WRAPPER = ROOT / "scripts" / "phase3f" / "existing-pod-secret-status.sh"
FAKE_TOKEN = "MLY_fixture-token-not-a-secret"
UNRESOLVED_REFERENCE = "{{ RUNPOD_SECRET_atlaslens_mapillary_access_token }}"


def _native_bash() -> str | None:
    if os.name == "nt":
        for candidate in (
            Path("C:/Program Files/Git/bin/bash.exe"),
            Path("C:/Program Files/Git/usr/bin/bash.exe"),
        ):
            if candidate.is_file():
                return str(candidate)
        return None
    return shutil.which("bash")


NATIVE_BASH = _native_bash()


def _row(identifier: str) -> dict[str, object]:
    return {
        "id": identifier,
        "computed_geometry": {"type": "Point", "coordinates": [32.85, 39.93]},
        "captured_at": "2024-01-01T00:00:00+00:00",
        "sequence": {"id": f"sequence-{identifier}"},
        "creator": {"id": f"creator-{identifier}"},
        "width": 1024,
        "height": 768,
    }


def _manual_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("phase3f_existing_pod_job_test", MANUAL_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _subprocess_environment(token: str | None) -> dict[str, str]:
    environment = {
        name: value
        for name in ("PATH", "SYSTEMROOT", "TEMP", "TMP")
        if (value := os.environ.get(name)) is not None
    }
    environment["PYTHONIOENCODING"] = "utf-8"
    if token is not None:
        environment["MAPILLARY_ACCESS_TOKEN"] = token
    return environment


def test_pagination_failure_subdivides_cell_and_resume_preserves_completed_page(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "work" / "metadata-pages.json"
    area = CityArea("Ankara", 32.85, 39.93, 0.01)
    state = _new_metadata_page_checkpoint((area,), "a" * 32)
    first_calls = 0

    def first_handler(_request: httpx.Request) -> httpx.Response:
        nonlocal first_calls
        first_calls += 1
        if first_calls == 1:
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
        return httpx.Response(503, text=f"unsafe {FAKE_TOKEN}")

    limits = ClientLimits(request_cap=20, page_cap=20, retry_cap=0)
    with httpx.Client(transport=httpx.MockTransport(first_handler)) as http_client:
        client = MapillaryClient(FAKE_TOKEN, limits=limits, http_client=http_client)
        with pytest.raises(MapillaryApiError, match="mapillary_api_server_retry_exhausted"):
            audit_metadata(
                client,
                (area,),
                checkpoint=state,
                checkpoint_path=checkpoint_path,
                run_id="a" * 32,
            )

    persisted = checkpoint_path.read_text(encoding="utf-8")
    assert FAKE_TOKEN not in persisted
    assert "access_token" not in persisted
    resumed = _load_metadata_page_checkpoint(
        checkpoint_path,
        areas=(area,),
        run_id="a" * 32,
    )
    observed: list[httpx.Request] = []

    def resume_handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return httpx.Response(200, json={"data": [_row("image-2")]})

    with httpx.Client(transport=httpx.MockTransport(resume_handler)) as http_client:
        client = MapillaryClient(FAKE_TOKEN, limits=limits, http_client=http_client)
        with pytest.raises(CoverageInsufficient):
            audit_metadata(
                client,
                (area,),
                checkpoint=resumed,
                checkpoint_path=checkpoint_path,
                run_id="a" * 32,
            )

    # The failed cursor is cleared and its cell is replaced in place by four
    # stable SW, SE, NW, NE children. The other 15 base cells retain their order.
    assert len(observed) == 19
    assert all("after" not in parse_qs(request.url.query.decode()) for request in observed)
    assert [
        parse_qs(request.url.query.decode())["bbox"][0] for request in observed
    ] == [
        *(child.as_query_value() for box in area.boxes[0:1] for child in quarter_bbox(box)),
        *(box.as_query_value() for box in area.boxes[1:]),
    ]
    document = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert document["schema"] == METADATA_PAGE_CHECKPOINT_SCHEMA
    assert document["subdivision_count"] == 1
    assert [row["mapillary_image_id"] for row in document["rows_by_city"]["Ankara"]] == [
        "image-1",
        "image-2",
    ]


def test_boundary_image_ids_are_deduplicated_in_stable_cell_order() -> None:
    area = CityArea("Ankara", 32.85, 39.93, 0.01)
    observed_bbox: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        query = parse_qs(request.url.query.decode())
        observed_bbox.append(query["bbox"][0])
        return httpx.Response(200, json={"data": [_row("boundary-image")]})

    limits = ClientLimits(
        request_cap=20,
        page_cap=20,
        metadata_item_cap=20,
        page_size=1,
        retry_cap=0,
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = MapillaryClient(FAKE_TOKEN, limits=limits, http_client=http_client)
        images = tuple(
            client.iter_images(area.boxes, include_thumbnail=False, item_cap=20)
        )

    assert [image.metadata.mapillary_image_id for image in images] == ["boundary-image"]
    assert observed_bbox == [box.as_query_value() for box in area.boxes]


def test_empty_v1_checkpoint_is_atomically_migrated_to_partition_plan(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "work" / "metadata-pages.json"
    area = CityArea("Ankara", 32.85, 39.93, 0.01)
    state = _new_metadata_page_checkpoint((area,), "c" * 32)
    _write_metadata_page_checkpoint(checkpoint_path, state)
    legacy = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    legacy["schema"] = LEGACY_METADATA_PAGE_CHECKPOINT_SCHEMA
    for key in (
        "base_cell_plan_sha256",
        "cell_plan_sha256",
        "cells_by_city",
        "subdivision_count",
    ):
        del legacy[key]
    checkpoint_path.write_text(json.dumps(legacy), encoding="utf-8")

    migrated = _load_metadata_page_checkpoint(
        checkpoint_path,
        areas=(area,),
        run_id="c" * 32,
    )

    persisted = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert persisted["schema"] == METADATA_PAGE_CHECKPOINT_SCHEMA
    assert persisted["cell_plan_sha256"] == migrated.cell_plan_sha256
    assert migrated.city_index == 0
    assert migrated.box_index == 0
    assert all(not rows for rows in migrated.rows_by_city.values())


def test_progressed_v1_checkpoint_partition_migration_fails_closed(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "work" / "metadata-pages.json"
    area = CityArea("Ankara", 32.85, 39.93, 0.01)
    state = _new_metadata_page_checkpoint((area,), "d" * 32)
    _write_metadata_page_checkpoint(checkpoint_path, state)
    legacy = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    legacy["schema"] = LEGACY_METADATA_PAGE_CHECKPOINT_SCHEMA
    legacy["box_index"] = 1
    for key in (
        "base_cell_plan_sha256",
        "cell_plan_sha256",
        "cells_by_city",
        "subdivision_count",
    ):
        del legacy[key]
    checkpoint_path.write_text(json.dumps(legacy), encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match="METADATA_PAGE_CHECKPOINT_PARTITION_MIGRATION_UNSAFE",
    ):
        _load_metadata_page_checkpoint(
            checkpoint_path,
            areas=(area,),
            run_id="d" * 32,
        )


def test_progressed_v2_checkpoint_migrates_to_v3_without_row_or_cursor_loss(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "work" / "metadata-pages.json"
    area = CityArea("Ankara", 32.85, 39.93, 0.01)
    state = _new_metadata_page_checkpoint((area,), "e" * 32)
    state.rows_by_city["Ankara"].append(
        RemoteImage(
            ImageMetadata.model_validate(
                {
                    "mapillary_image_id": "preserved-image",
                    "computed_geometry": {
                        "type": "Point",
                        "coordinates": (32.85, 39.93),
                    },
                    "captured_at": "2024-01-01T00:00:00+00:00",
                    "sequence_id": "sequence-preserved-image",
                    "creator_id": "creator-preserved-image",
                    "width_px": 1024,
                    "height_px": 768,
                }
            )
        )
    )
    _write_metadata_page_checkpoint(checkpoint_path, state)
    version_two = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    version_two["schema"] = LEGACY_V2_METADATA_PAGE_CHECKPOINT_SCHEMA
    version_two["cell_plan_sha256"] = version_two["base_cell_plan_sha256"]
    version_two["next_url"] = "https://graph.mapillary.com/images?after=cursor-v2"
    version_two["visited_page_sha256"] = ["f" * 64]
    expected_rows = version_two["rows_by_city"]["Ankara"]
    for key in ("base_cell_plan_sha256", "cells_by_city", "subdivision_count"):
        del version_two[key]
    checkpoint_path.write_text(json.dumps(version_two), encoding="utf-8")

    migrated = _load_metadata_page_checkpoint(
        checkpoint_path,
        areas=(area,),
        run_id="e" * 32,
    )

    persisted = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert persisted["schema"] == METADATA_PAGE_CHECKPOINT_SCHEMA
    assert migrated.next_url == "https://graph.mapillary.com/images?after=cursor-v2"
    assert migrated.visited_page_sha256 == ("f" * 64,)
    assert [
        row.metadata.mapillary_image_id for row in migrated.rows_by_city["Ankara"]
    ] == ["preserved-image"]
    assert persisted["rows_by_city"]["Ankara"] == expected_rows
    assert persisted["subdivision_count"] == 0


def test_failed_v2_first_request_migration_and_subdivision_use_one_atomic_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "work" / "metadata-pages.json"
    area = CityArea("Ankara", 32.85, 39.93, 0.01)
    state = _new_metadata_page_checkpoint((area,), "4" * 32)
    state.rows_by_city["Ankara"].append(
        RemoteImage(
            ImageMetadata.model_validate(
                {
                    "mapillary_image_id": "preserved-image",
                    "computed_geometry": {
                        "type": "Point",
                        "coordinates": (32.85, 39.93),
                    },
                    "captured_at": "2024-01-01T00:00:00+00:00",
                }
            )
        )
    )
    _write_metadata_page_checkpoint(checkpoint_path, state)
    version_two = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    version_two["schema"] = LEGACY_V2_METADATA_PAGE_CHECKPOINT_SCHEMA
    version_two["cell_plan_sha256"] = version_two["base_cell_plan_sha256"]
    version_two["next_url"] = None
    for key in ("base_cell_plan_sha256", "cells_by_city", "subdivision_count"):
        del version_two[key]
    checkpoint_path.write_text(json.dumps(version_two), encoding="utf-8")
    cloud_job_module = importlib.import_module("atlaslens_api.phase3f.cloud_job")
    original_atomic_write = cloud_job_module._atomic_private_json
    atomic_writes = 0

    def observe_atomic_write(path: Path, value: object) -> str:
        nonlocal atomic_writes
        atomic_writes += 1
        return original_atomic_write(path, value)

    monkeypatch.setattr(cloud_job_module, "_atomic_private_json", observe_atomic_write)

    result = migrate_and_subdivide_failed_pagination_checkpoint(
        checkpoint_path,
        areas=(area,),
        run_id="4" * 32,
    )

    persisted = _load_metadata_page_checkpoint(
        checkpoint_path,
        areas=(area,),
        run_id="4" * 32,
    )
    assert atomic_writes == 1
    assert result["rows_preserved"] == 1
    assert result["rows_sha256_preserved"] is True
    assert result["cursor_cleared"] is True
    assert result["failed_request_kind"] == "cell_first_page"
    assert persisted.subdivision_count == 1
    assert persisted.next_url is None
    assert [
        row.metadata.mapillary_image_id for row in persisted.rows_by_city["Ankara"]
    ] == ["preserved-image"]


def test_first_request_server_failure_persists_child_plan_for_next_resume(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "work" / "metadata-pages.json"
    area = CityArea("Ankara", 32.85, 39.93, 0.01)
    state = _new_metadata_page_checkpoint((area,), "1" * 32)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="safe fixture failure")

    limits = ClientLimits(request_cap=2, page_cap=2, retry_cap=0)
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = MapillaryClient(FAKE_TOKEN, limits=limits, http_client=http_client)
        with pytest.raises(MapillaryApiError, match="mapillary_api_server_retry_exhausted"):
            audit_metadata(
                client,
                (area,),
                checkpoint=state,
                checkpoint_path=checkpoint_path,
                run_id="1" * 32,
            )

    persisted = _load_metadata_page_checkpoint(
        checkpoint_path,
        areas=(area,),
        run_id="1" * 32,
    )
    assert persisted.next_url is None
    assert persisted.visited_page_sha256 == ()
    assert persisted.subdivision_count == 1
    assert [
        cell.bbox.as_query_value() for cell in persisted.cells_by_city["Ankara"][:4]
    ] == [child.as_query_value() for child in quarter_bbox(area.boxes[0])]
    assert [cell.depth for cell in persisted.cells_by_city["Ankara"][:4]] == [1, 1, 1, 1]
    assert persisted.rows_by_city["Ankara"] == []


def test_adaptive_partition_depth_limit_is_typed_and_preserves_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "work" / "metadata-pages.json"
    area = CityArea("Ankara", 32.85, 39.93, 0.08)
    state = _new_metadata_page_checkpoint((area,), "2" * 32)
    for _ in range(MAX_ADAPTIVE_PARTITION_DEPTH):
        _subdivide_failed_metadata_cell(
            state,
            areas=(area,),
            checkpoint_path=checkpoint_path,
        )
    before = checkpoint_path.read_bytes()

    with pytest.raises(
        Phase3FCloudJobError,
        match="MAPILLARY_PARTITION_DEPTH_LIMIT_REACHED",
    ):
        _subdivide_failed_metadata_cell(
            state,
            areas=(area,),
            checkpoint_path=checkpoint_path,
        )

    assert checkpoint_path.read_bytes() == before
    assert state.subdivision_count == MAX_ADAPTIVE_PARTITION_DEPTH


def test_adaptive_partition_minimum_area_is_typed_and_preserves_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "work" / "metadata-pages.json"
    area = CityArea("Ankara", 32.85, 39.93, 0.01)
    state = _new_metadata_page_checkpoint((area,), "3" * 32)
    for _ in range(2):
        _subdivide_failed_metadata_cell(
            state,
            areas=(area,),
            checkpoint_path=checkpoint_path,
        )
    before = checkpoint_path.read_bytes()

    with pytest.raises(
        Phase3FCloudJobError,
        match="MAPILLARY_PARTITION_MIN_AREA_REACHED",
    ):
        _subdivide_failed_metadata_cell(
            state,
            areas=(area,),
            checkpoint_path=checkpoint_path,
        )

    assert checkpoint_path.read_bytes() == before
    assert state.subdivision_count == 2


def test_local_partition_limit_errors_are_terminal() -> None:
    module = importlib.import_module("atlaslens_api.phase3f.local_first")
    assert (
        module._acquisition_failure_stage("MAPILLARY_PARTITION_DEPTH_LIMIT_REACHED")
        == "ACQUISITION_FAILED_TERMINAL"
    )
    assert (
        module._acquisition_failure_stage("MAPILLARY_PARTITION_MIN_AREA_REACHED")
        == "ACQUISITION_FAILED_TERMINAL"
    )
    assert (
        module._acquisition_failure_stage("MAPILLARY_API_SERVER_RETRY_EXHAUSTED")
        == "ACQUISITION_FAILED_RESUMABLE"
    )


def test_metadata_resume_retains_cursor_loop_guard(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "work" / "metadata-pages.json"
    area = CityArea("Ankara", 32.85, 39.93, 0.01)
    state = _new_metadata_page_checkpoint((area,), "b" * 32)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [], "paging": {"next": "/images?after=cursor-loop"}},
        )

    limits = ClientLimits(request_cap=3, page_cap=3, retry_cap=0)
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = MapillaryClient(FAKE_TOKEN, limits=limits, http_client=http_client)
        with pytest.raises(MapillarySafetyError, match="mapillary_paging_loop_detected"):
            audit_metadata(
                client,
                (area,),
                checkpoint=state,
                checkpoint_path=checkpoint_path,
                run_id="b" * 32,
            )


def test_manual_prepare_creates_only_run_parent_and_current_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _manual_module()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setattr(module.secrets, "token_hex", lambda _size: "c" * 32)
    monkeypatch.setattr(module, "_prepare_readiness", lambda *_args: None)
    monkeypatch.setattr(module, "_existing_cloud_jobs", lambda: ())
    monkeypatch.setattr(
        module,
        "_attest_mapillary_secret_inheritance",
        lambda: "RESOLVED_SECRET",
    )

    result = module.main(
        [
            "prepare-check",
            "--runtime-root",
            str(runtime),
            "--repository-root",
            str(ROOT),
            "--model",
            str(tmp_path / "model.safetensors"),
            "--vendor-root",
            str(tmp_path / "vendor"),
            "--source-commit",
            "1" * 40,
        ]
    )

    run_parent = runtime / ("c" * 32)
    assert result == 0
    assert run_parent.is_dir()
    assert sorted(path.name for path in run_parent.iterdir()) == []
    assert not (run_parent / "work").exists()
    assert not any(run_parent.glob("output-*"))
    assert json.loads((runtime / "current-root.json").read_text())["run_id"] == "c" * 32


def test_manual_start_is_single_job_fresh_output_and_stale_pid_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _manual_module()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    run_id = "d" * 32
    run_parent = runtime / run_id
    run_parent.mkdir()
    (runtime / "current-root.json").write_text(
        json.dumps({"schema": "atlaslens-phase3f-current-root-v1", "run_id": run_id})
    )
    monkeypatch.setattr(module, "_prepare_readiness", lambda *_args: None)
    monkeypatch.setattr(module, "_existing_cloud_jobs", lambda: ())
    monkeypatch.setattr(module, "_proc_identity", lambda _pid: ("123", "e" * 64))
    monkeypatch.setenv("MAPILLARY_ACCESS_TOKEN", FAKE_TOKEN)
    monkeypatch.setattr(
        module,
        "_attest_mapillary_secret_inheritance",
        lambda: "RESOLVED_SECRET",
    )
    fake_fcntl = SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *_args: None)
    monkeypatch.setattr(module.importlib, "import_module", lambda _name: fake_fcntl)
    commands: list[list[str]] = []
    child_environments: list[dict[str, str]] = []

    class FakeProcess:
        pid = 4321

        def __init__(self, command: list[str], **_kwargs: Any) -> None:
            commands.append(command)
            child_environments.append(_kwargs["env"])

        def poll(self) -> None:
            return None

        def wait(self, *, timeout: int) -> int:
            return 0

    monkeypatch.setattr(module.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    arguments = [
        "start-job",
        "--runtime-root",
        str(runtime),
        "--repository-root",
        str(ROOT),
        "--model",
        str(tmp_path / "model.safetensors"),
        "--vendor-root",
        str(tmp_path / "vendor"),
        "--source-commit",
        "1" * 40,
    ]

    assert module.main(arguments) == 0
    first_output = capsys.readouterr().out
    assert first_output.splitlines() == ["RESOLVED_SECRET", "PHASE3F_IN_POD_JOB_STARTED"]
    assert FAKE_TOKEN not in first_output
    assert len(commands) == 1
    assert FAKE_TOKEN not in "\n".join(commands[0])
    assert child_environments[0]["MAPILLARY_ACCESS_TOKEN"] == FAKE_TOKEN
    assert "RUNPOD_API_KEY" not in child_environments[0]
    assert commands[0].count(str(ROOT / "scripts" / "phase3f" / "cloud_job.py")) == 1
    assert "--resume" not in commands[0]
    assert not (run_parent / "work").exists()
    output_path = Path(commands[0][commands[0].index("--output-root") + 1])
    assert not output_path.exists()
    for private_file in run_parent.rglob("*"):
        if private_file.is_file():
            assert FAKE_TOKEN.encode() not in private_file.read_bytes()
    assert module.main(arguments) == 1
    assert len(commands) == 1

    monkeypatch.setattr(module, "_proc_identity", lambda _pid: ("changed", "f" * 64))
    assert module.main(arguments) == 1
    assert len(commands) == 1


def test_manual_log_projection_never_emits_unknown_text() -> None:
    module = _manual_module()
    assert module._safe_log_line("MAPILLARY_API_BAD_REQUEST") == "MAPILLARY_API_BAD_REQUEST"
    assert module._safe_log_line(f"Authorization: Bearer {FAKE_TOKEN}") == (
        "[REDACTED_UNSAFE_LOG_LINE]"
    )
    assert module._safe_log_line("https://graph.mapillary.com/images?after=private") == (
        "[REDACTED_UNSAFE_LOG_LINE]"
    )


def test_manual_vendor_hash_is_canonical_for_lf_and_crlf(tmp_path: Path) -> None:
    module = _manual_module()
    lf = b"line one\nline two\n"
    lf_path = tmp_path / "lf.py"
    crlf_path = tmp_path / "crlf.py"
    mixed_path = tmp_path / "mixed.py"
    lf_path.write_bytes(lf)
    crlf_path.write_bytes(lf.replace(b"\n", b"\r\n"))
    mixed_path.write_bytes(b"line one\r\nline two\n")
    expected = hashlib.sha256(lf).hexdigest()

    assert module._canonical_lf_sha256(lf_path, max_bytes=128) == expected
    assert module._canonical_lf_sha256(crlf_path, max_bytes=128) == expected
    with pytest.raises(module.ManualJobError, match="PHASE3F_VENDOR_MIXED_NEWLINES_REFUSED"):
        module._canonical_lf_sha256(mixed_path, max_bytes=128)


def test_manual_child_environment_excludes_unrelated_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _manual_module()
    monkeypatch.setenv("MAPILLARY_ACCESS_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("RUNPOD_API_KEY", "fixture-runpod-value")
    monkeypatch.setenv("UNRELATED_CREDENTIAL", "fixture-unrelated-value")
    monkeypatch.setenv("PATH", "/usr/bin")

    environment = module._child_environment(include_mapillary_token=True)

    assert environment["MAPILLARY_ACCESS_TOKEN"] == FAKE_TOKEN
    assert environment["PATH"] == "/usr/bin"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert "RUNPOD_API_KEY" not in environment
    assert "UNRELATED_CREDENTIAL" not in environment


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (FAKE_TOKEN, "RESOLVED_SECRET"),
        (UNRESOLVED_REFERENCE, "UNRESOLVED_RUNPOD_SECRET_REFERENCE"),
        (None, "MAPILLARY_ACCESS_TOKEN_MISSING"),
        ("fixture-invalid-format", "MAPILLARY_ACCESS_TOKEN_INVALID_FORMAT"),
    ],
)
def test_manual_secret_classification_is_presence_only(
    value: str | None,
    expected: str,
) -> None:
    module = _manual_module()
    assert module._classify_mapillary_secret(value) == expected


@pytest.mark.parametrize(
    ("token", "expected", "returncode"),
    [
        (FAKE_TOKEN, "RESOLVED_SECRET", 0),
        (UNRESOLVED_REFERENCE, "UNRESOLVED_RUNPOD_SECRET_REFERENCE", 1),
        (None, "MAPILLARY_ACCESS_TOKEN_MISSING", 1),
    ],
)
def test_secret_status_real_controller_and_child_subprocess_chain(
    token: str | None,
    expected: str,
    returncode: int,
) -> None:
    command = [sys.executable, str(MANUAL_SCRIPT), "secret-status"]
    result = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        env=_subprocess_environment(token),
        check=False,
        timeout=30,
        text=True,
    )

    assert result.returncode == returncode
    assert result.stdout.strip() == expected
    assert result.stderr == ""
    if token is not None:
        assert token not in result.stdout
        assert token not in result.stderr
        assert token not in "\n".join(command)


@pytest.mark.skipif(
    NATIVE_BASH is None,
    reason="native bash unavailable",
)
def test_secret_status_shell_wrapper_inherits_only_exported_environment(
    tmp_path: Path,
) -> None:
    assert NATIVE_BASH is not None
    environment = _subprocess_environment(FAKE_TOKEN)
    if os.name == "nt":
        launcher = tmp_path / "python3"
        executable = Path(sys.executable).as_posix()
        launcher.write_text(
            f"#!/usr/bin/env bash\nexec '{executable}' \"$@\"\n",
            encoding="utf-8",
        )
        launcher.chmod(0o755)
        environment["PATH"] = f"{tmp_path}{os.pathsep}{environment['PATH']}"
    command = [NATIVE_BASH, str(SECRET_STATUS_WRAPPER)]
    resolved = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        env=environment,
        check=False,
        timeout=30,
        text=True,
    )
    missing = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        env={
            key: value
            for key, value in environment.items()
            if key != "MAPILLARY_ACCESS_TOKEN"
        },
        check=False,
        timeout=30,
        text=True,
    )

    assert resolved.returncode == 0
    assert resolved.stdout.strip() == "RESOLVED_SECRET"
    assert missing.returncode == 1
    assert missing.stdout.strip() == "MAPILLARY_ACCESS_TOKEN_MISSING"
    combined = resolved.stdout + resolved.stderr + missing.stdout + missing.stderr
    assert FAKE_TOKEN not in combined
    assert FAKE_TOKEN not in "\n".join(command)


def test_failed_prepare_creates_no_current_root_and_start_never_launches(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    common = [
        "--runtime-root",
        str(runtime),
        "--repository-root",
        str(ROOT),
        "--model",
        str(tmp_path / "missing-model.safetensors"),
        "--vendor-root",
        str(tmp_path / "missing-vendor"),
        "--source-commit",
        "1" * 40,
    ]
    prepare_command = [sys.executable, str(MANUAL_SCRIPT), "prepare-check", *common]
    prepare = subprocess.run(
        prepare_command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        env=_subprocess_environment(None),
        check=False,
        timeout=30,
        text=True,
    )

    assert prepare.returncode == 1
    assert prepare.stdout.strip() == "MAPILLARY_ACCESS_TOKEN_MISSING"
    assert prepare.stderr == ""
    assert not (runtime / "current-root.json").exists()
    assert not any(runtime.glob("*/job-state.json"))
    assert not any(runtime.glob("*/job.log"))

    start_command = [sys.executable, str(MANUAL_SCRIPT), "start-job", *common]
    start = subprocess.run(
        start_command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        env=_subprocess_environment(FAKE_TOKEN),
        check=False,
        timeout=30,
        text=True,
    )

    assert start.returncode == 1
    assert start.stdout.strip() == "PHASE3F_CURRENT_ROOT_MISSING"
    assert start.stderr == ""
    assert FAKE_TOKEN not in start.stdout
    assert FAKE_TOKEN not in start.stderr
    assert FAKE_TOKEN not in "\n".join(start_command)
    assert not any(runtime.glob("*/job-state.json"))
    assert not any(runtime.glob("*/job.log"))


def test_manual_status_and_stop_use_only_bound_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _manual_module()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    run_id = "f" * 32
    run_parent = runtime / run_id
    run_parent.mkdir()
    (runtime / "current-root.json").write_text(
        json.dumps({"schema": "atlaslens-phase3f-current-root-v1", "run_id": run_id})
    )
    state = {
        "schema": "atlaslens-phase3f-in-pod-job-v1",
        "run_id": run_id,
        "pid": 4321,
        "proc_start_ticks": "123",
        "command_sha256": "e" * 64,
        "output_name": "output-1-fixture",
        "deadline_epoch": 123456,
        "resume": False,
    }
    (run_parent / "job-state.json").write_text(json.dumps(state))
    validated: list[int] = []
    terminated: list[int] = []
    monkeypatch.setattr(
        module,
        "_validated_process",
        lambda value: validated.append(value["pid"]) or value["pid"],
    )
    monkeypatch.setattr(
        module,
        "_terminate_process_group",
        lambda pid: terminated.append(pid),
    )
    monkeypatch.setattr(
        module,
        "_proc_identity",
        lambda _pid: (_ for _ in ()).throw(module.ManualJobError("PHASE3F_JOB_NOT_RUNNING")),
    )

    assert module.main(["status-job", "--runtime-root", str(runtime)]) == 0
    assert capsys.readouterr().out.strip() == "PHASE3F_IN_POD_JOB_RUNNING"
    assert module.main(["stop-job", "--runtime-root", str(runtime)]) == 0
    assert capsys.readouterr().out.strip() == "PHASE3F_IN_POD_JOB_STOPPED"
    assert validated == [4321, 4321]
    assert terminated == [4321]
