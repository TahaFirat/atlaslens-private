from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

from atlaslens_api.database import create_database_engine
from atlaslens_api.history import history_item
from atlaslens_api.repository import AnalysisHistoryFilters, SQLAlchemyAnalysisRepository
from atlaslens_api.schemas import (
    Analysis,
    AnalysisMode,
    AnalysisStatus,
    ImageSummary,
    Progress,
    SimulationSummary,
)


def _analysis(identifier: int, *, simulated: bool, created_at: datetime) -> Analysis:
    return Analysis(
        id=UUID(int=identifier),
        status=AnalysisStatus.QUEUED,
        analysis_mode=AnalysisMode.LOCAL_ONLY,
        created_at=created_at,
        expires_at=created_at + timedelta(hours=1),
        progress=Progress(stage="queued", percent=0, message_key="progress.queued"),
        image=ImageSummary(
            format="jpeg",
            width=640,
            height=480,
            megapixels=0.3072,
            sha256=f"{identifier:064x}",
            orientation_normalized=True,
            exif_present=False,
        ),
        evidence=[],
        candidates=[],
        warnings=[],
        timings_ms={},
        fusion_policy_version="phase1-fusion-v1",
        result_classification="simulated" if simulated else "real",
        simulation=(
            SimulationSummary(scenario_id="safe_demo") if simulated else None
        ),
    )


async def test_history_persists_simulation_and_lists_with_filters() -> None:
    repository = SQLAlchemyAnalysisRepository(create_database_engine("sqlite:///:memory:"))
    await repository.initialize()
    now = datetime.now(UTC)
    await repository.create(
        _analysis(1, simulated=False, created_at=now),
        storage_key=None,
        idempotency_hash=None,
    )
    await repository.create(
        _analysis(2, simulated=True, created_at=now + timedelta(seconds=1)),
        storage_key="a" * 32 + ".upload",
        idempotency_hash=None,
    )

    page, total = await repository.list_history(
        filters=AnalysisHistoryFilters(classification="simulated"), limit=10, offset=0
    )
    assert total == 1
    assert page[0].analysis.result_classification == "simulated"
    assert page[0].analysis.simulation is not None
    assert page[0].analysis.simulation.watermark == "SIMULATED DEVELOPMENT RESULT"
    summary = history_item(page[0])
    assert summary.source_retained is True
    assert summary.image_sha256 == f"{2:064x}"
    assert summary.provider_ids == []


async def test_history_pagination_is_newest_first() -> None:
    repository = SQLAlchemyAnalysisRepository(create_database_engine("sqlite:///:memory:"))
    await repository.initialize()
    now = datetime.now(UTC)
    for identifier in range(1, 4):
        await repository.create(
            _analysis(
                identifier,
                simulated=False,
                created_at=now + timedelta(seconds=identifier),
            ),
            storage_key=None,
            idempotency_hash=None,
        )

    page, total = await repository.list_history(
        filters=AnalysisHistoryFilters(), limit=1, offset=1
    )
    assert total == 3
    assert page[0].analysis.id == UUID(int=2)
