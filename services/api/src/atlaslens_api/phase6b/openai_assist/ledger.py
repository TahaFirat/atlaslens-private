from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from atlaslens_api.phase6b.openai_assist.models import (
    OpenAIReviewStructuredOutput,
    OpenAIReviewUsage,
)

CallFinalStatus = Literal[
    "success",
    "failure",
    "refused",
    "cancelled",
    "timeout",
    "quota_error",
    "invalid_output",
]

_MAX_CACHE_RESULT_BYTES = 64 * 1024


class _LedgerModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class OpenAIBudgetSummary(_LedgerModel):
    cloud_calls_today: int = Field(ge=0)
    estimated_spend_this_month_usd: Decimal = Field(ge=0, decimal_places=8)
    remaining_configured_budget_usd: Decimal = Field(ge=0, decimal_places=8)
    monthly_budget_usd: Decimal = Field(gt=0, decimal_places=8)


class OpenAICallReservation(_LedgerModel):
    allowed: bool
    reservation_id: int | None = Field(default=None, ge=1)
    reason_code: Literal[
        "reserved",
        "daily_call_limit_reached",
        "monthly_budget_exhausted",
        "per_analysis_call_limit_reached",
    ]
    summary: OpenAIBudgetSummary


class OpenAIUsageLedgerRecord(_LedgerModel):
    timestamp: datetime
    model: str = Field(min_length=1, max_length=120)
    prompt_version: str = Field(min_length=1, max_length=80)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    estimated_cost: Decimal | None = Field(default=None, ge=0, decimal_places=8)
    cost_estimate_version: str | None = Field(default=None, min_length=1, max_length=80)
    cost_basis: Literal["usage_based", "reservation_upper_bound", "cache_hit"] | None
    image_detail: Literal["low"]
    cache_hit: bool
    analysis_id: str = Field(min_length=1, max_length=80)
    status: str = Field(min_length=1, max_length=40, pattern=r"^[a-z_]+$")
    failure_code: str | None = Field(
        default=None, min_length=1, max_length=80, pattern=r"^[a-z0-9._:-]+$"
    )


class OpenAIUsageLedger(Protocol):
    def budget_summary(self, *, monthly_budget_usd: Decimal) -> OpenAIBudgetSummary: ...

    def get_cached(self, cache_key: str) -> OpenAIReviewStructuredOutput | None: ...

    def put_cached(
        self,
        cache_key: str,
        result: OpenAIReviewStructuredOutput,
        *,
        model: str,
        prompt_version: str,
        image_detail: Literal["low"],
        ttl_days: int,
    ) -> None: ...

    def reserve_call(
        self,
        *,
        analysis_id: str,
        model: str,
        prompt_version: str,
        image_detail: Literal["low"],
        reservation_cost_usd: Decimal,
        cost_estimate_version: str,
        monthly_budget_usd: Decimal,
        daily_call_limit: int,
        per_analysis_call_limit: int,
    ) -> OpenAICallReservation: ...

    def finalize_call(
        self,
        reservation_id: int,
        *,
        status: CallFinalStatus,
        usage: OpenAIReviewUsage | None,
        estimated_cost: Decimal | None,
        cost_estimate_version: str,
        failure_code: str | None = None,
    ) -> None: ...

    def record_cache_hit(
        self,
        *,
        analysis_id: str,
        model: str,
        prompt_version: str,
        image_detail: Literal["low"],
        cost_estimate_version: str,
    ) -> None: ...


class SQLiteOpenAIUsageLedger:
    """Local fail-closed budget ledger and structured-result cache."""

    def __init__(
        self,
        path: Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._path = path.expanduser().resolve()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = threading.RLock()
        self._initialize()

    def _initialize(self) -> None:
        if self._path.exists() and self._path.is_symlink():
            raise OSError("unsafe ledger path")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS openai_geo_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER,
                    estimated_cost_usd TEXT,
                    cost_estimate_version TEXT,
                    cost_basis TEXT,
                    image_detail TEXT NOT NULL,
                    cache_hit INTEGER NOT NULL CHECK (cache_hit IN (0, 1)),
                    analysis_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    failure_code TEXT
                );
                CREATE INDEX IF NOT EXISTS ix_openai_geo_usage_timestamp
                    ON openai_geo_usage(timestamp_utc);
                CREATE INDEX IF NOT EXISTS ix_openai_geo_usage_analysis
                    ON openai_geo_usage(analysis_id);
                CREATE TABLE IF NOT EXISTS openai_geo_cache (
                    cache_key TEXT PRIMARY KEY,
                    created_at_utc TEXT NOT NULL,
                    expires_at_utc TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    image_detail TEXT NOT NULL,
                    result_json TEXT NOT NULL
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None:
            raise ValueError("ledger clock must return a timezone-aware datetime")
        return now.astimezone(UTC)

    @staticmethod
    def _timestamp(value: datetime) -> str:
        return value.astimezone(UTC).isoformat(timespec="microseconds")

    @staticmethod
    def _period_starts(now: datetime) -> tuple[str, str]:
        day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        month = day.replace(day=1)
        return (
            SQLiteOpenAIUsageLedger._timestamp(day),
            SQLiteOpenAIUsageLedger._timestamp(month),
        )

    @staticmethod
    def _monthly_spend(connection: sqlite3.Connection, month_start: str) -> Decimal:
        rows = connection.execute(
            """
            SELECT estimated_cost_usd
            FROM openai_geo_usage
            WHERE timestamp_utc >= ? AND cache_hit = 0 AND estimated_cost_usd IS NOT NULL
            """,
            (month_start,),
        ).fetchall()
        spend = Decimal("0")
        for row in rows:
            try:
                spend += Decimal(str(row["estimated_cost_usd"]))
            except InvalidOperation as exc:
                raise sqlite3.DataError("invalid cost in local usage ledger") from exc
        return spend

    @staticmethod
    def _daily_calls(connection: sqlite3.Connection, day_start: str) -> int:
        row = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM openai_geo_usage
            WHERE timestamp_utc >= ? AND cache_hit = 0
            """,
            (day_start,),
        ).fetchone()
        return int(row["count"] if row is not None else 0)

    @classmethod
    def _summary(
        cls,
        connection: sqlite3.Connection,
        *,
        now: datetime,
        monthly_budget_usd: Decimal,
    ) -> OpenAIBudgetSummary:
        day_start, month_start = cls._period_starts(now)
        calls = cls._daily_calls(connection, day_start)
        spend = cls._monthly_spend(connection, month_start)
        return OpenAIBudgetSummary(
            cloud_calls_today=calls,
            estimated_spend_this_month_usd=spend,
            remaining_configured_budget_usd=max(Decimal("0"), monthly_budget_usd - spend),
            monthly_budget_usd=monthly_budget_usd,
        )

    def budget_summary(self, *, monthly_budget_usd: Decimal) -> OpenAIBudgetSummary:
        with self._lock, self._connect() as connection:
            return self._summary(
                connection,
                now=self._now(),
                monthly_budget_usd=monthly_budget_usd,
            )

    def reserve_call(
        self,
        *,
        analysis_id: str,
        model: str,
        prompt_version: str,
        image_detail: Literal["low"],
        reservation_cost_usd: Decimal,
        cost_estimate_version: str,
        monthly_budget_usd: Decimal,
        daily_call_limit: int,
        per_analysis_call_limit: int,
    ) -> OpenAICallReservation:
        now = self._now()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            summary = self._summary(
                connection,
                now=now,
                monthly_budget_usd=monthly_budget_usd,
            )
            analysis_calls = int(
                connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM openai_geo_usage
                    WHERE analysis_id = ? AND cache_hit = 0
                    """,
                    (analysis_id,),
                ).fetchone()["count"]
            )
            if analysis_calls >= per_analysis_call_limit:
                return OpenAICallReservation(
                    allowed=False,
                    reason_code="per_analysis_call_limit_reached",
                    summary=summary,
                )
            if summary.cloud_calls_today >= daily_call_limit:
                return OpenAICallReservation(
                    allowed=False,
                    reason_code="daily_call_limit_reached",
                    summary=summary,
                )
            if summary.estimated_spend_this_month_usd + reservation_cost_usd > monthly_budget_usd:
                return OpenAICallReservation(
                    allowed=False,
                    reason_code="monthly_budget_exhausted",
                    summary=summary,
                )
            cursor = connection.execute(
                """
                INSERT INTO openai_geo_usage(
                    timestamp_utc, model, prompt_version, estimated_cost_usd,
                    cost_estimate_version, cost_basis, image_detail, cache_hit,
                    analysis_id, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, 'reserved')
                """,
                (
                    self._timestamp(now),
                    model,
                    prompt_version,
                    str(reservation_cost_usd),
                    cost_estimate_version,
                    "reservation_upper_bound",
                    image_detail,
                    analysis_id,
                ),
            )
            if cursor.lastrowid is None:
                raise sqlite3.IntegrityError("usage reservation did not receive an identifier")
            reservation_id = int(cursor.lastrowid)
            updated = self._summary(
                connection,
                now=now,
                monthly_budget_usd=monthly_budget_usd,
            )
            return OpenAICallReservation(
                allowed=True,
                reservation_id=reservation_id,
                reason_code="reserved",
                summary=updated,
            )

    def finalize_call(
        self,
        reservation_id: int,
        *,
        status: CallFinalStatus,
        usage: OpenAIReviewUsage | None,
        estimated_cost: Decimal | None,
        cost_estimate_version: str,
        failure_code: str | None = None,
    ) -> None:
        cost_basis = "usage_based" if estimated_cost is not None else "reservation_upper_bound"
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT estimated_cost_usd FROM openai_geo_usage
                WHERE id = ? AND status = 'reserved'
                """,
                (reservation_id,),
            ).fetchone()
            if row is None:
                raise sqlite3.IntegrityError("usage reservation is missing or already finalized")
            retained_cost = str(estimated_cost) if estimated_cost is not None else row[0]
            cursor = connection.execute(
                """
                UPDATE openai_geo_usage
                SET input_tokens = ?, output_tokens = ?, total_tokens = ?,
                    estimated_cost_usd = ?, cost_estimate_version = ?, cost_basis = ?,
                    status = ?, failure_code = ?
                WHERE id = ? AND status = 'reserved'
                """,
                (
                    usage.input_tokens if usage is not None else None,
                    usage.output_tokens if usage is not None else None,
                    usage.total_tokens if usage is not None else None,
                    retained_cost,
                    cost_estimate_version,
                    cost_basis,
                    status,
                    failure_code,
                    reservation_id,
                ),
            )
            if cursor.rowcount != 1:
                raise sqlite3.IntegrityError("usage reservation could not be finalized")

    def record_cache_hit(
        self,
        *,
        analysis_id: str,
        model: str,
        prompt_version: str,
        image_detail: Literal["low"],
        cost_estimate_version: str,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO openai_geo_usage(
                    timestamp_utc, model, prompt_version, estimated_cost_usd,
                    cost_estimate_version, cost_basis, image_detail, cache_hit,
                    analysis_id, status
                ) VALUES (?, ?, ?, '0', ?, 'cache_hit', ?, 1, ?, 'cache_hit')
                """,
                (
                    self._timestamp(self._now()),
                    model,
                    prompt_version,
                    cost_estimate_version,
                    image_detail,
                    analysis_id,
                ),
            )

    def get_cached(self, cache_key: str) -> OpenAIReviewStructuredOutput | None:
        now = self._timestamp(self._now())
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM openai_geo_cache WHERE expires_at_utc <= ?", (now,))
            row = connection.execute(
                "SELECT result_json FROM openai_geo_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row is None:
                return None
            payload = str(row["result_json"])
            if len(payload.encode("utf-8")) > _MAX_CACHE_RESULT_BYTES:
                connection.execute("DELETE FROM openai_geo_cache WHERE cache_key = ?", (cache_key,))
                return None
            try:
                return OpenAIReviewStructuredOutput.model_validate_json(payload)
            except ValueError:
                connection.execute("DELETE FROM openai_geo_cache WHERE cache_key = ?", (cache_key,))
                return None

    def put_cached(
        self,
        cache_key: str,
        result: OpenAIReviewStructuredOutput,
        *,
        model: str,
        prompt_version: str,
        image_detail: Literal["low"],
        ttl_days: int,
    ) -> None:
        payload = result.model_dump_json()
        if len(payload.encode("utf-8")) > _MAX_CACHE_RESULT_BYTES:
            raise ValueError("structured review exceeds the cache bound")
        now = self._now()
        expires = now + timedelta(days=ttl_days)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO openai_geo_cache(
                    cache_key, created_at_utc, expires_at_utc, model,
                    prompt_version, image_detail, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    created_at_utc = excluded.created_at_utc,
                    expires_at_utc = excluded.expires_at_utc,
                    model = excluded.model,
                    prompt_version = excluded.prompt_version,
                    image_detail = excluded.image_detail,
                    result_json = excluded.result_json
                """,
                (
                    cache_key,
                    self._timestamp(now),
                    self._timestamp(expires),
                    model,
                    prompt_version,
                    image_detail,
                    payload,
                ),
            )

    def records(self) -> tuple[OpenAIUsageLedgerRecord, ...]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT timestamp_utc, model, prompt_version, input_tokens, output_tokens,
                       total_tokens, estimated_cost_usd, cost_estimate_version, cost_basis,
                       image_detail, cache_hit, analysis_id, status, failure_code
                FROM openai_geo_usage ORDER BY id
                """
            ).fetchall()
        return tuple(
            OpenAIUsageLedgerRecord(
                timestamp=datetime.fromisoformat(str(row["timestamp_utc"])),
                model=str(row["model"]),
                prompt_version=str(row["prompt_version"]),
                input_tokens=row["input_tokens"],
                output_tokens=row["output_tokens"],
                total_tokens=row["total_tokens"],
                estimated_cost=(
                    Decimal(str(row["estimated_cost_usd"]))
                    if row["estimated_cost_usd"] is not None
                    else None
                ),
                cost_estimate_version=row["cost_estimate_version"],
                cost_basis=row["cost_basis"],
                image_detail=row["image_detail"],
                cache_hit=bool(row["cache_hit"]),
                analysis_id=str(row["analysis_id"]),
                status=str(row["status"]),
                failure_code=row["failure_code"],
            )
            for row in rows
        )
