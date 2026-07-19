from __future__ import annotations

import asyncio
import time
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from atlaslens_api.image_processing import CloudSafeDerivative
from atlaslens_api.providers.base import (
    InvocationContext,
    ProviderDescriptor,
    ProviderOutcome,
    VisionClueResult,
    VisionHypothesis,
)
from atlaslens_api.schemas import AnalysisMode


class _StructuredHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=120)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    radius_km: float = Field(gt=0)
    confidence: float = Field(ge=0, le=1)
    granularity: str
    country_code: str | None = None
    visible_clues: list[str] = Field(min_length=1, max_length=6)


class _StructuredVisionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    abstained: bool
    hypotheses: list[_StructuredHypothesis] = Field(max_length=5)


_SYSTEM_PROMPT = (
    "Analyze only visible, non-personal geographic clues in the supplied image. "
    "Return at most five broad location hypotheses or abstain. Do not identify people, "
    "infer a private home address, or provide hidden reasoning. Use broad uncertainty."
)


class OpenAIVisionClueProvider:
    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        timeout_seconds: float,
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._client = (
            client if client is not None else (AsyncOpenAI(api_key=api_key) if api_key else None)
        )
        self.descriptor = ProviderDescriptor(
            id="openai-vision-clues",
            kind="vision_language",
            version="1.0.0",
            execution_boundary="cloud",
            criticality="optional",
            available=self._client is not None,
            unavailable_reason_code=None if self._client is not None else "missing_secret",
            model_name=model,
        )

    async def extract(
        self, derivative: CloudSafeDerivative, context: InvocationContext
    ) -> ProviderOutcome[VisionClueResult]:
        if context.mode != AnalysisMode.CLOUD_ASSISTED or not context.cloud_consent:
            return ProviderOutcome.skipped("privacy_denied")
        if self._client is None:
            return ProviderOutcome.skipped("missing_secret")

        started = time.monotonic()
        attempts = 0
        while attempts < 3:
            attempts += 1
            try:
                response = await asyncio.wait_for(
                    self._client.responses.parse(
                        model=self._model,
                        input=[
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": (
                                            "Report broad visible geographic clues and hypotheses."
                                        ),
                                    },
                                    {
                                        "type": "input_image",
                                        "image_url": derivative.data_url,
                                        "detail": "low",
                                    },
                                ],
                            },
                        ],
                        text_format=_StructuredVisionOutput,
                        max_output_tokens=1000,
                    ),
                    timeout=self._timeout_seconds,
                )
                parsed = _StructuredVisionOutput.model_validate(response.output_parsed)
                if parsed.abstained or not parsed.hypotheses:
                    return ProviderOutcome.abstained()
                hypotheses = [self._sanitize(item) for item in parsed.hypotheses]
                return ProviderOutcome.succeeded(VisionClueResult(hypotheses=hypotheses))
            except (ValidationError, AttributeError, TypeError, ValueError):
                return ProviderOutcome.failed(
                    "invalid_output",
                    retryable=False,
                    attempts=attempts,
                    duration_ms=int((time.monotonic() - started) * 1000),
                )
            except TimeoutError:
                transient = True
                failure_code = "timeout"
            except Exception as exc:  # SDK errors are sanitized below and never returned verbatim.
                transient, failure_code = self._classify_error(exc)
                if not transient:
                    return ProviderOutcome.failed(
                        failure_code,
                        retryable=False,
                        attempts=attempts,
                        duration_ms=int((time.monotonic() - started) * 1000),
                    )
            if not transient or attempts >= 3:
                return ProviderOutcome.failed(
                    failure_code,
                    retryable=transient,
                    attempts=attempts,
                    duration_ms=int((time.monotonic() - started) * 1000),
                )
            await asyncio.sleep(0.25 * (2 ** (attempts - 1)))

        return ProviderOutcome.failed(
            "transient_upstream",
            retryable=True,
            attempts=attempts,
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    @staticmethod
    def _sanitize(item: _StructuredHypothesis) -> VisionHypothesis:
        granularity = (
            item.granularity if item.granularity in {"city", "region", "country"} else "broad_area"
        )
        raw_country_code = (item.country_code or "").upper()
        country_code: str | None = (
            raw_country_code if len(raw_country_code) == 2 and raw_country_code.isalpha() else None
        )
        return VisionHypothesis(
            label=" ".join(item.label.split())[:120],
            latitude=item.latitude,
            longitude=item.longitude,
            radius_km=max(item.radius_km, 25.0),
            confidence=min(item.confidence, 0.40),
            granularity=granularity,
            country_code=country_code,
            visible_clues=[" ".join(clue.split())[:120] for clue in item.visible_clues[:6]],
        )

    @staticmethod
    def _classify_error(exc: Exception) -> tuple[bool, str]:
        name = type(exc).__name__.lower()
        status = getattr(exc, "status_code", None)
        if status in (401, 403) or "authentication" in name or "permission" in name:
            return False, "upstream_auth"
        if status == 429 or "ratelimit" in name:
            return True, "rate_limited"
        if status is not None and int(status) >= 500:
            return True, "transient_upstream"
        if "timeout" in name or "connection" in name:
            return True, "transient_upstream"
        return False, "internal_provider_error"
