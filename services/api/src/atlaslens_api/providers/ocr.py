from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from atlaslens_api.place_evidence.normalization import sanitize_unicode
from atlaslens_api.providers.base import (
    InvocationContext,
    OCRResult,
    ProviderDescriptor,
    ProviderOutcome,
)
from atlaslens_api.storage import LocalImageHandle

_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d ().-]{6,}\d)(?!\w)")
_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_LONG_ID = re.compile(r"(?<!\w)\d{6,20}(?!\w)")
_MIXED_IDENTIFIER = re.compile(
    r"\b(?=[A-Z0-9-]{5,16}\b)(?=[A-Z0-9-]*[A-Z])(?=[A-Z0-9-]*\d)[A-Z0-9-]+\b",
    re.IGNORECASE,
)
_TURKISH_PLATE = re.compile(r"(?<!\w)\d{2}\s?[A-Z]{1,3}\s?\d{2,4}(?!\w)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class RedactedOCRText:
    text: str
    sensitive: bool


def redact_sensitive_text(raw: str) -> RedactedOCRText:
    text = sanitize_unicode(raw, max_length=512)
    sensitive = False
    for pattern, replacement in (
        (_EMAIL, "[redacted-email]"),
        (_PHONE, "[redacted-phone]"),
        (_URL, "[redacted-url]"),
        (_TURKISH_PLATE, "[redacted-plate]"),
        (_LONG_ID, "[redacted-identifier]"),
        (_MIXED_IDENTIFIER, "[redacted-identifier]"),
    ):
        text, substitutions = pattern.subn(replacement, text)
        sensitive = sensitive or substitutions > 0
    return RedactedOCRText(text=" ".join(text.split())[:240], sensitive=sensitive)


def redact_ocr_text(raw: str) -> list[str]:
    snippets: list[str] = []
    for line in raw.splitlines():
        redacted = redact_sensitive_text(line).text
        if redacted:
            snippets.append(redacted[:160])
        if len(snippets) == 5:
            break
    return snippets


def _discover(command: str | None) -> str | None:
    if command:
        path = Path(command).expanduser()
        if path.is_file():
            return str(path.resolve())
    return shutil.which("tesseract")


def _run_tesseract(command: str, path: Path) -> str:
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    completed = subprocess.run(  # noqa: S603 - command is discovered/configured server-side
        [command, str(path), "stdout", "--psm", "6"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
        check=False,
        creationflags=creation_flags,
    )
    if completed.returncode != 0:
        raise RuntimeError("OCR process failed")
    return completed.stdout


class TesseractOCRProvider:
    def __init__(self, *, enabled: bool, command: str | None) -> None:
        self._command = _discover(command) if enabled else None
        reason = None if self._command else ("disabled" if not enabled else "missing_dependency")
        self.descriptor = ProviderDescriptor(
            id="tesseract-ocr",
            kind="ocr",
            version="1.0.0",
            execution_boundary="local",
            criticality="optional",
            available=self._command is not None,
            unavailable_reason_code=reason,
        )

    async def extract(
        self, handle: LocalImageHandle, context: InvocationContext
    ) -> ProviderOutcome[OCRResult]:
        if self._command is None:
            return ProviderOutcome.skipped(self.descriptor.unavailable_reason_code or "unavailable")
        if context.cancellation.is_set():
            return ProviderOutcome.skipped("unavailable")
        started = time.monotonic()
        try:
            raw = await asyncio.to_thread(_run_tesseract, self._command, handle.path)
            snippets = redact_ocr_text(raw)
            del raw
        except subprocess.TimeoutExpired:
            return ProviderOutcome.failed(
                "timeout",
                retryable=False,
                attempts=1,
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        except (OSError, RuntimeError):
            return ProviderOutcome.failed(
                "internal_provider_error",
                retryable=False,
                attempts=1,
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        if not snippets:
            return ProviderOutcome.abstained()
        return ProviderOutcome.succeeded(OCRResult(redacted_snippets=snippets))
