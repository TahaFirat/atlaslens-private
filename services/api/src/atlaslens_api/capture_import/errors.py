from __future__ import annotations

from atlaslens_api.corpus_pipeline import CorpusPipelineError

FFMPEG_NOT_AVAILABLE_MESSAGE = (
    "FFMPEG_NOT_AVAILABLE — provide an explicit existing safe executable path; "
    "AtlasLens will not download FFmpeg."
)


class CaptureImportError(CorpusPipelineError):
    """Safe capture error with a stable machine code and optional fixed message."""

    def __init__(self, code: str, *, message: str | None = None) -> None:
        super().__init__(code)
        if message is not None:
            self.args = (message,)


def ffmpeg_not_available() -> CaptureImportError:
    return CaptureImportError(
        "ffmpeg_not_available",
        message=FFMPEG_NOT_AVAILABLE_MESSAGE,
    )


__all__ = [
    "FFMPEG_NOT_AVAILABLE_MESSAGE",
    "CaptureImportError",
    "ffmpeg_not_available",
]
