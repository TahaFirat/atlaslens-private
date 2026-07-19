from __future__ import annotations


class CorpusPipelineError(RuntimeError):
    """Base error whose message is a stable, non-sensitive reason code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ManifestValidationError(CorpusPipelineError):
    pass


class RightsValidationError(CorpusPipelineError):
    pass


class AssetSafetyError(CorpusPipelineError):
    pass


class LeakageValidationError(CorpusPipelineError):
    pass


class CheckpointError(CorpusPipelineError):
    pass


class CheckpointCompatibilityError(CheckpointError):
    pass


class PipelineInterrupted(CorpusPipelineError):
    pass


__all__ = [
    "AssetSafetyError",
    "CheckpointCompatibilityError",
    "CheckpointError",
    "CorpusPipelineError",
    "LeakageValidationError",
    "ManifestValidationError",
    "PipelineInterrupted",
    "RightsValidationError",
]
