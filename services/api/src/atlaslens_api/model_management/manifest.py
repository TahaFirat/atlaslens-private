from __future__ import annotations

import json
from pathlib import Path

from .models import EmbeddingModelManifest, ModelManifest


def load_manifest(path: Path) -> ModelManifest:
    return ModelManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))


def default_manifest_path() -> Path:
    return Path(__file__).resolve().parents[5] / "config" / "models" / "geoclip-1.2.0.json"


def load_embedding_manifest(path: Path) -> EmbeddingModelManifest:
    return EmbeddingModelManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))


def siglip2_manifest_path() -> Path:
    return Path(__file__).resolve().parents[5] / "config" / "models" / "siglip2-b16-384.json"
