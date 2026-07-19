from __future__ import annotations

import os
from pathlib import Path


def default_cache_root() -> Path:
    if configured := os.environ.get("ATLAS_MODEL_CACHE"):
        return Path(configured).expanduser()
    if local := os.environ.get("LOCALAPPDATA"):
        return Path(local) / "AtlasLens" / "models"
    return Path.home() / ".local" / "share" / "AtlasLens" / "models"
