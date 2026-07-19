#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
API="$ROOT/services/api"
CACHE_ROOT=${ATLAS_MODEL_CACHE:-"${XDG_DATA_HOME:-$HOME/.local/share}/AtlasLens/models"}
SMOKE_IMAGE=${1:-}
command -v uv >/dev/null 2>&1 || { echo "Required command 'uv' was not found." >&2; exit 1; }
[ -n "$SMOKE_IMAGE" ] && [ -f "$SMOKE_IMAGE" ] || { echo "Usage: scripts/setup-models.sh <licensed-smoke-image>" >&2; exit 1; }
AVAILABLE_KB=$(df -Pk "$(dirname "$CACHE_ROOT")" 2>/dev/null | awk 'NR==2 {print $4}')
[ -z "${AVAILABLE_KB:-}" ] || [ "$AVAILABLE_KB" -ge 4194304 ] || { echo "At least 4 GB free disk space is required." >&2; exit 1; }
uv sync --project "$API" --frozen
uv run --project "$API" python -m atlaslens_api.model_management.cli --cache-root "$CACHE_ROOT" install geoclip
uv run --project "$API" python -m atlaslens_api.model_management.cli --cache-root "$CACHE_ROOT" verify geoclip
uv run --project "$API" python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU fallback')"
uv run --project "$API" python -m atlaslens_api.model_management.cli --cache-root "$CACHE_ROOT" test geoclip --image "$SMOKE_IMAGE" --device auto
printf '%s\n' "Next: uv run --project services/api uvicorn atlaslens_api.main:app --host 127.0.0.1 --port 8000"
