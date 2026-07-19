#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
api_dir="$repo_root/services/api"
web_dir="$repo_root/apps/web"
skip_docker="${SKIP_DOCKER_VALIDATION:-0}"
skip_e2e="${SKIP_E2E:-0}"
skip_install="${SKIP_INSTALL:-0}"

command -v uv >/dev/null || { echo "uv is required on PATH." >&2; exit 1; }
command -v npm >/dev/null || { echo "npm is required on PATH." >&2; exit 1; }

if [[ "$skip_install" != "1" ]]; then
  uv sync --project "$api_dir" --frozen
  npm --prefix "$web_dir" ci --no-audit --no-fund
fi

(
  cd "$api_dir"
  uv run ruff check .
  uv run mypy
  uv run pytest
)

(
  cd "$web_dir"
  npm run lint
  npm run typecheck
  npm run test
  npm run build
  npm run check:api
  if [[ "$skip_e2e" != "1" && -f playwright.config.ts ]]; then
    npm run test:e2e
  fi
)

bash "$repo_root/scripts/validate-contract.sh"

if [[ "$skip_docker" != "1" ]]; then
  command -v docker >/dev/null || {
    echo "Docker is required for Compose validation. Set SKIP_DOCKER_VALIDATION=1 only when recording this environmental blocker." >&2
    exit 1
  }
  POSTGRES_PASSWORD=compose-validation-only docker compose -f "$repo_root/docker-compose.yml" config --quiet
fi

echo "All requested checks passed."
