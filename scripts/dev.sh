#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
api_dir="$repo_root/services/api"
web_dir="$repo_root/apps/web"
target="${1:-all}"
install="${INSTALL_DEPS:-0}"

if [[ "${2:-}" == "--install" || "${1:-}" == "--install" ]]; then
  install=1
  [[ "${1:-}" == "--install" ]] && target=all
fi

case "$target" in
  api|web|all|dependencies) ;;
  *) echo "Usage: bash scripts/dev.sh [api|web|all|dependencies] [--install]" >&2; exit 2 ;;
esac

if [[ "$target" == "dependencies" ]]; then
  command -v docker >/dev/null || { echo "docker is required on PATH." >&2; exit 1; }
  exec docker compose -f "$repo_root/docker-compose.yml" up -d db
fi

command -v uv >/dev/null || { echo "uv is required on PATH." >&2; exit 1; }
command -v npm >/dev/null || { echo "npm is required on PATH." >&2; exit 1; }

if [[ "$install" == "1" ]]; then
  uv sync --project "$api_dir" --frozen
  npm --prefix "$web_dir" ci --no-audit --no-fund
fi

migrate() {
  if [[ -f "$api_dir/alembic.ini" ]]; then
    uv run --project "$api_dir" alembic -c "$api_dir/alembic.ini" upgrade head
  fi
}

run_api() {
  migrate
  exec uv run --project "$api_dir" uvicorn atlaslens_api.main:app --host 127.0.0.1 --port 8000
}

run_web() {
  exec npm --prefix "$web_dir" run dev
}

case "$target" in
  api) run_api ;;
  web) run_web ;;
  all)
    migrate
    uv run --project "$api_dir" uvicorn atlaslens_api.main:app --host 127.0.0.1 --port 8000 &
    api_pid=$!
    trap 'kill "$api_pid" 2>/dev/null || true; wait "$api_pid" 2>/dev/null || true' EXIT INT TERM
    sleep 2
    if ! kill -0 "$api_pid" 2>/dev/null; then
      wait "$api_pid"
      exit $?
    fi
    npm --prefix "$web_dir" run dev
    ;;
esac
