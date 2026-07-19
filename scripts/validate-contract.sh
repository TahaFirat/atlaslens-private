#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
web_dir="$repo_root/apps/web"
contract="$repo_root/packages/contracts/openapi.yaml"

command -v npm >/dev/null || { echo "npm is required on PATH." >&2; exit 1; }
[[ -d "$web_dir/node_modules" ]] || {
  echo "Frontend dependencies are missing. Run 'npm --prefix apps/web ci' first." >&2
  exit 1
}

temporary_output="$(mktemp "${TMPDIR:-/tmp}/atlaslens-openapi.XXXXXX.d.ts")"
trap 'rm -f "$temporary_output"' EXIT

(
  cd "$web_dir"
  npm exec -- openapi-typescript "$contract" --output "$temporary_output"
)
echo "OpenAPI contract parsed successfully."
