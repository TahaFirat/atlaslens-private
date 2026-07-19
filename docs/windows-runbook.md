# Windows development runbook

This runbook starts the existing AtlasLens API and web application on a local
Windows workstation. It does not install optional model/data artifacts unless an
operator runs their explicit management commands.

## First setup

From PowerShell at the repository root:

```powershell
Set-Location D:\geoSearch
Copy-Item .env.example .env
New-Item -ItemType Directory -Force C:\AtlasLensRuntime | Out-Null
$env:UV_PROJECT_ENVIRONMENT = "C:\AtlasLensRuntime\api-venv"
$env:UV_CACHE_DIR = "C:\AtlasLensRuntime\uv-cache"
uv sync --project .\services\api --frozen --all-groups
npm.cmd --prefix .\apps\web ci --no-audit --no-fund --cache C:\AtlasLensRuntime\npm-cache
```

Keep `.env` untracked. The safe default is `APP_ENV=production`, mock inference
off, operator-report API off, cloud analysis unconfigured, and temporary upload
retention off.

## Start both surfaces

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\dev.ps1 -RuntimeRoot C:\AtlasLensRuntime
```

The API is `http://127.0.0.1:8000`; the Vite client is
`http://127.0.0.1:5173`. The script applies current migrations, waits for
readiness/proxy identity, reports a child-process failure, and stops both children
when it exits.

For separate terminals:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-api.ps1 -RuntimeRoot C:\AtlasLensRuntime
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-web.ps1 -RuntimeRoot C:\AtlasLensRuntime
```

Docker is not required for the local SQLite path. Use `npm.cmd` when PowerShell
execution policy blocks `npm.ps1`.

`-RuntimeRoot` is optional; omitting it preserves the repository-local venv
fallback. The scripts launch `python -m alembic` and `python -m uvicorn`
through the selected Python rather than a stale console-script trampoline.
Startup also sets Hugging Face/Transformers/Datasets offline guards. It never
bootstraps model weights or datasets.

For an env-isolated startup/shutdown smoke that cannot read the repository
`.env`, use unused ports:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\dev.ps1 -RuntimeRoot C:\AtlasLensRuntime -IgnoreProjectEnv -SmokeTest -ApiPort 8105 -WebPort 5179
```

This waits for API, readiness and the Vite proxy, then uses the normal owned
process-tree cleanup. Existing healthy Phase 6B workers are retained. The
`-IgnoreProjectEnv` switch is for controlled validation only; ordinary
development uses the untracked project `.env`.

## Phase 6A SegFormer preparation

The API never downloads or loads the trusted training checkpoint at startup.
Inspect and prepare it explicitly from the repository root after staging the
reviewed NVIDIA base model locally:

```powershell
.\services\api\.venv\Scripts\python.exe .\scripts\inspect_segmentation_checkpoint.py .\last_checkpoint.pt
.\services\api\.venv\Scripts\python.exe .\scripts\prepare_segmentation_model.py .\last_checkpoint.pt --base-model-dir .\.local\models\base-segformer-b2-cityscapes --output .\.local\models\atlaslens-segformer-b2-v4
```

Then set the following values in the untracked `.env` and restart the API:

```text
PHASE6A_ENABLED=true
GEOCLIP_INTERNAL_TOP_K=50
GEOCLIP_CLUSTER_RADIUS_KM=40
SEGMENTATION_ENABLED=true
SEGMENTATION_MODEL_DIR=.local/models/atlaslens-segformer-b2-v4
SEGMENTATION_DEVICE=auto
REVERSE_GEOCODE_TOP_K=10
```

The current deployment selects EMA and has the reviewed exact 124-entry Mapillary
mapping. Restore/verify it without touching weights:

```powershell
.\services\api\.venv\Scripts\python.exe .\scripts\restore_mapillary_labels.py
```

The command verifies architecture, classifier count, inverse labels and a stable
safetensors digest. Exact base-model staging, GeoNames/OCR/index commands,
verification, and limitations are documented in
[`phase-6a-hybrid-pipeline.md`](phase-6a-hybrid-pipeline.md).

## Phase 6B local-worker diagnostics and smoke

No dataset is needed or permitted. Start with the offline diagnostic:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap_phase6b_models.ps1 -VerifyOnly
.\services\api\.venv\Scripts\atlas.exe diagnostics phase6b --json
```

Optional model preparation is an explicit network action. OSV-5M and PLONK need a
separate Python 3.10 installation; PaddleOCR uses an isolated Python 3.12
environment. These commands fetch pinned source/pretrained weights only and do not
make a provider operational unless its reviewed worker is bound and diagnostics
reports ready:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap_phase6b_models.ps1 -Models osv5m
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap_phase6b_models.ps1 -Models plonk
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap_phase6b_models.ps1 -Models paddleocr
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap_phase6b_models.ps1 -Models rapidocr
```

Start the isolated workers. The manager binds only loopback ports 8791-8793,
rejects foreign listeners and runs one real offline inference per selected worker
before returning success:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-phase6b-workers.ps1
.\services\api\.venv\Scripts\atlas.exe diagnostics phase6b --json
```

For a local smoke, provide a repository-owned or user-licensed image:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\smoke_phase6b.ps1 -Image C:\licensed\smoke.jpg
```

Stop only manager-owned identity-matched workers when the manual session ends:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\stop-phase6b-workers.ps1
```

The optional billable OpenAI smoke never runs automatically. Put
`OPENAI_API_KEY=` only in the untracked repository-root `.env`, enable the reviewed
OpenAI settings, then explicitly supply both flags:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\smoke_phase6b.ps1 -Image C:\licensed\smoke.jpg -UseOpenAI -CloudConsent
```

Keep `GPU_MAX_HEAVY_CONCURRENCY=1` and `GPU_MAX_RESIDENT_MODELS=1` on the inspected
8 GB GPU. For `isolated_worker_python_missing`, install the requested minor Python
without altering the API venv. For PaddleOCR, package presence is insufficient:
require reviewed weights plus real load/inference proof; RapidOCR likewise needs
its hash-bound real-inference receipt and `RAPIDOCR_DEVICE=cpu` for the reviewed
ONNX Runtime CPU receipt. The Paddle-disabled full HTTP fallback is accepted.
Only PLONK YFCC is currently accepted by
real inference; OSV/iNaturalist are prepared-only. Full setup, disk locations,
cost controls, evaluation and limitations are in
[`phase-6b-multimodel.md`](phase-6b-multimodel.md).

## Development simulation

The committed deterministic scenario is for UI/product development only. Enable
it explicitly in the untracked `.env`:

```text
APP_ENV=development
ENABLE_MOCK_INFERENCE=true
MOCK_SCENARIO=development_demo
MOCK_INFERENCE_MODE=primary
```

Restart the API. Every resulting analysis is classified `simulated` and visibly
watermarked. Production startup refuses this combination. Never use mock output
in evaluation, screenshots represented as real predictions, or accuracy claims.

## Local history and reports

History lists only current TTL-bound analysis records. Rerun works only for a
retained `local_only` source (`KEEP_UPLOADS=true`) and creates a new analysis;
otherwise the API returns a conflict. Cloud-assisted rerun always requires a new
upload and consent. Delete remains the supported early-erasure action.

For read-only local evaluation/model/dataset-QA dashboards:

```text
APP_ENV=development
OPERATOR_API_ENABLED=true
EVALUATION_REPORT_DIR=C:\private\atlaslens\reports\evaluations
DATASET_QA_REPORT_DIR=C:\private\atlaslens\reports\dataset-qa
VITE_ENABLE_OPERATOR_UI=true
```

Set `VITE_ENABLE_OPERATOR_UI=true` explicitly in the web process when these
operator workspaces are required. The switch shows navigation only; it cannot
bypass the API gate. Never expose this unauthenticated operator surface publicly.

## Verification

```powershell
# Frozen dependency restore plus the complete suite:
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\test.ps1 -RuntimeRoot C:\AtlasLensRuntime

# Already restored dependencies; Docker unavailable on this host:
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\test.ps1 -RuntimeRoot C:\AtlasLensRuntime -SkipInstall -SkipDockerValidation
```

For focused offline checks:

```powershell
cd services\api
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\mypy.exe src
.\.venv\Scripts\pytest.exe

cd ..\..\apps\web
npm.cmd run lint
npm.cmd run typecheck
npm.cmd test
npm.cmd run check:api
npm.cmd run build
```

Record Playwright, Docker/Compose, real map tiles, GPU, OCR, retrieval index, and
custom ONNX runtime as unavailable or not run when their actual prerequisites are
missing. Unit fixtures do not satisfy real-provider or production acceptance
gates.

The 2026-07-15 Phase 1 run restored `C:\AtlasLensRuntime\api-venv`,
`uv-cache` and `npm-cache` from frozen locks. Ruff, strict mypy, 624 backend
tests (3 skipped), frontend lint/typecheck/77 tests/build/OpenAPI, five
Playwright tests (3 conditional skips), Alembic `0008 (head)`, Phase 6B
diagnostics and deterministic startup/shutdown smoke passed. PaddleOCR (8793),
OSV-5M (8791), and PLONK YFCC (8792) passed health plus real inference using
existing artifacts. API 8000 and web 5173 remained ready; MegaLoc 8794,
hierarchical Phase 6C and the reference index remained disabled. Final free
space was C: 48.285 GiB and D: 10.368 GiB. No model/dataset download occurred.
Docker/Compose was the sole environment-specific test omission.

The source backup is
`C:\AtlasLensBackups\AtlasLens-source-prechange-20260715-171223.zip`.
The reviewed 484-file baseline is committed locally as
`c2bfc221bfc542ab94ef9e9e2d1900bd759b7895`. The authorized Git identity is
configured for this repository only; global Git configuration was not changed.
The Phase 1 runtime files are kept in the separate local
`build: restore relocatable Windows development runtime` commit. No remote,
push or pull request is implied. The pre-existing `asda.html` remains untracked
and outside the reviewed commits. Phase 2 has not started and cloud GPU has not
been provisioned.

According to the user, the external credential is stored only in ignored
`.env`; its value was not read or recorded during closure.

USER ACTION REQUIRED — external credential revocation cannot be verified locally.

## Short operator commands

```powershell
cd services\api
.\.venv\Scripts\atlas.exe models register-local --manifest C:\private\model\manifest.yaml
.\.venv\Scripts\atlas.exe models verify atlaslens-custom-geolocation
.\.venv\Scripts\atlas.exe benchmark compare --providers geoclip,atlaslens-custom-geolocation --manifest C:\licensed\eval\manifest.csv --asset-root C:\licensed\eval\images --allow-license CC0-1.0 --geoclip-report C:\private\reports\geoclip --custom-report C:\private\reports\custom --output C:\private\reports\comparison
.\.venv\Scripts\atlas.exe dataset qa --images C:\licensed\images --manifest C:\licensed\manifest.csv --output C:\private\reports\dataset-qa\run-1
```

Produce the GeoCLIP and custom reports independently with `atlas benchmark run`
on the same validated split before comparison.

## Related runbooks

- `custom-model-handoff.md`: strict local registration, verification, smoke test,
  evaluation, promotion, and removal.
- `dataset-qa.md`: read-only dataset diagnostics and reports.
- `evaluation-dashboard.md`: benchmark report catalog and interpretation.
- `phase-5b-licensed-reference-runbook.md`: reviewed reference-data workflow.
- `phase-6a-hybrid-pipeline.md`: checkpoint conversion, local scene provider,
  hybrid evidence, exact Windows commands, and preserved responsibility boundary.
- `phase-6b-multimodel.md`: multi-model diagnostics, isolated-worker seams,
  exact bootstrap/smoke/evaluation commands, OpenAI consent and cost controls.

Production authentication, distributed queues, observability, SLOs, release
automation, and fleet rollout are outside Phase 5C.
