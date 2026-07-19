# Phase 2 implementation plan

Status: repair gate in progress; provider/API contract will be frozen after the
direct and proxied capability checks pass.

## Gate 1 — Canonical local connectivity repair

Observed on 2026-07-10:

- Port `8000` was owned by a foreign Uvicorn application identifying itself as
  `NexusOSINT`; its `/api/v1/capabilities` returned `404`.
- Port `5173` was owned by a different Vite application, not AtlasLens.
- The running Vite proxy forwarded `/api` to the foreign service on `8000`, so
  direct and proxied requests returned the same Uvicorn `{"detail":"Not Found"}`.
- AtlasLens itself returned valid capability JSON through its existing proxy
  when run on free diagnostic ports `8100/5174`.

The permanent repair is defense in depth:

1. Root launchers preflight both configured ports and fail with PID/service
   guidance instead of attaching to unrelated processes.
2. API startup is accepted only after the AtlasLens health schema and direct
   capability response validate.
3. Vite loads proxy target and ports from the root environment explicitly;
   same-origin development keeps `VITE_API_BASE_URL` empty.
4. Launcher prints canonical URLs, checks the proxied capability resource, and
   cleans up child processes.
5. Proxy failures become a safe `502`; frontend differentiates unreachable,
   API 404, server error, and invalid response.
6. Unknown API routes return safe problem JSON and never SPA HTML.
7. A project favicon removes unrelated console noise.

Phase 2 implementation cannot begin until both
`127.0.0.1:8000/api/v1/capabilities` and
`127.0.0.1:5173/api/v1/capabilities` return valid AtlasLens JSON.

## Phase 2 boundary

Phase 2 adds observable, lazy, failure-isolated evidence providers and broad
global-baseline plumbing. It does not add licensed reference-image retrieval,
global image indexing, map constraints, spatial reranking, or geometric
verification. It never downloads a model at ordinary startup and never claims
an unavailable provider is operational.

## Provider architecture

- Extend existing protocols and outcomes; do not replace the Phase 1 pipeline.
- Add `ProviderRegistry`, `ProviderHealthService`, `ConfidencePolicy`,
  `ModelCacheManager`, and `AnalysisTelemetryService`.
- Registry entries are factories. Optional heavyweight providers have explicit
  `disabled`, `missing_dependency`, `missing_artifact`, `not_loaded`, `loading`,
  `ready`, and `failed` load states and single-flight initialization.
- Health/readiness inspect metadata only; they never import Torch, hash large
  artifacts, download weights, or initialize a model.
- Providers emit normalized evidence/hypotheses. Only centralized fusion creates
  final candidates or persists results.

## Local OCR

Keep Tesseract as the least-disruptive local adapter. Use its TSV output for
word boxes and engine confidence, normalize boxes to `[0,1]`, group blocks,
optionally consume OSD/script data, and redact emails, phones, long identifiers,
URLs, and likely complete plates before output crosses the adapter boundary.
Raw text/TSV is ephemeral and never logged or persisted. Tesseract remains
honestly unavailable on this Windows host until an operator installs a reviewed
binary and language data.

## Visual clues

Add an extensible structured clue schema and versioned label configuration for
script, language hints, driving side, road/sign/architecture/terrain/climate/
vegetation/utility/vehicle/plate-category/landmark/settlement observations and
contradictions. Preserve the optional consent-gated OpenAI structured provider.
A local zero-shot clue adapter is registered but disabled unless reviewed model
dependencies and artifacts are explicitly installed; it never emits placeholder
inference.

## Optional global baseline

GeoCLIP is represented in a versioned model manifest and an admin-only
list/install/verify CLI, but remains disabled by default. The official code is
MIT; a separately explicit production license for released weights/gallery was
not established during research, and the official constructor can implicitly
download a large CLIP backbone. Therefore:

- No artifact downloads occur in this task.
- `license_review_status=required` and operational status is false.
- Runtime uses offline/local-files-only semantics when an implementation is
  later enabled.
- Cache installation stages, size-bounds, hashes, and atomically promotes only
  verified official artifacts.
- The API never exposes the cache path or username.
- CUDA is opt-in; CPU fallback is policy-controlled; local model concurrency
  defaults to one.

## Scores, fusion, and abstention

Phase 2 freezes separate fields for raw provider score, score kind, calibration
state, nullable calibrated/display confidence, candidate classification, and
contradictions. Gallery softmax or similarity is never formatted as a
probability. Policy `phase2-fusion-v1` remains deterministic and geodesic:

- EXIF remains `metadata_exact`, rank-first, and not scene verification.
- OCR and quality never create coordinates alone.
- Model-only candidates are `model_broad`, unverified, and use at least a
  conservative `750 km` radius before calibration.
- Independent agreement may change rank/classification but cannot numerically
  inflate confidence or shrink below contributing uncertainty floors.
- Contradictions penalize/suppress; poor quality can gate inference; unsupported
  or weak signals abstain with a stable explanation.

## Shared contract extensions

Preserve all Phase 1 fields and add documented optional Phase 2 fields plus:

- Detailed provider and model resource endpoints.
- Provider descriptor/health/load/device/license/install metadata.
- OCR blocks with normalized boxes and redaction metadata.
- Structured visual clues and optional regions.
- Provider diagnostics/timings without secrets or paths.
- Candidate raw score, score kind, calibration state, classification, and
  contradiction codes.

OpenAPI is the source of truth; frontend types and runtime validation regenerate
in the same change. Unknown future evidence/provider fields render safely.

## Wave 2 ownership after repair gate

| Owner | Exclusive write scope | Deliverable |
|---|---|---|
| Backend agent | `services/api/**` | Registry/health/cache/CLI, OCR TSV, clue/global disabled adapters, telemetry, fusion and tests |
| Frontend agent | `apps/web/**` | Provider/evidence/overlay/candidate/diagnostic UX, connectivity errors and tests |
| DX agent | `infra/**`, `scripts/**`, `.github/**` | Canonical launchers, script/proxy/CI/model-command validation |
| Delivery lead | Root, `packages/contracts/**`, `docs/**` | Repair, frozen contract, integration, root config, final verification/state |

## Verification

Run repair probes, Phase 1 regression suites, new provider/fusion/frontend tests,
real-stack EXIF/no-EXIF/proxy E2E, lint/type/build, OpenAPI and generated-type
drift, model-manifest validation, script syntax, secret/fake/random scans, and
browser console/mobile checks. Docker is reported `unavailable` when the command
is absent; it is not required for ordinary local development.

