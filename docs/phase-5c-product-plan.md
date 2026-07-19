# Phase 5C product-completion plan

Status: complete (verified 2026-07-12). This is a bounded Phase 5 extension while
the external custom model trains. Phase 6 is not authorized. The still-blocked Phase 5B real-artifact,
10,000-reference and measurable-improvement gates remain unchanged.

## Wave 1 reconciliation

| Area | Reuse | Phase 5C gap |
|---|---|---|
| Provider foundation | Typed descriptors, outcomes, invocation context, GeoCLIP injection | One canonical inference adapter/coordinator, deployment modes and lifecycle |
| GeoCLIP | Verified local/offline Top-K baseline | Preserve unchanged behind the canonical adapter |
| OCR/place | RapidOCR, redaction and forward GeoNames implementation | Runtime is unavailable on this host; expose honest provider state |
| Retrieval/reranking | SigLIP2/FAISS seams, Phase 5B adapters and deterministic reranker | Runtime artifacts/index remain unavailable; do not recreate these modules |
| Map | MapLibre, marker/uncertainty behavior and bounded OSM adapter | Better in-frame failure UX and product workspace; real basemap remains unverified |
| History | Individual TTL-bound analysis persistence and deletion | Safe list/filter/page and rerun-with-retained-source behavior |
| Evaluation | Licensed manifests, metrics, reports and five-mode comparison | Safe report catalog/API/dashboard and custom-provider comparison seam |
| Dataset QA | Manifest containment, hashes, provenance and evaluation-overlap checks | Aggregated image/mask/GPS/leakage/distribution QA plus reports |
| Custom model | Provider injection and safe model-manager patterns | Versioned artifact manager, adapter, shadow and fail-closed promotion |
| Frontend | Accessible EN/TR upload, results, diagnostics and map | Compact investigation workspace, history/evaluation/QA navigation and simulation marking |

The frontend audit also identified truthful-progress, false-precision, SSE fallback,
map-failure and dense-card issues. Those are repaired additively; the visual system
is not redesigned.

## Frozen boundaries

### Canonical inference domain

`GeolocationInferenceProvider` is an internal adapter contract. Its request binds a
normalized private image handle or immutable bytes, dimensions, optional EXIF,
analysis mode, requested Top-K, cancellation, deadline and a safe trace ID. Its
result binds provider/model/runtime revisions, classification, raw candidates,
explicit score semantics, calibration state, runtime, warnings, safe failure and
provenance. Providers never create final product confidence.

The existing `GlobalGeolocationProvider` and GeoCLIP implementation remain valid and
are wrapped rather than rewritten. `ProviderEnsembleCoordinator` preserves every
source result, invokes independent eligible providers concurrently, excludes mock
results from real fusion/evaluation and keeps shadow output from changing user
ranking.

### Development mock

- Disabled by default and allowed only for `APP_ENV=development` or tests.
- `APP_ENV=production` plus `ENABLE_MOCK_INFERENCE=true` fails configuration/startup.
- Fixtures live only under `services/api/dev_fixtures/inference` with containment
  checks and deterministic validation.
- Simulated analyses carry `result_classification=simulated`, a bilingual warning
  key and visible watermark. History retains the classification, not the source
  image.
- Benchmark/evaluation adapters reject simulated providers/results.

### Custom trained artifact

Use a separate `atlaslens-trained-artifact-v1` manager; do not generalize the
GeoCLIP manager. Registration is local-only and verifies canonical manifest data,
an allowlisted format/runtime adapter, exact file set, byte limits, SHA-256, model
and dataset lineage, preprocessing/output revisions and license approval before an
atomic private-cache receipt is written.

Arbitrary pickle, `.pt`, `.pth`, `.ckpt`, remote code and `trust_remote_code` are
rejected. Initial generic execution is an optional reviewed ONNX coordinate-output
adapter; safetensors is accepted only with a repository-known architecture adapter.
An absent runtime dependency or unknown architecture stays disabled with a safe
reason. No dummy artifact or fallback prediction is created.

Deployment modes are `disabled`, `shadow`, `candidate` and `primary`:

- unverified artifacts are `disabled`;
- verified artifacts enter `shadow` only when immutable-input execution is safe;
- `candidate` may participate only under the documented coordinator policy;
- `primary` requires an immutable promotion receipt binding artifact, paired
  benchmark identities, minimum sample/subgroup gates, latency/safety checks,
  compatible output schema and explicit operator approval.

Live unlabeled shadow data can measure runtime/failure/overlap only, never accuracy.
Calibration remains a separate compatible receipt.

### Analysis and history

Existing analysis endpoints and payloads remain compatible. Optional additive
analysis fields record real/simulated classification, simulation warning and safe
provider comparison. History adds a paginated safe summary endpoint and uses the
existing detail/delete endpoints. It stores only current TTL-bound result records,
safe image hash/dimensions, provider versions, summaries, runtime and warnings.
Original images, original filenames and raw OCR are not history data. Rerun returns
a clear conflict when the source was not explicitly retained.

### Evaluation and dataset QA

Evaluation and QA HTTP endpoints are read-only catalogs of prebuilt, schema-checked
reports under configured private roots. Production operator access is disabled
unless explicitly configured with authentication. User uploads never become
evaluation/training data.

Dataset QA is an isolated read-only package. A versioned policy produces stable
safe issues, distributions and a fingerprint, plus JSON/CSV/Markdown/static HTML
and an opt-in metadata-free contact sheet. Inputs and outputs are contained,
non-symlink roots. Reports use opaque asset keys and omit absolute paths, raw EXIF,
raw OCR and exact GPS. No source file is mutated; `--apply` is not implemented
until a separately reviewed repair policy exists.

### Additive public schema freeze

The shared contract may add only:

- optional analysis classification/simulation/comparison fields;
- `GET /api/v1/analyses` and `POST /api/v1/analyses/{id}/rerun`;
- `GET /api/v1/providers` and `GET /api/v1/models` safe status views;
- read-only `GET /api/v1/evaluations[/{id}]`;
- read-only `GET /api/v1/datasets/qa[/{id}]`.

Existing multipart fields, status/detail/delete/SSE endpoints, candidate semantics,
GeoCLIP diagnostics, Phase 4/5B assessments and generated-client drift checks stay
unchanged. Operator mutation stays CLI-only in Phase 5C.

## Implementation ownership

- Agent E: canonical provider adapters, mock, custom adapter, coordinator, shadow
  execution boundary, pipeline concurrency and backend-focused tests.
- Agent F: existing-design workspace, result hierarchy/map states, simulation
  watermark, history/evaluation/QA screens, localization and frontend tests.
- Agent G: evaluation report catalog, dataset QA CLI/scanner/reporters and focused
  tests.
- Main agent: public schemas/OpenAPI/generated types, repository/API integration,
  security boundaries, custom artifact registration/promotion CLI, migrations if
  required, conflict resolution, full verification, documentation and state.

Agents do not create subagents. Shared-file edits are sequenced or owned by the
main agent.

## Acceptance and stop conditions

Phase 5C is complete only if the preserved GeoCLIP/EXIF/delete flows, explicit mock
flow, production mock refusal, missing-custom-model behavior, history, report
dashboards, dataset QA fixtures, full backend/frontend quality chains, contracts,
migrations and security scans pass. Any unexecuted gate is reported as not run;
any unavailable external artifact stays unavailable. No Phase 5B accuracy claim is
made and Phase 6 work does not begin.

## Operator documentation

- `custom-model-handoff.md` defines the strict already-trained artifact handoff,
  verification, shadow/candidate/primary promotion and removal workflow.
- `dataset-qa.md` defines read-only image/mask/GPS/leakage diagnostics and the
  private report catalog.
- `evaluation-dashboard.md` defines accepted prebuilt benchmark reports, simulated
  report rejection and interpretation limits.
- `windows-runbook.md` defines local startup, simulation, operator-report and
  verification commands.

This plan remains `in progress` until the final quality and runtime gate record is
written. Documentation does not convert an unavailable dependency or fixture-backed
test into a passed real-provider gate.
