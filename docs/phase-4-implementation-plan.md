# Phase 4 implementation plan

Status: implemented and verified on 2026-07-11. The mandatory submission repair
gate passed before Phase 4 implementation began. The sections below preserve the
diagnosis, contract freeze, implementation policy, and verification order.

## Mandatory repair gate

Phase 4 implementation is gated on a verified analysis-submission repair. The
current backend and proxy were exercised on the canonical Windows ports on
2026-07-11:

- direct capabilities `:8000` returned `200`;
- proxied capabilities `:5173` returned `200`;
- direct multipart analysis submission returned `202`;
- proxied multipart analysis submission returned `202`;
- both generated no-EXIF JPEG jobs completed with quality output and honest
  `insufficient_geographic_evidence` abstention.

The server endpoint, multipart names and accepted-response schema are therefore
not the cause of the reported pre-request failure.

### Root cause

`apps/web/src/api/client.ts::createAnalysis` calls `crypto.randomUUID()` while
constructing the optional idempotency header, before `XMLHttpRequest.send()`.
Browsers or execution contexts without `randomUUID`, or a browser implementation
that throws from it, reject the Promise with a native exception before any POST
is emitted. `apps/web/src/App.tsx::messageCode` recognizes only `ApiError`, so the
native exception loses its category and becomes the generic Turkish
`errors.unknown` message.

The original user's historical console trace is unavailable, but this code path
deterministically reproduces the reported combination: no POST and a generic
submission error. Current Playwright Chromium supports `randomUUID`, which is
why the existing EXIF E2E passes.

### Repair

1. Generate an idempotency key only through guarded browser crypto. Use
   `randomUUID` when available, fall back to `getRandomValues`, and omit the
   optional header if secure randomness is unavailable.
2. Wrap FormData/XHR construction, `open`, headers and `send` so synchronous
   browser API failures become a typed, localized `ApiError`.
3. Centralize relative API URL construction and keep development on `/api`.
4. Preserve exact multipart fields:
   `image`, `analysis_mode`, `cloud_processing_consent`, and
   `authorization_acknowledged`. Never set multipart `Content-Type` manually.
5. Normalize fetch/JSON/schema failures, preserve safe API `code` and
   `request_id`, and distinguish submission, backend, HTTP, invalid response,
   job failure and SSE degradation.
6. Keep a successful `202` and its analysis ID when SSE fails. Polling remains
   the source-of-truth fallback and stale subscriptions are closed.
7. Reset transient progress between analyses and retain double-submit guards.
8. Add focused real-client tests for missing `randomUUID`, multipart fields,
   optional idempotency header, `202` parsing, structured errors and synchronous
   setup failure.
9. Extend E2E with canonical no-EXIF abstention, explicit `202`, POST presence,
   progress and deletion.
10. Add Windows start scripts that apply Alembic before API startup; the manual
    Uvicorn command remains documented but must be preceded by migration.

### Repair contract freeze

No repair-stage OpenAPI change is required. The existing POST path, multipart
field names, boolean/enum serialization, `202 AnalysisAccepted` body, status URL,
events URL, delete URL and problem-details schema remain frozen.

The gate passes only after direct and proxied POST, EXIF, no-EXIF abstention,
SSE/polling, backend-unavailable behavior and browser console/network checks are
verified.

## Phase 4 boundaries after the gate

Preserve `atlaslens_api.retrieval` and add isolated packages:

- `reranking`: candidate hypotheses, antimeridian-safe clustering, feature
  extraction, versioned relative scoring and deterministic tie-breaking;
- `constraints`: disabled-by-default map providers, observations, cache and safe
  offline fixtures;
- `verification`: opaque reference resolution and bounded OpenCV geometry;
- `candidate_pipeline`: time/candidate/reference budgets, partial failure,
  contradiction policy, telemetry and final assessment/abstention.

The domain types stay distinct: `RetrievalHit`, `CandidateHypothesis`,
`MapConstraintObservation`, `GeometricVerificationResult`, `RerankResult`, and
`FinalCandidateAssessment`. Retrieval similarity and relative rank are never
presented as calibrated probability or geographic proof.

## Planned implementation ownership

- Submission repair/API integration: frontend client, SSE/polling, Windows start
  scripts and repair tests only.
- Reranking/candidate pipeline: backend `reranking` and `candidate_pipeline`
  packages, versioned config and tests.
- Map/reference resolution: backend `constraints` and opaque reference resolver,
  offline fixtures and tests.
- Geometry/Phase 4 UI: backend `verification`, generated fixtures, existing-design
  UI panels/localization and frontend tests.
- Main agent: shared schemas/OpenAPI, migrations, integration, conflict
  resolution, adversarial review, full verification and project state.

## Planned Phase 4 policies

- Geographic clustering uses deterministic geodesic distance, unit-vector
  centers, positive dispersion-based uncertainty, stable ties and source/hash
  diversity controls.
- Reranking config is versioned as `phase4-v1`; only available real features are
  scored. Raw values and weighted contributions remain separate.
- Map checks are optional and conservative. Disabled and offline-fixture adapters
  are implemented now; network/local-PBF adapters remain bounded interfaces.
- Reference assets use opaque keys resolved beneath a configured licensed root.
  Paths, filenames, credentials, signed URLs, blobs and descriptors never enter
  API responses or logs. Display defaults to denied.
- Geometry uses a bounded deterministic OpenCV CPU baseline with feature limits,
  robust fitting, degeneracy/coverage checks and `inconclusive` as the normal
  weak-signal outcome.
- The final pipeline bounds candidates, references and time; propagates
  cancellation; tolerates optional-provider failure; and abstains when evidence
  is insufficient or contradictory.

## Verification order

1. Repair unit/integration tests and canonical live direct/proxy flow.
2. Existing Phase 1-3 regression suite.
3. Clustering and reranking tests.
4. Constraints/cache/reference resolver tests.
5. Generated-fixture geometry tests.
6. Candidate pipeline and API contract tests.
7. Frontend/localization/accessibility/mobile tests.
8. Fresh and `0002` migration upgrades, CLI, security scans and Windows scripts.
9. Browser Network/console inspection and Playwright E2E.
10. Documentation and `PROJECT_STATE.md` only after observed results.

Phase 5 benchmarking and calibration and Phase 6 production scaling remain out
of scope.
